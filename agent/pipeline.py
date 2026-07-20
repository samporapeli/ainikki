"""
Orchestrates the full pipeline: Collect -> Dedup -> Cluster -> Score -> Enrich ->
Compose -> Overview -> Validate -> Write.

CLI example:
    ./agent run --topic ai --since 2026-07-16 --until 2026-07-16

run_pipeline() accepts raw_items_override/llm_client/enrich_client
parameters for testability (see tests/test_pipeline.py) — with these,
the entire pipeline can be run with mock data without network access.
In production these are left as None and the code makes real HTTP calls.
"""

import argparse
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import httpx

from agent.schema import Period, RawItem
from agent.collect.hn import fetch_hn
from agent.collect.base import save_raw
from agent.dedup import dedup_candidates, save_candidates
from agent.cluster import cluster_candidates
from agent.score import score_clusters, load_rubric, ScoreValidationError
from agent.enrich import enrich_candidates
from agent.compose import compose_items
from agent.overview import generate_overview
from agent.validate import assemble_briefing, EmptyBriefingError
from agent.write import write_briefing, WriteRoundtripError
from agent.llm import make_llm_call, load_models_config, resolve_step_config

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"
LLM_STEPS = ["cluster", "score", "compose", "overview"]


class ConfigPaths:
    def __init__(self, config_dir: Path, topic: str):
        self.models = config_dir / "models.yaml"
        self.persona = config_dir / "personas" / "ainikki_v1.yaml"
        self.guardrails = config_dir / "guardrails" / "guardrails_v1.yaml"
        self.golden_examples_dir = config_dir / "golden_examples"
        self.rubric = config_dir / "rubrics" / f"{topic}_scoring_rubric_v1.yaml"


def _resolve_llm_call(step: str, models_config: dict, config_paths: ConfigPaths,
                       overrides: dict[str, tuple[str | None, str | None]],
                       client: httpx.Client | None, models_used_out: dict[str, str]):
    """Builds an LlmCall function for a step AND records which model was actually
    used into models_used_out dict (for GenerationMeta traceability)."""
    provider_override, model_override = overrides.get(step, (None, None))
    cfg = resolve_step_config(step, models_config, model_override, provider_override)
    models_used_out[step] = f"{cfg.provider}/{cfg.model}"
    return make_llm_call(step, config_path=config_paths.models,
                          override_model=model_override, override_provider=provider_override,
                          client=client)


def run_pipeline(topic: str, period: Period, since: datetime, until: datetime,
                  config_dir: Path = Path("config"), data_dir: Path = Path("data"),
                  out_dir: Path | None = None, min_points: int = 20,
                  display_date: date | None = None,
                  model_overrides: dict[str, tuple[str | None, str | None]] | None = None,
                  raw_items_override: list[RawItem] | None = None,
                  llm_client: httpx.Client | None = None,
                  enrich_client: httpx.Client | None = None) -> Path:
    model_overrides = model_overrides or {}
    out_dir = out_dir or (data_dir / "output")
    config_paths = ConfigPaths(config_dir, topic)
    date_str = since.date().isoformat()
    all_warnings: list[str] = []
    dropped_stories: list = []

    # 1. Collect
    if raw_items_override is not None:
        raw_items = raw_items_override
        logger.info("collect: using raw_items_override (%d items) - test run", len(raw_items))
    else:
        try:
            raw_items = fetch_hn(since, until, min_points=min_points)
        except Exception as e:
            logger.warning("collect: HN adapter failed (%s) - continuing with empty list", e)
            raw_items = []
            all_warnings.append(f"Keruu: HN-adapteri epäonnistui: {e}")
    save_raw(topic, date_str, "hn", raw_items, data_dir=data_dir / "raw")
    logger.info("collect: %d raw items", len(raw_items))

    # 2. Dedup
    candidates = dedup_candidates(raw_items)
    save_candidates(topic, date_str, candidates, data_dir=data_dir / "dedup")
    logger.info("dedup: %d -> %d candidates", len(raw_items), len(candidates))

    models_config = load_models_config(config_paths.models)
    models_used: dict[str, str] = {}

    # 3. Cluster
    llm_cluster = _resolve_llm_call("cluster", models_config, config_paths, model_overrides,
                                     llm_client, models_used)
    cluster_result = cluster_candidates(candidates, llm_cluster)
    if cluster_result.warning:
        all_warnings.append(cluster_result.warning)
    logger.info("cluster: %d -> %d clusters", len(candidates), len(cluster_result.clusters))

    # 4. Score (CRITICAL - no fallback, ScoreValidationError crashes the entire run)
    llm_score = _resolve_llm_call("score", models_config, config_paths, model_overrides,
                                   llm_client, models_used)
    rubric = load_rubric(config_paths.rubric)
    scored = score_clusters(cluster_result.clusters, config_paths.rubric, llm_score)
    logger.info("score: %d selected (rubric %s)", len(scored), rubric.get("version"))

    # 5. Enrich
    owns_enrich_client = enrich_client is None
    ec = enrich_client or httpx.Client(timeout=15.0, follow_redirects=True)
    try:
        enrich_result = enrich_candidates(scored, client=ec)
        all_warnings.extend(enrich_result.warnings)
        dropped_stories = enrich_result.dropped_stories
        logger.info("enrich: %d -> %d items (content fetched successfully)",
                    len(scored), len(enrich_result.items))
    finally:
        if owns_enrich_client:
            ec.close()

    # 6. Compose
    llm_compose = _resolve_llm_call("compose", models_config, config_paths, model_overrides,
                                     llm_client, models_used)
    compose_result = compose_items(enrich_result.items, config_paths.persona,
                                    config_paths.guardrails, llm_compose,
                                    golden_examples_dir=config_paths.golden_examples_dir)
    all_warnings.extend(compose_result.warnings)
    logger.info("compose: %d items written", len(compose_result.items))

    # 7. Overview
    llm_overview = _resolve_llm_call("overview", models_config, config_paths, model_overrides,
                                      llm_client, models_used)
    overview_result = generate_overview(compose_result.items, llm_overview, topic)

    # 8. Validate (CRITICAL - EmptyBriefingError if nothing remains)
    briefing = assemble_briefing(
        topic=topic, period=period, period_start=since.date(),
        period_end=(until - timedelta(seconds=1)).date(),
        display_date=display_date,
        overview_result=overview_result, compose_result=compose_result,
        models_used=models_used, pipeline_version=PIPELINE_VERSION,
        rubric_version=str(rubric.get("version", "unknown")),
        dropped_stories=dropped_stories,
        extra_warnings=all_warnings,
    )

    # 9. Write
    path = write_briefing(briefing, output_dir=out_dir)
    logger.info("write: written %s (%d warnings)", path, len(briefing.warnings))
    return path


def _parse_model_overrides(raw_overrides: list[str]) -> dict[str, tuple[str | None, str | None]]:
    """Parses --model-override step=provider:model CLI arguments."""
    result: dict[str, tuple[str | None, str | None]] = {}
    for raw in raw_overrides:
        try:
            step, rest = raw.split("=", 1)
            provider, model = rest.split(":", 1)
        except ValueError:
            raise SystemExit(
                f"Invalid --model-override '{raw}' - expected format: step=provider:model"
            )
        result[step.strip()] = (provider.strip(), model.strip())
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AI news digest agent")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--period", choices=[p.value for p in Period], default="daily")
    parser.add_argument("--since", required=True, help="ISO date, e.g. 2026-07-16")
    parser.add_argument("--until", required=True, help="ISO date (inclusive)")
    parser.add_argument("--config-dir", default="config", type=Path)
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default=None, type=Path)
    parser.add_argument("--min-points", type=int, default=20)
    parser.add_argument("--model-override", action="append", default=[],
                         help="step=provider:model, e.g. score=openai:gpt-4o-mini. Can be specified multiple times.")
    parser.add_argument("--display-date", default=None,
                         help="Date shown to the user (default: same as --since)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(levelname)s %(name)s: %(message)s")

    since_date = date.fromisoformat(args.since)
    until_date = date.fromisoformat(args.until)
    since_dt = datetime.combine(since_date, time.min, tzinfo=timezone.utc)
    until_dt = datetime.combine(until_date + timedelta(days=1), time.min, tzinfo=timezone.utc)

    model_overrides = _parse_model_overrides(args.model_override)
    display_date = date.fromisoformat(args.display_date) if args.display_date else None

    try:
        path = run_pipeline(
            topic=args.topic, period=Period(args.period), since=since_dt, until=until_dt,
            config_dir=args.config_dir, data_dir=args.data_dir, out_dir=args.out_dir,
            min_points=args.min_points, display_date=display_date,
            model_overrides=model_overrides,
        )
    except ScoreValidationError as e:
        print(f"ERROR (score-step, critical): {e}", file=sys.stderr)
        sys.exit(1)
    except EmptyBriefingError as e:
        print(f"ERROR (no publishable data): {e}", file=sys.stderr)
        sys.exit(1)
    except WriteRoundtripError as e:
        print(f"ERROR (write failed): {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Written: {path}")


if __name__ == "__main__":
    main()
