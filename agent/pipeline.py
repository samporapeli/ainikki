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
import yaml

from agent.schema import Period, RawItem
from agent.collect.hn import fetch_hn
from agent.collect.rss import fetch_and_parse_rss
from agent.collect.base import save_raw
from agent.dedup import dedup_candidates, save_candidates
from agent.cluster import cluster_candidates
from agent.filter_topic import filter_topic
from agent.score import (score_clusters, load_rubric, load_previous_stories,
                          filter_previous_clusters, ScoreValidationError, ScoreResult)
from agent.enrich import enrich_candidates
from agent.compose import compose_items
from agent.overview import generate_overview
from agent.validate import assemble_briefing, EmptyBriefingError
from agent.write import write_briefing, WriteRoundtripError
from agent.llm import make_llm_call, load_models_config, resolve_step_config

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"
LLM_STEPS = ["cluster", "filter_topic", "score", "compose", "overview"]


class ConfigPaths:
    def __init__(self, config_dir: Path, topic: str):
        self.models = config_dir / "models.yaml"
        self.persona = config_dir / "personas" / "ainikki_v1.yaml"
        self.guardrails = config_dir / "guardrails" / "guardrails_v1.yaml"
        self.golden_examples_dir = config_dir / "golden_examples"
        self.rubric = config_dir / "rubrics" / f"{topic}_scoring_rubric_v1.yaml"


def _load_sources_config(topic: str, config_dir: Path) -> list[dict]:
    """Reads config/sources/{topic}.yaml and returns the source list."""
    path = config_dir / "sources" / f"{topic}.yaml"
    if not path.exists():
        logger.info("no sources config at %s, falling back to HN only", path)
        return [{"type": "hn", "min_points": 20}]
    data = yaml.safe_load(path.read_text())
    return data.get("sources", [])


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
        raw_items = []
        sources_config = _load_sources_config(topic, config_dir)

        for src in sources_config:
            source_type = src.get("type", "hn")
            source_name = src.get("name", source_type)
            try:
                if source_type == "hn":
                    items = fetch_hn(since, until, min_points=src.get("min_points", min_points))
                elif source_type == "rss":
                    items = fetch_and_parse_rss(src["url"])
                else:
                    logger.warning("collect: unknown source type '%s' - skipped", source_type)
                    continue
                raw_items.extend(items)
                save_raw(topic, date_str, source_name, items, data_dir=data_dir / "raw")
                logger.info("collect (%s): %d items", source_name, len(items))
            except Exception as e:
                logger.warning("collect (%s) failed: %s - continuing", source_name, e)
                all_warnings.append(f"Keruu ({source_name}): {e}")

    logger.info("collect: total %d raw items across all sources", len(raw_items))

    # 2. Dedup
    candidates = dedup_candidates(raw_items)
    save_candidates(topic, date_str, candidates, data_dir=data_dir / "dedup")
    logger.info("dedup: %d -> %d candidates", len(raw_items), len(candidates))

    models_config = load_models_config(config_paths.models)
    models_used: dict[str, str] = {}
    rubric = load_rubric(config_paths.rubric)

    # 3. Filter by topic relevance
    llm_filter = _resolve_llm_call("filter_topic", models_config, config_paths, model_overrides,
                                    llm_client, models_used)
    topic_description = rubric.get("topic_description", topic)
    filter_result = filter_topic(candidates, topic_description, llm_filter)
    if filter_result.warning:
        all_warnings.append(filter_result.warning)
    logger.info("filter_topic: %d -> %d candidates (%d dropped)",
                len(candidates), filter_result.n_kept, filter_result.n_dropped)

    # 4. Cluster
    llm_cluster = _resolve_llm_call("cluster", models_config, config_paths, model_overrides,
                                     llm_client, models_used)
    cluster_result = cluster_candidates(filter_result.candidates, llm_cluster)
    if cluster_result.warning:
        all_warnings.append(cluster_result.warning)
    logger.info("cluster: %d -> %d clusters", len(filter_result.candidates), len(cluster_result.clusters))

    # 5. Score (CRITICAL - no fallback, ScoreValidationError crashes the entire run)
    llm_score = _resolve_llm_call("score", models_config, config_paths, model_overrides,
                                   llm_client, models_used)
    min_items = rubric["items_per_briefing"]["min"]
    previous_stories = load_previous_stories(topic, since.date(), output_dir=out_dir)
    clusters_before = len(cluster_result.clusters)
    clusters = filter_previous_clusters(cluster_result.clusters, previous_stories)
    if len(clusters) < clusters_before:
        logger.info("cross-day dedup: filtered %d clusters already in previous digests",
                     clusters_before - len(clusters))
    try:
        scored = score_clusters(clusters, config_paths.rubric, llm_score,
                                previous_stories=previous_stories)
    except ScoreValidationError:
        logger.warning("score step failed, retrying once")
        scored = score_clusters(clusters, config_paths.rubric, llm_score,
                                previous_stories=previous_stories)
    initial_pool = [sc for sc in scored.scored if sc.rank <= scored.cutoff_rank]
    backfill_pool = [sc for sc in scored.scored if sc.rank > scored.cutoff_rank]
    logger.info("score: %d selected (cutoff %d, %d initial + %d backfill, rubric %s)",
                len(scored.scored), scored.cutoff_rank, len(initial_pool),
                len(backfill_pool), rubric.get("version"))

    # 5. Enrich
    owns_enrich_client = enrich_client is None
    ec = enrich_client or httpx.Client(timeout=15.0, follow_redirects=True)
    try:
        enrich_result = enrich_candidates(initial_pool, client=ec)
        all_warnings.extend(enrich_result.warnings)
        dropped_stories = enrich_result.dropped_stories
        logger.info("enrich: %d -> %d items (content fetched successfully)",
                    len(initial_pool), len(enrich_result.items))

        # 6. Compose
        llm_compose = _resolve_llm_call("compose", models_config, config_paths, model_overrides,
                                         llm_client, models_used)
        compose_result = compose_items(enrich_result.items, config_paths.persona,
                                        config_paths.guardrails, llm_compose,
                                        golden_examples_dir=config_paths.golden_examples_dir)
        all_warnings.extend(compose_result.warnings)
        logger.info("compose: %d items written", len(compose_result.items))

        # 6b. Backfill: if initial pool drops below minimum, try backfill pool
        if len(compose_result.items) < min_items and backfill_pool:
            logger.info("backfill: have %d items, need %d — trying %d backfill candidates",
                         len(compose_result.items), min_items, len(backfill_pool))
            for sc in backfill_pool:
                if len(compose_result.items) >= min_items:
                    break
                batch = enrich_candidates([sc], client=ec)
                all_warnings.extend(batch.warnings)
                dropped_stories.extend(batch.dropped_stories)
                if not batch.items:
                    continue
                batch_compose = compose_items(batch.items, config_paths.persona,
                                               config_paths.guardrails, llm_compose,
                                               golden_examples_dir=config_paths.golden_examples_dir)
                all_warnings.extend(batch_compose.warnings)
                compose_result.items.extend(batch_compose.items)
    finally:
        if owns_enrich_client:
            ec.close()

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
                         help="step=provider:model, e.g. score=openrouter:gpt-4o-mini. Can be specified multiple times.")
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
