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
import json
import logging
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from time import perf_counter

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
from agent.tts import synthesize, load_tts_config
from agent.tts_template import TtsTemplate

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.1.0"
LLM_STEPS = ["cluster", "filter_topic", "score", "compose", "overview"]


class ConfigPaths:
    def __init__(self, config_dir: Path, topic: str):
        self.topic_name = topic
        self.models = config_dir / "models.yaml"
        self.topics_dir = config_dir / "topics"
        self.topic_config = self.topics_dir / f"{topic}.yaml"
        path = config_dir / "personas" / f"{topic}_v1.yaml"
        self.persona = path if path.exists() else config_dir / "personas" / "ainikki_v1.yaml"
        self.guardrails = config_dir / "guardrails" / "guardrails_v1.yaml"
        self.golden_examples_dir = config_dir / "golden_examples"
        topic_overview_examples = config_dir / "overview_examples" / f"{topic}_v1.yaml"
        self.overview_examples = (
            topic_overview_examples if topic_overview_examples.exists()
            else config_dir / "overview_examples" / "ai_v1.yaml"
        )
        topic_rubric = config_dir / "rubrics" / f"{topic}_scoring_rubric_v1.yaml"
        self.rubric = (
            topic_rubric if topic_rubric.exists()
            else config_dir / "rubrics" / "ai_scoring_rubric_v1.yaml"
        )
        self.sources = config_dir / "sources.yaml"
        self.template_path = config_dir / "tts-templates" / "ainikki-oletus.yaml"


def _load_sources_config(config_paths: ConfigPaths) -> list[dict]:
    """Reads config/sources.yaml (all source definitions) and
    config/topics/{topic}.yaml (which sources to use for this topic).

    Returns a list of source dicts ready for the collect step.
    """
    shared_path = config_paths.sources
    if not shared_path.exists():
        logger.warning("no sources config at %s, falling back to HN only", shared_path)
        return [{"type": "hn", "min_points": 20}]

    shared = yaml.safe_load(shared_path.read_text())
    sources_map = {s["key"]: s for s in shared.get("sources", [])}

    topic_path = config_paths.topics_dir / f"{config_paths.topic_name}.yaml"
    if not topic_path.exists():
        logger.info("no topics config at %s, falling back to HN only", topic_path)
        return [{"type": "hn", "min_points": 20}]

    topic_data = yaml.safe_load(topic_path.read_text())
    source_keys = topic_data.get("sources", [])

    result = []
    for key in source_keys:
        src = sources_map.get(key)
        if src is None:
            logger.warning("collect: unknown source key '%s' - skipped", key)
            continue
        entry = dict(src)  # shallow copy
        result.append(entry)
    return result


def _resolve_llm_call(step: str, models_config: dict, config_paths: ConfigPaths,
                      overrides: dict[str, tuple[str | None, str | None]],
                      client: httpx.Client | None, models_used_out: dict[str, str],
                      llm_stats_out: dict[str, dict],
                      llm_prompts_out: dict[str, list[dict]] | None = None):
    """Builds an LlmCall function for a step AND records which model was actually
    used into models_used_out dict (for GenerationMeta traceability).

    Also accumulates LLM usage stats into llm_stats_out dict.
    Optionally captures prompts and raw responses into llm_prompts_out dict.
    """
    provider_override, model_override = overrides.get(step, (None, None))
    cfg = resolve_step_config(step, models_config, model_override, provider_override)
    models_used_out[step] = f"{cfg.provider}/{cfg.model}"

    llm_fn = make_llm_call(step, config_path=config_paths.models,
                           override_model=model_override, override_provider=provider_override,
                           client=client)

    step_stats = {
        "calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": None,
    }
    def _tracked_call(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        content, usage = llm_fn(system_prompt, user_prompt)
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        total_tokens = usage.get("total_tokens", 0) or 0
        cost = usage.get("cost")
        step_stats["calls"] += 1
        step_stats["total_prompt_tokens"] += prompt_tokens
        step_stats["total_completion_tokens"] += completion_tokens
        step_stats["total_tokens"] += total_tokens
        if cost is not None:
            step_stats["total_cost"] = (step_stats["total_cost"] or 0) + cost
        if llm_prompts_out is not None:
            llm_prompts_out[step].append({
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": content,
            })
        return content, usage

    llm_stats_out[step] = step_stats

    if llm_prompts_out is not None:
        llm_prompts_out[step] = []

    return _tracked_call


def run_pipeline(
    topic: str, period: Period, since: datetime, until: datetime,
    config_dir: Path = Path("config"), data_dir: Path = Path("data"),
    out_dir: Path | None = None, min_points: int = 20,
    display_date: date | None = None,
    model_overrides: dict[str, tuple[str | None, str | None]] | None = None,
    raw_items_override: list[RawItem] | None = None,
    llm_client: httpx.Client | None = None,
    enrich_client: httpx.Client | None = None,
) -> Path:

    model_overrides = model_overrides or {}
    out_dir = out_dir or (data_dir / "output")
    config_paths = ConfigPaths(config_dir, topic)
    if not config_paths.topic_config.exists():
        raise ValueError(f"Topic configuration not found: {config_paths.topic_config}")
    topic_config = yaml.safe_load(config_paths.topic_config.read_text())
    if (not isinstance(topic_config, dict)
            or not topic_config.get("target_audience")
            or not isinstance(topic_config.get("public"), bool)):
        raise ValueError(
            "Topic configuration must define target_audience and boolean public: "
            f"{config_paths.topic_config}"
        )
    target_audience = topic_config["target_audience"]
    effective_display_date = display_date or since.date()
    date_str = effective_display_date.isoformat()
    all_warnings: list[str] = []
    dropped_stories: list = []
    t_pipeline_start = perf_counter()
    step_durations: dict[str, float] = {}

    # 1. Collect
    t0 = perf_counter()
    if raw_items_override is not None:
        raw_items = raw_items_override
        logger.info("collect: using raw_items_override (%d items) - test run", len(raw_items))
    else:
        raw_items = []
        sources_config = _load_sources_config(config_paths)

        for src in sources_config:
            source_type = src.get("type", "hn")
            source_name = src.get("pretty_name", source_type)
            source_badge = src.get("badge") or src.get("pretty_name") or source_type
            try:
                if source_type == "hn":
                    items = fetch_hn(since, until, min_points=min_points)
                elif source_type == "rss":
                    items = fetch_and_parse_rss(src["feed_url"], since=since, until=until, source_badge=source_badge)
                else:
                    logger.warning("collect: unknown source type '%s' - skipped", source_type)
                    continue
                raw_items.extend(items)
                save_raw(topic, date_str, source_name, items, data_dir=data_dir / "raw")
                logger.info("collect (%s): %d items", source_name, len(items))
            except Exception as e:
                logger.warning("collect (%s) failed: %s - continuing", source_name, e)
                all_warnings.append(f"Keruu ({source_name}): {e}")

    step_durations["collect"] = round(perf_counter() - t0, 2)
    logger.info("collect: total %d raw items across all sources (%.2fs)", len(raw_items), step_durations["collect"])

    # 2. Dedup
    t0 = perf_counter()
    candidates = dedup_candidates(raw_items)
    save_candidates(topic, date_str, candidates, data_dir=data_dir / "dedup")
    step_durations["dedup"] = round(perf_counter() - t0, 2)
    logger.info("dedup: %d -> %d candidates (%.2fs)", len(raw_items), len(candidates), step_durations["dedup"])

    models_config = load_models_config(config_paths.models)
    models_used: dict[str, str] = {}
    llm_stats: dict[str, dict] = {}
    llm_prompts: dict[str, list[dict]] = {}
    rubric = load_rubric(config_paths.rubric)

    # 3. Filter by topic relevance
    t0 = perf_counter()
    llm_filter = _resolve_llm_call("filter_topic", models_config, config_paths, model_overrides,
                                    llm_client, models_used, llm_stats, llm_prompts)
    topic_description = topic_config.get("topic_description", topic)
    filter_result = filter_topic(candidates, topic_description, llm_filter)
    if filter_result.warning:
        all_warnings.append(filter_result.warning)
    step_durations["filter_topic"] = round(perf_counter() - t0, 2)
    logger.info("filter_topic: %d -> %d candidates (%d dropped, %.2fs)",
                len(candidates), filter_result.n_kept, filter_result.n_dropped, step_durations["filter_topic"])

    # 4. Cluster
    t0 = perf_counter()
    llm_cluster = _resolve_llm_call("cluster", models_config, config_paths, model_overrides,
                                     llm_client, models_used, llm_stats, llm_prompts)
    cluster_result = cluster_candidates(filter_result.candidates, llm_cluster)
    if cluster_result.warning:
        all_warnings.append(cluster_result.warning)
    step_durations["cluster"] = round(perf_counter() - t0, 2)
    logger.info("cluster: %d -> %d clusters (%.2fs)", len(filter_result.candidates), len(cluster_result.clusters), step_durations["cluster"])

    # 5. Score (CRITICAL - no fallback, ScoreValidationError crashes the entire run)
    t0 = perf_counter()
    llm_score = _resolve_llm_call("score", models_config, config_paths, model_overrides,
                                   llm_client, models_used, llm_stats, llm_prompts)
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
    step_durations["score"] = round(perf_counter() - t0, 2)
    logger.info("score: %d selected (cutoff %d, %d initial + %d backfill, rubric %s, %.2fs)",
                len(scored.scored), scored.cutoff_rank, len(initial_pool),
                len(backfill_pool), rubric.get("version"), step_durations["score"])

    # 6. Enrich & Compose
    owns_enrich_client = enrich_client is None
    ec = enrich_client or httpx.Client(timeout=15.0, follow_redirects=True)
    try:
        t0_enrich = perf_counter()
        enrich_result = enrich_candidates(initial_pool, client=ec)
        all_warnings.extend(enrich_result.warnings)
        dropped_stories = enrich_result.dropped_stories
        step_durations["enrich"] = round(perf_counter() - t0_enrich, 2)
        logger.info("enrich: %d -> %d items (content fetched successfully, %.2fs)",
                    len(initial_pool), len(enrich_result.items), step_durations["enrich"])

        t0_compose = perf_counter()
        llm_compose = _resolve_llm_call("compose", models_config, config_paths, model_overrides,
                                         llm_client, models_used, llm_stats, llm_prompts)
        compose_result = compose_items(enrich_result.items, config_paths.persona,
                                        config_paths.guardrails, llm_compose,
                                        topic=topic,
                                        target_audience=target_audience,
                                        golden_examples_dir=config_paths.golden_examples_dir)
        all_warnings.extend(compose_result.warnings)
        step_durations["compose"] = round(perf_counter() - t0_compose, 2)
        logger.info("compose: %d items written (%.2fs)", len(compose_result.items), step_durations["compose"])

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
                                               topic=topic,
                                               target_audience=target_audience,
                                               golden_examples_dir=config_paths.golden_examples_dir)
                all_warnings.extend(batch_compose.warnings)
                compose_result.items.extend(batch_compose.items)
    finally:
        if owns_enrich_client:
            ec.close()

    # 7. Overview
    t0 = perf_counter()
    llm_overview = _resolve_llm_call("overview", models_config, config_paths, model_overrides,
                                      llm_client, models_used, llm_stats, llm_prompts)
    overview_result = generate_overview(compose_result.items, llm_overview, topic,
                                         examples_path=config_paths.overview_examples)
    step_durations["overview"] = round(perf_counter() - t0, 2)

    tts_cfg = topic_config.get("tts", {})
    tts_text = ""
    tts_has_audio = False
    provider = tts_cfg.get("provider", "google-cloud")
    voice = tts_cfg.get("voice", "fi-FI-Chirp3-HD-Callirrhoe")
    if tts_cfg.get("enabled", False):
        t0_tts = perf_counter()
        try:
            logger.info("tts: generating audio for overview (provider=%s, voice=%s)", provider, voice)

            # Load and render TTS template if configured
            tts_template = TtsTemplate(config_paths.template_path)
            tts_text = tts_template.render({
                "overview_heading": overview_result.digest_topic,
                "overview_text": overview_result.overview,
                "items": compose_result.items
            })
            logger.info("tts: rendering template (provider=%s, voice=%s)", provider, voice)
            logger.info("tts: template output preview:\n%s", tts_text[:300])
            logger.debug("tts: full template output:\n%s", tts_text)

            tts_result = synthesize(tts_text, provider, voice)
            if tts_result:
                audio_path = out_dir / f"{topic}_{period.value}_{date_str}_audio.mp3"
                audio_path.write_bytes(tts_result.audio_bytes)
                tts_has_audio = True
                logger.info("tts: audio saved to %s", audio_path)
                step_durations["tts"] = round(perf_counter() - t0_tts, 2)
            else:
                n_bytes = len(tts_text.encode("utf-8"))
                msg = f"tts: synthesis returned no result (input {n_bytes} bytes exceeds limit)"
                logger.warning(msg)
                all_warnings.append(msg)
        except Exception as e:
            logger.error(
                "tts: generation failed for provider=%s voice=%s: %s (type: %s)",
                provider, voice, e, type(e).__name__, exc_info=e
            )

    models_used['tts'] = f"{provider}/{voice}"

    total_duration = round(perf_counter() - t_pipeline_start, 2)

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
        duration_seconds=total_duration,
        llm_stats=llm_stats,
        tts_text=tts_text,
    )

    # 10. Write pipeline debug data
    pipeline_dir = data_dir / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_file = pipeline_dir / f"{topic}_{period.value}_{date_str}.json"
    pipeline_debug = {
        "topic": topic,
        "period": period.value,
        "period_start": since.date().isoformat(),
        "period_end": (until - timedelta(seconds=1)).date().isoformat(),
        "total_duration_s": total_duration,
        "step_durations_s": step_durations,
        "llm_stats": llm_stats,
        "llm_prompts": llm_prompts,
        "steps": {
            "collect": {
                "total_raw": len(raw_items),
            },
            "dedup": {
                "candidates": len(candidates),
            },
            "filter_topic": {
                "n_kept": filter_result.n_kept,
                "n_dropped": filter_result.n_dropped,
            },
            "cluster": {
                "n_clusters": len(cluster_result.clusters),
                "clusters": [
                    {
                        "items": [
                            {"title": ri.title, "url": str(ri.url),
                             "source_type": ri.source_type.value,
                             "source_badge": ri.source_badge or (ri.source_type.value.upper() if ri.source_type.value != 'hackernews' else 'HN')}
                            for ri in c.items
                        ],
                        "cluster_reason": c.cluster_reason,
                    }
                    for c in cluster_result.clusters
                ],
            },
            "score": {
                "cutoff_rank": scored.cutoff_rank,
                "candidates": [
                    {
                        "title": sc.items[0].title,
                        "url": str(sc.items[0].url),
                        "rank": sc.rank,
                        "selection_reason": sc.selection_reason,
                        "n_urls": sc.n_urls,
                    }
                    for sc in scored.scored
                ],
            },
            "enrich": {
                "n_success": len(enrich_result.items),
                "n_failed": len(enrich_result.dropped_stories),
                "dropped": [
                    {"title": d.title, "url": d.url}
                    for d in enrich_result.dropped_stories
                ],
            },
            "compose": {
                "n_written": len(compose_result.items),
            },
            "overview": {
                "overview": overview_result.overview,
                "digest_topic": overview_result.digest_topic,
            },
            "tts": {
                "enabled": tts_cfg.get("enabled", False),
                "text": tts_text if tts_cfg.get("enabled") else None,
                "provider": provider,
                "voice": voice,
                "has_audio": tts_has_audio,
            },
        },
    }
    pipeline_file.write_text(
        json.dumps(pipeline_debug, ensure_ascii=False, indent=2)
    )
    logger.info("pipeline debug: written %s", pipeline_file)

    # 11. Write
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
