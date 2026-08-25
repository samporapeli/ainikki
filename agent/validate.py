"""
Validate-step: assembles data produced by all previous steps into a single
Briefing object. Pydantic validates the structure automatically in the
constructor — this module adds one additional check on top: if no items
survived (all were filtered out in score/enrich/compose), the entire run
fails visibly and no empty/misleading digest is written.
"""

import httpx
from datetime import date, datetime, timezone

from agent.dateformat import fi_date
from agent.schema import Briefing, DroppedStory, GenerationMeta, Period
from agent.compose import ComposeResult
from agent.overview import OverviewResult


class EmptyBriefingError(Exception):
    """No validated items remaining — all candidates were filtered out
    along the way (score/enrich/compose). Do not write an empty digest."""


def assemble_briefing(topic: str, period: Period, period_start: date, period_end: date,
                       overview_result: OverviewResult, compose_result: ComposeResult,
                       models_used: dict[str, str], pipeline_version: str,
                       rubric_version: str, display_date: date | None = None,
                       dropped_stories: list[DroppedStory] | None = None,
                       extra_warnings: list[str] | None = None,
                       duration_seconds: float | None = None,
                       llm_stats: dict[str, dict] | None = None,
                       tts_text: str = "") -> Briefing:

    if not compose_result.items:
        raise EmptyBriefingError(
            f"No validated items remaining for topic '{topic}' "
            f"({period_start} - {period_end}) — digest will not be written."
        )

    all_warnings = list(extra_warnings or [])
    if overview_result.warning:
        all_warnings.append(overview_result.warning)

    meta = GenerationMeta(
        models_used=models_used,
        generated_at=datetime.now(timezone.utc),
        duration_seconds=duration_seconds,
        pipeline_version=pipeline_version,
        persona=compose_result.persona_id,
        rubric_version=rubric_version,
        guardrails_version=compose_result.guardrails_version,
        golden_examples_version=compose_result.golden_examples_version,
        llm_stats=llm_stats or {},
    )

    effective_dd = display_date or period_start

    return Briefing(
        topic=topic, period=period, period_start=period_start, period_end=period_end,
        display_date=display_date, display_date_fi=fi_date(effective_dd.isoformat()),
        digest_topic=overview_result.digest_topic,
        overview=overview_result.overview,
        items=compose_result.items, meta=meta,
        dropped_stories=dropped_stories or [], warnings=all_warnings,
        tts_text=tts_text,
    )
