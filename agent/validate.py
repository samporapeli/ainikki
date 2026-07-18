"""
Validate-step: kokoaa kaikkien edellisten vaiheiden tuottaman datan
yhdeksi Briefing-objektiksi. Pydantic validoi rakenteen automaattisesti
konstruktorissa - tämä moduuli lisää sen päälle vain yhden ylimääräisen
tarkistuksen: jos yksikään item ei selvinnyt asti (kaikki karsiutuivat
score/enrich/compose-vaiheissa), koko ajo epäonnistuu näkyvästi eikä
kirjoiteta tyhjää/harhaanjohtavaa koostetta.
"""

from datetime import date, datetime, timezone

from agent.dateformat import fi_date
from agent.schema import Briefing, DroppedStory, GenerationMeta, Period
from agent.compose import ComposeResult
from agent.overview import OverviewResult


class EmptyBriefingError(Exception):
    """Ei yhtään validoitua itemiä jäljellä - kaikki kandidaatit karsiutuivat
    matkan varrella (score/enrich/compose). Ei kirjoiteta tyhjää koostetta."""


def assemble_briefing(topic: str, period: Period, period_start: date, period_end: date,
                       overview_result: OverviewResult, compose_result: ComposeResult,
                       models_used: dict[str, str], pipeline_version: str,
                       rubric_version: str, display_date: date | None = None,
                       dropped_stories: list[DroppedStory] | None = None,
                       extra_warnings: list[str] | None = None) -> Briefing:
    if not compose_result.items:
        raise EmptyBriefingError(
            f"Ei yhtään validoitua itemiä jäljellä topicille '{topic}' "
            f"({period_start} - {period_end}) - koostetta ei kirjoiteta."
        )

    all_warnings = list(extra_warnings or [])
    all_warnings.extend(compose_result.warnings)
    if overview_result.warning:
        all_warnings.append(overview_result.warning)

    meta = GenerationMeta(
        models_used=models_used,
        generated_at=datetime.now(timezone.utc),
        pipeline_version=pipeline_version,
        persona=compose_result.persona_id,
        rubric_version=rubric_version,
        guardrails_version=compose_result.guardrails_version,
        golden_examples_version=compose_result.golden_examples_version,
    )

    effective_dd = display_date or period_start

    return Briefing(
        topic=topic, period=period, period_start=period_start, period_end=period_end,
        display_date=display_date, display_date_fi=fi_date(effective_dd.isoformat()),
        overview=overview_result.overview,
        items=compose_result.items, meta=meta,
        dropped_stories=dropped_stories or [], warnings=all_warnings,
    )
