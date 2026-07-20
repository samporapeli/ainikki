"""
Overview-step: writes a 2-3 sentence overview for the entire digest,
using ONLY the headline+summary texts already produced by the Compose
step — never raw sources or article content. This keeps the context
small and fact-error risk minimal, since the texts have already passed
through guardrails once.

Special case compared to other non-critical steps: Briefing.overview is
a REQUIRED field in the schema, so simply "dropping it" is not an option
when the LLM fails. Instead, a deterministic fallback is used (composed
directly from already-validated headlines, without LLM) — always works,
but visibly logged and marked as a warning.
"""

import json
import logging
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import NewsItem
from agent.cluster import LlmCall

logger = logging.getLogger(__name__)


def _build_system_prompt(topic: str) -> str:
    return f"""Olet uutistoimittaja. Saat listan artikkeleista aiheesta "{topic}".
Artikkelit on jo suodatettu aiheen mukaan, joten älä erikseen kerro mistä aiheesta on kyse.

Kirjoita 2-3 lauseen yleiskatsaus: nosta esiin yhteiset teemat ja
keskeiset toimijat nimeltä (kerro lyhyesti kuka tai mikä entiteetti on kyseessä, jos
se ei ole ilmeistä). Jos aiheet ovat hajanaisia, kerro lyhyesti mistä aiheista koosteessa puhutaan.

Käytä VAIN annettuja otsikoita ja yhteenvetoja, älä lisää tietoa joita niissä ei ole.

Vastaa VAIN JSON-muodossa:
{{"overview": "..."}}"""


class OverviewResponse(BaseModel):
    overview: str


class OverviewResult(NamedTuple):
    overview: str
    warning: str | None


def build_overview_prompt(items: list[NewsItem], topic: str) -> tuple[str, str]:
    ordered = sorted(items, key=lambda i: i.rank)
    lines = [f"{i.rank}. {i.headline} — {i.summary}" for i in ordered]
    user_prompt = "Tämän päivän jutut:\n" + "\n".join(lines)
    return _build_system_prompt(topic), user_prompt


def _fallback_overview(items: list[NewsItem], max_headlines: int = 3) -> str:
    """Deterministic, non-LLM fallback. Not as good as a model-written
    synthesis, but ALWAYS works — schema requires the overview field."""
    ordered = sorted(items, key=lambda i: i.rank)
    top_headlines = [i.headline for i in ordered[:max_headlines]]
    return "Tämän päivän aiheita: " + "; ".join(top_headlines) + "."


def generate_overview(items: list[NewsItem], llm_call: LlmCall,
                       topic: str) -> OverviewResult:
    if not items:
        return OverviewResult(overview="Ei julkaistavia juttuja tälle ajalle.", warning=None)

    system_prompt, user_prompt = build_overview_prompt(items, topic)
    raw_response = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = OverviewResponse(**parsed)
        if not response.overview.strip():
            raise ValueError("overview is empty")
        return OverviewResult(overview=response.overview.strip(), warning=None)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        fallback = _fallback_overview(items)
        warning = f"Overview generation failed ({e}) - deterministic fallback used"
        logger.warning(warning)
        return OverviewResult(overview=fallback, warning=warning)
