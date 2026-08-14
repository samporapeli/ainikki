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
    return f"""Saat listan artikkeleista aiheesta "{topic}".

Tee kaksi asiaa:

1) Tiivistelmäotsikko (digest_topic): 2-8 sanan otsikko, joka kuvaa koosteen
   keskeistä teemaa.
   - Asiallinen ja informatiivinen, ei klikkiotsikko
   - Vältä ylisanoja (mullistava, uskomaton, shokeeraava), kysymysmuotoa
     ja liioittelua
   - Otsikon on perustuttava alla oleviin artikkeleihin, älä keksi
   - Suomen kielellä

2) Yleiskatsaus (overview): 2-3 lauseen yhteenveto koko koosteesta.
   - Älä kerro mistä aiheesta on kyse, äläkä aloita toteamuksella kuten
     "Tämän päivän uutiset käsittelevät"
   - Tiivistä kunkin jutun pääasia
   - Jos jutut linkittyvät toisiinsa, mainitse yhteys — muuten älä pakota
   - Upota toimijoiden nimet luontevasti lauseisiin

Käytä VAIN annettuja otsikoita ja yhteenvetoja, älä lisää tietoa jota niissä
ole.

Esimerkkejä hyvistä tiivistelmäotsikoista:
- Avoimet kielimallit haastavat kaupalliset toimijat
- Tekoälyagentit astuvat työelämään
- Euroopan tekoälysääntely kiristyy
- Tekstin tunnistus ja muistin rajoitukset esillä
- Oppimisen tulevaisuus tekoälyn aikakaudella

Vastaa VAIN JSON-muodossa:
{{"digest_topic": "...", "overview": "..."}}"""


class OverviewResponse(BaseModel):
    overview: str
    digest_topic: str


class OverviewResult(NamedTuple):
    overview: str
    digest_topic: str
    warning: str | None


def build_overview_prompt(items: list[NewsItem], topic: str) -> tuple[str, str]:
    ordered = sorted(items, key=lambda i: i.rank)
    lines = [f"{i.rank}. {i.headline} — {i.summary}" for i in ordered]
    user_prompt = "\n".join(lines)
    return _build_system_prompt(topic), user_prompt


def _fallback_overview(items: list[NewsItem], max_headlines: int = 3) -> str:
    """Deterministic, non-LLM fallback for the overview text.
    Not as good as a model-written synthesis, but ALWAYS works."""
    ordered = sorted(items, key=lambda i: i.rank)
    top_headlines = [i.headline for i in ordered[:max_headlines]]
    return "Tämän päivän aiheita: " + "; ".join(top_headlines) + "."


def _fallback_digest_topic(topic: str) -> str:
    """Deterministic fallback when LLM fails to produce a digest topic.
    Returns the config topic name as-is — better than nothing."""
    return topic


def generate_overview(items: list[NewsItem], llm_call: LlmCall,
                       topic: str) -> OverviewResult:
    if not items:
        return OverviewResult(overview="Ei julkaistavia juttuja tälle ajalle.",
                               digest_topic="", warning=None)

    system_prompt, user_prompt = build_overview_prompt(items, topic)
    raw_response = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = OverviewResponse(**parsed)
        if not response.overview.strip():
            raise ValueError("overview is empty")
        digest_topic = response.digest_topic.strip() if response.digest_topic else ""
        return OverviewResult(overview=response.overview.strip(),
                               digest_topic=digest_topic, warning=None)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        fallback_overview = _fallback_overview(items)
        fallback_digest = _fallback_digest_topic(topic)
        warning = f"Overview generation failed ({e}) - deterministic fallback used"
        logger.warning(warning)
        return OverviewResult(overview=fallback_overview,
                               digest_topic=fallback_digest, warning=warning)
