"""
Overview-step: kirjoittaa 2-3 lauseen yleiskatsauksen koko koosteelle,
käyttäen VAIN jo Compose-stepin tuottamia headline+summary-tekstejä -
ei koskaan raakalähteitä tai artikkelisisältöä. Tämä pitää kontekstin
pienenä ja faktavirheriskin minimaalisena, koska tekstit on jo kertaalleen
guardrailsien läpi vietyjä.

Erikoistapaus muihin ei-kriittisiin stepeihin verrattuna: Briefing.overview
on PAKOLLINEN kenttä schemassa, joten pelkkä "pudota pois" ei ole vaihtoehto
jos LLM epäonnistuu. Sen sijaan käytetään determinististä fallbackia (koostettu
suoraan jo validoiduista otsikoista ilman LLM:ää) - aina toimiva, mutta
näkyvästi lokitettu ja merkitty warningiksi.
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
    """Deterministinen, ei-LLM-pohjainen varakeino. Ei yhtä hyvä kuin mallin
    kirjoittama synteesi, mutta AINA toimiva - schema vaatii overview-kentän."""
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
            raise ValueError("overview on tyhjä")
        return OverviewResult(overview=response.overview.strip(), warning=None)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        fallback = _fallback_overview(items)
        warning = f"overview-step epäonnistui ({e}) - käytetty deterministinen fallback"
        logger.warning(warning)
        return OverviewResult(overview=fallback, warning=warning)
