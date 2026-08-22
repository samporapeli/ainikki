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
from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import BaseModel, Field, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import NewsItem
from agent.cluster import LlmCall

logger = logging.getLogger(__name__)


class OverviewExample(BaseModel):
    source: str
    digest_topic: str
    style: str = "asiallinen"


class OverviewExamplesConfig(BaseModel):
    version: str
    examples: list[OverviewExample]

    @classmethod
    def load(cls, path: Path) -> "OverviewExamplesConfig":
        return cls(**yaml.safe_load(path.read_text()))


def _build_system_prompt(topic: str, examples: list[OverviewExample]) -> str:
    examples_by_style = {"asiallinen": [], "napakka poikkeus": []}
    for example in examples:
        examples_by_style.setdefault(example.style, []).append(example)

    example_sections = []
    for style, style_examples in examples_by_style.items():
        if not style_examples:
            continue
        lines = [f'- Artikkeleissa: "{e.source}" → "{e.digest_topic}"'
                 for e in style_examples]
        example_sections.append(f"{style.capitalize()}:\n" + "\n".join(lines))
    examples_text = "\n\n".join(example_sections)

    return f"""Saat listan artikkeleista aiheesta "{topic}".

Tee kaksi asiaa:

1) Tiivistelmäotsikko: kirjoita 2–12 sanan otsikko, joka kertoo,
   mikä juuri tässä koosteessa on kiinnostavaa tai erityistä.
   - Otsikon tulee perustua annettuihin artikkeleihin.
   - Pyri erottuvuuteen: nimeä tuotteita, yrityksiä tai ilmiöitä, jos ne
     muodostavat koosteessa selkeän kokonaisuuden.
   - Vältä geneerisiä otsikoita, jotka voisivat kuvata lähes mitä tahansa
     koosteessa, kuten "tekoäly kehittyy", "uusia edistysaskeleita
     teknologiassa" tai "ala jatkaa kasvuaan".
   - Oletustyyli on asiallinen ja informatiivinen. Suurin osa otsikoista
     kuuluu tähän tyyliin.
   - Vain silloin kun jokin yksittäinen juttu on aineistossa selvästi
     koominen, ironinen tai itsestään absurdi, voit käyttää yhtä harkittua
     napakkaa tai kuivaa ilmausta. Käytä kevyttä sävyä harvoin, ei useissa
     peräkkäisissä koosteissa eikä vain vaihtelun vuoksi.
   - Kohdista kevyt sävy instituutioon, yritykseen tai järjestelmään, älä
     jutun inhimilliseen uhriin. Jos jutun ydin on vakava seuraus ihmiselle,
     pysy asiallisena.
   - Vältä ylisanoja, kysymysmuotoa ja liioittelua.
   - Jos jutut eivät linkity toisiinsa, valitse yksi vahvin tarina otsikon
     ytimeksi. Älä pakota kaikkia teemoja samaan otsikkoon.
   - Jos artikkeleita on vain yksi tai aineisto on hyvin ohut, tiivistä
     suoraan sen ydinsisältö äläkä keksi laajempaa kehystä.
   - Kirjoita suomeksi.

2) Yleiskatsaus: kirjoita koko koosteesta 2–3 virkkeen ja noin 250–400
   merkin yhteenveto.
   - Mene suoraan asiaan. Älä aloita kertomalla, mistä aiheesta uutiset
     käsittelevät.
   - Tiivistä tärkeimpien juttujen pääasiat ja mainitse, mitä muuta
     käsiteltiin.
   - Mainitse juttujen välinen yhteys vain, jos sellainen on.
   - Pidä yleiskatsaus neutraalina ja informatiivisena riippumatta otsikon
     tyylistä. Yleiskatsaus ei ole leikillinen.
   - Upota toimijoiden nimet luontevasti lauseisiin.
   - Käytä vain annettuja otsikoita ja yhteenvetoja. Älä lisää tietoa,
     jota niissä ei ole, äläkä arvaile lähteissä mainitsemattomia syitä
     tai seurauksia.

Esimerkkejä (asiallinen oletus ja harvinainen napakka poikkeus):
{examples_text}

Vastaa VAIN JSON-muodossa:
{{"tiivistelmäotsikko": "...", "yleiskatsaus": "..."}}"""


class OverviewResponse(BaseModel):
    overview: str = Field(validation_alias="yleiskatsaus")
    digest_topic: str = Field(validation_alias="tiivistelmäotsikko")


class OverviewResult(NamedTuple):
    overview: str
    digest_topic: str
    warning: str | None


def build_overview_prompt(items: list[NewsItem], topic: str,
                          examples: list[OverviewExample] | None = None) -> tuple[str, str]:
    ordered = sorted(items, key=lambda i: i.rank)
    lines = [f"{i.rank}. {i.headline} — {i.summary}" for i in ordered]
    user_prompt = "\n".join(lines)
    return _build_system_prompt(topic, examples or []), user_prompt


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
                       topic: str, examples_path: Path | None = None) -> OverviewResult:
    if not items:
        return OverviewResult(overview="Ei julkaistavia juttuja tälle ajalle.",
                               digest_topic="", warning=None)

    examples = []
    if examples_path and examples_path.exists():
        examples = OverviewExamplesConfig.load(examples_path).examples
    system_prompt, user_prompt = build_overview_prompt(items, topic, examples)
    raw_response, _usage = llm_call(system_prompt, user_prompt)

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
