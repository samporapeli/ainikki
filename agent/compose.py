"""
Compose-step: kirjoittaa suomenkielisen headline+summary jokaiselle
Enrich-stepin läpäisseelle itemille. Ainoa step joka käyttää persoonaa -
Score/Cluster eivät saa sisältää persoona-vaikutusta (ks. keskustelussa
sovittu erottelu: editorial-arvio vs. ääni ovat eri kysymyksiä).

Ei-kriittinen step per item (ks. README "Failure-policy stepeittäin"):
yhden itemin epäonnistunut/rikki menevä vastaus pudottaa VAIN sen itemin,
ei kaada koko ajoa - toisin kuin Score, jossa koko valinta on yhden
LLM-kutsun varassa.

Yksi LLM-kutsu per item (ei kaikille kerralla) - pieni konteksti,
rinnakkaistettavissa myöhemmin (esim. asyncio.gather), ja yhden itemin
epäonnistuminen on eristetty muista.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import EnrichedCandidate, NewsItem, RawItem, Source
from agent.cluster import LlmCall

logger = logging.getLogger(__name__)


# --- Konfiguraatioiden lataus ----------------------------------------------

class Persona(BaseModel):
    name: str
    version: str
    target_audience: str
    tone_description: str
    voice_traits: list[str]
    avoid: list[str]
    golden_examples_ref: str

    @classmethod
    def load(cls, path: Path) -> "Persona":
        return cls(**yaml.safe_load(path.read_text()))

    @property
    def id(self) -> str:
        return f"{self.name}-{self.version}"


class GoldenExample(BaseModel):
    headline: str
    summary: str


class GoldenExamplesConfig(BaseModel):
    version: str
    examples: list[GoldenExample]

    @classmethod
    def load(cls, path: Path) -> "GoldenExamplesConfig":
        return cls(**yaml.safe_load(path.read_text()))


class GuardrailsConfig(BaseModel):
    version: str
    rules: list[str]

    @classmethod
    def load(cls, path: Path) -> "GuardrailsConfig":
        return cls(**yaml.safe_load(path.read_text()))


# --- Prompt construction -------------------------------------------------

def build_compose_system_prompt(persona: Persona, golden_examples: GoldenExamplesConfig,
                                 guardrails: GuardrailsConfig) -> str:
    voice = "\n".join(f"- {t}" for t in persona.voice_traits)
    avoid = "\n".join(f"- {a}" for a in persona.avoid)
    rules = "\n".join(f"- {r}" for r in guardrails.rules)
    examples = "\n\n".join(f"{ex.headline}\n{ex.summary}" for ex in golden_examples.examples)

    return f"""Olet AI-uutiskoosteen toimittaja "{persona.name}".

Kohdeyleisö: {persona.target_audience}

Sävy: {persona.tone_description}

Tyylipiirteet:
{voice}

Vältä:
{avoid}

Säännöt (pätevät aina, riippumatta äänestä):
{rules}

Esimerkkejä aiemmista julkaisuistasi (pidä sama ääni):
{examples}

Tehtäväsi: kirjoita annetulle uutiselle suomenkielinen headline (max 12 sanaa)
ja summary (2-4 lausetta) yllä kuvatulla äänellä ja säännöillä. Käytä VAIN
annettua sisältöä, älä lisää tietoa jota ei ole siinä.

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{{"headline": "...", "summary": "..."}}"""


def build_compose_user_prompt(item: EnrichedCandidate) -> str:
    primary = item.items[0]
    return f"""Alkuperäinen otsikko: {primary.title}

Sisältö:
{item.content}"""


# --- Execution ----------------------------------------------------------

class ComposeResponse(BaseModel):
    headline: str
    summary: str


class ComposeResult(NamedTuple):
    items: list[NewsItem]
    warnings: list[str]
    persona_id: str
    guardrails_version: str
    golden_examples_version: str


def _raw_item_to_source(raw_item: RawItem) -> Source:
    return Source(
        url=raw_item.url, title=raw_item.title, source_type=raw_item.source_type,
        published_at=raw_item.published_at, raw_signal=raw_item.raw_signal,
    )


def compose_items(enriched: list[EnrichedCandidate], persona_path: Path,
                   guardrails_path: Path, llm_call: LlmCall,
                   golden_examples_dir: Path = Path("config/golden_examples")) -> ComposeResult:
    persona = Persona.load(persona_path)
    golden_examples = GoldenExamplesConfig.load(golden_examples_dir / persona.golden_examples_ref)
    guardrails = GuardrailsConfig.load(guardrails_path)

    system_prompt = build_compose_system_prompt(persona, golden_examples, guardrails)

    news_items: list[NewsItem] = []
    warnings: list[str] = []

    for item in enriched:
        primary_title = item.items[0].title
        user_prompt = build_compose_user_prompt(item)
        raw_response = llm_call(system_prompt, user_prompt)

        try:
            parsed = json.loads(strip_code_fences(raw_response))
            response = ComposeResponse(**parsed)
            if not response.headline.strip() or not response.summary.strip():
                raise ValueError("headline or summary is empty")
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            msg = f"Kirjoitus epäonnistui '{primary_title}': {e} - pudotettu koosteesta"
            logger.warning(msg)
            warnings.append(msg)
            continue

        news_items.append(NewsItem(
            headline=response.headline,
            summary=response.summary,
            sources=[_raw_item_to_source(ri) for ri in item.items],
            rank=item.rank,
            tags=[],
            selection_reason=item.selection_reason,
        ))

    return ComposeResult(
        items=news_items, warnings=warnings,
        persona_id=persona.id, guardrails_version=guardrails.version,
        golden_examples_version=golden_examples.version,
    )
