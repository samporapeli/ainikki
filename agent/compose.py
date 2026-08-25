"""
Compose-step: writes a Finnish headline+summary for each item that passed
the Enrich step. The only step that uses the persona — Score/Cluster must
not include persona influence (see the agreed separation: editorial
assessment vs. voice are separate concerns).

Non-critical step per item (see README "Failure-policy per step"):
a single item's failed/broken response drops ONLY that item, it does not
crash the entire run — unlike Score, where the whole selection depends on
a single LLM call.

One LLM call per item (not all at once) — small context, parallelizable
later (e.g. asyncio.gather), and a single item's failure is isolated
from others.
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import EnrichedCandidate, NewsItem, RawItem, Source
from agent.cluster import LlmCall
from random import choice

logger = logging.getLogger(__name__)

MAX_HEADLINE_CHARS = 256
MAX_SUMMARY_CHARS = 4096


# --- Configuration loading ----------------------------------------------

class Persona(BaseModel):
    name: str
    version: str
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
                                 guardrails: GuardrailsConfig, topic: str,
                                 target_audience: str) -> str:
    voice = "\n".join(f"- {t}" for t in persona.voice_traits)
    avoid = "\n".join(f"- {a}" for a in persona.avoid)
    rules = "\n".join(f"- {r}" for r in guardrails.rules)
    examples = "\n\n".join(f"{ex.headline}\n{ex.summary}" for ex in golden_examples.examples)

    return f"""Kirjoitat {topic}-aiheiselle uutiskoosteelle tekstiä persoonan "{persona.name}" äänellä.

Kohdeyleisö: {target_audience}

Sävy: {persona.tone_description}

Tyylipiirteet:
{voice}

Vältä:
{avoid}

Säännöt (pätevät aina, riippumatta äänestä):
{rules}

Esimerkkejä aiemmista julkaisuistasi (pidä sama ääni):
{examples}

Tehtävä:
- otsikko: suomenkielinen otsikko, enintään 12 sanaa
- tiivistelmä: suomenkielinen tiivistelmä, 2–4 virkettä

Käytä vain annettua sisältöä. Älä lisää tietoa, jota lähde ei mainitse.
Tekninen enimmäispituus on otsikolle {MAX_HEADLINE_CHARS} merkkiä ja
tiivistelmälle {MAX_SUMMARY_CHARS} merkkiä.

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{{"otsikko": "...", "tiivistelmä": "..."}}"""



def build_compose_user_prompt(item: EnrichedCandidate) -> str:
    content_source = item.content_source or item.items[0]
    return f"""Alkuperäinen otsikko: {content_source.title}

Sisältö:
{item.content}"""


# --- Execution ----------------------------------------------------------

class ComposeResponse(BaseModel):
    headline: str = Field(validation_alias="otsikko")
    summary: str = Field(validation_alias="tiivistelmä")


class ComposeResult(NamedTuple):
    items: list[NewsItem]
    warnings: list[str]
    persona_id: str
    guardrails_version: str
    golden_examples_version: str
    persona_path: Optional[Path] = None


def _parse_compose_response(raw_response: str) -> ComposeResponse:
    cleaned = strip_code_fences(raw_response).strip()
    if not cleaned:
        raise ValueError("empty response")
    parsed = json.loads(cleaned)
    response = ComposeResponse(**parsed)
    if not response.headline.strip() or not response.summary.strip():
        raise ValueError("headline or summary is empty")
    return response


def _validate_output_lengths(response: ComposeResponse) -> str | None:
    errors = []
    if len(response.headline) > MAX_HEADLINE_CHARS:
        errors.append(f"headline ylittää {MAX_HEADLINE_CHARS} merkin rajan")
    if len(response.summary) > MAX_SUMMARY_CHARS:
        errors.append(f"summary ylittää {MAX_SUMMARY_CHARS} merkin rajan")
    return "; ".join(errors) if errors else None


def _raw_item_to_source(raw_item: RawItem) -> Source:
    return Source(
        url=raw_item.url, title=raw_item.title, source_type=raw_item.source_type,
        source_badge=raw_item.source_badge,
        published_at=raw_item.published_at, raw_signal=raw_item.raw_signal,
    )


def compose_items(enriched: list[EnrichedCandidate], persona_path: Path,
                   guardrails_path: Path, llm_call: LlmCall,
                   topic: str = "",
                   target_audience: str = "",
                   golden_examples_dir: Path = Path("config/golden_examples")) -> ComposeResult:
    persona = Persona.load(persona_path)
    golden_examples = GoldenExamplesConfig.load(golden_examples_dir / persona.golden_examples_ref)
    guardrails = GuardrailsConfig.load(guardrails_path)

    system_prompt = build_compose_system_prompt(persona, golden_examples, guardrails, topic,
                                                 target_audience)

    news_items: list[NewsItem] = []
    warnings: list[str] = []

    for item in enriched:
        content_source = item.content_source or item.items[0]
        primary_title = content_source.title
        user_prompt = build_compose_user_prompt(item)
        raw_response, _usage = llm_call(system_prompt, user_prompt)

        if not strip_code_fences(raw_response).strip():
            logger.warning("compose: empty response for '%s', retrying once", primary_title)
            raw_response, _usage = llm_call(system_prompt, user_prompt)

        try:
            response = _parse_compose_response(raw_response)
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            msg = f"Kirjoitus epäonnistui '{primary_title}': {e} - pudotettu koosteesta"
            logger.warning(msg)
            warnings.append(msg)
            continue

        length_error = _validate_output_lengths(response)
        if length_error:
            logger.warning("compose: %s for '%s', retrying once", length_error, primary_title)
            retry_prompt = (user_prompt + "\n\n"
                            "Lyhennä vastausta: edellinen vastaus ylitti sallitun "
                            "merkkirajan. Vastaa uudelleen samoilla JSON-kentillä "
                            "ja noudata annettuja enimmäispituuksia.")
            retry_raw_response, _usage = llm_call(system_prompt, retry_prompt)
            try:
                response = _parse_compose_response(retry_raw_response)
                retry_length_error = _validate_output_lengths(response)
                if retry_length_error:
                    raise ValueError(retry_length_error)
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
        persona_path=persona_path,
    )

