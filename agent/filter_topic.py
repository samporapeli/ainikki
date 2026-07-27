"""
Filter-topic step: removes candidates that are not about the configured
topic (e.g. AI/ML) before scoring. This is a hard gate — unlike the
scoring rubric's relevance criterion, which competes with other criteria
like community signal, this step has a single job: is the story about
this topic or not?

Non-critical step: if the LLM response is broken, all clusters pass
through to scoring (with a warning). The scoring rubric still applies
as a secondary relevance check.
"""

import json
import logging
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import Candidate, ClusteredCandidate, RawItem

logger = logging.getLogger(__name__)


class FilterResult(NamedTuple):
    candidates: list[Candidate]
    n_kept: int
    n_dropped: int
    warning: str | None


class TopicJudgment(BaseModel):
    index: int
    keep: bool


class TopicFilterResponse(BaseModel):
    results: list[TopicJudgment]


def build_filter_prompt(candidates: list[Candidate],
                        topic_description: str) -> tuple[str, str]:
    topic_lines = []
    for i, c in enumerate(candidates):
        primary = c.primary
        total_signal = sum(
            v for item in c.items for v in item.raw_signal.values()
            if isinstance(v, (int, float))
        )
        topic_lines.append(
            f"[{i}] {primary.title} "
            f"(signaali: {total_signal})"
        )
    user_prompt = "Ehdokkaat:\n" + "\n".join(topic_lines)

    system_prompt = f"""Saist listan uutisehdokkaita. Tehtäväsi: jokainen ehdokas on
joko aiheeseen liittyvä tai ei. Pääsääntö: kyseessä on aihe ({topic_description}),
ei pelkkä sivumaininta tai aiheeseen liittyvä työkalu.

Esimerkkejä (AI/ML-aihe):

Kyllä:
- "GPT-5 julkaistu" → uusi LLM, selvästi AI-uutinen
- "Tutkijat kehittivät uuden koneoppimismenetelmän" → AI-tutkimus
- "AI:n vaikutus työelämään" → AI-aiheinen keskustelu, Relevantti
- "Mallin arvioinnista selvisi että..." → AI-malli keskustelussa
- "Inflect-Micro-v2: TTS-malli julkaistu" → puheen syntetisointi, koneoppiminen

Ei:
- "Python-lintteri Ruff päivitetty" → ei AI:ää, pelkkä kehitystyökalu
- "Uusi TypeScript-kääntäjä" → ohjelmointikieli, ei tekoälyä
- "Robottipysäköinti lentoasemalla" → robotiikka ilman AI/ML-yhteyttä
- "GrapheneOS puhelimen suojelu" → tietoturva, ei tekoälyä

Vastaa VAIN JSON-muodossa:
{{"results": [{{"index": 0, "keep": true}}, {{"index": 1, "keep": false}}, ...]}}"""

    return system_prompt, user_prompt


def filter_topic(candidates: list[Candidate],
                 topic_description: str,
                 llm_call) -> FilterResult:
    """Filters candidates by topic relevance.

    Non-critical: on any failure, returns all candidates unchanged with a warning.
    """
    if not candidates:
        return FilterResult(candidates=[], n_kept=0, n_dropped=0, warning=None)

    system_prompt, user_prompt = build_filter_prompt(candidates, topic_description)
    raw_response = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = TopicFilterResponse(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        warning = f"Topic filter degraded: model response is not valid JSON ({e})"
        logger.warning(warning)
        return FilterResult(candidates=candidates, n_kept=len(candidates),
                            n_dropped=0, warning=warning)

    keep_indices = {j.index for j in response.results if j.keep}
    kept = [c for i, c in enumerate(candidates) if i in keep_indices]
    dropped = len(candidates) - len(kept)

    if not kept:
        warning = "Topic filter dropped all candidates — passing all through to scoring"
        logger.warning(warning)
        return FilterResult(candidates=candidates, n_kept=len(candidates),
                            n_dropped=0, warning=warning)

    logger.info("topic filter: %d/%d candidates kept", len(kept), len(candidates))
    return FilterResult(candidates=kept, n_kept=len(kept),
                        n_dropped=dropped, warning=None)
