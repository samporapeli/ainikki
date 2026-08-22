"""
Score-step: selects and ranks clustered candidates based on the rubric
(see config/rubrics/*.yaml). This is a CRITICAL step (see README.md
"Failure-policy per step") — unlike the cluster step, there is NO silent
degradation when the model response is broken. There is no principled way
to guess "which stories are important" if structured output fails — so
score_clusters() raises a ScoreValidationError instead of producing an
arbitrary/wrong selection.

Context is kept small as in the cluster step: the prompt only receives
the title and the number of distinct URLs per candidate — not full article
content or popularity signals.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import Briefing, ClusteredCandidate, ScoredCandidate
from agent.cluster import LlmCall


class ScoreValidationError(Exception):
    """Model response did not meet structural or rubric requirements.
    Intentionally NO fallback — see module docstring."""


class ScoreAssignment(BaseModel):
    candidate_index: int
    rank: int
    selection_reason: str


class ScoreResponse(BaseModel):
    selected: list[ScoreAssignment]
    cutoff_rank: int


class ScoreResult(NamedTuple):
    scored: list[ScoredCandidate]
    cutoff_rank: int


def load_rubric(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_previous_stories(topic: str, since: date, days: int = 7,
                          output_dir: Path = Path("data/output")) -> list[dict]:
    stories: list[dict] = []
    for path in output_dir.glob(f"{topic}_*.json"):
        match = re.search(r"(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not match:
            continue
        file_date = date.fromisoformat(match.group(1))
        if file_date > since or file_date < since - timedelta(days=days):
            continue
        try:
            briefing = Briefing(**json.loads(path.read_text()))
        except Exception:
            continue
        for item in briefing.items:
            url = str(item.sources[0].url) if item.sources else ""
            stories.append({"title": item.headline, "url": url})
    return stories


def _normalize_for_compare(url: str) -> str:
    return url.split("#")[0].rstrip("/")


def filter_previous_clusters(clusters: list[ClusteredCandidate],
                              previous_stories: list[dict]) -> list[ClusteredCandidate]:
    """Hard-filter: remove clusters whose primary URL was already published."""
    if not previous_stories:
        return clusters
    seen = {_normalize_for_compare(s["url"]) for s in previous_stories if s.get("url")}
    filtered = []
    for c in clusters:
        primary_url = _normalize_for_compare(str(c.items[0].url))
        if primary_url in seen:
            continue
        filtered.append(c)
    return filtered


def build_examples_text(examples: dict) -> str:
    """Build examples section for the score prompt from rubric YAML data."""
    good = examples.get("good", [])
    bad = examples.get("bad", [])
    parts = []
    if good:
        good_lines = "\n".join(f'   - "{ex["title"]}" -> {ex["reason"]}' for ex in good)
        parts.append(f"   Hyva valinta:\n{good_lines}")
    if bad:
        bad_lines = "\n".join(f'   - "{ex["title"]}" -> {ex["reason"]}' for ex in bad)
        parts.append(f"\n   Huono valinta:\n{bad_lines}")
    return "\n".join(parts)


def build_score_prompt(clusters: list[ClusteredCandidate], rubric: dict,
                       previous_stories: list[dict] | None = None) -> tuple[str, str]:
    criteria_lines = []
    for c in rubric.get("criteria", []):
        criteria_lines.append(f"- {c['name']} (paino: {c['weight']}): {c['description'].strip()}")
    criteria_text = "\n".join(criteria_lines)

    exclude_lines = "\n".join(f"- {e}" for e in rubric.get("exclude", []))

    examples = rubric.get("examples", {})
    examples_text = build_examples_text(examples)

    previous_lines = ""
    if previous_stories:
        items = "\n".join(f"- {s['title']} ({s['url']})" for s in previous_stories)
        previous_lines = f"""
Seuraavat jutut on jo käsitelty aiemmissa koosteissa. Älä valitse samoja uutisia uudelleen.
Jos juttua on jo käsitelty aiemmin, sen voi sisällyttää vain, jos kyseessä on uusi näkökulma
samaan tarinaan. Muussa tapauksessa juttu jätetään pois (jo sisällytetty aiemmin):

{items}
"""

    min_items = rubric["items_per_briefing"]["min"]
    max_items = rubric["items_per_briefing"]["max"]
    pool_max = 3 * max_items

    system_prompt = f"""Arvioit päivän uutisehdokaslistaa. Tehtävässäsi on kaksi osaa:

1. VALITSE: Käy LÄPI kaikki ehdokkaat ja valitse kaikki jotka ovat relevantteja
   tai kiinnostavia. Älä ole valikoiva — mieluumman liikaa kuin liian vähän.
   Tavoitale vähintään {pool_max} valittua, ellei listassa ole niin montaa relevanttia.
   Käytä seuraavia kriteerejä arvioinnissa:

{criteria_text}

   Älä valitse ehdokkaita jotka ovat:
{exclude_lines}

   Esimerkkejä valinnoista:
{examples_text}
{previous_lines}

   Jokaiselle valitulle: anna rank (1 = tärkein), ja lyhyt selection_reason.

2. CUTOFF: Määritä cutoff_rank — se on korkein rank, joka kuuluu päivän
   TÄRKEIMPIEN juttujen joukkoon. Ensimmäinen uutiskokoonpano tehdään
   cutoff_rankin asti (rank ≤ cutoff). cutoff_rank:n pitää olla vähintään
   {min_items} ja enintään {max_items}. cutoff_rank:n pitää olla AINA
   selvästi pienempi kuin valittujen kokonaismäärä, jotta backfill-poolissa
   on ehdokkaita jos ensisijaisia tarvitsee korvata.

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{{"selected": [{{"candidate_index": 0, "rank": 1, "selection_reason": "..."}}, ...],
 "cutoff_rank": 2}}"""

    lines = []
    for i, cluster in enumerate(clusters):
        primary = cluster.items[0]
        lines.append(f"[{i}] {primary.title}")
    user_prompt = "Ehdokkaat:\n" + "\n".join(lines)

    return system_prompt, user_prompt


def _normalize_response(response: ScoreResponse, n_candidates: int, max_items: int) -> ScoreResponse:
    """Fix common LLM failures: duplicate indices, rank gaps/duplicates,
    and selected count exceeding pool_max (capped silently instead of failing)."""
    pool_max = 3 * max_items
    seen: set[int] = set()
    deduped: list[ScoreAssignment] = []
    for s in sorted(response.selected, key=lambda x: x.rank):
        if s.candidate_index not in seen:
            seen.add(s.candidate_index)
            deduped.append(s)
    # Cap at pool_max — model occasionally overshoots, no need to crash
    deduped = deduped[:pool_max]
    for i, s in enumerate(deduped, 1):
        s.rank = i
    cutoff = min(response.cutoff_rank, len(deduped))
    return ScoreResponse(selected=deduped, cutoff_rank=cutoff)


def _validate_response(response: ScoreResponse, n_candidates: int, min_items: int, max_items: int) -> None:
    n_selected = len(response.selected)
    pool_max = 3 * max_items
    if not (min_items <= n_selected <= pool_max):
        raise ScoreValidationError(
            f"number of selected items ({n_selected}) outside bounds [{min_items}, {pool_max}]"
        )

    backfill_size = n_selected - response.cutoff_rank

    indices = [s.candidate_index for s in response.selected]
    if any(i < 0 or i >= n_candidates for i in indices):
        raise ScoreValidationError(f"candidate_index out of bounds [0, {n_candidates - 1}]")


def score_clusters(clusters: list[ClusteredCandidate], rubric_path: Path,
                   llm_call: LlmCall,
                   previous_stories: list[dict] | None = None) -> ScoreResult:
    if not clusters:
        return ScoreResult(scored=[], cutoff_rank=0)

    rubric = load_rubric(rubric_path)
    min_items = rubric["items_per_briefing"]["min"]
    max_items = rubric["items_per_briefing"]["max"]

    system_prompt, user_prompt = build_score_prompt(clusters, rubric, previous_stories)
    raw_response, _usage = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = ScoreResponse(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ScoreValidationError(f"model response is not valid JSON: {e}") from e

    response = _normalize_response(response, len(clusters), max_items)
    _validate_response(response, len(clusters), min_items, max_items)

    result = [
        ScoredCandidate(
            items=clusters[s.candidate_index].items,
            cluster_reason=clusters[s.candidate_index].cluster_reason,
            n_urls=clusters[s.candidate_index].n_urls,
            rank=s.rank,
            selection_reason=s.selection_reason,
        )
        for s in response.selected
    ]
    result.sort(key=lambda sc: sc.rank)
    return ScoreResult(scored=result, cutoff_rank=response.cutoff_rank)
