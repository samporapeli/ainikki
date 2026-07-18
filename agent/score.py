"""
Score-step: valitsee ja järjestää klusteroidut candidaatit rubriikin (ks.
config/rubrics/*.yaml) perusteella. Tämä on KRIITTINEN step (ks. README.md
"Failure-policy stepeittäin") - toisin kuin cluster-step, tässä EI degradoida
hiljaisesti jos mallin vastaus on rikki. Ei ole periaatteellista tapaa arvata
"mitkä jutut ovat tärkeitä" jos structured output epäonnistuu - siksi
score_clusters() heittää ScoreValidationError-poikkeuksen sen sijaan että
tuottaisi mielivaltaisen/väärän valinnan.

Konteksti pidetään pienenä kuten cluster-stepissäkin: promptiin menee vain
otsikko, lähdetyyppi, yhdistetty signaali ja lähteiden lukumäärä per
klusteroitu candidate - ei täyttä artikkelisisältöä.
"""

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import ClusteredCandidate, ScoredCandidate
from agent.cluster import LlmCall


class ScoreValidationError(Exception):
    """Mallin vastaus ei täyttänyt rakenteellisia tai rubriikin asettamia
    vaatimuksia. Tarkoituksella EI fallbackata - ks. moduulin docstring."""


class ScoreAssignment(BaseModel):
    candidate_index: int
    rank: int
    selection_reason: str


class ScoreResponse(BaseModel):
    selected: list[ScoreAssignment]


def load_rubric(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_score_prompt(clusters: list[ClusteredCandidate], rubric: dict) -> tuple[str, str]:
    criteria_lines = []
    for c in rubric.get("criteria", []):
        criteria_lines.append(f"- {c['name']} (paino: {c['weight']}): {c['description'].strip()}")
    criteria_text = "\n".join(criteria_lines)

    exclude_lines = "\n".join(f"- {e}" for e in rubric.get("exclude", []))

    min_items = rubric["items_per_briefing"]["min"]
    max_items = rubric["items_per_briefing"]["max"]

    system_prompt = f"""Olet uutistoimittaja joka valitsee päivän tärkeimmät jutut annetusta
ehdokaslistasta. Käytä seuraavia kriteerejä arvioinnissa:

{criteria_text}

Älä valitse ehdokkaita jotka ovat:
{exclude_lines}

Valitse {min_items}-{max_items} ehdokasta, järjestä ne tärkeysjärjestykseen (1 = tärkein,
ei toistuvia sijoituksia), ja kirjoita jokaiselle lyhyt (1 lause) selection_reason joka
perustelee valinnan annetuilla kriteereillä.

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{{"selected": [{{"candidate_index": 0, "rank": 1, "selection_reason": "..."}}, ...]}}"""

    lines = []
    for i, cluster in enumerate(clusters):
        primary = cluster.items[0]
        total_signal = sum(
            v for item in cluster.items for v in item.raw_signal.values()
            if isinstance(v, (int, float))
        )
        lines.append(
            f"[{i}] {primary.title} "
            f"(lähde: {primary.source_type.value}, yhdistetty signaali: {total_signal}, "
            f"riippumattomia lähteitä: {len(cluster.items)})"
        )
    user_prompt = "Ehdokkaat:\n" + "\n".join(lines)

    return system_prompt, user_prompt


def _validate_response(response: ScoreResponse, n_candidates: int, min_items: int, max_items: int) -> None:
    n_selected = len(response.selected)
    if not (min_items <= n_selected <= max_items):
        raise ScoreValidationError(
            f"valittujen määrä ({n_selected}) rubriikin rajojen [{min_items}, {max_items}] ulkopuolella"
        )

    indices = [s.candidate_index for s in response.selected]
    if len(indices) != len(set(indices)):
        raise ScoreValidationError("sama candidate_index valittu useammin kuin kerran")
    if any(i < 0 or i >= n_candidates for i in indices):
        raise ScoreValidationError(f"candidate_index putoaa rajojen [0, {n_candidates - 1}] ulkopuolelle")

    ranks = sorted(s.rank for s in response.selected)
    if ranks != list(range(1, n_selected + 1)):
        raise ScoreValidationError(
            f"rank-arvojen pitää olla 1..{n_selected} ilman aukkoja/duplikaatteja, saatiin {ranks}"
        )


def score_clusters(clusters: list[ClusteredCandidate], rubric_path: Path,
                    llm_call: LlmCall) -> list[ScoredCandidate]:
    if not clusters:
        return []

    rubric = load_rubric(rubric_path)
    min_items = rubric["items_per_briefing"]["min"]
    max_items = rubric["items_per_briefing"]["max"]

    system_prompt, user_prompt = build_score_prompt(clusters, rubric)
    raw_response = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = ScoreResponse(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ScoreValidationError(f"mallin vastaus ei ole validia JSON:ia: {e}") from e

    _validate_response(response, len(clusters), min_items, max_items)

    result = [
        ScoredCandidate(
            items=clusters[s.candidate_index].items,
            cluster_reason=clusters[s.candidate_index].cluster_reason,
            rank=s.rank,
            selection_reason=s.selection_reason,
        )
        for s in response.selected
    ]
    result.sort(key=lambda sc: sc.rank)
    return result
