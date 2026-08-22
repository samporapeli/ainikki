"""
Cluster-step: groups candidates that cover the same story together,
even if they have different URLs (e.g. TechCrunch and the original
company blog both covering the same release). This differs from the
dedup step, which only merged literally identical URLs.

The LLM call is injected as a parameter (Protocol LlmCall) so that:
1) it can be tested without a real model/network (see tests/test_cluster.py)
2) model independence is maintained — agent/llm.py (later) just implements
   this same interface, reading config/models.yaml for which model to use.

Context is kept intentionally small: the prompt only receives the title,
source type, and signal per candidate — not full article content.
"""

import json
import logging
import re
from typing import Protocol, NamedTuple

from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import Candidate, ClusteredCandidate

logger = logging.getLogger(__name__)


class LlmCall(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        """Returns (model response text, usage dict).
        Usage dict has keys: prompt_tokens, completion_tokens, total_tokens, cost.
        """
        ...


class ClusterResult(NamedTuple):
    """Cluster step result + visible warning if structural fallback was triggered.
    warning=None means clustering succeeded normally.
    The calling pipeline.py decides whether to add the warning to
    Briefing.warnings, but does not crash the run — cluster is a
    non-critical step, degrading to singletons is safe (no wrong data,
    just lost optimization).
    """
    clusters: list[ClusteredCandidate]
    warning: str | None


class ClusterAssignment(BaseModel):
    candidate_indices: list[int]
    primary_position: int  # zero-based position within candidate_indices
    reason: str | None = None


class ClusterResponse(BaseModel):
    clusters: list[ClusterAssignment]


_TITLE_STOPWORDS = {
    "a", "an", "and", "are", "for", "from", "how", "in", "is", "of",
    "on", "the", "to", "with", "new", "this", "that", "when", "what",
    "ai", "tekoäly", "uusi", "uuden", "ja", "että",
}


def _title_terms(title: str) -> set[str]:
    words = re.findall(r"[\w-]{4,}", title.lower())
    return {word for word in words if word not in _TITLE_STOPWORDS}


def _evidence_groups(indices: list[int], candidates: list[Candidate]) -> list[list[int]]:
    """Split a model cluster into groups with concrete title overlap."""
    terms = [_title_terms(candidates[idx].primary.title) for idx in indices]
    groups: list[list[int]] = []
    unseen = set(range(len(indices)))
    while unseen:
        component = {unseen.pop()}
        changed = True
        while changed:
            changed = False
            for i in list(unseen):
                if any(len(terms[i] & terms[j]) >= 2 for j in component):
                    unseen.remove(i)
                    component.add(i)
                    changed = True
        groups.append([indices[i] for i in sorted(component)])
    return groups


SYSTEM_PROMPT = """Saat listan uutisehdokkaita otsikoineen.

Tämä on varmennusvaihe: tarkista, löytyykö ehdokkaista eri URL:eilla
artikkeleita, jotka käsittelevät täsmälleen samaa konkreettista uutistapahtumaa.
Ryhmittele vain sellaiset ehdokkaat, joiden voit otsikoiden perusteella varmasti
tunnistaa samaksi tarinaksi.

ÄLÄ ryhmittele yhteisen teeman perusteella. Sama yritys, tuote, teknologia,
aihepiiri, käyttötapaus tai yleinen vaikutus ei riitä. Esimerkiksi kaksi eri
mallijulkaisua, kaksi eri agenttityökalua tai kaksi tekoälyn vaikutuksia käsittelevää
artikkelia ovat erillisiä uutisia, elleivät ne käsittele täsmälleen samaa tapahtumaa.
Palauta vain vähintään kaksi ehdokasta sisältävät ryhmät. Singleton-ehdokkaita ei
tarvitse palauttaa: koodi käsittelee kaikki puuttuvat ehdokkaat automaattisesti
erillisinä ryhminä. Jos et löydä varmasti samaa tarinaa käsitteleviä ehdokkaita,
palauta tyhjä lista.

Valitse jokaiselle ryhmälle päälähde (primary_position): ilmoita sen nollapohjainen
paikka candidate_indices-listassa. Suosi alkuperäistä julkaisijaa
ja suoraa lähdettä aggregaattorin, keskustelulangan tai myöhemmän kommentaarin sijaan.
Kirjoita reason-kenttään konkreettinen perustelu: mainitse yhteinen tuote, julkaisu,
tapahtuma tai päätös ja kerro, miksi eri URL:t käsittelevät samaa tarinaa. Älä käytä
pelkkää ilmausta kuten "sama julkaisu" tai "sama aihe".

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{"clusters": [{"candidate_indices": [2, 7], "primary_position": 0, "reason": "Molemmat käsittelevät Aurora-mallin samaa julkaisua eri URL:eilla."}]}
Jos sopivia ryhmiä ei ole:
{"clusters": []}"""


def build_cluster_prompt(candidates: list[Candidate]) -> tuple[str, str]:
    lines = []
    for i, c in enumerate(candidates):
        primary = c.primary
        signal_str = ", ".join(f"{k}={v}" for k, v in primary.raw_signal.items())
        published = primary.published_at.isoformat() if primary.published_at else "tuntematon"
        lines.append(
            f"[{i}] {primary.title} (URL: {primary.url}, lähde: {primary.source_type.value}, "
            f"julkaistu: {published}, {signal_str})"
        )
    user_prompt = "Ehdokkaat:\n" + "\n".join(lines)
    return SYSTEM_PROMPT, user_prompt


def _validate_assignments(response: ClusterResponse, n_candidates: int) -> bool:
    """Validates the indices in explicitly returned multi-item groups."""
    seen: set[int] = set()
    for cluster in response.clusters:
        if not cluster.candidate_indices:
            return False
        if cluster.primary_position < 0 or cluster.primary_position >= len(cluster.candidate_indices):
            return False
        for index in cluster.candidate_indices:
            if index < 0 or index >= n_candidates or index in seen:
                return False
            seen.add(index)
    return True


def _recover_missing(response: ClusterResponse, n_candidates: int,
                     candidates: list[Candidate]) -> ClusterResult:
    """Accept partial clustering: use what the model got right,
    treat missing/duplicate indices as singletons."""
    seen: set[int] = set()
    valid_clusters = []
    rejected_merges = 0
    for cluster in response.clusters:
        clean_indices = [i for i in cluster.candidate_indices if i not in seen
                         and 0 <= i < n_candidates]
        if not clean_indices:
            continue
        for i in clean_indices:
            seen.add(i)
        groups = _evidence_groups(clean_indices, candidates)
        rejected_merges += max(0, len(groups) - 1)
        primary = min(max(cluster.primary_position, 0), len(clean_indices) - 1)
        primary_idx = clean_indices[primary]
        for group in groups:
            group_primary = primary_idx if primary_idx in group else group[0]
            ordered_indices = [group_primary] + [idx for idx in group if idx != group_primary]
            valid_clusters.append(ClusteredCandidate(
                items=[item for idx in ordered_indices for item in candidates[idx].items],
                cluster_reason=(
                    cluster.reason if len(group) > 1 or len(clean_indices) == 1
                    else "fallback: insufficient title evidence for merge"
                ),
                n_urls=len(ordered_indices),
            ))

    missing = [i for i in range(n_candidates) if i not in seen]
    for idx in missing:
        valid_clusters.append(ClusteredCandidate(
            items=candidates[idx].items,
            cluster_reason="fallback: not covered by model response",
            n_urls=1,
        ))

    if missing:
        logger.info("cluster: recovered %d missing indices as singletons", len(missing))
    if rejected_merges:
        logger.info("cluster: rejected %d merge(s) without shared title evidence", rejected_merges)
    return ClusterResult(clusters=valid_clusters, warning=None)


def _fallback_singletons(candidates: list[Candidate], reason: str) -> ClusterResult:
    """If the model response doesn't validate, don't crash — degrade safely
    to a state where each candidate is its own cluster (same outcome as
    without this step). Clustering benefit is lost on this run, but
    nothing breaks. Degradation is NOT silent: logged at WARNING level
    and a warning message is returned for Briefing.warnings.
    """
    warning = f"Clustering degraded to singletons: {reason}"
    logger.warning(warning)
    clusters = [ClusteredCandidate(items=c.items, cluster_reason="fallback: clustering failed")
                for c in candidates]
    return ClusterResult(clusters=clusters, warning=warning)


def cluster_candidates(candidates: list[Candidate], llm_call: LlmCall) -> ClusterResult:
    if not candidates:
        return ClusterResult(clusters=[], warning=None)

    system_prompt, user_prompt = build_cluster_prompt(candidates)
    raw_response, _usage = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = ClusterResponse(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        return _fallback_singletons(candidates, f"model response is not valid JSON ({e})")

    if not _validate_assignments(response, len(candidates)):
        return _fallback_singletons(candidates, "model response has invalid candidate assignments")

    if len({i for c in response.clusters for i in c.candidate_indices}) < len(candidates):
        return _recover_missing(response, len(candidates), candidates)

    result = []
    rejected_merges = 0
    for cluster in response.clusters:
        groups = _evidence_groups(cluster.candidate_indices, candidates)
        if len(groups) > 1:
            rejected_merges += len(groups) - 1
        for group in groups:
            primary_idx = cluster.candidate_indices[cluster.primary_position]
            if primary_idx not in group:
                primary_idx = group[0]
            ordered_indices = [primary_idx] + [idx for idx in group if idx != primary_idx]
            merged_items = [item for idx in ordered_indices for item in candidates[idx].items]
            result.append(ClusteredCandidate(
                items=merged_items,
                cluster_reason=(
                    cluster.reason if len(group) > 1 or len(cluster.candidate_indices) == 1
                    else "fallback: insufficient title evidence for merge"
                ),
                n_urls=len(ordered_indices),
            ))

    warning = None
    if rejected_merges:
        warning = f"Clustering rejected {rejected_merges} merge(s) without shared title evidence"
        logger.warning(warning)
    return ClusterResult(clusters=result, warning=warning)
