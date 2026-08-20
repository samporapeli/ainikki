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
    """Cluster step result + visible warning if fallback was triggered.
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
    primary_index: int  # index within candidate_indices list, not global
    reason: str | None = None


class ClusterResponse(BaseModel):
    clusters: list[ClusterAssignment]


SYSTEM_PROMPT = """Olet uutisanalyytikko. Saat listan uutisehdokkaita otsikoineen.
Tehtäväsi: ryhmittele ehdokkaat jotka käsittelevät SAMAA tarinaa/tapahtumaa yhteen,
vaikka ne olisivat eri lähteistä eri URL:eilla. ÄLÄ yhdistä ehdokkaita jotka vain
käsittelevät samaa aihepiiriä (esim. kaksi eri mallijulkaisua ei ole sama tarina).

Jokaisen ehdokkaan pitää kuulua täsmälleen yhteen ryhmään, myös ne joilla ei ole paria
(ne muodostavat oman yhden-ehdokkaan ryhmänsä).

Valitse jokaiselle ryhmälle ehdotettu päälähde (primary_index): suosi alkuperäistä
julkaisijaa (esim. yhtiön oma blogi) aggregaattorin/keskustelulangan sijaan.

Vastaa VAIN JSON-muodossa, ei muuta tekstiä:
{"clusters": [{"candidate_indices": [0, 3], "primary_index": 0, "reason": "molemmat käsittelevät samaa julkaisua"}, ...]}"""


def build_cluster_prompt(candidates: list[Candidate]) -> tuple[str, str]:
    lines = []
    for i, c in enumerate(candidates):
        primary = c.primary
        signal_str = ", ".join(f"{k}={v}" for k, v in primary.raw_signal.items())
        lines.append(f"[{i}] {primary.title} (lähde: {primary.source_type.value}, {signal_str})")
    user_prompt = "Ehdokkaat:\n" + "\n".join(lines)
    return SYSTEM_PROMPT, user_prompt


def _validate_full_coverage(response: ClusterResponse, n_candidates: int) -> bool:
    """Ensures every input index appears exactly once.
    This is the safety net that prevents model errors (missing/duplicated
    index, index out of bounds) from propagating down the pipeline.
    """
    seen: list[int] = []
    for cluster in response.clusters:
        if not cluster.candidate_indices:
            return False
        if cluster.primary_index < 0 or cluster.primary_index >= len(cluster.candidate_indices):
            return False
        seen.extend(cluster.candidate_indices)

    if len(seen) != n_candidates:
        return False
    if set(seen) != set(range(n_candidates)):
        return False
    if len(seen) != len(set(seen)):  # duplicate index in some cluster
        return False
    return True


def _recover_missing(response: ClusterResponse, n_candidates: int,
                     candidates: list[Candidate]) -> ClusterResult:
    """Accept partial clustering: use what the model got right,
    treat missing/duplicate indices as singletons."""
    seen: set[int] = set()
    valid_clusters = []
    for cluster in response.clusters:
        clean_indices = [i for i in cluster.candidate_indices if i not in seen
                         and 0 <= i < n_candidates]
        if not clean_indices:
            continue
        for i in clean_indices:
            seen.add(i)
        primary = min(cluster.primary_index, len(clean_indices) - 1)
        valid_clusters.append(ClusteredCandidate(
            items=[item for idx in clean_indices for item in candidates[idx].items],
            cluster_reason=cluster.reason,
        ))

    missing = [i for i in range(n_candidates) if i not in seen]
    for idx in missing:
        valid_clusters.append(ClusteredCandidate(
            items=candidates[idx].items,
            cluster_reason="fallback: not covered by model response",
        ))

    if missing:
        logger.info("cluster: recovered %d missing indices as singletons", len(missing))
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

    if not _validate_full_coverage(response, len(candidates)):
        return _recover_missing(response, len(candidates), candidates)

    result = []
    for cluster in response.clusters:
        ordered_indices = [cluster.candidate_indices[cluster.primary_index]] + [
            idx for j, idx in enumerate(cluster.candidate_indices) if j != cluster.primary_index
        ]
        merged_items = []
        for idx in ordered_indices:
            merged_items.extend(candidates[idx].items)
        result.append(ClusteredCandidate(items=merged_items, cluster_reason=cluster.reason))

    return ClusterResult(clusters=result, warning=None)
