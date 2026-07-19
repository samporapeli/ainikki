"""
Cluster-step: ryhmittelee samaa tarinaa käsittelevät kandidaatit yhteen,
vaikka niillä olisi eri URL (esim. TechCrunch ja alkuperäinen yhtiön blogi
molemmat samasta julkaisusta). Tämä on ero dedup-stepiin, joka yhdisti
vain kirjaimellisesti saman URL:n.

LLM-kutsu on injektoitu parametrina (Protocol LlmCall), jotta:
1) tätä voi testata ilman oikeaa mallia/verkkoyhteyttä (ks. tests/test_cluster.py)
2) malliriippumattomuus säilyy - agent/llm.py (myöhemmin) vain toteuttaa
   tämän saman rajapinnan lukien config/models.yaml:sta mitä mallia käyttää

Konteksti pidetään pienenä tarkoituksella: promptiin menee VAIN otsikko,
lähdetyyppi ja signaali per candidate - ei täyttä artikkelisisältöä.
"""

import json
import logging
from typing import Protocol, NamedTuple

from pydantic import BaseModel, ValidationError

from agent.json_utils import strip_code_fences
from agent.schema import Candidate, ClusteredCandidate

logger = logging.getLogger(__name__)


class LlmCall(Protocol):
    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the model's raw response as text (expects JSON)."""
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
    primary_index: int  # indeksi candidate_indices-listan SISÄLLÄ, ei globaali
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
    if len(seen) != len(set(seen)):  # duplikaatti-indeksi jossain ryhmässä
        return False
    return True


def _fallback_singletons(candidates: list[Candidate], reason: str) -> ClusterResult:
    """If the model response doesn't validate, don't crash — degrade safely
    to a state where each candidate is its own cluster (same outcome as
    without this step). Clustering benefit is lost on this run, but
    nothing breaks. Degradation is NOT silent: logged at WARNING level
    and a warning message is returned for Briefing.warnings.
    """
    warning = f"Ryhmittely degradoitui singletoneihin: {reason}"
    logger.warning(warning)
    clusters = [ClusteredCandidate(items=c.items, cluster_reason="fallback: klusterointi epäonnistui")
                for c in candidates]
    return ClusterResult(clusters=clusters, warning=warning)


def cluster_candidates(candidates: list[Candidate], llm_call: LlmCall) -> ClusterResult:
    if not candidates:
        return ClusterResult(clusters=[], warning=None)

    system_prompt, user_prompt = build_cluster_prompt(candidates)
    raw_response = llm_call(system_prompt, user_prompt)

    try:
        parsed = json.loads(strip_code_fences(raw_response))
        response = ClusterResponse(**parsed)
    except (json.JSONDecodeError, ValidationError) as e:
        return _fallback_singletons(candidates, f"mallin vastaus ei ole validia JSON:ia ({e})")

    if not _validate_full_coverage(response, len(candidates)):
        return _fallback_singletons(candidates, "mallin vastaus ei kattanut kaikkia kandidaatteja täsmälleen kerran")

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
