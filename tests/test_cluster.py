import json
from datetime import datetime, timezone
from pathlib import Path

from agent.cluster import cluster_candidates, build_cluster_prompt
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits
from agent.schema import RawItem, SourceType, Candidate


def _build_test_candidates() -> list[Candidate]:
    """9 candidates: 8 from the original HN fixture + 1 TechCrunch article
    covering the SAME Anthropic story at a different URL than the HN link.
    Dedup does not merge these (different URL) — that's exactly the
    Cluster step's job.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])

    techcrunch_item = RawItem(
        title="Anthropic's new Fable and Mythos models draw mixed reactions",
        url="https://techcrunch.com/2026/07/14/anthropic-fable-mythos-reaction/",
        source_type=SourceType.other,
        published_at=datetime(2026, 7, 14, 15, 0, tzinfo=timezone.utc),
        raw_signal={"outlet": "techcrunch"},
        origin_id="tc_1",
        fetched_at=datetime.now(timezone.utc),
    )

    return dedup_candidates(hn_items + [techcrunch_item])


def _mock_llm_group_anthropic_stories(system_prompt: str, user_prompt: str) -> str:
    """Simulates model response: finds the Anthropic and TechCrunch lines
    in the prompt and groups them, rest become singletons."""
    lines = user_prompt.splitlines()[1:]  # skip "Candidates:" header line
    anthropic_idx = next(i for i, l in enumerate(lines) if "Fable 5 and Claude Mythos 5" in l)
    techcrunch_idx = next(i for i, l in enumerate(lines) if "mixed reactions" in l)
    all_indices = set(range(len(lines)))
    remaining = all_indices - {anthropic_idx, techcrunch_idx}

    clusters = [{"candidate_indices": [anthropic_idx, techcrunch_idx], "primary_index": 0,
                 "reason": "molemmat käsittelevät Fable/Mythos-julkaisua"}]
    for idx in remaining:
        clusters.append({"candidate_indices": [idx], "primary_index": 0, "reason": None})

    return json.dumps({"clusters": clusters})


def test_cross_url_clustering():
    candidates = _build_test_candidates()
    assert len(candidates) == 9, f"expected 9 candidates before clustering, got {len(candidates)}"

    result = cluster_candidates(candidates, _mock_llm_group_anthropic_stories)
    clustered = result.clusters

    assert result.warning is None, "successful clustering should not produce a warning"
    assert len(clustered) == 8, f"expected 8 groups after clustering, got {len(clustered)}"

    merged = next(c for c in clustered if len(c.items) == 2)
    assert merged.items[0].source_type == SourceType.hn, "primary source should be HN (company news), not TechCrunch"
    assert merged.items[1].source_type == SourceType.other
    assert merged.cluster_reason is not None


def test_singletons_preserved():
    candidates = _build_test_candidates()
    result = cluster_candidates(candidates, _mock_llm_group_anthropic_stories)
    singleton_titles = {c.items[0].title for c in result.clusters if len(c.items) == 1}
    assert "New paper: chain-of-thought reasoning does not always reflect model computation" in singleton_titles


def test_fallback_on_malformed_json():
    candidates = _build_test_candidates()
    result = cluster_candidates(candidates, lambda s, u: "tämä ei ole JSON:ia ollenkaan")
    assert len(result.clusters) == len(candidates), "fallback should have the same number of groups as candidates"
    assert all(c.cluster_reason and "fallback" in c.cluster_reason for c in result.clusters)
    assert result.warning is not None, "fallback must NOT be silent - warning must be returned"
    assert "degraded" in result.warning


def test_fallback_on_incomplete_coverage():
    """Model forgot one candidate entirely in the response — must detect and fallback."""
    candidates = _build_test_candidates()

    def bad_response(system_prompt: str, user_prompt: str) -> str:
        # only first 5 candidates mentioned out of 9 - incomplete coverage
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(5)]
        return json.dumps({"clusters": clusters})

    result = cluster_candidates(candidates, bad_response)
    assert len(result.clusters) == len(candidates)
    assert all("fallback" in (c.cluster_reason or "") for c in result.clusters)
    assert result.warning is not None


def test_prompt_stays_lightweight():
    """Verifies that full article content doesn't leak into the prompt —
    only headline, source type, and signal per candidate."""
    candidates = _build_test_candidates()
    _, user_prompt = build_cluster_prompt(candidates)
    assert "story_text" not in user_prompt
    assert len(user_prompt) < 2000, f"prompt unexpectedly long ({len(user_prompt)} chars) - context growing?"


def test_code_fenced_json_is_parsed():
    """Verifies that model responses wrapped in code fences are cleaned up."""
    candidates = _build_test_candidates()

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None}
                    for i in range(len(candidates))]
        inner = json.dumps({"clusters": clusters})
        return f"```json\n{inner}\n```"

    result = cluster_candidates(candidates, mock_llm_code_fence)
    assert result.warning is None
    assert len(result.clusters) == len(candidates)

