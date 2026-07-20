import json
from pathlib import Path

import pytest

from agent.score import (build_score_prompt, load_rubric, load_previous_stories,
                          filter_previous_clusters, score_clusters, ScoreValidationError)
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits
from agent.schema import ClusteredCandidate

FIXTURE_RUBRIC = Path("tests/fixtures/rubric_test.yaml")
REAL_RUBRIC = Path("config/rubrics/ai_scoring_rubric_v1.yaml")


def _build_test_clusters() -> list[ClusteredCandidate]:
    """4 clustered candidates (here: singleton clusters — cluster step's
    exact behavior is not this test's responsibility, see tests/test_cluster.py)."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)
    return [ClusteredCandidate(items=c.items, cluster_reason=None) for c in candidates]


def test_build_score_prompt_includes_rubric_content():
    clusters = _build_test_clusters()
    rubric = load_rubric(FIXTURE_RUBRIC)
    system_prompt, user_prompt = build_score_prompt(clusters, rubric)

    assert "uutuusarvo" in system_prompt
    assert "markkinointisisältö" in system_prompt  # from exclude list
    assert "2-3" in system_prompt  # min-max from test rubric
    assert "story_text" not in user_prompt  # no full content in prompt


def test_score_clusters_happy_path():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # select candidates 0 and 3, rank order reversed to test sorting
        return json.dumps({"selected": [
            {"candidate_index": 3, "rank": 2, "selection_reason": "kiinnostava tutkimustulos"},
            {"candidate_index": 0, "rank": 1, "selection_reason": "laaja vaikutus, uusi mallijulkaisu"},
        ]})

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)

    assert len(result) == 2
    assert result[0].rank == 1
    assert result[0].items[0].title.startswith("Anthropic announces")
    assert result[1].rank == 2
    assert "tutkimustulos" in result[1].selection_reason


def test_raises_on_malformed_json():
    clusters = _build_test_clusters()
    with pytest.raises(ScoreValidationError, match="JSON"):
        score_clusters(clusters, FIXTURE_RUBRIC, lambda s, u: "ei json:ia")


def test_raises_on_count_outside_rubric_bounds():
    """Test rubric has min=2, max=3 — try selecting only 1 (too few)."""
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "ainoa valinta"},
        ]})

    with pytest.raises(ScoreValidationError, match="outside rubric bounds"):
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)


def test_raises_on_duplicate_candidate_index():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "a"},
            {"candidate_index": 0, "rank": 2, "selection_reason": "b"},
        ]})

    with pytest.raises(ScoreValidationError, match="more than once"):
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)


def test_raises_on_invalid_rank_sequence():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # ranks 1 and 1 (duplicate) instead of two selections with ranks 1 and 2
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "a"},
            {"candidate_index": 1, "rank": 1, "selection_reason": "b"},
        ]})

    with pytest.raises(ScoreValidationError, match="rank"):
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)


def test_prompt_builds_with_real_production_rubric():
    """Integration test with the real config/rubrics/ai_scoring_rubric_v1.yaml file,
    to ensure the test fixture does not drift out of sync with the real config."""
    clusters = _build_test_clusters()
    rubric = load_rubric(REAL_RUBRIC)
    system_prompt, user_prompt = build_score_prompt(clusters, rubric)
    assert "aihepiirin ydinolennaisuus" in system_prompt
    assert "aihepiirien monipuolisuus" in system_prompt
    assert "3-10" in system_prompt


def test_rejects_irrelevant_candidates():
    """Verifies score.py correctly handles the case where some candidates
    are rejected by relevance (mock LLM selects only a subset)."""
    clusters = _build_test_clusters()
    n_clusters = len(clusters)

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # select only the first 2, rest are "dropped" (not selected)
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 2, "selection_reason": "AI-aiheinen"},
            {"candidate_index": 1, "rank": 1, "selection_reason": "AI-aiheinen"},
        ]})

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)
    assert len(result) == 2, f"expected 2 selected, got {len(result)}"
    assert result[0].rank == 1
    assert result[1].rank == 2


def test_code_fenced_json_is_parsed():
    """Verifies that model responses wrapped in code fences are cleaned up."""
    clusters = _build_test_clusters()

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        inner = json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "tärkein"},
            {"candidate_index": 1, "rank": 2, "selection_reason": "toinen"},
        ]})
        return f"```json\n{inner}\n```"

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm_code_fence)
    assert len(result) == 2
    assert result[0].rank == 1


def _make_briefing_json(topic: str, date_str: str, headlines: list[str]) -> dict:
    items = []
    for i, h in enumerate(headlines):
        items.append({
            "headline": h,
            "summary": f"Yhteenveto jutulle {h}",
            "sources": [{"url": f"https://example.com/{date_str}/{i}", "title": h, "source_type": "hackernews"}],
            "rank": i + 1,
        })
    return {
        "topic": topic,
        "period": "daily",
        "period_start": date_str,
        "period_end": date_str,
        "overview": "Testi.",
        "items": items,
        "meta": {
            "models_used": {},
            "generated_at": f"{date_str}T12:00:00Z",
            "pipeline_version": "0.1.0",
            "persona": "test",
            "rubric_version": "v1",
            "guardrails_version": "v1",
            "golden_examples_version": "v1",
        },
    }


def test_load_previous_stories_returns_recent(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)

    briefing_3d = _make_briefing_json("ai", "2026-07-17", ["Story A", "Story B"])
    (tmp_path / "ai_daily_2026-07-17.json").write_text(json.dumps(briefing_3d))

    briefing_1d = _make_briefing_json("ai", "2026-07-19", ["Story C"])
    (tmp_path / "ai_daily_2026-07-19.json").write_text(json.dumps(briefing_1d))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    titles = [s["title"] for s in stories]
    assert "Story A" in titles
    assert "Story B" in titles
    assert "Story C" in titles


def test_load_previous_stories_excludes_outside_window(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)

    briefing_old = _make_briefing_json("ai", "2026-07-10", ["Old Story"])
    (tmp_path / "ai_daily_2026-07-10.json").write_text(json.dumps(briefing_old))

    briefing_recent = _make_briefing_json("ai", "2026-07-19", ["Recent Story"])
    (tmp_path / "ai_daily_2026-07-19.json").write_text(json.dumps(briefing_recent))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    titles = [s["title"] for s in stories]
    assert "Recent Story" in titles
    assert "Old Story" not in titles


def test_load_previous_stories_includes_previous_day(tmp_path):
    """The previous day's digest (file_date == since) should be included,
    as it covers stories from the day before the current run."""
    from datetime import date
    today = date(2026, 7, 20)

    briefing_prev = _make_briefing_json("ai", "2026-07-19", ["Yesterday Story"])
    (tmp_path / "ai_daily_2026-07-19.json").write_text(json.dumps(briefing_prev))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    assert len(stories) == 1
    assert stories[0]["title"] == "Yesterday Story"


def test_load_previous_stories_excludes_future(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)

    briefing_future = _make_briefing_json("ai", "2026-07-21", ["Future Story"])
    (tmp_path / "ai_daily_2026-07-21.json").write_text(json.dumps(briefing_future))

    briefing_current = _make_briefing_json("ai", "2026-07-20", ["Current Story"])
    (tmp_path / "ai_daily_2026-07-20.json").write_text(json.dumps(briefing_current))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    titles = [s["title"] for s in stories]
    assert "Current Story" in titles
    assert "Future Story" not in titles


def test_load_previous_stories_skips_malformed_files(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)

    (tmp_path / "ai_daily_2026-07-19.json").write_text("not json")
    briefing = _make_briefing_json("ai", "2026-07-18", ["Valid Story"])
    (tmp_path / "ai_daily_2026-07-18.json").write_text(json.dumps(briefing))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    assert len(stories) == 1
    assert stories[0]["title"] == "Valid Story"


def test_load_previous_stories_empty_dir(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)
    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    assert stories == []


def test_load_previous_stories_ignores_different_topic(tmp_path):
    from datetime import date
    today = date(2026, 7, 20)

    briefing_other = _make_briefing_json("ml", "2026-07-19", ["ML Story"])
    (tmp_path / "ml_daily_2026-07-19.json").write_text(json.dumps(briefing_other))

    stories = load_previous_stories("ai", today, output_dir=tmp_path)
    assert stories == []


def test_build_score_prompt_includes_previous_stories():
    clusters = _build_test_clusters()
    rubric = load_rubric(FIXTURE_RUBRIC)
    previous = [{"title": "Yesterday Story", "url": "https://example.com/1"}]

    system_prompt, _ = build_score_prompt(clusters, rubric, previous_stories=previous)
    assert "Yesterday Story" in system_prompt
    assert "aiemmissa koosteissa" in system_prompt


def test_build_score_prompt_omits_previous_when_empty():
    clusters = _build_test_clusters()
    rubric = load_rubric(FIXTURE_RUBRIC)

    system_prompt, _ = build_score_prompt(clusters, rubric, previous_stories=[])
    assert "aiemmissa koosteissa" not in system_prompt


def test_filter_previous_clusters_removes_matching_urls():
    from datetime import datetime, timezone
    from agent.schema import RawItem, SourceType

    item_a = RawItem(title="Story A", url="https://example.com/a", source_type=SourceType.hn,
                      raw_signal={"points": 100}, origin_id="1", fetched_at=datetime.now(timezone.utc))
    item_b = RawItem(title="Story B", url="https://example.com/b", source_type=SourceType.hn,
                      raw_signal={"points": 200}, origin_id="2", fetched_at=datetime.now(timezone.utc))
    clusters = [
        ClusteredCandidate(items=[item_a]),
        ClusteredCandidate(items=[item_b]),
    ]
    previous = [{"title": "Story A", "url": "https://example.com/a"}]

    result = filter_previous_clusters(clusters, previous)
    assert len(result) == 1
    assert result[0].items[0].title == "Story B"


def test_filter_previous_clusters_handles_fragment_urls():
    from datetime import datetime, timezone
    from agent.schema import RawItem, SourceType

    item = RawItem(title="Story", url="https://example.com/post#fnref:3",
                    source_type=SourceType.hn, raw_signal={"points": 100},
                    origin_id="1", fetched_at=datetime.now(timezone.utc))
    clusters = [ClusteredCandidate(items=[item])]
    previous = [{"title": "Story", "url": "https://example.com/post"}]

    result = filter_previous_clusters(clusters, previous)
    assert len(result) == 0


def test_filter_previous_clusters_handles_trailing_slash():
    from datetime import datetime, timezone
    from agent.schema import RawItem, SourceType

    item = RawItem(title="Story", url="https://example.com/post/",
                    source_type=SourceType.hn, raw_signal={"points": 100},
                    origin_id="1", fetched_at=datetime.now(timezone.utc))
    clusters = [ClusteredCandidate(items=[item])]
    previous = [{"title": "Story", "url": "https://example.com/post"}]

    result = filter_previous_clusters(clusters, previous)
    assert len(result) == 0


def test_filter_previous_clusters_empty_previous():
    from datetime import datetime, timezone
    from agent.schema import RawItem, SourceType

    item = RawItem(title="Story", url="https://example.com/a",
                    source_type=SourceType.hn, raw_signal={"points": 100},
                    origin_id="1", fetched_at=datetime.now(timezone.utc))
    clusters = [ClusteredCandidate(items=[item])]

    result = filter_previous_clusters(clusters, [])
    assert len(result) == 1

