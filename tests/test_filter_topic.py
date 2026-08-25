import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import yaml

from agent.filter_topic import filter_topic, build_filter_prompt
from agent.schema import Candidate, RawItem, SourceType


def _make_candidate(title: str, points: int = 100) -> Candidate:
    item = RawItem(
        title=title,
        url="https://example.com/test",
        source_type=SourceType.hn,
        published_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
        raw_signal={"points": points, "num_comments": 10},
        origin_id="hn_1",
        fetched_at=datetime.now(timezone.utc),
    )
    return Candidate(normalized_url="https://example.com/test", items=[item])


def test_filters_out_non_ai():
    candidates = [
        _make_candidate("GPT-5 julkaistu"),
        _make_candidate("Python-lintteri Ruff päivitetty"),
        _make_candidate("Robottipysäköinti lentoasemalla"),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return json.dumps({"results": [
            {"index": 0, "keep": True},
            {"index": 1, "keep": False},
            {"index": 2, "keep": False},
        ]}), {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75, "cost": 0.00005}

    result = filter_topic(candidates, "tekoäly ja koneoppiminen", mock_llm)
    assert result.n_kept == 1
    assert result.n_dropped == 2
    assert result.warning is None
    assert len(result.candidates) == 1
    assert "GPT-5" in result.candidates[0].primary.title


def test_keeps_all_when_all_relevant():
    candidates = [
        _make_candidate("GPT-5 julkaistu"),
        _make_candidate("Claude 5 arvosteltu"),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return json.dumps({"results": [
            {"index": 0, "keep": True},
            {"index": 1, "keep": True},
        ]}), {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75, "cost": 0.00005}

    result = filter_topic(candidates, "tekoäly", mock_llm)
    assert result.n_kept == 2
    assert result.n_dropped == 0
    assert result.warning is None


def test_fallback_when_all_dropped():
    """If the model drops everything, pass all through with a warning."""
    candidates = [
        _make_candidate("GPT-5 julkaistu"),
        _make_candidate("Claude 5 arvosteltu"),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return json.dumps({"results": [
            {"index": 0, "keep": False},
            {"index": 1, "keep": False},
        ]}), {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75, "cost": 0.00005}

    result = filter_topic(candidates, "tekoäly", mock_llm)
    assert result.n_kept == 2
    assert result.n_dropped == 0
    assert result.warning is not None
    assert "all" in result.warning.lower()


def test_fallback_on_malformed_json():
    candidates = [_make_candidate("GPT-5 julkaistu")]
    result = filter_topic(candidates, "tekoäly", lambda s, u: ("ei ole JSON:ia", {}))
    assert result.n_kept == 1
    assert result.warning is not None
    assert "degraded" in result.warning.lower()


def test_empty_input():
    result = filter_topic([], "tekoäly", lambda s, u: ("", {}))
    assert result.n_kept == 0
    assert result.warning is None


def test_prompt_stays_lightweight():
    candidates = [_make_candidate("GPT-5 julkaistu"), _make_candidate("Claude 5")]
    topic_description = """Tekoäly

PIDÄ:
- tekoälymallit ja AI-työkalut

JÄTÄ POIS:
- yleinen robotiikka ilman AI-yhteyttä"""
    system_prompt, user_prompt = build_filter_prompt(candidates, topic_description)
    assert "story_text" not in user_prompt
    assert len(user_prompt) < 1000
    assert "Aihe:\nTekoäly" in system_prompt
    assert "PIDÄ:" in system_prompt
    assert "JÄTÄ POIS:" in system_prompt
    assert "Arvioi vain otsikon perusteella." in system_prompt


def test_ai_filter_prompt_uses_explicit_scope():
    topic_data = yaml.safe_load(Path("config/topics/ai.yaml").read_text())
    system_prompt, _ = build_filter_prompt(
        [_make_candidate("Tekoäly muuttaa työelämää")], topic_data["topic_description"]
    )

    assert "tekoälyn vaikutukset työhön" in system_prompt
    assert "tekoälyyn liittyvä luottamus" in system_prompt
    assert "yleinen robotiikka" in system_prompt


def test_code_fenced_json_is_parsed():
    candidates = [_make_candidate("GPT-5 julkaistu")]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        inner = json.dumps({"results": [{"index": 0, "keep": True}]})
        return f"```json\n{inner}\n```", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00001}

    result = filter_topic(candidates, "tekoäly", mock_llm)
    assert result.n_kept == 1
    assert result.warning is None
