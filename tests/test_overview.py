import json
from typing import Tuple

from agent.overview import build_overview_prompt, generate_overview
from agent.schema import NewsItem, Source, SourceType


def _make_news_item(headline: str, summary: str, rank: int) -> NewsItem:
    source = Source(url=f"https://example.com/{rank}", title=headline, source_type=SourceType.hn)
    return NewsItem(headline=headline, summary=summary, sources=[source], rank=rank,
                     selection_reason="testi")


def test_build_overview_prompt_uses_only_headline_and_summary():
    items = [
        _make_news_item("Uusi malli julkaistu", "Lyhyt kuvaus mallista.", rank=2),
        _make_news_item("Tutkimus yllättää", "Tutkijat löysivät jotain uutta.", rank=1),
    ]
    _, user_prompt = build_overview_prompt(items, "testi-aihe")

    assert "Tutkimus yllättää" in user_prompt
    assert "Uusi malli julkaistu" in user_prompt
    assert not user_prompt.startswith("Jutut"), "should not prefix the list"
    assert "Tämän päivän" not in user_prompt, "should not prime filler framing"
    # rank order: lowest rank first
    assert user_prompt.index("1. Tutkimus yllättää") < user_prompt.index("2. Uusi malli julkaistu")


def test_generate_overview_happy_path():
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1),
             _make_news_item("Juttu B", "Kuvaus B.", rank=2)]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return json.dumps({"overview": "Päivän isoin aihe oli uusi mallijulkaisu.",
                           "digest_topic": "Uusi mallijulkaisu puhuttaa"}), {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80, "cost": 0.0001}

    result = generate_overview(items, mock_llm, "testi-aihe")
    assert result.overview == "Päivän isoin aihe oli uusi mallijulkaisu."
    assert result.digest_topic == "Uusi mallijulkaisu puhuttaa"
    assert result.warning is None


def test_generate_overview_fallback_on_malformed_response():
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1),
             _make_news_item("Juttu B", "Kuvaus B.", rank=2),
             _make_news_item("Juttu C", "Kuvaus C.", rank=3)]

    result = generate_overview(items, lambda s, u: ("ei json:ia ollenkaan", {}), "testi-aihe")

    assert result.warning is not None
    assert "fallback" in result.warning
    assert "Juttu A" in result.overview
    assert "Juttu B" in result.overview
    assert "Juttu C" in result.overview
    assert result.digest_topic == "testi-aihe"


def test_generate_overview_empty_items_skips_llm_call():
    call_count = {"n": 0}

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        call_count["n"] += 1
        return json.dumps({"overview": "ei pitäisi tulla tänne",
                           "digest_topic": "ei pitäisi"}), {}

    result = generate_overview([], mock_llm, "testi-aihe")

    assert call_count["n"] == 0, "with empty item list, LLM should not be called at all"
    assert "Ei julkaistavia" in result.overview
    assert result.digest_topic == ""
    assert result.warning is None


def test_fallback_respects_rank_order_not_list_order():
    """Verifies fallback uses the rank field, not the list's original order."""
    items = [
        _make_news_item("Kolmas", "...", rank=3),
        _make_news_item("Ensimmäinen", "...", rank=1),
        _make_news_item("Toinen", "...", rank=2),
    ]
    result = generate_overview(items, lambda s, u: ("rikki", {}), "testi-aihe")
    idx_first = result.overview.index("Ensimmäinen")
    idx_second = result.overview.index("Toinen")
    idx_third = result.overview.index("Kolmas")
    assert idx_first < idx_second < idx_third


def test_code_fenced_json_is_parsed():
    """Verifies that model responses wrapped in code fences are cleaned up."""
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1)]

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        inner = json.dumps({"overview": "Koodiblokki-yleiskatsaus.",
                            "digest_topic": "Koodiblokin aihe"})
        return f"```json\n{inner}\n```", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00001}

    result = generate_overview(items, mock_llm_code_fence, "testi-aihe")
    assert result.overview == "Koodiblokki-yleiskatsaus."
    assert result.digest_topic == "Koodiblokin aihe"
    assert result.warning is None

