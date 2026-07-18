"""
Ajo: python -m tests.test_overview (projektin juuresta)
"""

import json

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
    _, user_prompt = build_overview_prompt(items)

    assert "Tutkimus yllättää" in user_prompt
    assert "Uusi malli julkaistu" in user_prompt
    # rank-järjestys: pienin rank ensin
    assert user_prompt.index("Tutkimus yllättää") < user_prompt.index("Uusi malli julkaistu")


def test_generate_overview_happy_path():
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1),
             _make_news_item("Juttu B", "Kuvaus B.", rank=2)]

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"overview": "Päivän isoin aihe oli uusi mallijulkaisu."})

    result = generate_overview(items, mock_llm)
    assert result.overview == "Päivän isoin aihe oli uusi mallijulkaisu."
    assert result.warning is None


def test_generate_overview_fallback_on_malformed_response():
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1),
             _make_news_item("Juttu B", "Kuvaus B.", rank=2),
             _make_news_item("Juttu C", "Kuvaus C.", rank=3)]

    result = generate_overview(items, lambda s, u: "ei json:ia ollenkaan")

    assert result.warning is not None
    assert "fallback" in result.warning
    assert "Juttu A" in result.overview
    assert "Juttu B" in result.overview
    assert "Juttu C" in result.overview


def test_generate_overview_empty_items_skips_llm_call():
    call_count = {"n": 0}

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        call_count["n"] += 1
        return json.dumps({"overview": "ei pitäisi tulla tänne"})

    result = generate_overview([], mock_llm)

    assert call_count["n"] == 0, "tyhjällä item-listalla ei pitäisi kutsua LLM:ää ollenkaan"
    assert "Ei julkaistavia" in result.overview
    assert result.warning is None


def test_fallback_respects_rank_order_not_list_order():
    """Varmistaa että fallback käyttää rank-kenttää, ei listan alkuperäistä järjestystä."""
    items = [
        _make_news_item("Kolmas", "...", rank=3),
        _make_news_item("Ensimmäinen", "...", rank=1),
        _make_news_item("Toinen", "...", rank=2),
    ]
    result = generate_overview(items, lambda s, u: "rikki")
    idx_first = result.overview.index("Ensimmäinen")
    idx_second = result.overview.index("Toinen")
    idx_third = result.overview.index("Kolmas")
    assert idx_first < idx_second < idx_third


def test_code_fenced_json_is_parsed():
    """Varmistaa että mallin koodiblokkiin käärityt JSON-vastaukset siivoutuvat."""
    items = [_make_news_item("Juttu A", "Kuvaus A.", rank=1)]

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        inner = json.dumps({"overview": "Koodiblokki-yleiskatsaus."})
        return f"```json\n{inner}\n```"

    result = generate_overview(items, mock_llm_code_fence)
    assert result.overview == "Koodiblokki-yleiskatsaus."
    assert result.warning is None


if __name__ == "__main__":
    test_build_overview_prompt_uses_only_headline_and_summary()
    test_generate_overview_happy_path()
    test_generate_overview_fallback_on_malformed_response()
    test_generate_overview_empty_items_skips_llm_call()
    test_fallback_respects_rank_order_not_list_order()
    test_code_fenced_json_is_parsed()
    print("\nKaikki testit läpi.")
