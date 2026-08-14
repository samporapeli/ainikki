from datetime import date

import pytest

from agent.validate import assemble_briefing, EmptyBriefingError
from agent.compose import ComposeResult
from agent.overview import OverviewResult
from agent.schema import NewsItem, Source, SourceType, Period


def _make_news_item(rank: int) -> NewsItem:
    source = Source(url=f"https://example.com/{rank}", title=f"Otsikko {rank}", source_type=SourceType.hn)
    return NewsItem(headline=f"Headline {rank}", summary=f"Summary {rank}.", sources=[source],
                     rank=rank, selection_reason="testi")


def test_assemble_briefing_happy_path():
    compose_result = ComposeResult(
        items=[_make_news_item(1), _make_news_item(2)], warnings=[],
        persona_id="ainikki-v1", guardrails_version="v1", golden_examples_version="v1",
    )
    overview_result = OverviewResult(overview="Päivän kooste.", digest_topic="", warning=None)

    briefing = assemble_briefing(
        topic="ai", period=Period.daily, period_start=date(2026, 7, 16), period_end=date(2026, 7, 16),
        overview_result=overview_result, compose_result=compose_result,
        models_used={"score": "openrouter/openai/gpt-4o-mini"}, pipeline_version="0.1.0",
        rubric_version="ai-v1",
    )

    assert briefing.topic == "ai"
    assert len(briefing.items) == 2
    assert briefing.meta.persona == "ainikki-v1"
    assert briefing.meta.rubric_version == "ai-v1"
    assert briefing.warnings == []


def test_assemble_briefing_collects_warnings_from_all_steps():
    compose_result = ComposeResult(
        items=[_make_news_item(1)], warnings=[],
        persona_id="ainikki-v1", guardrails_version="v1", golden_examples_version="v1",
    )
    overview_result = OverviewResult(overview="Fallback-kooste.", digest_topic="", warning="overview fallbackasi")

    briefing = assemble_briefing(
        topic="ai", period=Period.daily, period_start=date(2026, 7, 16), period_end=date(2026, 7, 16),
        overview_result=overview_result, compose_result=compose_result,
        models_used={}, pipeline_version="0.1.0", rubric_version="ai-v1",
        extra_warnings=["cluster degradoitui singletoneihin", "compose epäonnistui jutulle X"],
    )

    assert len(briefing.warnings) == 3
    assert "cluster degradoitui" in briefing.warnings[0]
    assert "compose epäonnistui" in briefing.warnings[1]
    assert "overview fallbackasi" in briefing.warnings[2]


def test_assemble_briefing_raises_on_empty_items():
    compose_result = ComposeResult(
        items=[], warnings=["kaikki epäonnistuivat"],
        persona_id="ainikki-v1", guardrails_version="v1", golden_examples_version="v1",
    )
    overview_result = OverviewResult(overview="Ei julkaistavia juttuja tälle ajalle.", digest_topic="", warning=None)

    with pytest.raises(EmptyBriefingError, match="ai"):
        assemble_briefing(
            topic="ai", period=Period.daily, period_start=date(2026, 7, 16), period_end=date(2026, 7, 16),
            overview_result=overview_result, compose_result=compose_result,
            models_used={}, pipeline_version="0.1.0", rubric_version="ai-v1",
        )

