import json
from datetime import date
from pathlib import Path

from agent.write import write_briefing
from agent.schema import Briefing, GenerationMeta, NewsItem, Source, SourceType, Period
from datetime import datetime, timezone


def _make_briefing() -> Briefing:
    source = Source(url="https://example.com/1", title="Otsikko", source_type=SourceType.hn)
    item = NewsItem(headline="Headline", summary="Summary.", sources=[source], rank=1,
                     selection_reason="testi")
    meta = GenerationMeta(
        models_used={"score": "openrouter/openai/gpt-4o-mini"},
        generated_at=datetime.now(timezone.utc), pipeline_version="0.1.0",
        persona="ainikki-v1", rubric_version="ai-v1",
        guardrails_version="v1", golden_examples_version="v1",
    )
    return Briefing(topic="ai", period=Period.daily, period_start=date(2026, 7, 16),
                     period_end=date(2026, 7, 16), overview="Yleiskatsaus.", items=[item], meta=meta)


def test_write_briefing_creates_correctly_named_file(tmp_path):
    briefing = _make_briefing()
    path = write_briefing(briefing, output_dir=tmp_path)
    assert path.name == "ai_daily_2026-07-16.json"
    assert path.exists()


def test_write_briefing_content_is_valid_json_matching_schema(tmp_path):
    briefing = _make_briefing()
    path = write_briefing(briefing, output_dir=tmp_path)

    raw = json.loads(path.read_text())
    assert raw["topic"] == "ai"
    assert raw["items"][0]["headline"] == "Headline"
    assert raw["meta"]["persona"] == "ainikki-v1"

    reloaded = Briefing(**raw)
    assert reloaded.topic == briefing.topic


def test_write_briefing_roundtrip_preserves_item_count(tmp_path):
    briefing = _make_briefing()
    path = write_briefing(briefing, output_dir=tmp_path)
    reloaded = Briefing(**json.loads(path.read_text()))
    assert len(reloaded.items) == len(briefing.items)

