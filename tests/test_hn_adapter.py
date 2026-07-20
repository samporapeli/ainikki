import json
from pathlib import Path

from agent.collect.hn import parse_hn_hits
from agent.collect.base import save_raw, load_raw
from agent.schema import SourceType


def test_parse_basic_fields():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])

    assert len(items) == 8, f"expected 8 items, got {len(items)}"

    first = items[0]
    assert first.title == "Anthropic announces Claude Fable 5 and Claude Mythos 5"
    assert str(first.url) == "https://www.anthropic.com/news/fable-mythos"
    assert first.source_type == SourceType.hn
    assert first.raw_signal["points"] == 412
    assert first.raw_signal["num_comments"] == 231
    assert first.origin_id == "hn_111111"


def test_null_url_fallback():
    """Ask HN post has no url field — must fall back to HN's own page."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])

    ask_hn_item = next(i for i in items if i.origin_id == "hn_333333")
    assert str(ask_hn_item.url) == "https://news.ycombinator.com/item?id=333333", \
        f"fallback URL wrong: {ask_hn_item.url}"


def test_published_at_parsed():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])
    assert items[0].published_at is not None
    assert items[0].published_at.year == 2026


def test_cache_roundtrip(tmp_path):
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])

    save_raw("ai", "2026-07-14", "hn", items, data_dir=tmp_path)
    loaded = load_raw("ai", "2026-07-14", "hn", data_dir=tmp_path)

    assert len(loaded) == len(items)
    assert loaded[0].title == items[0].title
    assert str(loaded[0].url) == str(items[0].url)

