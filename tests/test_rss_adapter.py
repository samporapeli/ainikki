import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent.collect.rss import parse_rss_feed, fetch_and_parse_rss

FIXTURE = Path("tests/fixtures/rss_sample.xml").read_text(encoding="utf-8")


def test_parse_rss_feed_returns_correct_items():
    items = parse_rss_feed(FIXTURE, "https://example.com/feed")
    assert len(items) == 3  # the empty <item/> is skipped

    ai = items[0]
    assert ai.title == "A story about AI and machine learning"
    assert str(ai.url) == "https://example.com/ai-story"
    assert ai.source_type.value == "rss"
    assert ai.origin_id.startswith("rss:")
    assert ai.published_at is not None
    assert "AI" in ai.raw_signal.get("tags", "")

    linux = items[1]
    assert "Linux" in linux.raw_signal.get("tags", "")

    nodate = items[2]
    assert nodate.title == "Story without published date"
    assert nodate.published_at is None


def test_parse_rss_feed_sets_fetched_at():
    now = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
    items = parse_rss_feed(FIXTURE, "https://example.com/feed", fetched_at=now)
    for item in items:
        assert item.fetched_at == now


def test_parse_rss_feed_empty():
    items = parse_rss_feed("", "https://example.com/feed")
    assert items == []


def test_fetch_and_parse_rss_with_mock():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=FIXTURE)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    items = fetch_and_parse_rss("https://example.com/feed", client=client)
    assert len(items) == 3
    assert str(items[0].url) == "https://example.com/ai-story"