"""
Testaa HN-adapterin parsintalogiikkaa mock-datalla, ei vaadi verkkoyhteyttä.
Ajo: python -m tests.test_hn_adapter (projektin juuresta)
"""

import json
import tempfile
import shutil
from pathlib import Path

from agent.collect.hn import parse_hn_hits
from agent.collect.base import save_raw, load_raw
from agent.schema import SourceType


def test_parse_basic_fields():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])

    assert len(items) == 4, f"odotettiin 4 itemiä, saatiin {len(items)}"

    first = items[0]
    assert first.title == "Anthropic announces Claude Fable 5 and Claude Mythos 5"
    assert str(first.url) == "https://www.anthropic.com/news/fable-mythos"
    assert first.source_type == SourceType.hn
    assert first.raw_signal["points"] == 412
    assert first.raw_signal["num_comments"] == 231
    assert first.origin_id == "hn_111111"


def test_null_url_fallback():
    """Ask HN -postauksella ei ole url-kenttää - pitää fallbackata HN:n omaan sivuun."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])

    ask_hn_item = next(i for i in items if i.origin_id == "hn_333333")
    assert str(ask_hn_item.url) == "https://news.ycombinator.com/item?id=333333", \
        f"fallback-URL väärin: {ask_hn_item.url}"


def test_published_at_parsed():
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    items = parse_hn_hits(fixture["hits"])
    assert items[0].published_at is not None
    assert items[0].published_at.year == 2026


def test_cache_roundtrip(tmp_path=None):
    """Varmistaa että save_raw/load_raw-kierros säilyttää datan ehjänä."""
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    try:
        fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
        items = parse_hn_hits(fixture["hits"])

        save_raw("ai", "2026-07-14", "hn", items, data_dir=tmp_path)
        loaded = load_raw("ai", "2026-07-14", "hn", data_dir=tmp_path)

        assert len(loaded) == len(items)
        assert loaded[0].title == items[0].title
        assert str(loaded[0].url) == str(items[0].url)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    test_parse_basic_fields()
    test_null_url_fallback()
    test_published_at_parsed()
    test_cache_roundtrip()
    print("\nKaikki testit läpi.")
