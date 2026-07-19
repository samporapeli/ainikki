import json
from datetime import datetime, timezone
from pathlib import Path

from agent.dedup import normalize_url, dedup_candidates, save_candidates, load_candidates
from agent.collect.hn import parse_hn_hits
from agent.schema import RawItem, SourceType


def test_normalize_url_strips_tracking_and_www():
    a = "https://WWW.Example.com/article/some-post/?utm_source=hn&ref=news&id=1"
    b = "https://example.com/article/some-post?id=1"
    assert normalize_url(a) == normalize_url(b), \
        f"odotettiin samaa normalisoitua URL:ia:\n{normalize_url(a)}\n{normalize_url(b)}"



def test_normalize_url_keeps_different_paths_distinct():
    a = normalize_url("https://example.com/article-one")
    b = normalize_url("https://example.com/article-two")
    assert a != b


def test_dedup_merges_cross_source_duplicate():
    """Simuloi tilannetta jossa sama artikkeli tuli sekä HN:stä että RSS:stä."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])

    # same article (Anthropic news) also comes from "RSS source", different URL format
    rss_duplicate = RawItem(
        title="Anthropic launches Fable 5 and Mythos 5",
        url="https://www.anthropic.com/news/fable-mythos/?utm_source=rss",
        source_type=SourceType.rss,
        published_at=datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc),
        raw_signal={"feed_name": "anthropic-blog"},
        origin_id="rss_anthropic_1",
        fetched_at=datetime.now(timezone.utc),
    )

    all_items = hn_items + [rss_duplicate]
    candidates = dedup_candidates(all_items)

    # 4 uniikkia HN-storya, joista yksi (Anthropic) yhdistyy RSS-duplikaatin kanssa
    assert len(candidates) == 4, f"odotettiin 4 kandidaattia, saatiin {len(candidates)}"

    anthropic_candidate = next(
        c for c in candidates if "anthropic.com" in c.normalized_url
    )
    assert len(anthropic_candidate.items) == 2, \
        f"odotettiin 2 yhdistettyä itemiä, saatiin {len(anthropic_candidate.items)}"

    source_types = {i.source_type for i in anthropic_candidate.items}
    assert source_types == {SourceType.hn, SourceType.rss}


def test_dedup_signal_ordering():
    """Korkeamman yhdistetyn signaalin item pitäisi olla items[0] (primary)."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)

    anthropic_candidate = next(c for c in candidates if "anthropic.com" in c.normalized_url)
    # only item in this group without duplicates - verify primary still works
    assert anthropic_candidate.primary.raw_signal["points"] == 412


def test_cache_roundtrip(tmp_path):
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)

    save_candidates("ai", "2026-07-14", candidates, data_dir=tmp_path)
    loaded = load_candidates("ai", "2026-07-14", data_dir=tmp_path)

    assert len(loaded) == len(candidates)
    assert {c.normalized_url for c in loaded} == {c.normalized_url for c in candidates}

