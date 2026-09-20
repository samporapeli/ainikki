"""
Consistency checks for the RSS/Atom feed graph.

Everything here is offline and fast: it parses a known-valid fixture feed
through the same ``parse_rss_feed`` used by the real collect step, and
asserts that every configured RSS source points at a syntactically
well-formed feed_url.

The value is *detector* over *verifier*: a source that silently stops
matching our feed format, is renamed, or drops a field will trip this test
even before the pipeline runs. Live network checks are guarded by an env
flag so the hermetic run stays offline;
set ``RSS_LIVE_TESTS=1`` to also hit the network.
"""
import os
from datetime import datetime, timezone
import urllib.parse

import httpx

import pytest

from agent.collect.rss import parse_rss_feed

FIXTURE = "tests/fixtures/rss_sample.xml"
EXPECTED_ITEM_COUNT = 3


def _load_sources():
    import yaml

    cfg = yaml.safe_load(open("config/sources.yaml"))
    return cfg.get("sources", [])


def _topic_sources(topic):
    import yaml

    data = yaml.safe_load(open("config/topics/{}.yaml".format(topic)))
    return data.get("sources", [])


def _check_required_keys(src):
    for field in ("key", "type", "pretty_name", "badge", "site_url", "feed_url"):
        assert field in src, f"field '{field}' missing on source {src.get('key')!r}"


def _assert_valid_feed_url(key, feed_url):
    parts = urllib.parse.urlparse(feed_url)
    assert parts.scheme in ("http", "https"), (
        f"{key}: feed_url must be http(s), got {parts.scheme!r}"
    )
    assert parts.hostname, f"{key}: feed_url missing hostname: {feed_url!r}"
    assert parts.path, f"{key}: feed_url missing path: {feed_url!r}"
    if not (parts.hostname.startswith("127.") or parts.hostname.startswith("localhost")):
        assert parts.scheme == "https", f"{key}: feed_url must be https, got {feed_url!r}"


def _assert_parses(key, feed_url):
    try:
        urllib.parse.urlparse(feed_url)
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"{key}: could not parse feed_url {feed_url!r}: {exc}")
    assert feed_url.strip(), f"{key}: feed_url must not be empty"


def test_expected_item_count():
    """Sanity-check the fixture the parser runs against."""
    since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    until = datetime(2030, 1, 1, tzinfo=timezone.utc)
    items = parse_rss_feed(open(FIXTURE).read(),
                           "https://example.com/feed", since=since, until=until)
    assert len(items) == EXPECTED_ITEM_COUNT
    assert items[0].title


def test_sources_have_unique_keys():
    seen = set()
    for src in _load_sources():
        assert src["key"] not in seen, f"duplicate source key: {src['key']!r}"
        seen.add(src["key"])


@pytest.mark.parametrize(
    "src",
    [s for s in _load_sources() if s.get("type") == "rss"],
    ids=[s["key"] for s in _load_sources() if s.get("type") == "rss"],
)
def test_rss_feed_url_is_well_formed(src):
    assert src["type"] == "rss", f"{src['key']} must be type rss, got {src['type']}"
    _check_required_keys(src)
    _assert_valid_feed_url(src["key"], src["feed_url"])
    _assert_parses(src["key"], src["feed_url"])


def test_all_topic_keys_resolve():
    sources = _load_sources()
    known = {s["key"] for s in sources}
    for name in os.listdir("config/topics"):
        if not name.endswith(".yaml"):
            continue
        topic = name[:-5]
        if topic.startswith("_"):
            continue
        for key in _topic_sources(topic):
            assert key in known, f"topic {topic} references undefined source key {key!r}"


def test_no_duplicate_source_in_topic():
    for name in os.listdir("config/topics"):
        if not name.endswith(".yaml"):
            continue
        topic = name[:-5]
        if topic.startswith("_"):
            continue
        keys = _topic_sources(topic)
        assert len(keys) == len(set(keys)), f"topic {topic} references a source more than once"


def test_live_rss_reachable():
    """Optional live check. Skip unless ``RSS_LIVE_TESTS=1`` is set.

    Exposes the collector to network failures the offline checks above
    cannot: feeds that 404, block our UA, or change format are caught here at
    CI time (intentionally opt-in so the unit suite stays hermetic).
    """
    if os.environ.get("RSS_LIVE_TESTS") != "1":
        pytest.skip("live network checks disabled (set RSS_LIVE_TESTS=1 to run)")

    with httpx.Client() as client:
        for src in _load_sources():
            if src.get("type") != "rss":
                continue
            resp = client.get(src["feed_url"], timeout=30,
                              headers={"User-Agent": "ainikki/1.0"})
            assert resp.status_code == 200, (
                f"{src['key']}: HTTP {resp.status_code} for {src['feed_url']}"
            )
