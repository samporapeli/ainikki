"""
RSS/Atom feed adapter. Follows the same pattern as hn.py:
- parse_rss_feed() is a pure function (testable without network)
- fetch_and_parse_rss() is the convenience function (real HTTP)

Each RSS entry becomes a RawItem with source_type=rss.
The feed URL is the origin_id seed (stable per feed).
"""

import logging
from datetime import datetime, timezone
from typing import Any

import feedparser

from agent.schema import RawItem, SourceType

logger = logging.getLogger(__name__)


def parse_rss_feed(raw_xml: str, feed_url: str,
                   fetched_at: datetime | None = None,
                   since: datetime | None = None,
                   until: datetime | None = None) -> list[RawItem]:
    """Pure function: RSS/Atom XML string -> list[RawItem].
    Handles both RSS 2.0 and Atom feeds.

    Each entry's link must be an absolute http(s) URL; relative or
    missing links produce a warning and the entry is skipped.

    If since/until are provided, entries outside the window are skipped.
    Entries without a parsed published_at are kept (we can't filter).
    """
    parsed = feedparser.parse(raw_xml)
    now = fetched_at or datetime.now(timezone.utc)
    items: list[RawItem] = []

    for entry in parsed.entries:
        title = entry.get("title", "").strip()
        if not title:
            continue

        link = _resolve_link(entry, feed_url)
        if not link:
            continue

        published = _parse_published(entry)
        if published and since and published < since:
            continue
        if published and until and published >= until:
            continue
        categories = entry.get("tags", [])
        tag_names = [t.get("term", "") for t in categories if t.get("term")]
        summary_fragment = entry.get("summary", "") or ""
        summary_len = len(summary_fragment.strip())

        items.append(RawItem(
            title=title,
            url=link,
            source_type=SourceType.rss,
            published_at=published,
            raw_signal={"summary_chars": summary_len, "tags": ",".join(tag_names)},
            origin_id=f"rss:{feed_url}:{entry.get('id', entry.get('link', ''))}",
            fetched_at=now,
        ))

    logger.info("rss parse: %d entries from %s", len(items), feed_url)
    return items


def _resolve_link(entry: dict[str, Any], feed_url: str) -> str | None:
    link = entry.get("link", "")
    if not link:
        return None
    # feedparser may return a dict for Atom entries with rel=alternate
    if isinstance(link, dict):
        link = link.get("href", "")
    link = str(link).strip()
    if not link.startswith("http"):
        return None
    return link


def _parse_published(entry: dict[str, Any]) -> datetime | None:
    raw = entry.get("published_parsed") or entry.get("updated_parsed")
    if not raw:
        return None
    try:
        return datetime(*raw[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def fetch_and_parse_rss(rss_url: str, client: Any = None,
                        since: datetime | None = None,
                        until: datetime | None = None) -> list[RawItem]:
    if client is not None:
        resp = client.get(rss_url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        raw_xml = resp.text
    else:
        import httpx
        with httpx.Client(timeout=15.0, follow_redirects=True) as c:
            resp = c.get(rss_url)
            resp.raise_for_status()
            raw_xml = resp.text

    return parse_rss_feed(raw_xml, rss_url, since=since, until=until)