"""
Hacker News adapter using Algolia's public search API
(https://hn.algolia.com/api) — no authentication required.

Intentionally two separate functions:
- fetch_hn_raw(): makes the HTTP call, returns raw JSON
- parse_hn_hits(): pure function, converts JSON hits into RawItems

This separation allows testing the parsing logic without network access
(see tests/test_hn_adapter.py, uses a mock response).
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from agent.schema import RawItem, SourceType

ALGOLIA_BASE = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hn_raw(since: datetime, until: datetime, min_points: int = 20,
                  hits_per_page: int = 100, max_pages: int = 5,
                  client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Fetches HN stories from a time range via the Algolia API.
    Returns raw 'hit' dicts as-is; parsing is done separately.

    NOTE: this function makes real HTTP calls — won't work in this
    sandbox (network restricted), but works as-is when run in an
    environment with normal internet access.
    """
    owns_client = client is None
    client = client or httpx.Client(timeout=15.0)
    all_hits: list[dict[str, Any]] = []
    try:
        for page in range(max_pages):
            params = {
                "tags": "story",
                "numericFilters": (
                    f"created_at_i>={int(since.timestamp())},"
                    f"created_at_i<{int(until.timestamp())},"
                    f"points>={min_points}"
                ),
                "hitsPerPage": hits_per_page,
                "page": page,
            }
            resp = client.get(ALGOLIA_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", [])
            all_hits.extend(hits)
            if page >= data.get("nbPages", 1) - 1:
                break
    finally:
        if owns_client:
            client.close()
    return all_hits


def parse_hn_hits(hits: list[dict[str, Any]]) -> list[RawItem]:
    """Pure function: HN Algolia hits -> RawItem list.
    Also handles text posts (Ask HN etc.) that have no url field —
    falls back to HN's own discussion page URL for those.
    """
    items: list[RawItem] = []
    now = datetime.now(timezone.utc)

    for hit in hits:
        object_id = hit.get("objectID")
        if not object_id:
            continue  # cannot build fallback URL without id

        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        title = hit.get("title")
        if not title:
            continue  # no usable title, skip

        published_at = None
        if hit.get("created_at"):
            try:
                published_at = datetime.fromisoformat(hit["created_at"].replace("Z", "+00:00"))
            except ValueError:
                published_at = None

        items.append(RawItem(
            title=title,
            url=url,
            source_type=SourceType.hn,
            source_badge="HN",
            published_at=published_at,
            raw_signal={
                "points": hit.get("points") or 0,
                "num_comments": hit.get("num_comments") or 0,
            },
            origin_id=f"hn_{object_id}",
            fetched_at=now,
        ))

    return items


def fetch_hn(since: datetime, until: datetime, min_points: int = 20) -> list[RawItem]:
    """Convenience function: fetch and parse in one call."""
    raw_hits = fetch_hn_raw(since, until, min_points=min_points)
    return parse_hn_hits(raw_hits)
