"""
Hacker News -adapteri, käyttää Algolian julkista hakurajapintaa
(https://hn.algolia.com/api) - ei vaadi autentikointia.

Tarkoituksella kaksi erillistä funktiota:
- fetch_hn_raw(): tekee verkkokutsun, palauttaa raakan JSON:in
- parse_hn_hits(): puhdas funktio, muuntaa JSON-hitit RawItemeiksi

Tämä jako mahdollistaa parse-logiikan testaamisen ilman verkkoyhteyttä
(ks. tests/test_hn_adapter.py, käyttää mock-vastausta).
"""

from datetime import datetime, timezone
from typing import Any

import httpx

from agent.schema import RawItem, SourceType

ALGOLIA_BASE = "https://hn.algolia.com/api/v1/search_by_date"


def fetch_hn_raw(since: datetime, until: datetime, min_points: int = 20,
                  hits_per_page: int = 100, max_pages: int = 5,
                  client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Hakee HN-storyt annetulta aikaväliltä Algolian API:sta.
    Palauttaa raakat 'hit'-dictit sellaisenaan, parsinta tehdään erikseen.

    HUOM: tämä funktio tekee oikean verkkokutsun - ei toimi tässä
    sandboxissa (verkko rajattu), mutta toimii sellaisenaan ajettuna
    ympäristössä jolla on normaali internet-pääsy.
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
    """Puhdas funktio: HN Algolia -hitit -> RawItem-lista.
    Käsittelee myös tekstipostaukset (Ask HN yms.) joilla ei ole url-kenttää -
    näille käytetään HN:n oman keskustelusivun URL:ia fallbackina.
    """
    items: list[RawItem] = []
    now = datetime.now(timezone.utc)

    for hit in hits:
        object_id = hit.get("objectID")
        if not object_id:
            continue  # ei voida muodostaa fallback-URL:ia ilman id:tä

        url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        title = hit.get("title")
        if not title:
            continue  # ei käyttökelpoista otsikkoa, ohitetaan

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
    """Kätevyysfunktio: hakee ja parsii yhdellä kutsulla."""
    raw_hits = fetch_hn_raw(since, until, min_points=min_points)
    return parse_hn_hits(raw_hits)
