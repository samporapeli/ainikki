import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from agent.enrich import extract_article_text, enrich_candidates, DEFAULT_MAX_CHARS
from agent.schema import RawItem, SourceType, ScoredCandidate

CLEAN_HTML = Path("tests/fixtures/article_clean.html").read_text()
PAYWALL_HTML = Path("tests/fixtures/article_paywall.html").read_text()

ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /\n"
ROBOTS_DISALLOW_ALL = "User-agent: *\nDisallow: /\n"


def _make_scored(title: str, url: str, rank: int) -> ScoredCandidate:
    item = RawItem(
        title=title, url=url, source_type=SourceType.hn,
        published_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
        raw_signal={"points": 100}, origin_id=f"test_{rank}",
        fetched_at=datetime.now(timezone.utc),
    )
    return ScoredCandidate(items=[item], cluster_reason=None, rank=rank,
                            selection_reason="testivalinta")


def test_extract_clean_article():
    text = extract_article_text(CLEAN_HTML, url="https://example.com/article")
    assert text is not None
    assert "Mythos 5" in text
    assert "export controls" in text
    # navigation/footer/comments must not leak through
    assert "Copyright 2026 Anthropic" not in text
    assert "user123" not in text


def test_extract_paywall_returns_none():
    text = extract_article_text(PAYWALL_HTML, url="https://example.com/paywalled")
    assert text is None


def test_max_chars_truncation():
    long_paragraphs = "\n\n".join(
        f"<p>Tämä on testikappale numero {i} joka toistaa samaa sisältöä pituuden kasvattamiseksi "
        f"jotta max_chars-rajaus tulee varmasti testattua kunnolla ilman ongelmia.</p>"
        for i in range(200)
    )
    html = f"<html><body><article>{long_paragraphs}</article></body></html>"
    text = extract_article_text(html, max_chars=500)
    assert text is not None
    assert len(text) <= 500


def test_enrich_candidates_mixed_outcomes():
    """3 kandidaattia: 1 onnistuu, 1 on paywallin takana, 1 palauttaa HTTP 500 -
    varmistaa että vain onnistunut jää jäljelle ja molemmat epäonnistumiset näkyvät warningeina."""
    scored = [
        _make_scored("Toimiva artikkeli", "https://good.example.com/article", rank=1),
        _make_scored("Tilaajaseinän takana", "https://paywall.example.com/article", rank=2),
        _make_scored("Palvelin kaatuu", "https://broken.example.com/article", rank=3),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "good.example.com":
            return httpx.Response(200, text=CLEAN_HTML)
        elif host == "paywall.example.com":
            return httpx.Response(200, text=PAYWALL_HTML)
        elif host == "broken.example.com":
            return httpx.Response(500, text="Internal Server Error")
        raise AssertionError(f"odottamaton host: {host}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert len(result.items) == 1, f"odotettiin 1 onnistunutta, saatiin {len(result.items)}"
    assert result.items[0].items[0].title == "Toimiva artikkeli"
    assert result.items[0].content is not None
    assert "Mythos 5" in result.items[0].content

    assert len(result.warnings) == 2, f"odotettiin 2 warningia, saatiin {len(result.warnings)}"
    assert any("Tilaajaseinän" in w for w in result.warnings)
    assert any("Palvelin kaatuu" in w for w in result.warnings)

    assert len(result.dropped_stories) == 2, f"odotettiin 2 pudotettua juttua, saatiin {len(result.dropped_stories)}"
    assert any(d.title == "Tilaajaseinän takana" for d in result.dropped_stories)
    assert any(d.title == "Palvelin kaatuu" for d in result.dropped_stories)


def test_enrich_preserves_scoring_fields():
    """Varmistaa ettei rank/selection_reason/cluster_reason katoa enrich-vaiheessa."""
    scored = [_make_scored("Artikkeli", "https://good2.example.com/article", rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=CLEAN_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert result.items[0].rank == 1
    assert result.items[0].selection_reason == "testivalinta"


def test_robots_txt_allows_all():
    """Host whose robots.txt allows everything -> item is processed normally."""
    scored = [_make_scored("Sallittu artikkeli", "https://allowed.example.com/article", rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        return httpx.Response(200, text=CLEAN_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert len(result.items) == 1
    assert len(result.warnings) == 0
    assert "Mythos 5" in result.items[0].content


def test_robots_txt_disallows_all():
    """Host whose robots.txt disallows everything -> item is dropped, warning mentions robots.txt."""
    scored = [_make_scored("Kielletty artikkeli", "https://blocked.example.com/article", rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ROBOTS_DISALLOW_ALL)
        return httpx.Response(200, text=CLEAN_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert len(result.items) == 0
    assert len(result.warnings) == 1
    assert "robots.txt" in result.warnings[0]
    assert "Kielletty artikkeli" in result.warnings[0]
    assert len(result.dropped_stories) == 1
    assert result.dropped_stories[0].title == "Kielletty artikkeli"


def test_robots_txt_404_assumed_allowed():
    """If robots.txt is unavailable (404), assume allowed -> item is processed."""
    scored = [_make_scored("Artikkeli ilman robots.txt", "https://no-robots.example.com/article", rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text=CLEAN_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert len(result.items) == 1
    assert len(result.warnings) == 0


def test_robots_cache_per_domain():
    """Verifies robots.txt is fetched only once per domain (cache works)."""
    scored = [
        _make_scored("Artikkeli A", "https://cached.example.com/article-a", rank=1),
        _make_scored("Artikkeli B", "https://cached.example.com/article-b", rank=2),
    ]

    robots_fetch_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            robots_fetch_count["n"] += 1
            return httpx.Response(200, text=ROBOTS_ALLOW_ALL)
        return httpx.Response(200, text=CLEAN_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = enrich_candidates(scored, client=client)

    assert len(result.items) == 2
    assert robots_fetch_count["n"] == 1, f"robots.txt pitäisi haetaa vain kerran, haettiin {robots_fetch_count['n']} kertaa"

