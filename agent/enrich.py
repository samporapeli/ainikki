"""
Enrich-step: hakee täyden artikkelisisällön Score-stepin valitsemille
itemeille. Ei LLM-kutsuja - puhdas HTTP + HTML-erottelu.

Tehdään VASTA scoringin jälkeen, ei kaikille kandidaateille (ks.
pipeline_design.md) - säästää verkkokutsuja ja kontekstia, koska
haetaan täysi sisältö vain jutuille jotka oikeasti päätyvät koosteeseen.

Ei-kriittinen step (ks. README "Failure-policy stepeittäin"): jos yhden
itemin haku/erottelu epäonnistuu (esim. paywall, sivu alhaalla, JS-renderöity
sisältö jota trafilatura ei tunnista), se item pudotetaan koosteesta.
Degradointi EI ole hiljaista - jokainen pudotus lokitetaan ja palautuu
warning-listana, sama periaate kuin cluster-stepissä.
"""

import logging
from typing import NamedTuple
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from agent.schema import ScoredCandidate, EnrichedCandidate, DroppedStory

logger = logging.getLogger(__name__)

USER_AGENT = "ai-briefing-agent/0.1 (+https://github.com/samporapeli/ainikki; henkilokohtainen projekti)"

# Kontekstin hallinta: rajataan artikkelin pituus ennen Compose-stepille
# antamista. Karkea leikkaus - parannettavissa myöhemmin (esim. leikkaa
# viimeisen kappaleen rajalta täyden merkkirajan sijaan).
DEFAULT_MAX_CHARS = 6000


def fetch_html_raw(url: str, client: httpx.Client) -> str:
    """Hakee sivun raa'an HTML:n. Oikea verkkokutsu - ei toimi tässä
    sandboxissa, mutta toimii sellaisenaan normaalissa ympäristössä."""
    resp = client.get(url, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def extract_article_text(html: str, url: str | None = None,
                          max_chars: int = DEFAULT_MAX_CHARS) -> str | None:
    """Puhdas funktio: HTML -> puhdistettu artikkelisisältö tai None jos
    käyttökelpoista sisältöä ei löytynyt (esim. paywall/consent-seinä/tyhjä sivu).
    Testattavissa ilman verkkoyhteyttä, ks. tests/test_enrich.py.
    """
    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False,
        favor_precision=True,
    )
    if not text or len(text.strip()) < 100:
        return None  # liian vähän/ei mitään sisältöä - todennäköisesti epäonnistunut erottelu
    return text[:max_chars]


def _is_allowed_by_robots(url: str, client: httpx.Client,
                           cache: dict[str, bool]) -> bool:
    """Tarkistaa robots.txt:n salliako haun annetulle URL:ille.
    Cachee robots.txt-tuloksen per domain yhden enrich_candidates()-kutsun sisällä.

    Jos robots.txt:n haku epäonnistuu (404, timeout, virhe), oletetaan sallittu.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin in cache:
        return cache[origin]

    robots_url = f"{origin}/robots.txt"
    try:
        resp = client.get(robots_url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        allowed = rp.can_fetch(USER_AGENT, url)
    except Exception:
        # Jos robots.txt ei ole saatavilla tai-parse epäonnistuu, oletetaan sallittu
        allowed = True

    cache[origin] = allowed
    return allowed


class EnrichResult(NamedTuple):
    items: list[EnrichedCandidate]
    dropped_stories: list[DroppedStory]
    warnings: list[str]


def enrich_candidates(scored: list[ScoredCandidate], client: httpx.Client,
                       max_chars: int = DEFAULT_MAX_CHARS) -> EnrichResult:
    kept: list[EnrichedCandidate] = []
    dropped_stories: list[DroppedStory] = []
    warnings: list[str] = []
    robots_cache: dict[str, bool] = {}

    for sc in scored:
        primary = sc.items[0]
        url_str = str(primary.url)

        if not _is_allowed_by_robots(url_str, client, robots_cache):
            msg = f"enrich estetty robots.txt:llä '{primary.title}' ({primary.url}) - pudotettu koosteesta"
            logger.warning(msg)
            warnings.append(msg)
            dropped_stories.append(DroppedStory(title=primary.title, url=url_str))
            continue

        content = None
        try:
            html = fetch_html_raw(url_str, client=client)
            content = extract_article_text(html, url=url_str, max_chars=max_chars)
        except httpx.HTTPError as e:
            content = None
            error_note = str(e)
        else:
            error_note = "sisältöä ei pystytty erottamaan (paywall/tyhjä sivu?)" if content is None else None

        if content is None:
            msg = f"enrich epäonnistui '{primary.title}' ({primary.url}): {error_note} - pudotettu koosteesta"
            logger.warning(msg)
            warnings.append(msg)
            dropped_stories.append(DroppedStory(title=primary.title, url=url_str))
            continue

        kept.append(EnrichedCandidate(
            items=sc.items, cluster_reason=sc.cluster_reason,
            rank=sc.rank, selection_reason=sc.selection_reason,
            content=content,
        ))

    return EnrichResult(items=kept, dropped_stories=dropped_stories, warnings=warnings)
