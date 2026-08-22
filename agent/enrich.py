"""
Enrich-step: fetches full article content for items selected by the Score
step. No LLM calls — pure HTTP + HTML extraction.

Done ONLY after scoring, not for all candidates (see pipeline_design.md)
— saves HTTP calls and context, since full content is only fetched for
stories that actually make it into the digest.

Non-critical step (see README "Failure-policy per step"): if one item's
fetch/extraction fails (e.g. paywall, down page, JS-rendered content that
trafilatura cannot recognize), that item is dropped from the digest.
Degradation is NOT silent — each drop is logged and returned as a warning,
same principle as in the cluster step.
"""

import logging
from typing import NamedTuple
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

from agent.schema import ScoredCandidate, EnrichedCandidate, DroppedStory

logger = logging.getLogger(__name__)

USER_AGENT = "ai-briefing-agent/0.1 (+https://github.com/samporapeli/ainikki; henkilokohtainen projekti)"

# Context management: truncate article length before passing to Compose step.
# Rough cut — can be improved later (e.g. cut at paragraph boundary instead
# of at exact character limit).
DEFAULT_MAX_CHARS = 6000


def fetch_html_raw(url: str, client: httpx.Client) -> str:
    """Fetches raw HTML from a page. Real HTTP call — won't work in this
    sandbox, but works as-is in a normal environment."""
    resp = client.get(url, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.text


def extract_article_text(html: str, url: str | None = None,
                          max_chars: int = DEFAULT_MAX_CHARS) -> str | None:
    """Pure function: HTML -> cleaned article text or None if no usable
    content was found (e.g. paywall/consent wall/blank page).
    Testable without network, see tests/test_enrich.py.
    """
    text = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=False,
        favor_precision=True,
    )
    if not text or len(text.strip()) < 100:
        return None  # too little/no content — likely failed extraction
    return text[:max_chars]


def _is_allowed_by_robots(url: str, client: httpx.Client,
                           cache: dict[str, bool]) -> bool:
    """Checks robots.txt to see if fetching the given URL is allowed.
    Caches the robots.txt result per domain within one enrich_candidates() call.

    If robots.txt fetching fails (404, timeout, error), assumes allowed.
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
        # If robots.txt is unavailable or parse fails, assume allowed
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
        content = None
        content_source = None
        failures: list[str] = []

        for source in sc.items:
            url_str = str(source.url)
            if not _is_allowed_by_robots(url_str, client, robots_cache):
                failures.append(f"'{source.title}' estetty robots.txt:llä")
                continue

            try:
                html = fetch_html_raw(url_str, client=client)
                content = extract_article_text(html, url=url_str, max_chars=max_chars)
            except httpx.HTTPError as e:
                failures.append(f"'{source.title}': {e}")
                continue

            if content is None:
                failures.append(f"'{source.title}': sisältöä ei voitu poimia")
                continue

            content_source = source
            break

        if content is None or content_source is None:
            msg = f"Rikastaminen epäonnistui '{primary.title}' ({primary.url})"
            if failures:
                msg += ": " + "; ".join(failures)
            msg += " - pudotettu koosteesta"
            logger.warning(msg)
            warnings.append(msg)
            dropped_stories.append(DroppedStory(title=primary.title, url=str(primary.url)))
            continue

        if content_source is not primary:
            msg = (f"Rikastaminen epäonnistui ensisijaisesta lähteestä '{primary.title}'; "
                   f"käytetään lähdettä '{content_source.title}'")
            logger.warning(msg)
            warnings.append(msg)

        kept.append(EnrichedCandidate(
            items=sc.items, cluster_reason=sc.cluster_reason,
            n_urls=sc.n_urls,
            rank=sc.rank, selection_reason=sc.selection_reason,
            content=content, content_source=content_source,
        ))

    return EnrichResult(items=kept, dropped_stories=dropped_stories, warnings=warnings)
