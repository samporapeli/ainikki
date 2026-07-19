"""
Pydantic-mallit koko putkelle. Ks. keskustelussa sovitut periaatteet:
- RawItem: collect-vaiheen tuotos, yksi per lähteen yksittäinen löytö
- Source/NewsItem/Briefing: lopullisen JSON:in muoto (klusteroinnin jälkeen)
"""

from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, HttpUrl, Field


class SourceType(str, Enum):
    hn = "hackernews"
    rss = "rss"
    reddit = "reddit"
    arxiv = "arxiv"
    other = "other"


# --- Collect-vaihe --------------------------------------------------------

class RawItem(BaseModel):
    """Single raw finding from one source, before dedup/clustering.
    Saved as-is to data/raw/{topic}/{date}/{source}.json.
    """
    title: str
    url: HttpUrl
    source_type: SourceType
    published_at: datetime | None = None
    raw_signal: dict[str, int | float | str] = Field(default_factory=dict)
    origin_id: str  # adapter-specific unique id (e.g. HN objectID), debug/tracing
    fetched_at: datetime


# --- Dedup-vaihe -----------------------------------------------------------

class Candidate(BaseModel):
    """Yksi uniikki URL (normalisoitu) ja kaikki RawItemit jotka siihen viittasivat.
    Ei vielä editorial-valintaa 'parhaasta' - se tehdään Cluster/Score-vaiheissa.
    items[0] on provisorinen (korkein yhdistetty signaali), ei lopullinen päälähde.
    """
    normalized_url: str
    items: list[RawItem]

    @property
    def primary(self) -> RawItem:
        return self.items[0]


class ClusteredCandidate(BaseModel):
    """RawItems for one story after clustering, possibly from different URLs.
    items[0] = model-suggested primary source. This is still provisional —
    Score step makes the final editorial judgment, this only groups.
    """
    items: list[RawItem]
    cluster_reason: str | None = None  # short model rationale for clustering, debug/transparency


# --- Post-clustering / final form ------------------------------------------

class Source(BaseModel):
    """A single link/source related to the story. [0] in list = primary source."""
    url: HttpUrl
    title: str
    source_type: SourceType
    published_at: datetime | None = None
    raw_signal: dict[str, int | float | str] = Field(default_factory=dict)


class ScoredCandidate(BaseModel):
    """ClusteredCandidate + editorial-arvio: sijoitus ja perustelu.
    Vain valitut kandidaatit päätyvät tähän muotoon - hylätyt eivät etene."""
    items: list[RawItem]
    cluster_reason: str | None = None
    rank: int
    selection_reason: str


class EnrichedCandidate(ScoredCandidate):
    """ScoredCandidate + cleaned article content from the primary source.
    Only the primary source (items[0]) is enriched in v1 — secondary sources
    stay at metadata level since compose uses only the primary in v1.
    """
    content: str | None = None  # None = fetch/extraction failed for this item


class DroppedStory(BaseModel):
    """Story that was selected in the score step but dropped after enrich
    (e.g. paywall, blank page, robots.txt). Shown to the reader."""
    title: str
    url: str


class NewsItem(BaseModel):
    headline: str
    summary: str
    sources: list[Source]
    rank: int
    tags: list[str] = Field(default_factory=list)
    selection_reason: str | None = None


class Period(str, Enum):
    daily = "daily"
    weekly = "weekly"
    monthly = "monthly"


class GenerationMeta(BaseModel):
    models_used: dict[str, str]
    generated_at: datetime
    pipeline_version: str
    persona: str
    rubric_version: str
    guardrails_version: str
    golden_examples_version: str


class Briefing(BaseModel):
    topic: str
    period: Period
    period_start: date
    period_end: date
    display_date: date | None = None  # päivä joka näkyy käyttäjälle; None = period_start
    display_date_fi: str | None = None  # suomalainen muoto (esim. "tiistai 18.7.2026")
    overview: str
    items: list[NewsItem]
    meta: GenerationMeta
    dropped_stories: list[DroppedStory] = Field(default_factory=list)
    # Non-critical degradations (e.g. cluster step failed and fell back to
    # singletons) are recorded here — visible in the final JSON, not hidden
    # in logs only. Empty list = all steps worked as expected.
    warnings: list[str] = Field(default_factory=list)

    @property
    def effective_display_date(self) -> date:
        return self.display_date or self.period_start
