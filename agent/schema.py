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
    """Yhden lähteen tuottama yksittäinen raakalöytö, ennen dedupia/klusterointia.
    Tallennetaan sellaisenaan data/raw/{topic}/{date}/{source}.json -tiedostoon.
    """
    title: str
    url: HttpUrl
    source_type: SourceType
    published_at: datetime | None = None
    raw_signal: dict[str, int | float | str] = Field(default_factory=dict)
    origin_id: str  # adapterikohtainen uniikki id (esim. HN objectID), debug/jäljitys
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
    """Yhden tarinan RawItemit klusteroinnin jälkeen, mahdollisesti eri URL:eista.
    items[0] = mallin ehdottama päälähde. Tämä on vielä provisorinen -
    Score-step tekee lopullisen editorial-arvion, tämä vain ryhmittelee.
    """
    items: list[RawItem]
    cluster_reason: str | None = None  # mallin lyhyt perustelu ryhmittelylle, debug/läpinäkyvyys


# --- Klusteroinnin jälkeinen / lopullinen muoto ---------------------------

class Source(BaseModel):
    """Yksi linkki/lähde joka liittyy juttuun. [0] listassa = päälähde."""
    url: HttpUrl
    title: str
    source_type: SourceType
    published_at: datetime | None = None
    raw_signal: dict[str, int | float | str] = Field(default_factory=dict)


class ScoredCandidate(BaseModel):
    """ClusteredCandidate + editorial-arvio: sijoitus ja perustelu.
    Vain valitut candidaatit päätyvät tähän muotoon - hylätyt eivät etene."""
    items: list[RawItem]
    cluster_reason: str | None = None
    rank: int
    selection_reason: str


class EnrichedCandidate(ScoredCandidate):
    """ScoredCandidate + päälähteen puhdistettu artikkelisisältö.
    Vain päälähde (items[0]) rikastetaan v1:ssä - toissijaiset lähteet
    pysyvät metadatatasolla, koska compose käyttää v1:ssä vain päälähdettä.
    """
    content: str | None = None  # None = haku/erottelu epäonnistui tälle itemille


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
    overview: str
    items: list[NewsItem]
    meta: GenerationMeta
    # Ei-kriittiset degradoinnit (esim. cluster-step epäonnistui ja fallbackasi
    # singletoneihin) kirjataan tänne - näkyy lopullisessa JSON:issa, ei jää
    # vain lokeihin piiloon. Tyhjä lista = kaikki stepit toimivat odotetusti.
    warnings: list[str] = Field(default_factory=list)
