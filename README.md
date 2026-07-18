# Ainikki — AI-uutiskoosteagentti

Kerää uutisia Hacker Newsista, poistaa duplikaatit, ryhmittelee, arvioi, rikastaa ja
kirjoittaa suomenkielisen koosteen.
Astro-sivusto renderöi tulokset staattiseksi HTML:ksi.

## Asennus

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Aja testit

    python3 -m tests.test_pipeline   # integraatiotesti, kattaa koko putken

Jokaiselle moduulille on oma testitiedostonsa (`tests/test_*.py`), ajettavissa
samalla tavalla yksitellen.

## Aja oikeasti (kun testit ovat vihreällä)

    export OPENROUTER_API_KEY=...
    ./pipeline.sh --topic ai --since 2026-07-16 --until 2026-07-16 --verbose

`--verbose` näyttää INFO-tason lokit jokaisesta vaiheesta (montako itemiä
kuhunkin vaiheeseen meni/tuli ulos) - suositellaan ensimmäisillä ajoilla.

AB-vertailu eri mallien välillä ilman config-tiedoston muokkaamista:

    ./pipeline.sh --topic ai --since 2026-07-16 --until 2026-07-16 \
      --model-override score=openai:gpt-4o \
      --model-override compose=anthropic:claude-opus-4-8

## Renderöi ja deployaa sivusto

    cd site && npm install && npm run build

Generoitu HTML päätyy `site/dist/`. Deployaus:

    DEPLOY_TARGET=user@host:/var/www/ainikki ./site/deploy.sh

## Vaiheet

| # | Vaihe | LLM? | Sisään | Ulos | Konfiguraatio |
|---|---|---|---|---|---|
| 1 | **Collect** | Ei | source-URL:t, aikaikkuna | `list[RawItem]` per lähde | `sources/{topic}.yaml` |
| 2 | **Dedup** | Ei | kaikki RawItemit | `list[Candidate]` (exact-URL dedup) | — |
| 3 | **Cluster** | Kyllä | deduped candidates | `list[Candidate]` ryhmiteltynä | — |
| 4 | **Score** | Kyllä | klusteroidut candidaatit | ranking + `selection_reason` | `rubrics/{topic}_v*.yaml` |
| 5 | **Enrich** | Ei | top N valittua | täysi artikkelisisältö per item | — |
| 6 | **Compose** | Kyllä (per item) | 1 itemin sisältö + persona + guardrails | headline + summary suomeksi | `personas/*`, `guardrails/*`, `golden_examples/*` |
| 7 | **Overview** | Kyllä | kaikki composed itemit | 1-2 lauseen päivän yleiskatsaus | — |
| 8 | **Validate** | Ei | kaikki composed + overview | `Briefing`-objekti, Pydantic-validoitu | `schema.py` |
| 9 | **Write** | Ei | validoitu Briefing | JSON levylle | — |

## Failure-policy stepeittäin

- **Collect**: yhden lähdeadapterin epäonnistuminen skippaa sen lähde, jatka muilla.
- **Dedup**: ei pitäisi koskaan epäonnistua ellei syöte ole rikki.
- **Cluster / Enrich**: ei-kriittisiä. Cluster degradoituu singleton-ryhmiin,
  Enrich pudottaa yksittäisen itemin. Molemmat lokittavat WARNING-tasolla.
- **Score / Validate**: kriittisiä. Jos structured output on rikki, putki kaatuu
  näkyvästi sen sijaan että julkaistaan mielivaltainen valinta.
- **Compose**: yhden itemin epäonnistuminen -> pudota se pois, älä kaataa koko ajoa.
- **Write**: kirjoituksen jälkeen tiedosto luetaan takaisin ja validoidaan uudelleen.

## Mallit ja niiden roolit

Jokainen pipeline-askeli käyttää omaansa mallia. JSON-tuloksessa `models_used`
-kenttä kertoo minkä mallin kukin askeli käytti. Sivustolla nämä näkyvät
otsikon alla pieninä badgeina:

| Askeli     | Rooli                                      |
| ---------- | ------------------------------------------ |
| `cluster`  | Ryhmittele uutiset aihepiireittäin         |
| `score`    | Arvioi uutisten laatu ja uutisarvo         |
| `compose`  | Kirjoita otsikot ja yhteenveto (toimittaja)|
| `overview`  | Kirjoita päivän yleiskatsaus               |

## Suunnitteluperiaatteet

- **Enrich vasta scoringin jälkeen** - säästää verkkokutsuja ja kontekstia, koska
  haetaan täysi sisältö vain jutuille jotka oikeasti päätyvät koosteeseen.
- **Yksi LLM-kutsu per item Compose-stepissä** - pieni konteksti, rinnakkaistettavissa
  myöhemmin, ja yhden itemin epäonnistuminen on eristetty muista.
- **Pydantic-validointi** Write-jälkeen on turvaverkko joka estää rikkinäisen JSON:in
  päätymisen levylle ja sitä kautta Astro-buildiin asti.
- **Config vs. runtime-override** - `config/models.yaml` määrittää oletukset, CLI-flagit
  (`--model-override`) voittavat ne AB-vertailua varten.

## Käyttöoikeudet

Aseta ympäristömuuttuja sen providerin mukaan jota config/models.yaml käyttää:
- OpenRouter: `OPENROUTER_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Paikallinen (esim. Ollama): ei avainta, mutta `base_url` pitää olla oikein

## Tunnetut rajoitukset

- Vain yksi collect-lähde (HN). RSS-adapteri puuttuu.
- Cluster-step on otsikkopohjainen, ei täysin luotettava.
- Enrich käyttää trafilaturaa, ei toimi JS-renderöidyillä sivuilla.
- Compose käyttää vain päälähdettä, monilähde-synteesi v2:een.
- Ei asyncia - Compose ajaa itemit peräkkäin.

## Seuraavat askeleet

1. RSS-adapteri lisää toisen lähteen
2. Weekly/monthly-ajon oikea testaus
