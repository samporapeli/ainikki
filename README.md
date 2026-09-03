# Ainikki — AI news digest agent

Collects news from Hacker News and RSS feeds, deduplicates, clusters,
scores, enriches, and writes Finnish-language digests. Two topics:
**tekoäly** (daily) and **teknologia** (weekly). The Astro site renders
output as static HTML.

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Run tests

    python3 -m pytest

Each module has its own test file under `tests/test_*.py`.

## Run the pipeline

    export OPENROUTER_API_KEY=...
    ./pipeline.sh --topic ai --since 2026-07-16 --until 2026-07-16 --verbose

`--verbose` shows INFO-level logs for each step (item counts in/out) —
recommended for first runs.

A/B compare models without editing config, or run against Ollama:

    ./pipeline.sh --topic ai --since 2026-07-16 --until 2026-07-16 \
      --model-override score=openrouter:openai/gpt-4o \
      --model-override compose=openrouter:anthropic/claude-sonnet-5

    OLLAMA_HOST="http://localhost:11434/v1" \
      ./pipeline.sh --topic test --since 2026-07-16 --until 2026-07-16 \
      --verbose  # runs test steps configured with local provider

Run the full end-to-end test topic:

    OPENROUTER_API_KEY='...' TELEGRAM_BOT_TOKEN='...' \
      TELEGRAM_CHAT_ID_TEST='...' ./test.sh

The test writes its `test` output like a normal pipeline run to
`data/output/` and `data/pipeline/`. Start Astro yourself with
`cd site && npm run dev` and open the printed test URL. The production build
and deploy exclude topics whose configuration sets `public: false`.
The default test date follows the timestamps in the checked-in RSS fixtures;
override `AINIKKI_TEST_DATE` only when using matching fixture data.

## Build and deploy the site

    cd site && npm install && npm run build

Generated HTML lands in `site/dist/`. Deploy:

    DEPLOY_TARGET=user@host:/var/www/ainikki ./site/deploy.sh

Every topic configuration defines `public`. Local development builds include
all topics, while `site/deploy.sh` publishes only topics with `public: true`.

## Automation

### Daily digest (AI, daily at 06:30)

    OPENROUTER_API_KEY='sk-...' DEPLOY_TARGET='user@host:/var/www/ainikki' \
      TELEGRAM_BOT_TOKEN='...' TELEGRAM_CHAT_ID_AI='...' TELEGRAM_ERROR_CHAT_ID='...' \
      /path/to/ainikki/daily.sh >> /path/to/ainikki-cron.log 2>&1

### Weekly digest (Teknologia, Monday at 06:30)

    OPENROUTER_API_KEY='sk-...' DEPLOY_TARGET='user@host:/var/www/ainikki' \
      TELEGRAM_BOT_TOKEN='...' TELEGRAM_CHAT_ID_TEKNOLOGIA='...' TELEGRAM_ERROR_CHAT_ID='...' \
      /path/to/ainikki/weekly.sh >> /path/to/ainikki-cron.log 2>&1

Set keys and target directly in the cron entry, not in profile files.
Telegram is optional: if bot token and chat ID are missing, Telegram
steps are silently skipped.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | LLM access (OpenRouter) |
| `OLLAMA_HOST` | No | Base URL for local provider (e.g. `http://localhost:11434/v1`) |
| `GOOGLE_CLOUD_API_TOKEN` | No | API key for Google Cloud Text-to-Speech audio synthesis |
| `DEPLOY_TARGET` | Yes | `user@host:/path` for rsync deploy |
| `TELEGRAM_BOT_TOKEN` | No | Bot token for Telegram notifications |
| `TELEGRAM_CHAT_ID_AI` | No | Chat ID for daily AI digest channel |
| `TELEGRAM_CHAT_ID_TEKNOLOGIA` | No | Chat ID for weekly Teknologia digest channel |
| `TELEGRAM_ERROR_CHAT_ID` | No | Chat ID for error alerts (shared across topics) |
## Pipeline steps

| # | Step | LLM? | In | Out | Config |
|---|------|------|----|-----|--------|
| 1 | **Collect** | No | source URLs, time window | `list[RawItem]` per source | `sources/{topic}.yaml` |
| 2 | **Dedup** | No | all RawItems | `list[Candidate]` (exact-URL dedup) | — |
| 3 | **Filter topic** | Yes | deduped candidates | topic-relevant candidates | `topics/{topic}.yaml` |
| 4 | **Cluster** | Yes | filtered candidates | `list[Candidate]` grouped | — |
| 5 | **Score** | Yes | clustered candidates | ranking + `selection_reason`, plus a backfill pool below the cutoff | `rubrics/{topic}_v*.yaml` |
| 6 | **Enrich** | No | top N selected | extracted, truncated article text per item | — |
| 7 | **Compose** | Yes (per item) | 1 item content + persona + guardrails | headline + summary in Finnish | `personas/*`, `guardrails/*`, `golden_examples/*` |
| 8 | **Overview** | Yes | all composed items | 1–2 sentence daily overview | — |
| 9 | **Validate** | No | all composed + overview | `Briefing` object, Pydantic-validated | `schema.py` |
| 10 | **Write** | No | validated Briefing | JSON to disk | — |

## Failure policies

- **Collect**: one adapter failure skips that source, continues with others.
- **Dedup**: should never fail unless input is broken.
- **Cluster / Enrich**: non-critical. Cluster degrades to singletons,
  Enrich drops individual items. Both log at WARNING level.
- **Score / Validate**: critical. If structured output is broken, the
  pipeline crashes rather than publishing an arbitrary selection.
- **Compose**: one item fails → drop it, don't crash the whole run.
- **Write**: after writing, the file is read back and re-validated as
  a roundtrip safety net.

## Models and their roles

Each pipeline step uses its own model. The `models_used` field in the
JSON output shows which model ran each step. These appear as small lines
under the headline on the site.

| Step | Role |
|------|------|
| `filter_topic` | Keep stories relevant to the topic |
| `cluster` | Group stories by topic |
| `score` | Evaluate newsworthiness |
| `compose` | Write headlines and summaries (journalist) |
| `overview` | Write the daily overview |

### Providers

Every LLM step talks the OpenAI-compatible `/v1/chat/completions` API. Provider
**names** and provider **types** are separate concepts. Provider names are
configuration keys; `type` controls how the response is parsed:

- `openrouter` reads OpenRouter's real top-level `cost` field.
- `openai_compatible` uses the standard format and reports cost as
  unavailable (`N/A` on the site).

Providers are defined only under `providers:` in `config/models.yaml`. Add or
rename entries there; topic files then reference them by name in their
`.models:` list.

```yaml
# config/models.yaml
providers:
  openrouter:
    type: openrouter
    base_url: "https://openrouter.ai/api/v1"
    api_key_env: "OPENROUTER_API_KEY"
  local:
    type: openai_compatible
    base_url_env: "OLLAMA_HOST"
    timeout: 300
```

`base_url` and `api_key` support ``$VAR`` and ``${VAR:-default}``
interpolation. The `*_env` fields name environment variables. Keep secrets and
private hostnames out of the committed config.

Topics override one or more step chains with a prioritized list (`first` is
tried first, then each fallback on request failure):

```yaml
# config/topics/ai.yaml
steps:
  filter_topic:
    models:
      - provider: local
        model: "qwen3.6:35b-a3b"
      - provider: openrouter
        model: "openai/gpt-4o-mini"
```

Per-run `--model-override step=provider:model` replaces one step's chain.


### Pipeline

- **Enrich only after scoring** — saves HTTP calls and context by
  fetching full content only for items that make the cut.
- **Backfill pool** — Score ranks more candidates than the digest needs;
  if Enrich/Compose drops push the count below the minimum, the pipeline
  backfills from the next-best-ranked candidates instead of publishing
  too few items.
- **Cross-day dedup** — clusters whose primary URL was already published
  in the last 7 days are filtered out before scoring.
- **One LLM call per item in Compose** — small context, failure
  isolation, parallelizable later.
- **Pydantic validation after Write** — roundtrip safety net that
  prevents broken JSON from reaching the Astro build.
- **Config vs. runtime override** — `config/models.yaml` sets
  defaults, CLI flags (`--model-override`) override for A/B testing.

### Frontend

The site is a small daily paper, not a SaaS dashboard or modern feed
page. The guiding question: "Does this help the reader focus on today's
stories?" If not, remove it.

- **Typography first** — spacing, hierarchy, and line length carry
  visual weight, not cards, shadows, or decorations.
- **Pleasant reading experience** — generous whitespace, comfortable
  line height, readability first.
- **Invisible UI** — the reader notices the text, not the elements.
- **System fonts** — no external fonts, the typeface is the design.
- **No motion** — no animations, hover states are restrained.
- **Zero JS** — static HTML, inline CSS, no external resources.

## Credentials

Set `OPENROUTER_API_KEY` in the environment or in the cron entry.

## Known limitations

- Cluster step is title-based, not fully reliable.
- Enrich uses trafilatura, doesn't work on JS-rendered pages.
- Compose uses only the primary source, multi-source synthesis is v2.
- No async — Compose runs items sequentially.
- Weekly topic currently only has HN + Yle Tiede feeds — more sources
  would improve coverage.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned work.
