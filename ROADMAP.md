# Roadmap

## Done

- Nine-step pipeline: collect, dedup, cluster, score, enrich, compose, overview, validate, write
- Pytest test suite with integration and unit tests
- `run_digest.sh` for topic-specific scheduled runs, deploy, and Telegram notification
- Astro static site with newspaper-style design and small interaction scripts
- Plausible Analytics
- Finnish date formatting with weekday names
- OG tags and favicon for link previews
- Dropped stories surfaced as warnings in the digest
- `/tietoja/` pipeline explanation page
- Source domain shown in article links
- Persona, rubric, guardrails, and golden examples as versioned config
- Prompt improvements for multi-source digest context
- Cross-day story dedup (skip stories already published in the last week)
- Backfill pool: score returns extra candidates so enrich/compose drops
  don't push the digest below the minimum item count
- Retry compose once on an empty LLM response before dropping the item
- OpenRouter-only model provider (dropped OpenAI/Anthropic/local support)
- Colophon component: model list moved out of the header into a
  collapsed `<details>` at the bottom of the digest, step names aligned
  with `/tietoja/` (was three inconsistent label sets across the site)
- Dropped "olet toimittaja" framing from LLM prompts (compose/score/
  overview) - the app doesn't claim to do actual journalism, see
  `planning/identity-and-voice.md`
- Failure alerts for pipeline and deployment errors in `run_digest.sh`
- Pipeline observability: per-step status and duration, per-call provider/model
  attempts, fallback/retry outcomes, and failure context in pipeline data and
  `/läpivalaisu/`
- Multi-source collection with per-topic source lists and HN/RSS adapters
- Weekly and monthly digest periods alongside daily runs
- Topic-specific golden examples selected through each topic configuration

## Next


## Ideas

- Experimenting with Telegram message formats (current vs. full digest content vs. summary + links only)
- Groundedness check after Compose/Overview to catch factual drift against
  source material; report findings as non-blocking warnings.
- Link sources directly from the overview: give the Overview model stable
  source references that the site renderer turns into links.
- Async compose step for parallel LLM calls
- Reddit and ArXiv adapters
- New topic candidates:
  design, mechanical keyboards, electronic music
- **Per-topic persona** — `ConfigPaths` in `agent/pipeline.py` still loads
  one persona regardless of topic, limiting voice and audience adaptation.
  Guardrails are intentionally shared across topics. See
  `planning/identity-and-voice.md`.
