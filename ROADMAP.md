# Roadmap

## Done

- Nine-step pipeline: collect, dedup, cluster, score, enrich, compose, overview, validate, write
- Pytest test suite with integration and unit tests
- `daily.sh` for automated daily run, deploy, and Telegram notification
- Astro static site with newspaper-style design, zero JS
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

## Next

- **Failure alerting for `daily.sh`** — right now a pipeline crash
  (e.g. `ScoreValidationError`) just exits and gets logged to
  `ainikki-cron.log`; nobody is notified. Send a Telegram/email message
  on failure, not just on success.
- **Multi-source collection** — currently HN-only, which limits topic
  diversity and weakens the rubric's "source independence" criterion
  (multiple HN submissions of the same link aren't independent sources).
  See `planning/multi-source-and-topics.md` for source candidates (Yle
  RSS, HS, Reddit, topic-specific feeds) and the adapter design.
  - Per-topic source configuration (`config/sources/`)
  - RSS adapter as second source type
- **Per-topic persona/guardrails** — `ConfigPaths` in `agent/pipeline.py`
  only parameterizes the rubric by topic; persona and guardrails are
  hardcoded to one file regardless of topic. Blocks adding a topic with
  a different voice/audience. See `planning/identity-and-voice.md`.
- Lightweight groundedness check after Compose/Overview (LLM-as-judge
  or heuristic entity/number overlap against source text) — log as a
  non-blocking warning, consistent with the existing failure-policy
  philosophy, to catch drift now that nothing reviews output before
  publish.
- Parameterize `daily.sh` for multiple topics
- Extract Telegram logic into its own script

## Ideas

- Experimenting with Telegram message formats (current vs. full digest content vs. summary + links only)
- Weekly or monthly digest variants — could fit lower-velocity topics
  (e.g. mechanical keyboards) better than a daily cadence
- Async compose step for parallel LLM calls
- Reddit and ArXiv adapters
- Topic-specific golden examples
- New topic candidates once multi-source collection exists — see
  `planning/multi-source-and-topics.md` for source fit per topic:
  Linux/FOSS, design, mechanical keyboards, electronic music

## Open questions

- How should Ainikki's identity/branding work once there's more than
  one persona (different topics likely want different voices/target
  audiences)? Is "Ainikki" one author across all topics, or does each
  topic get its own named persona under the Ainikki umbrella? See
  `planning/identity-and-voice.md` for the options considered — no
  decision made yet.
