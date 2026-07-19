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

## Next

- Parameterize `daily.sh` for multiple topics
- Extract Telegram logic into its own script
- Per-topic source configuration (`config/sources/`)
- RSS adapter as second source type

## Ideas

- Experimenting with Telegram message formats (current vs. full digest content vs. summary + links only))
- Weekly or monthly digest variants
- Async compose step for parallel LLM calls
- Reddit and ArXiv adapters
- Topic-specific golden examples
