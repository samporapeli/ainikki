# AGENTS.md

## Language and comment rules

- **Do not comment on names.** The name of a file, function, or other identifier should speak for itself. If a name needs explanation, it is a bad name - choose a better name instead.
- Do not comment on trivial things (e.g. `set -euo pipefail` does not need a comment).
- Comments are "why", not "what". If "what" is not self-evident from the code, the code is too hard.

## Security

- Do not hardcode or reference local development machine details in committed files: absolute paths, hostnames, usernames, or other environment-specific values. Docs should not assume a specific folder structure.
- Never read, log, or expose secrets, API keys, tokens, or credentials.
- Never commit secrets or credentials to the repository.
- When handling credentials in code, always use environment variables or secure secret management.
- When handling credentials for humans, remind about safe practices: do not paste secrets into chat, do not share tokens openly, rotate compromised credentials immediately.
- Committed docs (planning, roadmap, commit messages) should read as general technical rationale, not a transcript of the maintainer's personal reasoning or preferences. Keep personal working notes in gitignored `*.local.md` files instead.

## Code style

- Python 3.13, pytest - tests `def test_*()` + `pytest` or `python -m pytest`
- No unnecessary dependencies: httpx, pydantic, pyyaml, trafilatura, pytest - all already in `requirements.txt`
- No emojis or unnecessary decorations in code

## Language

- **Finnish**: site pages, LLM prompts, config files (rubrics, personas, guardrails, golden examples), and any text the LLM writes (summaries, headlines, overviews)
- **English**: code, code comments, README, ROADMAP, AGENTS.md, commit messages, test descriptions

## Refactoring

- **YAGNI** (You Aren't Gonna Need It). Do not leave code or documentation "just in case". Build what is needed now, not what might be needed later.
- **DRY** (Don't Repeat Yourself) applies to code and documentation alike. However, do not force DRY if it introduces unnecessary indirection or complexity - pragmatic duplication is acceptable.
- When adding a feature, actively consider if something else becomes obsolete. Remove it after verifying it can be done.
- Complete refactors cleanly. Do not leave backwards compatibility code rotting. Confirm with the user whether to backfill old data or drop fallbacks.

## Pipeline

Collect → Dedup → Cluster → Score → Enrich → Compose → Overview → Validate → Write

- Failure policy: cluster = graceful degradation, score = hard failure, compose = per-item
- OpenRouter LLM, paid models required
- `strip_code_fences()` for all LLM responses before `json.loads()`
- `response_format: {"type": "json_object"}` in OpenAI-compatible requests

## Site

Astro, static HTML, zero JS. Script `pipeline.sh` = data pipeline, not site build.
- Page `/tietoja/` describes the pipeline and its steps. Keep it up to date when modifying collection or composition logic.

## Commits

- Conventional commit style: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:` etc.
- Subject line max 50 characters, no period at the end
- Body line wrap at 72 characters
- Commit messages in English. Preserve original Finnish terms when referencing Finnish text changes (e.g. `fix: typo (tomittaja -> toimittaja)`)
- Only stage changes relevant to the commit
- Prefer single-line commits, but do not force brevity over clarity. If a commit message starts to read like documentation, the explanation belongs in code comments or docs instead
- After committing, verify the commit contains exactly what was intended - nothing extra, nothing missing
- Never commit or amend without explicit user permission. When amending, verify the updated message is still accurate
- Run tests before committing, do not commit if tests fail
- Propose commits with a drafted message when it seems appropriate
