# AGENTS.md

## Language and comment rules

- **Do not comment on names.** The name of a file, function, or other identifier should speak for itself. If a name needs explanation, it is a bad name - choose a better name instead.
- Do not comment on trivial things (e.g. `set -euo pipefail` does not need a comment).
- Comments are "why", not "what". If "what" is not self-evident from the code, the code is too hard.

## Code style

- Python 3.13, pytest - tests `def test_*()` + `pytest` or `python -m pytest`
- All prompts and config in Finnish, code comments in Finnish
- No unnecessary dependencies: httpx, pydantic, pyyaml, trafilatura, pytest - all already in `requirements.txt`
- No emojis or unnecessary decorations in code

## Pipeline

Collect → Dedup → Cluster → Score → Enrich → Compose → Overview → Validate → Write

- Failure policy: cluster = graceful degradation, score = hard failure, compose = per-item
- OpenRouter LLM, paid models required
- `strip_code_fences()` for all LLM responses before `json.loads()`
- `response_format: {"type": "json_object"}` in OpenAI-compatible requests

## Site

Astro, static HTML, zero JS. Script `pipeline.sh` = data pipeline, not site build.

## Commits

- Commit messages in English
- Conventional commit style: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:` etc.
- Subject line max 50 characters, no period at the end
- Body line wrap at 72 characters
- Avoid lengthy explanations
- Always ask for user verification before committing
