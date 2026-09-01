"""
JSON utilities: clean LLM responses before json.loads().

Many models (especially GPT-4o-mini via OpenRouter) wrap their JSON responses
in Markdown code fences (```json\n{...}\n```) even when the prompt asked for
plain JSON. This module strips such markers.

Always use strip_code_fences() before calling json.loads(raw_response).
"""

import re

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n"  # opening ``` (optionally with language tag)
    r"(.*?)"                       # content (non-greedy)
    r"\n\s*```\s*$",              # closing ```
    re.DOTALL,
)


def strip_code_fences(text: str | None) -> str:
    """Strip Markdown code fence markers from an LLM response.

    - If text is None or empty, returns empty string.
    - If text starts with ``` and ends with ```, removes markers and returns content.
    - Otherwise returns text with .strip() applied.
    """
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return text

    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()

    return text
