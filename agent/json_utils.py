"""
JSON-apufunktiot: siivoaa LLM-vastauksia ennen json.loads-kutsua.

Moni malli (erityisesti GPT-4o-mini OpenRouterin kautta) kääri JSON-vastauksensa
Markdown-koodiblokkeihin (```json\n{...}\n```) vaikka promptissa pyydettiin
puhdasta JSON:ia. Tämä moduuli siivoaa tällaiset merkinnät pois.

Käytä strip_code_fences() aina ennen json.loads(raw_response) -kutsua.
"""

import re

_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n"  # avaus ``` (mahdollisesti kielimerkinnällä)
    r"(.*?)"                       # sisältö (non-greedy)
    r"\n\s*```\s*$",              # sulkava ```
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Strip Markdown code fence markers from an LLM response.

    - If text starts with ``` and ends with ```, removes markers and returns content.
    - Otherwise returns text with .strip() applied.
    - Empty/whitespace response returned as-is.
    """
    text = text.strip()
    if not text:
        return text

    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()

    return text
