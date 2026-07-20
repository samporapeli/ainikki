"""
Write-step: writes the validated Briefing as a JSON file to data/output/.
After writing, the file is immediately re-read and re-validated — a cheap
but useful safety net: if serialization (e.g. HttpUrl/datetime type
conversion) fails somehow, it is caught right away instead of causing a
late Astro build failure.
"""

import json
from pathlib import Path

from agent.schema import Briefing


class WriteRoundtripError(Exception):
    """File written to disk failed validation on re-read."""


def write_briefing(briefing: Briefing, output_dir: Path = Path("data/output")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{briefing.topic}_{briefing.period.value}_{briefing.effective_display_date.isoformat()}.json"
    path = output_dir / filename

    payload = briefing.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        reloaded = Briefing(**json.loads(path.read_text()))
    except Exception as e:
        raise WriteRoundtripError(f"Written file {path} failed validation on re-read: {e}") from e

    if reloaded.topic != briefing.topic or len(reloaded.items) != len(briefing.items):
        raise WriteRoundtripError(f"Roundtrip check failed for {path}: data does not match original")

    return path
