"""
Write-step: kirjoittaa validoidun Briefingin JSON-tiedostoksi data/output/-kansioon.
Kirjoituksen jälkeen tiedosto luetaan heti takaisin ja validoidaan uudelleen -
tämä on halpa mutta hyödyllinen turvaverkko: jos serialisointi (esim. HttpUrl/
datetime-tyyppien JSON-muunnos) menisi jotenkin pieleen, se huomataan heti
eikä jää odottamaan että Astro-build kaatuu myöhemmin.
"""

import json
from pathlib import Path

from agent.schema import Briefing


class WriteRoundtripError(Exception):
    """Levylle kirjoitettu tiedosto ei validoitunut takaisin luettaessa."""


def write_briefing(briefing: Briefing, output_dir: Path = Path("data/output")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{briefing.topic}_{briefing.period.value}_{briefing.period_start.isoformat()}.json"
    path = output_dir / filename

    payload = briefing.model_dump(mode="json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        reloaded = Briefing(**json.loads(path.read_text()))
    except Exception as e:
        raise WriteRoundtripError(f"Kirjoitettu tiedosto {path} ei validoitunut takaisin luettaessa: {e}") from e

    if reloaded.topic != briefing.topic or len(reloaded.items) != len(briefing.items):
        raise WriteRoundtripError(f"Roundtrip-tarkistus epäonnistui tiedostolle {path}: data ei täsmää alkuperäiseen")

    return path
