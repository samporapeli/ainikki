"""
Yhteinen rajapinta collect-adaptereille. Jokainen lähde (hn.py, rss.py, ...)
toteuttaa saman muotoisen fetch()-funktion, jotta uuden lähteen lisääminen
ei vaadi ydinlogiikan koskemista - vain uusi tiedosto tähän kansioon.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.schema import RawItem


def save_raw(topic: str, date_str: str, source_name: str, items: list[RawItem],
             data_dir: Path = Path("data/raw")) -> Path:
    """Tallentaa collect-vaiheen tuloksen levylle cacheksi.
    Näin epäonnistunut myöhempi step ei pakota fetchaamaan uudestaan.
    """
    out_dir = data_dir / topic / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{source_name}.json"
    payload = [item.model_dump(mode="json") for item in items]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def load_raw(topic: str, date_str: str, source_name: str,
             data_dir: Path = Path("data/raw")) -> list[RawItem]:
    """Loads previously cached raw data from disk."""
    path = data_dir / topic / date_str / f"{source_name}.json"
    payload = json.loads(path.read_text())
    return [RawItem(**item) for item in payload]


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
