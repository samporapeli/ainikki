"""
Yhteinen päivämäärän muotoilu Pythonin ja Astro-sivuston välillä.
Single source of truth - tämä funktio tuottaa saman tuloksen
kuin Astro-sivuston fiDate(). Jos muutat tätä, päivitä myös
site/src/pages/ -tiedostojen fiDate() vastaavaksi.
"""

WEEKDAYS = [
    "sunnuntai", "maanantai", "tiistai", "keskiviikko",
    "torstai", "perjantai", "lauantai",
]


def fi_date(iso_date: str) -> str:
    """Muuttaa ISO-päivämäärän (YYYY-MM-DD) suomalaiseen muotoon: 'tiistai 18.7.2026'."""
    from datetime import date as _date
    y, m, d = (int(x) for x in iso_date.split("-"))
    dt = _date(y, m, d)
    wd = WEEKDAYS[(dt.weekday() + 1) % 7]
    return f"{wd} {d}.{m}.{y}"
