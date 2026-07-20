"""
Shared date formatting between Python and the Astro site.
Single source of truth — this function produces the same output as the
site's fiDate(). If you change this, update fiDate() in site/src/pages/
accordingly.
"""

WEEKDAYS = [
    "sunnuntai", "maanantai", "tiistai", "keskiviikko",
    "torstai", "perjantai", "lauantai",
]


def fi_date(iso_date: str) -> str:
    """Converts an ISO date (YYYY-MM-DD) to Finnish format: 'tiistai 18.7.2026'."""
    from datetime import date as _date
    y, m, d = (int(x) for x in iso_date.split("-"))
    dt = _date(y, m, d)
    wd = WEEKDAYS[(dt.weekday() + 1) % 7]
    return f"{wd} {d}.{m}.{y}"
