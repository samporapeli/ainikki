"""
Dedup-step: yhdistää RawItemit joilla on sama (normalisoitu) URL.
Ei LLM-kutsuja - puhdas, deterministinen, nopeasti testattava koodi.

HUOM rajanveto: tämä EI ratkaise "sama tarina eri URL:eista" -ongelmaa
(esim. TechCrunch ja The Verge molemmat uutisoivat samasta asiasta eri
linkillä) - se on Cluster-stepin (LLM-pohjainen) vastuulla. Tämä step
yhdistää vain kirjaimellisesti saman resurssin eri esiintymät.
"""

import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from agent.schema import RawItem, Candidate

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "spm", "igshid",
}


def normalize_url(url: str) -> str:
    """Normalisoi URL:n vertailukelpoiseksi avaimeksi:
    - scheme aina https (vertailua varten, ei muuta alkuperäistä dataa)
    - host pieniksi kirjaimiksi, www.-etuliite pois
    - trailing slash pois (paitsi juuripolku)
    - tracking-parametrit pois query-stringistä, loput aakkosjärjestykseen
    - fragment (#...) pois kokonaan
    """
    parts = urlsplit(url)

    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = parts.path.rstrip("/") or "/"

    query_pairs = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in TRACKING_PARAMS
    ]
    query_pairs.sort()
    query = urlencode(query_pairs)

    return urlunsplit(("https", netloc, path, query, ""))


def dedup_candidates(raw_items: list[RawItem]) -> list[Candidate]:
    """Ryhmittelee RawItemit normalisoidun URL:n mukaan.
    Ryhmän sisällä järjestys: korkein yhdistetty numeerinen signaali
    (esim. points + num_comments) ensin - provisorinen, ei lopullinen valinta.
    """
    groups: dict[str, list[RawItem]] = {}
    for item in raw_items:
        key = normalize_url(str(item.url))
        groups.setdefault(key, []).append(item)

    def signal_score(item: RawItem) -> float:
        return sum(v for v in item.raw_signal.values() if isinstance(v, (int, float)))

    candidates = []
    for normalized_url, items in groups.items():
        items_sorted = sorted(items, key=signal_score, reverse=True)
        candidates.append(Candidate(normalized_url=normalized_url, items=items_sorted))

    return candidates


def save_candidates(topic: str, date_str: str, candidates: list[Candidate],
                     data_dir: Path = Path("data/dedup")) -> Path:
    out_dir = data_dir / topic / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "candidates.json"
    payload = [c.model_dump(mode="json") for c in candidates]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


def load_candidates(topic: str, date_str: str,
                     data_dir: Path = Path("data/dedup")) -> list[Candidate]:
    path = data_dir / topic / date_str / "candidates.json"
    payload = json.loads(path.read_text())
    return [Candidate(**c) for c in payload]
