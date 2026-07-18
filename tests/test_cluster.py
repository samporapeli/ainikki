"""
Ajo: python -m tests.test_cluster (projektin juuresta)

Testaa klusterointilogiikkaa mock-datalla, ei vaadi verkkoyhteyttä.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from agent.cluster import cluster_candidates, build_cluster_prompt
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits
from agent.schema import RawItem, SourceType, Candidate


def _build_test_candidates() -> list[Candidate]:
    """5 candidaattia: 4 alkuperäisestä HN-fixturesta + 1 TechCrunch-artikkeli
    joka käsittelee SAMAA Anthropic-uutista eri URL:illa kuin HN-linkki.
    Dedup ei yhdistä näitä (eri URL) - se on juuri Cluster-stepin tehtävä.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])

    techcrunch_item = RawItem(
        title="Anthropic's new Fable and Mythos models draw mixed reactions",
        url="https://techcrunch.com/2026/07/14/anthropic-fable-mythos-reaction/",
        source_type=SourceType.other,
        published_at=datetime(2026, 7, 14, 15, 0, tzinfo=timezone.utc),
        raw_signal={"outlet": "techcrunch"},
        origin_id="tc_1",
        fetched_at=datetime.now(timezone.utc),
    )

    return dedup_candidates(hn_items + [techcrunch_item])


def _mock_llm_group_anthropic_stories(system_prompt: str, user_prompt: str) -> str:
    """Simuloi mallin vastausta: löytää promptista Anthropic- ja TechCrunch-rivit
    ja ryhmittelee ne, loput jäävät yhden hengen ryhmiksi."""
    lines = user_prompt.splitlines()[1:]  # ohita "Ehdokkaat:"-otsikkorivi
    anthropic_idx = next(i for i, l in enumerate(lines) if "Fable 5 and Claude Mythos 5" in l)
    techcrunch_idx = next(i for i, l in enumerate(lines) if "mixed reactions" in l)
    all_indices = set(range(len(lines)))
    remaining = all_indices - {anthropic_idx, techcrunch_idx}

    clusters = [{"candidate_indices": [anthropic_idx, techcrunch_idx], "primary_index": 0,
                 "reason": "molemmat käsittelevät Fable/Mythos-julkaisua"}]
    for idx in remaining:
        clusters.append({"candidate_indices": [idx], "primary_index": 0, "reason": None})

    return json.dumps({"clusters": clusters})


def test_cross_url_clustering():
    candidates = _build_test_candidates()
    assert len(candidates) == 5, f"odotettiin 5 candidaattia ennen klusterointia, saatiin {len(candidates)}"

    result = cluster_candidates(candidates, _mock_llm_group_anthropic_stories)
    clustered = result.clusters

    assert result.warning is None, "onnistuneen klusteroinnin ei pitäisi tuottaa warningia"
    assert len(clustered) == 4, f"odotettiin 4 ryhmää klusteroinnin jälkeen, saatiin {len(clustered)}"

    merged = next(c for c in clustered if len(c.items) == 2)
    assert merged.items[0].source_type == SourceType.hn, "päälähteen pitäisi olla HN (yhtiön uutinen), ei TechCrunch"
    assert merged.items[1].source_type == SourceType.other
    assert merged.cluster_reason is not None


def test_singletons_preserved():
    candidates = _build_test_candidates()
    result = cluster_candidates(candidates, _mock_llm_group_anthropic_stories)
    singleton_titles = {c.items[0].title for c in result.clusters if len(c.items) == 1}
    assert "New paper: chain-of-thought reasoning does not always reflect model computation" in singleton_titles


def test_fallback_on_malformed_json():
    candidates = _build_test_candidates()
    result = cluster_candidates(candidates, lambda s, u: "tämä ei ole JSON:ia ollenkaan")
    assert len(result.clusters) == len(candidates), "fallbackissa pitäisi olla yhtä monta ryhmää kuin candidaattia"
    assert all(c.cluster_reason and "fallback" in c.cluster_reason for c in result.clusters)
    assert result.warning is not None, "fallback EI saa olla hiljainen - warning pitää palautua"
    assert "degradoitui" in result.warning


def test_fallback_on_incomplete_coverage():
    """Malli unohti yhden candidaatin kokonaan vastauksesta - pitää havaita ja fallbackata."""
    candidates = _build_test_candidates()

    def bad_response(system_prompt: str, user_prompt: str) -> str:
        # vain 3 ekaa candidaattia mainittu 5:stä - puutteellinen kattavuus
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(3)]
        return json.dumps({"clusters": clusters})

    result = cluster_candidates(candidates, bad_response)
    assert len(result.clusters) == len(candidates)
    assert all("fallback" in (c.cluster_reason or "") for c in result.clusters)
    assert result.warning is not None


def test_prompt_stays_lightweight():
    """Varmistaa ettei promptiin vahingossa vuoda täyttä artikkelisisältöä -
    vain otsikko, lähdetyyppi ja signaali per candidate."""
    candidates = _build_test_candidates()
    _, user_prompt = build_cluster_prompt(candidates)
    assert "story_text" not in user_prompt
    assert len(user_prompt) < 2000, f"prompti yllättävän pitkä ({len(user_prompt)} merkkiä) - konteksti kasvamassa?"


def test_code_fenced_json_is_parsed():
    """Varmistaa että mallin koodiblokkiin käärityt JSON-vastaukset siivoutuvat."""
    candidates = _build_test_candidates()

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None}
                    for i in range(len(candidates))]
        inner = json.dumps({"clusters": clusters})
        return f"```json\n{inner}\n```"

    result = cluster_candidates(candidates, mock_llm_code_fence)
    assert result.warning is None
    assert len(result.clusters) == len(candidates)


if __name__ == "__main__":
    test_cross_url_clustering()
    test_singletons_preserved()
    test_fallback_on_malformed_json()
    test_fallback_on_incomplete_coverage()
    test_prompt_stays_lightweight()
    test_code_fenced_json_is_parsed()
    print("\nKaikki testit läpi.")
