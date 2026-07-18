"""
Ajo: python -m tests.test_score (projektin juuresta)
"""

import json
from pathlib import Path

from agent.score import build_score_prompt, load_rubric, score_clusters, ScoreValidationError
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits
from agent.schema import ClusteredCandidate

FIXTURE_RUBRIC = Path("tests/fixtures/rubric_test.yaml")
REAL_RUBRIC = Path("config/rubrics/ai_scoring_rubric_v1.yaml")


def _build_test_clusters() -> list[ClusteredCandidate]:
    """4 klusteroitua candidaattia (tässä: singleton-klustereita, cluster-stepin
    tarkka toiminta ei ole tämän testin vastuulla - ks. tests/test_cluster.py)."""
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)
    return [ClusteredCandidate(items=c.items, cluster_reason=None) for c in candidates]


def test_build_score_prompt_includes_rubric_content():
    clusters = _build_test_clusters()
    rubric = load_rubric(FIXTURE_RUBRIC)
    system_prompt, user_prompt = build_score_prompt(clusters, rubric)

    assert "uutuusarvo" in system_prompt
    assert "markkinointisisältö" in system_prompt  # exclude-listalta
    assert "2-3" in system_prompt  # min-max testirubriikista
    assert "story_text" not in user_prompt  # ei täyttä sisältöä promptissa


def test_score_clusters_happy_path():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # valitaan candidaatit 0 ja 3, rank-järjestys käännettynä testataksemme sortin
        return json.dumps({"selected": [
            {"candidate_index": 3, "rank": 2, "selection_reason": "kiinnostava tutkimustulos"},
            {"candidate_index": 0, "rank": 1, "selection_reason": "laaja vaikutus, uusi mallijulkaisu"},
        ]})

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)

    assert len(result) == 2
    assert result[0].rank == 1
    assert result[0].items[0].title.startswith("Anthropic announces")
    assert result[1].rank == 2
    assert "tutkimustulos" in result[1].selection_reason


def test_raises_on_malformed_json():
    clusters = _build_test_clusters()
    try:
        score_clusters(clusters, FIXTURE_RUBRIC, lambda s, u: "ei json:ia")
        assert False, "olisi pitänyt heittää ScoreValidationError"
    except ScoreValidationError as e:
        assert "JSON" in str(e)


def test_raises_on_count_outside_rubric_bounds():
    """Testirubriikissa min=2, max=3 - yritetään valita 1 (liian vähän)."""
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "ainoa valinta"},
        ]})

    try:
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)
        assert False, "olisi pitänyt heittää ScoreValidationError"
    except ScoreValidationError as e:
        assert "rajojen" in str(e)


def test_raises_on_duplicate_candidate_index():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "a"},
            {"candidate_index": 0, "rank": 2, "selection_reason": "b"},
        ]})

    try:
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)
        assert False, "olisi pitänyt heittää ScoreValidationError"
    except ScoreValidationError as e:
        assert "useammin" in str(e)


def test_raises_on_invalid_rank_sequence():
    clusters = _build_test_clusters()

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # rankit 1 ja 1 (duplikaatti) kahden valinnan sijaan 1 ja 2
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "a"},
            {"candidate_index": 1, "rank": 1, "selection_reason": "b"},
        ]})

    try:
        score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)
        assert False, "olisi pitänyt heittää ScoreValidationError"
    except ScoreValidationError as e:
        assert "rank" in str(e)


def test_prompt_builds_with_real_production_rubric():
    """Integraatiotesti oikealla config/rubrics/ai_scoring_rubric_v1.yaml -tiedostolla,
    jotta testifixture ei pääse ajautumaan pois synkasta oikean configin kanssa."""
    clusters = _build_test_clusters()
    rubric = load_rubric(REAL_RUBRIC)
    system_prompt, user_prompt = build_score_prompt(clusters, rubric)
    assert "aihepiirin ydinolennaisuus" in system_prompt
    assert "aihepiirien monipuolisuus" in system_prompt
    assert "3-10" in system_prompt


def test_rejects_irrelevant_candidates():
    """Varmistaa että score.py käsittelee oikein tilanteen jossa osa
    candidaateista hylätään relevanssin perusteella (mock-LLM valitsee
    vain osan)."""
    clusters = _build_test_clusters()
    n_clusters = len(clusters)

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        # valitaan vain 2 ekaa, loput "hylätään" (ei valita)
        return json.dumps({"selected": [
            {"candidate_index": 0, "rank": 2, "selection_reason": "AI-aiheinen"},
            {"candidate_index": 1, "rank": 1, "selection_reason": "AI-aiheinen"},
        ]})

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm)
    assert len(result) == 2, f"odotettiin 2 valittua, saatiin {len(result)}"
    assert result[0].rank == 1
    assert result[1].rank == 2


def test_code_fenced_json_is_parsed():
    """Varmistaa että mallin koodiblokkiin käärityt JSON-vastaukset siivoutuvat."""
    clusters = _build_test_clusters()

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        inner = json.dumps({"selected": [
            {"candidate_index": 0, "rank": 1, "selection_reason": "tärkein"},
            {"candidate_index": 1, "rank": 2, "selection_reason": "toinen"},
        ]})
        return f"```json\n{inner}\n```"

    result = score_clusters(clusters, FIXTURE_RUBRIC, mock_llm_code_fence)
    assert len(result) == 2
    assert result[0].rank == 1


if __name__ == "__main__":
    test_build_score_prompt_includes_rubric_content()
    test_score_clusters_happy_path()
    test_raises_on_malformed_json()
    test_raises_on_count_outside_rubric_bounds()
    test_raises_on_duplicate_candidate_index()
    test_raises_on_invalid_rank_sequence()
    test_prompt_builds_with_real_production_rubric()
    test_rejects_irrelevant_candidates()
    test_code_fenced_json_is_parsed()
    print("\nKaikki testit läpi.")
