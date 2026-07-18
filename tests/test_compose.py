import json
from datetime import datetime, timezone
from pathlib import Path

from agent.compose import (
    build_compose_system_prompt, compose_items,
    Persona, GoldenExamplesConfig, GuardrailsConfig,
)
from agent.schema import RawItem, SourceType, EnrichedCandidate

PERSONA_PATH = Path("config/personas/ainikki_v1.yaml")
GUARDRAILS_PATH = Path("config/guardrails/guardrails_v1.yaml")
GOLDEN_EXAMPLES_DIR = Path("config/golden_examples")


def _make_enriched(title: str, content: str, rank: int) -> EnrichedCandidate:
    item = RawItem(
        title=title, url=f"https://example.com/{rank}", source_type=SourceType.hn,
        published_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
        raw_signal={"points": 200}, origin_id=f"test_{rank}",
        fetched_at=datetime.now(timezone.utc),
    )
    return EnrichedCandidate(items=[item], cluster_reason=None, rank=rank,
                              selection_reason=f"testiperuste {rank}", content=content)


def test_system_prompt_includes_all_three_configs():
    persona = Persona.load(PERSONA_PATH)
    golden_examples = GoldenExamplesConfig.load(GOLDEN_EXAMPLES_DIR / persona.golden_examples_ref)
    guardrails = GuardrailsConfig.load(GUARDRAILS_PATH)

    prompt = build_compose_system_prompt(persona, golden_examples, guardrails)

    assert "ainikki" in prompt
    assert "clickbait" in prompt  # persona.avoid
    assert "Käytä VAIN annetussa lähdemateriaalissa mainittuja faktoja." in prompt  # guardrails-sääntö
    assert "Anthropic julkaisi Claude Opus 4.8" in prompt  # golden example
    assert '{"headline"' in prompt  # JSON-muotovaatimus
    # v2: suomen kielen säännöt
    assert "yhdyssanat" in prompt
    assert "desimaalipilkku" in prompt or "desimaalierottimena" in prompt
    assert "Oxford" in prompt


def test_compose_items_happy_path():
    items = [
        _make_enriched("Uusi malli julkaistu", "Yhtiö julkaisi uuden mallin tänään...", rank=1),
        _make_enriched("Tutkimuspaperi ilmestyi", "Tutkijat osoittivat että...", rank=2),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        if "Uusi malli" in user_prompt:
            return json.dumps({"headline": "Yhtiö julkaisi uuden mallin",
                                "summary": "Malli tuo parannuksia aiempaan verrattuna."})
        return json.dumps({"headline": "Uusi tutkimus haastaa aiemman käsityksen",
                            "summary": "Tutkijat löysivät yllättävän tuloksen."})

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 2
    assert len(result.warnings) == 0
    assert result.persona_id == "ainikki-v1"
    assert result.guardrails_version == "v1"
    assert result.golden_examples_version == "v1"

    first = result.items[0]
    assert first.headline == "Yhtiö julkaisi uuden mallin"
    assert first.rank == 1
    assert first.selection_reason == "testiperuste 1"
    assert len(first.sources) == 1
    assert first.sources[0].source_type == SourceType.hn


def test_partial_failure_drops_only_bad_item():
    items = [
        _make_enriched("Toimiva juttu", "Sisältö...", rank=1),
        _make_enriched("Rikkinäinen vastaus", "Sisältö...", rank=2),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        if "Toimiva" in user_prompt:
            return json.dumps({"headline": "Toimiva otsikko", "summary": "Toimiva yhteenveto."})
        return "tämä ei ole JSON:ia"

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1, "vain onnistunut item pitäisi jäädä"
    assert result.items[0].headline == "Toimiva otsikko"
    assert len(result.warnings) == 1
    assert "Rikkinäinen vastaus" in result.warnings[0]


def test_empty_headline_is_rejected():
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]

    def mock_llm(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"headline": "   ", "summary": "Jotain tekstiä."})

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 0
    assert len(result.warnings) == 1


def test_code_fenced_json_is_parsed():
    """Varmistaa että mallin koodiblokkiin käärityt JSON-vastaukset siivoutuvat."""
    items = [_make_enriched("Artikkeli", "Sisältö...", rank=1)]

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> str:
        inner = json.dumps({"headline": "Koodiblokkiotsikko", "summary": "Koodiblokkikuvailu."})
        return f"```json\n{inner}\n```"

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm_code_fence,
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Koodiblokkiotsikko"

