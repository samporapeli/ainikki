import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from agent.compose import (
    build_compose_system_prompt, build_compose_user_prompt, compose_items,
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

    prompt = build_compose_system_prompt(
        persona, golden_examples, guardrails, "ai", "Testiyleisö"
    )

    assert "ainikki" in prompt
    assert "clickbait" in prompt  # persona.avoid
    assert "Käytä VAIN annetussa lähdemateriaalissa mainittuja faktoja." in prompt  # guardrails rule
    assert "AGENTS.md yleistyy koodausagenttien yhteisenä tiedostomuotona" in prompt  # golden example
    assert '{"otsikko"' in prompt  # JSON format requirement
    # v2: Finnish language rules
    assert "yhdyssanat" in prompt
    assert "desimaalipilkku" in prompt or "desimaalierottimena" in prompt
    assert "Oxford" in prompt


def test_compose_uses_fallback_content_source():
    item = _make_enriched("Ensisijainen otsikko", "Fallback-artikkelin sisältö", rank=1)
    item.content_source = RawItem(
        title="Toissijainen otsikko", url="https://example.com/fallback",
        source_type=SourceType.rss,
        published_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
        raw_signal={}, origin_id="fallback", fetched_at=datetime.now(timezone.utc),
    )

    prompt = build_compose_user_prompt(item)

    assert "Toissijainen otsikko" in prompt
    assert "Ensisijainen otsikko" not in prompt
    assert "Fallback-artikkelin sisältö" in prompt


def test_compose_items_happy_path():
    items = [
        _make_enriched("Uusi malli julkaistu", "Yhtiö julkaisi uuden mallin tänään...", rank=1),
        _make_enriched("Tutkimuspaperi ilmestyi", "Tutkijat osoittivat että...", rank=2),
    ]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        if "Uusi malli" in user_prompt:
            return json.dumps({"otsikko": "Yhtiö julkaisi uuden mallin",
                                "tiivistelmä": "Malli tuo parannuksia aiempaan verrattuna."}), {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80, "cost": 0.0001}
        return json.dumps({"otsikko": "Uusi tutkimus haastaa aiemman käsityksen",
                            "tiivistelmä": "Tutkijat löysivät yllättävän tuloksen."}), {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80, "cost": 0.0001}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 2
    assert len(result.warnings) == 0
    assert result.persona_id == "ainikki-v1"
    assert result.guardrails_version == "v1"
    assert result.golden_examples_version == "v2"

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

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        if "Toimiva" in user_prompt:
            return json.dumps({"otsikko": "Toimiva otsikko", "tiivistelmä": "Toimiva yhteenveto."}), {}
        return "tämä ei ole JSON:ia", {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1, "only the successful item should remain"
    assert result.items[0].headline == "Toimiva otsikko"
    assert len(result.warnings) == 1
    assert "Rikkinäinen vastaus" in result.warnings[0]


def test_empty_headline_is_rejected():
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        return json.dumps({"otsikko": "   ", "tiivistelmä": "Jotain tekstiä."}), {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 0
    assert len(result.warnings) == 1


def test_code_fenced_json_is_parsed():
    """Verifies that model responses wrapped in code fences are cleaned up."""
    items = [_make_enriched("Artikkeli", "Sisältö...", rank=1)]

    def mock_llm_code_fence(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        inner = json.dumps({"otsikko": "Koodiblokkiotsikko", "tiivistelmä": "Koodiblokkikuvailu."})
        return f"```json\n{inner}\n```", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.00001}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm_code_fence,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Koodiblokkiotsikko"


def test_compose_retries_on_empty_response():
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]
    call_count = 0

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "", {}
        return json.dumps({"otsikko": "Toistettu otsikko", "tiivistelmä": "Toistettu yhteenveto."}), {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Toistettu otsikko"
    assert call_count == 2


def test_compose_retries_on_overlong_output():
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]
    call_count = 0

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({
                "otsikko": "x" * 257,
                "tiivistelmä": "Lyhyt yhteenveto.",
            }), {}
        return json.dumps({
            "otsikko": "Korjattu otsikko",
            "tiivistelmä": "Korjattu yhteenveto.",
        }), {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Korjattu otsikko"
    assert call_count == 2


def test_compose_gives_up_after_retry():
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]
    call_count = 0

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        nonlocal call_count
        call_count += 1
        return "", {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 0
    assert len(result.warnings) == 1
    assert call_count == 2


def test_compose_retries_on_empty_code_fence():
    """LLM returns code fences with empty content — raw response is non-empty
    but strip_code_fences extracts empty string. Retry should fire."""
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]
    call_count = 0

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "```json\n\n```", {}
        return json.dumps({"otsikko": "Toistettu otsikko", "tiivistelmä": "Toistettu yhteenveto."}), {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Toistettu otsikko"
    assert call_count == 2


def test_compose_retries_on_none_response():
    """LLM returns None (e.g. content filter / empty message) on first try."""
    items = [_make_enriched("Juttu", "Sisältö...", rank=1)]
    call_count = 0

    def mock_llm(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, {}  # type: ignore[return-value]
        return json.dumps({"otsikko": "Toistettu otsikko", "tiivistelmä": "Toistettu yhteenveto."}), {}

    result = compose_items(items, PERSONA_PATH, GUARDRAILS_PATH, mock_llm,
                            topic="test",
                            target_audience="Testiyleisö",
                            golden_examples_dir=GOLDEN_EXAMPLES_DIR)

    assert len(result.items) == 1
    assert result.items[0].headline == "Toistettu otsikko"
    assert call_count == 2
