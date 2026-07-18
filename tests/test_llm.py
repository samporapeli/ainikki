"""
Ajo: python -m tests.test_llm (projektin juuresta)

Ei oikeaa verkkoyhteyttä eikä API-avainta - httpx.MockTransport simuloi
palvelimen vastauksen, jotta pyynnön rakentaminen ja vastauksen parsinta
tulee testattua molemmille provider-muodoille (OpenAI-yhteensopiva ja Anthropic).
"""

import json
from pathlib import Path

import httpx

from agent.llm import (
    resolve_step_config, make_llm_call,
    _post_openai_compatible, _post_anthropic,
)
from agent.cluster import cluster_candidates
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits

FIXTURE_CONFIG = Path("tests/fixtures/models_test.yaml")


def test_resolve_step_config_basic():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    cfg = resolve_step_config("cluster", config)
    assert cfg.provider == "openrouter"
    assert cfg.model == "openai/gpt-4o-mini"


def test_resolve_step_config_override_wins():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    cfg = resolve_step_config("cluster", config, override_model="gpt-4o", override_provider="openai")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"


def test_resolve_step_config_missing_model_raises():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    try:
        resolve_step_config("broken_step", config)
        assert False, "olisi pitänyt heittää ValueError"
    except ValueError as e:
        assert "malli" in str(e).lower() or "model" in str(e).lower()


def test_resolve_step_config_local_requires_base_url():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    cfg = resolve_step_config("local_step", config)
    assert cfg.provider == "local"
    assert cfg.base_url == "http://localhost:11434/v1"


def test_openai_compatible_request_and_parse():
    """Simuloi OpenRouter/OpenAI-vastausta, varmistaa oikean pyynnön muodon ja parsinnan."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "moi maailma"}}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system-ohje", "käyttäjän kysymys", client,
    )

    assert result == "moi maailma"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-4o-mini"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_no_json_format_when_disabled():
    """expect_json=False poistaa response_format -parametrin pyynnöstä."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "vapaa teksti"}}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system-ohje", "käyttäjän kysymys", client,
        expect_json=False,
    )

    assert result == "vapaa teksti"
    assert "response_format" not in captured["body"]


def test_anthropic_request_and_parse():
    """Simuloi Anthropicin /v1/messages-vastausta, joka on eri muotoinen kuin OpenAI:n."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "moi maailma anthropicilta"}]
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = _post_anthropic("claude-sonnet-4-6", "system-ohje", "käyttäjän kysymys",
                              "fake-anthropic-key", client)

    assert result == "moi maailma anthropicilta"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["body"]["system"] == "system-ohje"
    assert captured["headers"]["x-api-key"] == "fake-anthropic-key"


def test_make_llm_call_end_to_end_with_cluster():
    """Kokonaisketju: config -> make_llm_call -> cluster_candidates,
    mock-transportilla joka palauttaa oikean HN-fixturen datalla validin klusterivastauksen.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)  # 4 candidaattia

    def handler(request: httpx.Request) -> httpx.Response:
        # ei klusteroida mitään yhteen tässä testissä - kaikki singleton-ryhmiksi
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(4)]
        content = json.dumps({"clusters": clusters})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_call = make_llm_call("cluster", config_path=FIXTURE_CONFIG, client=mock_client)

    result = cluster_candidates(candidates, llm_call)

    assert result.warning is None
    assert len(result.clusters) == 4


if __name__ == "__main__":
    test_resolve_step_config_basic()
    test_resolve_step_config_override_wins()
    test_resolve_step_config_missing_model_raises()
    test_resolve_step_config_local_requires_base_url()
    test_openai_compatible_request_and_parse()
    test_openai_compatible_no_json_format_when_disabled()
    test_anthropic_request_and_parse()
    test_make_llm_call_end_to_end_with_cluster()
    print("\nKaikki testit läpi.")
