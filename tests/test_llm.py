import json
from pathlib import Path

import httpx
import pytest

from agent.llm import (
    resolve_step_config, make_llm_call,
    _post_openai_compatible,
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
    with pytest.raises(ValueError, match="malli|model"):
        resolve_step_config("broken_step", config)


def test_openai_compatible_request_and_parse():
    """Simulates OpenRouter/OpenAI response, verifies correct request format and parsing."""
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
    """expect_json=False removes the response_format parameter from the request."""
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


def test_make_llm_call_end_to_end_with_cluster():
    """End-to-end chain: config -> make_llm_call -> cluster_candidates,
    with mock transport returning a valid cluster response using real HN fixture data.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)  # 4 candidates

    def handler(request: httpx.Request) -> httpx.Response:
        # don't cluster anything together in this test - all become singletons
        clusters = [{"candidate_indices": [i], "primary_index": 0, "reason": None} for i in range(4)]
        content = json.dumps({"clusters": clusters})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_call = make_llm_call("cluster", config_path=FIXTURE_CONFIG, client=mock_client)

    result = cluster_candidates(candidates, llm_call)

    assert result.warning is None
    assert len(result.clusters) == 4

