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
            "choices": [{"message": {"content": "moi maailma"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0001},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system-ohje", "käyttäjän kysymys", client,
    )

    assert result == "moi maailma"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["cost"] == 0.0001
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
            "choices": [{"message": {"content": "vapaa teksti"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "cost": 0.00005},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system-ohje", "käyttäjän kysymys", client,
        expect_json=False,
    )

    assert result == "vapaa teksti"
    assert "response_format" not in captured["body"]
    assert usage["total_tokens"] == 8


def test_openai_compatible_no_cost_field():
    """Providers like OpenAI and Ollama don't return cost in usage."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ei kustannusta"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system-ohje", "käyttäjän kysymys", client,
    )

    assert result == "ei kustannusta"
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 10
    assert usage["total_tokens"] == 30
    assert usage.get("cost") is None
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_make_llm_call_end_to_end_with_cluster():
    """End-to-end chain: config -> make_llm_call -> cluster_candidates,
    with mock transport returning a valid cluster response using real HN fixture data.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)  # 8 candidates

    def handler(request: httpx.Request) -> httpx.Response:
        # don't cluster anything together in this test - all become singletons
        clusters = [{"candidate_indices": [i], "primary_position": 0, "reason": None} for i in range(8)]
        content = json.dumps({"clusters": clusters})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "cost": 0.0002},
        })

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_call = make_llm_call("cluster", config_path=FIXTURE_CONFIG, client=mock_client)

    result = cluster_candidates(candidates, llm_call)

    assert result.warning is None
    assert len(result.clusters) == 8


def test_openai_compatible_null_content_returns_empty_string():
    """Safety blocks or filter triggers returning null content should yield empty string."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system", "user", client,
    )

    assert result == ""
    assert isinstance(result, str)


def test_openai_compatible_empty_choices_returns_empty_string():
    """Responses with empty choices array should yield empty string."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system", "user", client,
    )

    assert result == ""
    assert isinstance(result, str)


def test_openai_compatible_retries_on_502(monkeypatch):
    """502 Bad Gateway should be retried with backoff."""
    monkeypatch.setattr("agent.llm._RETRY_BASE_DELAY", 0.0)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "toipui"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, _ = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system", "user", client,
    )

    assert result == "toipui"
    assert attempts == 2


def test_openai_compatible_retries_on_timeout(monkeypatch):
    """Network/read timeouts should be retried."""
    monkeypatch.setattr("agent.llm._RETRY_BASE_DELAY", 0.0)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "vastattu"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, _ = _post_openai_compatible(
        "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
        "system", "user", client,
    )

    assert result == "vastattu"
    assert attempts == 2


def test_openai_compatible_exhausted_retries_raises(monkeypatch):
    """When retries are exhausted on 500, HTTPStatusError is raised."""
    monkeypatch.setattr("agent.llm._RETRY_BASE_DELAY", 0.0)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        _post_openai_compatible(
            "https://openrouter.ai/api/v1", "fake-key", "openai/gpt-4o-mini",
            "system", "user", client,
        )

    assert attempts == 6  # 1 initial + 5 retries
