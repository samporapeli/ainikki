"""Tests for ``agent/llm.py``.

Every test that exercises HTTP (``_post_openai_compatible`` or
``make_llm_call``) uses ``httpx.MockTransport`` — no real network or API keys.
"""

import json
from pathlib import Path

import httpx
import pytest

from agent.llm import (
    ProviderEndpoint,
    _post_openai_compatible,
    _resolve_provider,
    make_llm_call,
    resolve_step_config,
    resolve_step_configs,
)
from agent.cluster import cluster_candidates
from agent.dedup import dedup_candidates
from agent.collect.hn import parse_hn_hits

FIXTURE_CONFIG = Path("tests/fixtures/models_test.yaml")


# -- Resolution tests ---------------------------------------------------------

def test_resolve_step_config_basic():
    """The primary entry in a step's model chain is resolved."""
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    cfg = resolve_step_config("cluster", config)
    assert cfg.provider == "local"
    assert cfg.model == "ornith:35b"


def test_resolve_step_config_override_wins():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    cfg = resolve_step_config(
        "cluster", config,
        override_model="gpt-4o",
        override_provider="openai",
    )
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"


def test_resolve_step_config_missing_model_raises():
    import yaml
    config = yaml.safe_load(FIXTURE_CONFIG.read_text())
    with pytest.raises(ValueError, match="malli|model"):
        resolve_step_config("broken_step", config)


# -- _post_openai_compatible tests -------------------------------------------

def _make_capture(request: httpx.Request) -> dict[str, str | dict]:
    return {
        "url": str(request.url),
        "body": json.loads(request.content),
    }


def test_openai_compatible_request_and_parse():
    """An OpenRouter response includes its real top-level cost."""
    captures = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captures["body"] = _make_capture(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "moi maailma"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15},
            "cost": 0.0001,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
        type="openrouter",
    )
    result, usage = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system-ohje",
        user_prompt="käyttäjän kysymys",
        client=client,
    )

    assert result == "moi maailma"
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["cost"] == 0.0001
    assert captures["body"]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    body = captures["body"]["body"]
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["messages"][0]["role"] == "system"
    assert body["response_format"] == {"type": "json_object"}


def test_openai_compatible_no_json_format_when_disabled():
    """expect_json=False removes the response_format parameter."""
    captures = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captures["body"] = _make_capture(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "vapaa teksti"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3,
                      "total_tokens": 8, "cost": 0.00005},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    result, usage = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system-ohje",
        user_prompt="käyttäjän kysymys",
        client=client,
        expect_json=False,
    )
    assert result == "vapaa teksti"
    assert "response_format" not in captures["body"]["body"]
    assert usage["total_tokens"] == 8


def test_openai_compatible_ignores_cost_field():
    """OpenAI-compatible endpoints do not report provider cost."""
    captures = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captures["url"] = str(request.url)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ei kustannusta"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10,
                      "total_tokens": 30},
            "cost": 0.123,
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="local", base_url="https://local.invalid/v1",
        auth_headers=None, type="openai_compatible",
    )
    result, usage = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system-ohje",
        user_prompt="käyttäjän kysymys",
        client=client,
    )
    assert result == "ei kustannusta"
    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 10
    assert usage["total_tokens"] == 30
    assert usage.get("cost") is None
    assert captures["url"] == "https://local.invalid/v1/chat/completions"


def test_openai_compatible_null_content_returns_empty_string():
    """Safety blocks or filter triggers returning ``null`` yield empty
    string, not ``"None"``."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0,
                      "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    result, _ = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        client=client,
    )
    assert result == ""
    assert isinstance(result, str)


def test_openai_compatible_empty_choices_returns_empty_string():
    """Responses with empty ``choices`` array yield empty string."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0,
                      "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    result, _ = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        client=client,
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
            "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                      "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    result, _ = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        client=client,
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
            "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                      "total_tokens": 10},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    result, _ = _post_openai_compatible(
        provider=provider,
        model="openai/gpt-4o-mini",
        system_prompt="system",
        user_prompt="user",
        client=client,
    )
    assert result == "vastattu"
    assert attempts == 2


def test_openai_compatible_exhausted_retries_raises(monkeypatch):
    """When retries are exhausted on 500, ``HTTPStatusError`` is raised."""
    monkeypatch.setattr("agent.llm._RETRY_BASE_DELAY", 0.0)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="Service Unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ProviderEndpoint(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        auth_headers=[{"Authorization": "Bearer fake-key"}],
    )
    with pytest.raises(httpx.HTTPStatusError):
        _post_openai_compatible(
            provider=provider,
            model="openai/gpt-4o-mini",
            system_prompt="system",
            user_prompt="user",
            client=client,
        )
    assert attempts == 6  # 1 initial + 5 retries


# -- make_llm_call test --------------------------------------------------------

def test_make_llm_call_end_to_end_with_cluster():
    """End-to-end chain: config -> make_llm_call -> cluster_candidates,
    with mock transport returning a valid cluster response using real HN fixture
    data.
    """
    fixture = json.loads(Path("tests/fixtures/hn_response_sample.json").read_text())
    hn_items = parse_hn_hits(fixture["hits"])
    candidates = dedup_candidates(hn_items)  # 8 candidates

    def handler(request: httpx.Request) -> httpx.Response:
        clusters = [{"candidate_indices": [i], "primary_position": 0,
                     "reason": None} for i in range(8)]
        content = json.dumps({"clusters": clusters})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "total_tokens": 150, "cost": 0.0002},
        })

    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    llm_call = make_llm_call("cluster", config_path=FIXTURE_CONFIG,
                             client=mock_client)

    result = cluster_candidates(candidates, llm_call)

    assert result.warning is None
    assert len(result.clusters) == 8


def test_make_llm_call_uses_fallback_model(monkeypatch):
    monkeypatch.setattr("agent.llm._RETRY_BASE_DELAY", 0.0)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        requests.append(model)
        if len(requests) == 1:
            return httpx.Response(401, json={"error": "unauthorized"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "fallback"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result, usage = make_llm_call("cluster", FIXTURE_CONFIG, client=client)("s", "u")
    assert result == "fallback"
    assert requests == ["ornith:35b", "openai/gpt-4o-mini"]
    assert usage["provider"] == "openrouter"
    assert usage["model"] == "openai/gpt-4o-mini"


def test_resolve_step_configs_returns_fallback_chain():
    config = _fixture_config()
    chain = resolve_step_configs("cluster", config)
    assert [(entry.provider, entry.model) for entry in chain] == [
        ("local", "ornith:35b"),
        ("openrouter", "openai/gpt-4o-mini"),
    ]


# -- Provider resolution tests ------------------------------------------------

def _fixture_config():
    import yaml
    return yaml.safe_load(FIXTURE_CONFIG.read_text())


def test_resolve_step_config_with_provider_block(monkeypatch):
    """Provider type, URL, and credentials come from global config."""
    monkeypatch.setenv("OPENROUTER_PROD_API_KEY", "prod-key-123")
    config = _fixture_config()
    cfg = resolve_step_config("cluster", config, override_provider="openrouter")
    prov = _resolve_provider(cfg, config)
    assert prov.type == "openrouter"
    assert prov.base_url == "https://openrouter.ai/api/v1"
    assert prov.auth_headers == [{"Authorization": "Bearer prod-key-123"}]


def test_resolve_local_provider_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://test.local:11434/v1")
    config = _fixture_config()
    cfg = resolve_step_config("compose", config)
    prov = _resolve_provider(cfg, config)
    assert prov.provider == "local"
    assert prov.type == "openai_compatible"
    assert prov.base_url == "http://test.local:11434/v1"
    assert prov.auth_headers is None


def test_resolve_provider_unknown_raises():
    config = _fixture_config()
    cfg = resolve_step_config("cluster", config, override_provider="bogus")
    with pytest.raises(ValueError, match="Unknown provider 'bogus'"):
        _resolve_provider(cfg, config)


def test_resolve_local_provider_missing_env_var_raises(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    config = _fixture_config()
    cfg = resolve_step_config("cluster", config)
    with pytest.raises(ValueError, match="OLLAMA_HOST"):
        _resolve_provider(cfg, config)
