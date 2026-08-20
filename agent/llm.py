"""
Model-agnostic LLM call layer. Reads config/models.yaml to determine which
model each step uses, and returns a function implementing the LlmCall
interface expected by cluster.py (and later score.py/compose.py):
    (system_prompt: str, user_prompt: str) -> str

Currently only OpenRouter is supported. The adapter speaks the OpenAI-compatible
/v1/chat/completions interface, so adding other OpenAI-compatible providers
later requires only a config change.

HTTP call and request construction are separate for testability — same
principle as in hn.py. Tests (tests/test_llm.py) use httpx.MockTransport,
no real network or API keys needed.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 10.0

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass
class StepModelConfig:
    provider: str
    model: str
    max_tokens: int = 1024


def load_models_config(path: Path = Path("config/models.yaml")) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_step_config(step: str, config: dict,
                         override_model: str | None = None,
                         override_provider: str | None = None) -> StepModelConfig:
    """Merges config file values with any runtime overrides
    (e.g. CLI flags for A/B testing). Override always wins over config.
    """
    step_cfg = config.get("steps", {}).get(step, {})
    provider = override_provider or step_cfg.get("provider") or config.get("default_provider")
    model = override_model or step_cfg.get("model")

    if not provider:
        raise ValueError(f"No provider defined for step '{step}'")
    if not model:
        raise ValueError(f"No model defined for step '{step}' and no override given")

    return StepModelConfig(provider=provider, model=model,
                           max_tokens=step_cfg.get("max_tokens", 1024))


def _post_openai_compatible(base_url: str, api_key: str | None, model: str,
                              system_prompt: str, user_prompt: str,
                              client: httpx.Client,
                              expect_json: bool = True) -> tuple[str, dict]:
    """Sends a chat completion request to an OpenAI-compatible endpoint.

    Retries on 429 with exponential backoff (_MAX_RETRIES attempts).
    If expect_json=True, adds response_format: {"type": "json_object"}
    to the request, forcing the model to return valid JSON.

    Returns (content, usage) where usage is a dict with keys:
    prompt_tokens, completion_tokens, total_tokens, cost.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    if expect_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    for attempt in range(_MAX_RETRIES + 1):
        resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        if resp.status_code != 429:
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") or {}
            return content, usage
        if attempt == _MAX_RETRIES:
            resp.raise_for_status()
        delay = _RETRY_BASE_DELAY * (2 ** attempt)
        retry_after = resp.headers.get("retry-after")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        logger.warning("429 rate-limited (attempt %d/%d), retrying in %.0fs...",
                        attempt + 1, _MAX_RETRIES + 1, delay)
        time.sleep(delay)

    raise RuntimeError("Retries exhausted")  # unreachable, but satisfies mypy


def make_llm_call(step: str, config_path: Path = Path("config/models.yaml"),
                  override_model: str | None = None, override_provider: str | None = None,
                  client: httpx.Client | None = None):
    """Returns a function implementing the LlmCall interface (cluster.py,
    later score/compose).

    The LlmCall signature is (system_prompt, user_prompt) -> (content, usage)
    where usage is a dict with prompt_tokens, completion_tokens, total_tokens,
    and cost fields from OpenRouter's response.

    The client parameter is for testability — inject an httpx.MockTransport-based
    client in tests instead of making real HTTP calls.
    """
    config = load_models_config(config_path)
    step_cfg = resolve_step_config(step, config, override_model, override_provider)

    def _call(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        owns_client = client is None
        c = client or httpx.Client(timeout=60.0)
        try:
            api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
            return _post_openai_compatible(OPENROUTER_BASE_URL, api_key, step_cfg.model,
                                           system_prompt, user_prompt, c)
        finally:
            if owns_client:
                c.close()

    return _call
