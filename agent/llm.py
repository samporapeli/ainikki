"""
Malliriippumaton LLM-kutsukerros. Lukee config/models.yaml:sta mitä mallia
mikäkin step käyttää, ja palauttaa funktion joka toteuttaa cluster.py:n
(ja myöhemmin score.py/compose.py:n) odottaman LlmCall-rajapinnan:
    (system_prompt: str, user_prompt: str) -> str

Verkkokutsu ja pyynnön rakentaminen ovat erillään testattavuuden vuoksi -
sama periaate kuin hn.py:ssä. Testit (tests/test_llm.py) käyttävät
httpx.MockTransport-mekanismia, ei oikeaa verkkoyhteyttä eikä API-avainta.
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

PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class StepModelConfig:
    provider: str
    model: str
    base_url: str | None = None  # pakollinen jos provider == "local"
    max_tokens: int = 1024


def load_models_config(path: Path = Path("config/models.yaml")) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_step_config(step: str, config: dict,
                         override_model: str | None = None,
                         override_provider: str | None = None) -> StepModelConfig:
    """Yhdistää config-tiedoston ja mahdolliset ajokohtaiset overridet
    (esim. CLI-flagit AB-vertailua varten). Override voittaa aina configin.
    """
    step_cfg = config.get("steps", {}).get(step, {})
    provider = override_provider or step_cfg.get("provider") or config.get("default_provider")
    model = override_model or step_cfg.get("model")

    if not provider:
        raise ValueError(f"Ei provideria määritelty stepille '{step}'")
    if not model:
        raise ValueError(f"Ei mallia määritelty stepille '{step}' eikä overridea annettu")
    if provider == "local" and not step_cfg.get("base_url"):
        raise ValueError(f"provider='local' vaatii base_url-kentän config/models.yaml:ssa stepille '{step}'")

    return StepModelConfig(provider=provider, model=model, base_url=step_cfg.get("base_url"),
                           max_tokens=step_cfg.get("max_tokens", 1024))


def _post_openai_compatible(base_url: str, api_key: str | None, model: str,
                             system_prompt: str, user_prompt: str,
                             client: httpx.Client,
                             expect_json: bool = True) -> str:
    """Toimii OpenAI:lle, OpenRoutrille ja mille tahansa OpenAI-yhteensopivalle
    /chat/completions-rajapinnalle (esim. Ollama paikallisesti).

    Retries on 429 with exponential backoff (_MAX_RETRIES attempts).
    Jos expect_json=True, lisää response_format: {"type": "json_object"}
    pyyntöön pakottaen mallin palauttamaan puhdasta JSON:ia.
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
            return data["choices"][0]["message"]["content"]
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


def _post_anthropic(model: str, system_prompt: str, user_prompt: str,
                     api_key: str | None, client: httpx.Client,
                     max_tokens: int = 1024) -> str:
    headers = {
        "x-api-key": api_key or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    resp = client.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
    return "\n".join(text_blocks)


def make_llm_call(step: str, config_path: Path = Path("config/models.yaml"),
                   override_model: str | None = None, override_provider: str | None = None,
                   client: httpx.Client | None = None):
    """Palauttaa funktion joka toteuttaa LlmCall-rajapinnan (cluster.py, myöhemmin score/compose).

    client-parametri on testattavuutta varten - injektoi httpx.MockTransport-pohjainen
    client testeissä sen sijaan että tehtäisiin oikea verkkokutsu.
    """
    config = load_models_config(config_path)
    step_cfg = resolve_step_config(step, config, override_model, override_provider)

    def _call(system_prompt: str, user_prompt: str) -> str:
        owns_client = client is None
        c = client or httpx.Client(timeout=60.0)
        try:
            if step_cfg.provider == "anthropic":
                api_key = os.environ.get(PROVIDER_API_KEY_ENV["anthropic"])
                return _post_anthropic(step_cfg.model, system_prompt, user_prompt, api_key, c,
                                       max_tokens=step_cfg.max_tokens)

            base_url = step_cfg.base_url or PROVIDER_BASE_URLS.get(step_cfg.provider)
            if not base_url:
                raise ValueError(f"Tuntematon provider '{step_cfg.provider}' eikä base_url annettu")
            api_key_env = PROVIDER_API_KEY_ENV.get(step_cfg.provider, f"{step_cfg.provider.upper()}_API_KEY")
            api_key = os.environ.get(api_key_env)
            return _post_openai_compatible(base_url, api_key, step_cfg.model, system_prompt, user_prompt, c)
        finally:
            if owns_client:
                c.close()

    return _call
