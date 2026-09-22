"""
Model-agnostic LLM call layer. Reads config/models.yaml to determine which
model each step uses, and returns a function implementing the LlmCall
interface expected by cluster.py (and later score.py/compose.py):
    (system_prompt: str, user_prompt: str) -> str

Named providers are configured via ``config/models.yaml`` under
``providers:``. Each provider can use any supported endpoint and its own
authentication scheme (env var, literal key, or no auth). The adapter
normalizes provider-specific chat APIs behind one internal interface:
Ollama uses ``/api/chat`` and OpenRouter uses ``/v1/chat/completions``.

HTTP call and request construction are separate for testability — same
principle as in hn.py. Tests (tests/test_llm.py) use httpx.MockTransport,
no real network or API keys needed.
"""

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2
_TIMEOUT_RETRIES = 1
_RETRY_BASE_DELAY = 10.0

_INTERP_PATTERN = re.compile(
    r'\$\{([^}]+(?::(?:-[^}]+)?)?)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
)


@dataclass
class StepModelConfig:
    provider: str
    model: str
    temperature: float = 0.2
    thinking: bool | None = None


@dataclass
class ProviderEndpoint:
    """Resolved endpoint for a named provider configuration."""
    provider: str
    base_url: str
    auth_headers: list[dict[str, str]] | None
    type: str = "ollama"
    timeout: float = 60.0


def _interpolate_vars(value: str) -> str:
    """Replace ``$VAR``, ``${VAR}``, and ``${VAR:-default}`` with env values."""

    def replacer(match: re.Match) -> str:
        brace = match.group(1)
        simple = match.group(2)
        if brace is not None:
            name, sep, default = brace.partition(":")
            val = os.environ.get(name)
            if val is None and sep:
                return default[1:] if default.startswith("-") else default
            return val or ""
        name = simple
        val = os.environ.get(name)
        if val is None:
            raise ValueError(f"Environment variable '{name}' is not set")
        return val

    return _INTERP_PATTERN.sub(replacer, value)


def load_models_config(path: Path = Path("config/models.yaml")) -> dict:
    return yaml.safe_load(path.read_text())


def resolve_step_configs(
    step: str,
    config: dict,
    model_entries: list[dict] | None = None,
    override_model: str | None = None,
    override_provider: str | None = None,
) -> list[StepModelConfig]:
    """Resolve the ordered model chain for one pipeline step."""
    if override_model or override_provider:
        entries = model_entries or config.get("steps", {}).get(step, {}).get("models") or []
        if not entries:
            if not override_model or not override_provider:
                raise ValueError(f"No models defined for step '{step}'")
            entries = [{}]
        first = entries[0]
        provider = override_provider or first.get("provider")
        model = override_model or first.get("model")
        if not provider:
            raise ValueError(f"No provider defined for step '{step}'")
        if not model:
            raise ValueError(f"No model defined for step '{step}'")
        return [StepModelConfig(
            provider=provider,
            model=model,
            temperature=first.get("temperature", 0.2),
            thinking=first.get("thinking"),
        )]

    entries = model_entries
    if entries is None:
        entries = config.get("steps", {}).get(step, {}).get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No models defined for step '{step}'")

    result = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Model entry {index} for step '{step}' must be a mapping")
        provider = entry.get("provider")
        model = entry.get("model")
        if not provider:
            raise ValueError(f"No provider defined for model {index} of step '{step}'")
        if not model:
            raise ValueError(f"No model defined for model {index} of step '{step}'")
        result.append(StepModelConfig(
            provider=str(provider).strip(),
            model=str(model).strip(),
            temperature=entry.get("temperature", 0.2),
            thinking=entry.get("thinking"),
        ))
    return result


def resolve_step_config(
    step: str,
    config: dict,
    override_model: str | None = None,
    override_provider: str | None = None,
) -> StepModelConfig:
    """Resolve the primary model for a step.

    Callers that need fallback behavior should use ``resolve_step_configs``.
    """
    return resolve_step_configs(
        step, config, override_model=override_model,
        override_provider=override_provider,
    )[0]


def _post_chat_completion(
    provider: ProviderEndpoint,
    model: str,
    system_prompt: str,
    user_prompt: str,
    client: httpx.Client,
    expect_json: bool = True,
    temperature: float = 0.2,
    thinking: bool | None = None,
    step: str = "",
) -> tuple[str, dict]:
    """Send a provider-specific chat request with shared retry behavior."""
    base_url = provider.base_url.rstrip("/")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if provider.type == "ollama":
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        endpoint = f"{base_url}/api/chat"
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if expect_json:
            payload["format"] = "json"
        if thinking is not None:
            payload["think"] = thinking
    else:
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}
        if thinking is False:
            payload["reasoning"] = {"effort": "none"}

    headers: dict[str, str] = {}
    for h in provider.auth_headers or []:
        for k, v in h.items():
            headers[k] = v

    step_prefix = f"[{step}] " if step else ""
    http_attempts: list[dict] = []
    for attempt in range(_MAX_RETRIES + 1):
        t0 = time.perf_counter()
        logger.info(
            "%sLLM request started: %s/%s %s (attempt %d/%d)",
            step_prefix,
            provider.provider,
            model,
            endpoint,
            attempt + 1,
            _MAX_RETRIES + 1,
        )
        try:
            resp = client.post(endpoint, json=payload, headers=headers)
            duration_ms = round((time.perf_counter() - t0) * 1000)
            if resp.status_code == 429 or resp.status_code >= 500 or resp.status_code == 408:
                if attempt == _MAX_RETRIES:
                    http_attempts.append({
                        "status": "failed",
                        "error": f"HTTP {resp.status_code}",
                        "duration_ms": duration_ms,
                        "response": resp.text,
                        "reasoning": None,
                    })
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        error.llm_attempts = http_attempts
                        raise
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass
                logger.warning(
                    "LLM request returned HTTP %d in %dms (attempt %d/%d), retrying in %.0fs...",
                    resp.status_code,
                    duration_ms,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    delay,
                )
                http_attempts.append({
                    "status": "failed",
                    "error": f"HTTP {resp.status_code}",
                    "duration_ms": duration_ms,
                    "response": resp.text,
                    "reasoning": None,
                })
                time.sleep(delay)
                continue

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as error:
                http_attempts.append({
                    "status": "failed",
                    "error": f"HTTP {resp.status_code}",
                    "duration_ms": duration_ms,
                    "response": resp.text,
                    "reasoning": None,
                })
                error.llm_attempts = http_attempts
                raise
            try:
                data = resp.json()
            except ValueError as error:
                http_attempts.append({
                    "status": "failed",
                    "error": "invalid JSON response",
                    "duration_ms": duration_ms,
                    "response": resp.text,
                    "reasoning": None,
                })
                error.llm_attempts = http_attempts
                error.llm_response = resp.text
                raise
            if provider.type == "ollama":
                message = data.get("message") or {}
                content = str(message.get("content") or "")
                prompt_tokens = data.get("prompt_eval_count") or 0
                completion_tokens = data.get("eval_count") or 0
                cost = None
                reasoning = message.get("thinking") or message.get("reasoning")
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cost": cost,
                    "duration_ms": duration_ms,
                    "reasoning": reasoning,
                }
            else:
                choices = data.get("choices") or []
                message = choices[0].get("message") or {} if choices else {}
                content = str(message.get("content") or "")
                usage = dict(data.get("usage") or {})
                prompt_tokens = usage.get("prompt_tokens", 0) or 0
                completion_tokens = usage.get("completion_tokens", 0) or 0
                cost = usage.get("cost")
                reasoning = message.get("reasoning")
                usage["cost"] = cost
                usage["duration_ms"] = duration_ms
                usage["reasoning"] = reasoning

            cost_str = f", cost={cost}" if cost is not None else ""
            logger.info(
                "%sLLM request completed: %s/%s in %dms (prompt=%d, completion=%d%s)",
                step_prefix,
                provider.provider,
                model,
                duration_ms,
                prompt_tokens,
                completion_tokens,
                cost_str,
            )
            http_attempts.append({
                "status": "success",
                "error": None,
                "duration_ms": duration_ms,
                "response": content,
                "reasoning": reasoning if isinstance(reasoning, str) else None,
            })
            usage["_http_attempts"] = http_attempts
            return content, usage
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            duration_ms = round((time.perf_counter() - t0) * 1000)
            retry_limit = _TIMEOUT_RETRIES if isinstance(e, httpx.TimeoutException) else _MAX_RETRIES
            http_attempts.append({
                "status": "failed",
                "error": str(e) or type(e).__name__,
                "duration_ms": duration_ms,
                "response": None,
                "reasoning": None,
            })
            if attempt >= retry_limit:
                e.llm_attempts = http_attempts
                raise
            delay = _RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "LLM request failed with %s in %dms (attempt %d/%d), retrying in %.0fs...",
                type(e).__name__,
                duration_ms,
                attempt + 1,
                retry_limit + 1,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError("Retries exhausted")  # unreachable, but satisfies mypy


def _resolve_provider(
    step_config: StepModelConfig,
    config: dict,
) -> ProviderEndpoint:
    """Resolve a named provider from the global provider configuration."""
    providers = config.get("providers") or {}
    cfg = providers.get(step_config.provider)
    if not isinstance(cfg, dict):
        names = list(providers)
        raise ValueError(
            f"Unknown provider '{step_config.provider}' (existing: {names})"
        )

    provider_type = cfg.get("type", "ollama")
    if provider_type not in {"ollama", "openrouter"}:
        raise ValueError(
            f"Unknown provider type '{provider_type}' for '{step_config.provider}'"
        )

    base_url_env = cfg.get("base_url_env")
    if base_url_env:
        base_url = os.environ.get(base_url_env)
        if not base_url:
            raise ValueError(
                f"Provider '{step_config.provider}' is missing base_url "
                f"(base_url_env '{base_url_env}' not set)"
            )
    else:
        base_url = cfg.get("base_url")
        if base_url is not None:
            base_url = _interpolate_vars(str(base_url))
    if not base_url:
        raise ValueError(f"Provider '{step_config.provider}' has no base_url")

    api_key = cfg.get("api_key")
    if api_key is not None:
        api_key = _interpolate_vars(str(api_key))
    if not api_key and cfg.get("api_key_env"):
        api_key = os.environ.get(cfg["api_key_env"])

    auth_headers = [{"Authorization": f"Bearer {api_key}"}] if api_key else None
    timeout = float(cfg.get("timeout") or 60.0)
    return ProviderEndpoint(
        provider=step_config.provider,
        type=provider_type,
        base_url=base_url,
        auth_headers=auth_headers,
        timeout=timeout,
    )


def make_llm_call(
    step: str,
    config_path: Path = Path("config/models.yaml"),
    override_model: str | None = None,
    override_provider: str | None = None,
    step_models_override: list[dict] | None = None,
    client: httpx.Client | None = None,
):
    """Return an LLM call function with an ordered model fallback chain."""
    config = load_models_config(config_path)
    model_chain = resolve_step_configs(
        step,
        config,
        model_entries=step_models_override,
        override_model=override_model,
        override_provider=override_provider,
    )

    def _call(system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        last_error: Exception | None = None
        attempts: list[dict] = []
        for index, step_cfg in enumerate(model_chain):
            provider = None
            model_attempts: list[dict] = []
            owns_client = client is None
            c = None
            try:
                provider = _resolve_provider(step_cfg, config)
                c = client or httpx.Client(timeout=provider.timeout)
                content, usage = _post_chat_completion(
                    provider=provider,
                    model=step_cfg.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    client=c,
                    temperature=step_cfg.temperature,
                    thinking=step_cfg.thinking,
                    step=step,
                )
                usage = dict(usage)
                model_attempts = usage.pop("_http_attempts", [])
                attempts.extend({
                    **attempt,
                    "provider": provider.provider,
                    "model": step_cfg.model,
                } for attempt in model_attempts)
                usage["provider"] = provider.provider
                usage["model"] = step_cfg.model
                usage["attempts"] = attempts
                return content, usage
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                model_attempts = getattr(error, "llm_attempts", [])
                attempts.extend({
                    **attempt,
                    "provider": step_cfg.provider,
                    "model": step_cfg.model,
                } for attempt in model_attempts)
                if not model_attempts:
                    attempts.append({
                        "provider": step_cfg.provider,
                        "model": step_cfg.model,
                        "status": "failed",
                        "error": str(error),
                        "duration_ms": 0,
                        "response": getattr(error, "llm_response", None),
                        "reasoning": getattr(error, "llm_reasoning", None),
                    })
                if index + 1 < len(model_chain):
                    logger.warning(
                        "LLM model %s/%s failed (%s); trying fallback %s/%s",
                        provider.provider,
                        step_cfg.model,
                        error,
                        model_chain[index + 1].provider,
                        model_chain[index + 1].model,
                    )
            finally:
                if owns_client and c is not None:
                    c.close()

        assert last_error is not None
        try:
            last_error.llm_attempts = attempts
        except AttributeError:
            pass
        raise last_error

    return _call
