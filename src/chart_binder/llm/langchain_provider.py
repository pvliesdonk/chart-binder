"""LangChain chat model factory for chart-binder.

Creates provider-agnostic BaseChatModel instances for use with LangChain agents
and structured output. Follows patterns from questfoundry.

Supported providers:
- Ollama (local models via OLLAMA_HOST)
- OpenAI (GPT models via OPENAI_API_KEY)
- Anthropic (Claude models via ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)


class LangChainProviderError(Exception):
    """Raised when LangChain provider creation fails."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


# Provider default models
PROVIDER_DEFAULTS: dict[str, str | None] = {
    "ollama": None,  # Require explicit model specification
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
}


def get_default_model(provider_name: str) -> str | None:
    """Get default model for a provider.

    Returns None for providers that require explicit model specification.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic).

    Returns:
        Default model name, or None if provider requires explicit model.
    """
    return PROVIDER_DEFAULTS.get(provider_name.lower())


def create_chat_model(
    provider_name: str,
    model: str,
    **kwargs: Any,
) -> BaseChatModel:
    """Create a LangChain BaseChatModel.

    This is the primary factory for getting chat models for use with
    LangChain agents and structured output.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic).
        model: Model name/identifier.
        **kwargs: Additional provider-specific options:
            - temperature: Sampling temperature (0.0 = deterministic)
            - max_tokens: Maximum tokens in response
            - timeout: Request timeout in seconds
            - host/base_url: API endpoint URL
            - api_key: API key (or uses env var)
            - num_ctx: Context window for Ollama (default 32768)

    Returns:
        Configured BaseChatModel.

    Raises:
        LangChainProviderError: If provider unavailable or misconfigured.
    """
    provider_lower = provider_name.lower()

    if provider_lower == "ollama":
        chat_model = _create_ollama_model(model, **kwargs)
    elif provider_lower == "openai":
        chat_model = _create_openai_model(model, **kwargs)
    elif provider_lower == "anthropic":
        chat_model = _create_anthropic_model(model, **kwargs)
    else:
        log.error("Unknown LangChain provider: %s", provider_lower)
        raise LangChainProviderError(provider_lower, f"Unknown provider: {provider_lower}")

    log.info("Created LangChain chat model: provider=%s model=%s", provider_lower, model)
    return chat_model


def _create_ollama_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create Ollama chat model.

    Args:
        model: Model name (e.g., "qwen3:8b", "llama3.2").
        **kwargs: Options including:
            - host: Ollama server URL (or OLLAMA_HOST env var)
            - temperature: Sampling temperature
            - num_ctx: Context window size (default 32768)
            - num_predict: Max tokens to generate
            - timeout: Request timeout

    Returns:
        Configured ChatOllama model.

    Raises:
        LangChainProviderError: If langchain-ollama not installed or OLLAMA_HOST not set.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        log.error("langchain-ollama not installed")
        raise LangChainProviderError(
            "ollama",
            "langchain-ollama not installed. Run: uv add langchain-ollama",
        ) from e

    host = kwargs.get("host") or kwargs.get("base_url") or os.getenv("OLLAMA_HOST")
    if not host:
        log.error("OLLAMA_HOST not configured")
        raise LangChainProviderError(
            "ollama",
            "OLLAMA_HOST not configured. Set OLLAMA_HOST environment variable.",
        )

    model_kwargs: dict[str, Any] = {
        "model": model,
        "base_url": host,
        "num_ctx": kwargs.get("num_ctx", 32768),  # Default 32k context
    }

    if "temperature" in kwargs:
        model_kwargs["temperature"] = kwargs["temperature"]

    if "num_predict" in kwargs:
        model_kwargs["num_predict"] = kwargs["num_predict"]
    elif "max_tokens" in kwargs:
        model_kwargs["num_predict"] = kwargs["max_tokens"]

    if "timeout" in kwargs:
        model_kwargs["timeout"] = kwargs["timeout"]

    return ChatOllama(**model_kwargs)


def _is_reasoning_model(model: str) -> bool:
    """Check if model is an OpenAI reasoning model (o1/o3 families).

    Reasoning models have different API constraints:
    - No temperature parameter (they control their own reasoning)
    - No tool/function calling support
    - Use max_completion_tokens instead of max_tokens

    Args:
        model: Model name to check.

    Returns:
        True if model is from the o1 or o3 family.
    """
    model_lower = model.lower()
    return model_lower.startswith("o1") or model_lower.startswith("o3")


def _create_openai_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create OpenAI chat model.

    Args:
        model: Model name (e.g., "gpt-4o-mini", "gpt-4o").
        **kwargs: Options including:
            - api_key: OpenAI API key (or OPENAI_API_KEY env var)
            - base_url: Custom API endpoint
            - temperature: Sampling temperature (ignored for o1/o3)
            - max_tokens: Maximum tokens in response
            - timeout: Request timeout

    Returns:
        Configured ChatOpenAI model.

    Raises:
        LangChainProviderError: If langchain-openai not installed or API key not set.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        log.error("langchain-openai not installed")
        raise LangChainProviderError(
            "openai",
            "langchain-openai not installed. Run: uv add langchain-openai",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not configured")
        raise LangChainProviderError(
            "openai",
            "API key required. Set OPENAI_API_KEY environment variable.",
        )

    model_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
    }

    if base_url := kwargs.get("base_url"):
        model_kwargs["base_url"] = base_url

    # Reasoning models (o1, o3) don't support temperature
    if not _is_reasoning_model(model):
        if "temperature" in kwargs:
            model_kwargs["temperature"] = kwargs["temperature"]
    else:
        log.debug("Reasoning model detected (%s), skipping temperature", model)

    if "max_tokens" in kwargs:
        model_kwargs["max_tokens"] = kwargs["max_tokens"]

    if "timeout" in kwargs:
        model_kwargs["timeout"] = kwargs["timeout"]

    return ChatOpenAI(**model_kwargs)


def _create_anthropic_model(model: str, **kwargs: Any) -> BaseChatModel:
    """Create Anthropic chat model.

    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514").
        **kwargs: Options including:
            - api_key: Anthropic API key (or ANTHROPIC_API_KEY env var)
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens in response
            - timeout: Request timeout

    Note:
        Anthropic does not support the seed parameter.

    Returns:
        Configured ChatAnthropic model.

    Raises:
        LangChainProviderError: If langchain-anthropic not installed or API key not set.
    """
    try:
        from langchain_anthropic import ChatAnthropic  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        log.error("langchain-anthropic not installed")
        raise LangChainProviderError(
            "anthropic",
            "langchain-anthropic not installed. Run: uv add langchain-anthropic",
        ) from e

    api_key = kwargs.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY not configured")
        raise LangChainProviderError(
            "anthropic",
            "API key required. Set ANTHROPIC_API_KEY environment variable.",
        )

    model_kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
    }

    if "temperature" in kwargs:
        model_kwargs["temperature"] = kwargs["temperature"]

    if "max_tokens" in kwargs:
        model_kwargs["max_tokens"] = kwargs["max_tokens"]

    if "timeout" in kwargs:
        model_kwargs["timeout"] = kwargs["timeout"]

    return ChatAnthropic(**model_kwargs)


## Tests


def test_get_default_model():
    """Test default model retrieval."""
    assert get_default_model("ollama") is None
    assert get_default_model("openai") == "gpt-4o-mini"
    assert get_default_model("anthropic") == "claude-sonnet-4-20250514"
    assert get_default_model("unknown") is None


def test_is_reasoning_model():
    """Test reasoning model detection."""
    assert _is_reasoning_model("o1") is True
    assert _is_reasoning_model("o1-mini") is True
    assert _is_reasoning_model("o3") is True
    assert _is_reasoning_model("o3-mini") is True
    assert _is_reasoning_model("gpt-4o") is False
    assert _is_reasoning_model("gpt-4o-mini") is False


def test_create_chat_model_unknown_provider():
    """Test error on unknown provider."""
    import pytest

    with pytest.raises(LangChainProviderError, match="Unknown provider"):
        create_chat_model("unknown_provider", "model")


def test_create_ollama_no_host(monkeypatch):
    """Test Ollama error when OLLAMA_HOST not set."""
    import pytest

    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    with pytest.raises(LangChainProviderError, match="OLLAMA_HOST not configured"):
        create_chat_model("ollama", "llama3.2")


def test_create_openai_no_key(monkeypatch):
    """Test OpenAI error when OPENAI_API_KEY not set."""
    import pytest

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LangChainProviderError, match="API key required"):
        create_chat_model("openai", "gpt-4o-mini")


def test_create_anthropic_no_key(monkeypatch):
    """Test Anthropic error when ANTHROPIC_API_KEY not set or module not installed."""
    import pytest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Either API key missing or langchain-anthropic not installed
    with pytest.raises(LangChainProviderError, match="(API key required|not installed)"):
        create_chat_model("anthropic", "claude-sonnet-4-20250514")
