"""LLM provider plugin system for Chart-Binder.

Provides a flexible plugin architecture for multiple LLM backends:
- Ollama (local models)
- OpenAI (GPT-4o, GPT-4o-mini)

The system is designed to be extensible for future providers.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger(__name__)


@dataclass
class LLMMessage:
    """Single message in an LLM conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM backends must implement this interface to be used
    with the Chart-Binder adjudication system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification and logging."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        """Send a chat completion request to the LLM.

        Args:
            messages: List of messages in the conversation
            model: Model ID to use (provider-specific)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 = deterministic)
            timeout: Request timeout in seconds

        Returns:
            LLMResponse with the model's response
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        ...


class OllamaProvider(LLMProvider):
    """Ollama provider for local LLM inference.

    Connects to a local Ollama server for running models like
    Llama, Mistral, Phi, etc.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama3.2",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = httpx.Client(timeout=60.0)

    @property
    def name(self) -> str:
        return "ollama"

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                },
                raw_response=data,
            )
        except httpx.HTTPError as e:
            log.error(f"Ollama request failed: {e}")
            raise LLMProviderError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse Ollama response: {e}")
            raise LLMProviderError(f"Failed to parse Ollama response: {e}") from e

    def is_available(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """List available models from Ollama."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except (httpx.HTTPError, json.JSONDecodeError):
            return []


class OpenAIProvider(LLMProvider):
    """OpenAI provider for GPT models.

    Supports GPT-4o, GPT-4o-mini, and other OpenAI models.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.getenv(api_key_env)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._client = httpx.Client(timeout=60.0)

    @property
    def name(self) -> str:
        return "openai"

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMProviderError("OpenAI API key not configured")

        model = model or self.default_model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})

            return LLMResponse(
                content=content,
                model=model,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
                raw_response=data,
            )
        except httpx.HTTPStatusError as e:
            log.error(f"OpenAI request failed with status {e.response.status_code}: {e}")
            raise LLMProviderError(f"OpenAI request failed: {e}") from e
        except httpx.HTTPError as e:
            log.error(f"OpenAI request failed: {e}")
            raise LLMProviderError(f"OpenAI request failed: {e}") from e
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            log.error(f"Failed to parse OpenAI response: {e}")
            raise LLMProviderError(f"Failed to parse OpenAI response: {e}") from e

    def is_available(self) -> bool:
        return bool(self.api_key)


class LLMProviderError(Exception):
    """Exception raised when an LLM provider operation fails."""

    pass


class ProviderRegistry:
    """Registry for LLM providers.

    Manages provider instances and provides factory methods
    for creating providers from configuration.
    """

    _providers: dict[str, type[LLMProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }

    def __init__(self) -> None:
        self._instances: dict[str, LLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: type[LLMProvider]) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_class

    @classmethod
    def available_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    def get_provider(
        self,
        name: str,
        **kwargs: Any,
    ) -> LLMProvider:
        """Get or create a provider instance.

        Args:
            name: Provider name (e.g., "ollama", "openai")
            **kwargs: Provider-specific configuration

        Returns:
            Configured provider instance
        """
        cache_key = f"{name}:{hash(frozenset(kwargs.items()))}"

        if cache_key not in self._instances:
            if name not in self._providers:
                raise ValueError(
                    f"Unknown provider: {name}. Available: {self.available_providers()}"
                )

            provider_class = self._providers[name]
            self._instances[cache_key] = provider_class(**kwargs)

        return self._instances[cache_key]

    def create_from_config(self, config: dict[str, Any]) -> LLMProvider:
        """Create a provider from configuration dictionary.

        Args:
            config: LLM configuration with provider, model_id, etc.

        Returns:
            Configured provider instance
        """
        provider_name = config.get("provider", "ollama")
        model_id = config.get("model_id", "llama3.2")

        kwargs: dict[str, Any] = {"default_model": model_id}

        if provider_name == "ollama":
            if base_url := config.get("ollama_base_url"):
                kwargs["base_url"] = base_url
        elif provider_name == "openai":
            if api_key_env := config.get("api_key_env"):
                kwargs["api_key_env"] = api_key_env
            if api_key := config.get("api_key"):
                kwargs["api_key"] = api_key
            if base_url := config.get("openai_base_url"):
                kwargs["base_url"] = base_url

        return self.get_provider(provider_name, **kwargs)


## Tests


def test_llm_message():
    """Test LLMMessage dataclass."""
    msg = LLMMessage(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_llm_response():
    """Test LLMResponse dataclass."""
    resp = LLMResponse(content="Hello!", model="test-model")
    assert resp.content == "Hello!"
    assert resp.model == "test-model"
    assert resp.usage == {}


def test_provider_registry_available():
    """Test listing available providers."""
    providers = ProviderRegistry.available_providers()
    assert "ollama" in providers
    assert "openai" in providers


def test_provider_registry_get():
    """Test getting provider instances."""
    registry = ProviderRegistry()
    provider = registry.get_provider("ollama")
    assert provider.name == "ollama"
    assert isinstance(provider, OllamaProvider)


def test_ollama_provider_unavailable():
    """Test Ollama provider availability check when not running."""
    provider = OllamaProvider(base_url="http://localhost:99999")
    assert provider.is_available() is False


def test_openai_provider_unavailable_no_key():
    """Test OpenAI provider availability without API key."""
    provider = OpenAIProvider(api_key=None, api_key_env="NONEXISTENT_KEY_12345")
    assert provider.is_available() is False


def test_create_from_config_ollama():
    """Test creating Ollama provider from config."""
    registry = ProviderRegistry()
    config = {
        "provider": "ollama",
        "model_id": "mistral",
        "ollama_base_url": "http://localhost:11434",
    }
    provider = registry.create_from_config(config)
    assert provider.name == "ollama"
    assert isinstance(provider, OllamaProvider)


def test_create_from_config_openai():
    """Test creating OpenAI provider from config."""
    registry = ProviderRegistry()
    config = {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "api_key": "test-key",
    }
    provider = registry.create_from_config(config)
    assert provider.name == "openai"
    assert isinstance(provider, OpenAIProvider)
