from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class HttpCacheConfig(BaseModel):
    """HTTP cache configuration."""

    directory: Path = Field(default=Path(".cache/http"))
    ttl_seconds: int = Field(default=86400, ge=0)
    enabled: bool = Field(default=True)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    music_graph_path: Path = Field(default=Path("musicgraph.sqlite"))
    charts_path: Path = Field(default=Path("charts.sqlite"))
    decisions_path: Path = Field(default=Path("decisions.sqlite"))


class LiveSourcesConfig(BaseModel):
    """Live sources API configuration."""

    # API credentials (read from env vars if not provided)
    acoustid_api_key: str | None = Field(default=None)
    discogs_token: str | None = Field(default=None)
    spotify_client_id: str | None = Field(default=None)
    spotify_client_secret: str | None = Field(default=None)

    # Rate limits
    musicbrainz_rate_limit: float = Field(default=1.0, ge=0)  # req/sec
    acoustid_rate_limit: float = Field(default=3.0, ge=0)  # req/sec
    discogs_rate_limit: int = Field(default=25, ge=0)  # req/min

    # Cache TTLs (seconds)
    cache_ttl_musicbrainz: int = Field(default=3600, ge=0)  # 1 hour
    cache_ttl_discogs: int = Field(default=86400, ge=0)  # 24 hours
    cache_ttl_spotify: int = Field(default=7200, ge=0)  # 2 hours
    cache_ttl_wikidata: int = Field(default=604800, ge=0)  # 7 days
    cache_ttl_acoustid: int = Field(default=86400, ge=0)  # 24 hours


class LLMProviderType(StrEnum):
    """Supported LLM provider types."""

    OLLAMA = "ollama"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    """LLM adjudication configuration (Epic 13)."""

    # Enable/disable LLM adjudication
    enabled: bool = Field(default=False)

    # Provider: "ollama" (local) or "openai"
    provider: LLMProviderType = Field(default=LLMProviderType.OLLAMA)

    # Model ID (provider-specific)
    # Ollama: llama3.2, mistral, phi3, etc.
    # OpenAI: gpt-4o, gpt-4o-mini, etc.
    model_id: str = Field(default="llama3.2")

    # API configuration
    api_key_env: str = Field(default="OPENAI_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434")
    openai_base_url: str = Field(default="https://api.openai.com/v1")

    # Request settings
    timeout_s: float = Field(default=30.0, ge=1.0)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Confidence thresholds
    auto_accept_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # Prompt versioning for A/B testing
    prompt_template_version: str = Field(default="v1")

    # Review queue database path
    review_queue_path: Path = Field(default=Path("review_queue.sqlite"))


class Config(BaseModel):
    """
    Main configuration for chart-binder.

    Loads from TOML file with optional environment variable overrides.
    """

    http_cache: HttpCacheConfig = Field(default_factory=HttpCacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    live_sources: LiveSourcesConfig = Field(default_factory=LiveSourcesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    offline_mode: bool = Field(default=False)

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        """
        Load configuration from TOML file with environment variable overrides.

        Environment variables take precedence and follow the pattern:
        CHART_BINDER_<SECTION>_<KEY> (e.g., CHART_BINDER_HTTP_CACHE_TTL_SECONDS)

        All values are gathered into a single dictionary first, then validated
        by Pydantic to ensure consistent type checking and coercion.
        """
        config_dict: dict[str, object] = {}

        if config_path and config_path.exists():
            config_dict = tomllib.loads(config_path.read_text())

        config_dict = cls._merge_env_overrides(config_dict)
        return cls.model_validate(config_dict)

    @classmethod
    def _merge_env_overrides(cls, config_dict: dict[str, object]) -> dict[str, object]:
        """
        Merge environment variable overrides into config dictionary.

        Returns a new dictionary with env vars applied, ready for Pydantic validation.
        """
        env_prefix = "CHART_BINDER_"

        if offline := os.getenv(f"{env_prefix}OFFLINE_MODE"):
            config_dict["offline_mode"] = offline.lower() in ("true", "1", "yes")

        http_cache = config_dict.setdefault("http_cache", {})
        if not isinstance(http_cache, dict):
            http_cache = {}
            config_dict["http_cache"] = http_cache

        if cache_dir := os.getenv(f"{env_prefix}HTTP_CACHE_DIRECTORY"):
            http_cache["directory"] = cache_dir

        if cache_ttl := os.getenv(f"{env_prefix}HTTP_CACHE_TTL_SECONDS"):
            http_cache["ttl_seconds"] = cache_ttl

        if cache_enabled := os.getenv(f"{env_prefix}HTTP_CACHE_ENABLED"):
            http_cache["enabled"] = cache_enabled.lower() in ("true", "1", "yes")

        database = config_dict.setdefault("database", {})
        if not isinstance(database, dict):
            database = {}
            config_dict["database"] = database

        if db_music_path := os.getenv(f"{env_prefix}DATABASE_MUSIC_GRAPH_PATH"):
            database["music_graph_path"] = db_music_path

        if db_charts_path := os.getenv(f"{env_prefix}DATABASE_CHARTS_PATH"):
            database["charts_path"] = db_charts_path

        if db_decisions_path := os.getenv(f"{env_prefix}DATABASE_DECISIONS_PATH"):
            database["decisions_path"] = db_decisions_path

        # Live sources config
        live_sources = config_dict.setdefault("live_sources", {})
        if not isinstance(live_sources, dict):
            live_sources = {}
            config_dict["live_sources"] = live_sources

        # API credentials from env
        if acoustid_key := os.getenv("ACOUSTID_API_KEY"):
            live_sources["acoustid_api_key"] = acoustid_key
        if discogs_token := os.getenv("DISCOGS_TOKEN"):
            live_sources["discogs_token"] = discogs_token
        if spotify_id := os.getenv("SPOTIFY_CLIENT_ID"):
            live_sources["spotify_client_id"] = spotify_id
        if spotify_secret := os.getenv("SPOTIFY_CLIENT_SECRET"):
            live_sources["spotify_client_secret"] = spotify_secret

        # Rate limits
        if mb_rate := os.getenv(f"{env_prefix}LIVE_SOURCES_MUSICBRAINZ_RATE_LIMIT"):
            live_sources["musicbrainz_rate_limit"] = mb_rate
        if acoustid_rate := os.getenv(f"{env_prefix}LIVE_SOURCES_ACOUSTID_RATE_LIMIT"):
            live_sources["acoustid_rate_limit"] = acoustid_rate
        if discogs_rate := os.getenv(f"{env_prefix}LIVE_SOURCES_DISCOGS_RATE_LIMIT"):
            live_sources["discogs_rate_limit"] = discogs_rate

        # LLM config
        llm = config_dict.setdefault("llm", {})
        if not isinstance(llm, dict):
            llm = {}
            config_dict["llm"] = llm

        if llm_enabled := os.getenv(f"{env_prefix}LLM_ENABLED"):
            llm["enabled"] = llm_enabled.lower() in ("true", "1", "yes")
        if llm_provider := os.getenv(f"{env_prefix}LLM_PROVIDER"):
            llm["provider"] = llm_provider
        if llm_model := os.getenv(f"{env_prefix}LLM_MODEL_ID"):
            llm["model_id"] = llm_model
        if llm_api_key_env := os.getenv(f"{env_prefix}LLM_API_KEY_ENV"):
            llm["api_key_env"] = llm_api_key_env
        if llm_ollama_url := os.getenv(f"{env_prefix}LLM_OLLAMA_BASE_URL"):
            llm["ollama_base_url"] = llm_ollama_url
        if llm_timeout := os.getenv(f"{env_prefix}LLM_TIMEOUT_S"):
            llm["timeout_s"] = llm_timeout
        if llm_max_tokens := os.getenv(f"{env_prefix}LLM_MAX_TOKENS"):
            llm["max_tokens"] = llm_max_tokens
        if llm_auto_accept := os.getenv(f"{env_prefix}LLM_AUTO_ACCEPT_THRESHOLD"):
            llm["auto_accept_threshold"] = llm_auto_accept
        if llm_review := os.getenv(f"{env_prefix}LLM_REVIEW_THRESHOLD"):
            llm["review_threshold"] = llm_review

        return config_dict


## Tests


def test_config_defaults():
    config = Config()
    assert config.offline_mode is False
    assert config.http_cache.enabled is True
    assert config.http_cache.ttl_seconds == 86400
    assert config.http_cache.directory == Path(".cache/http")
    assert config.database.music_graph_path == Path("musicgraph.sqlite")
    assert config.database.charts_path == Path("charts.sqlite")


def test_config_from_dict():
    config = Config.model_validate(
        {
            "offline_mode": True,
            "http_cache": {"ttl_seconds": 3600, "directory": "/tmp/cache"},
        }
    )
    assert config.offline_mode is True
    assert config.http_cache.ttl_seconds == 3600
    assert config.http_cache.directory == Path("/tmp/cache")


def test_config_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    monkeypatch.setenv(  # pyright: ignore[reportUnknownMemberType]
        "CHART_BINDER_OFFLINE_MODE", "true"
    )
    monkeypatch.setenv(  # pyright: ignore[reportUnknownMemberType]
        "CHART_BINDER_HTTP_CACHE_TTL_SECONDS", "7200"
    )
    monkeypatch.setenv(  # pyright: ignore[reportUnknownMemberType]
        "CHART_BINDER_HTTP_CACHE_DIRECTORY", "/custom/cache"
    )

    config = Config.load()
    assert config.offline_mode is True
    assert config.http_cache.ttl_seconds == 7200
    assert config.http_cache.directory == Path("/custom/cache")


def test_config_load_nonexistent_file():
    config = Config.load(Path("/nonexistent/config.toml"))
    assert config.offline_mode is False
    assert config.http_cache.ttl_seconds == 86400


def test_config_llm_defaults():
    """Test LLM configuration defaults."""
    config = Config()
    assert config.llm.enabled is False
    assert config.llm.provider == "ollama"
    assert config.llm.model_id == "llama3.2"
    assert config.llm.auto_accept_threshold == 0.85
    assert config.llm.review_threshold == 0.60


def test_config_llm_from_dict():
    """Test LLM configuration from dict."""
    config = Config.model_validate(
        {
            "llm": {
                "enabled": True,
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "auto_accept_threshold": 0.90,
            }
        }
    )
    assert config.llm.enabled is True
    assert config.llm.provider == "openai"
    assert config.llm.model_id == "gpt-4o-mini"
    assert config.llm.auto_accept_threshold == 0.90


def test_config_llm_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test LLM configuration from environment variables."""
    monkeypatch.setenv("CHART_BINDER_LLM_ENABLED", "true")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_PROVIDER", "openai")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_MODEL_ID", "gpt-4o")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()
    assert config.llm.enabled is True
    assert config.llm.provider == "openai"
    assert config.llm.model_id == "gpt-4o"
