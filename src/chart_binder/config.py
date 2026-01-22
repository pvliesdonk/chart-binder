from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class BackendMode(StrEnum):
    """MusicBrainz backend mode."""

    API = "api"
    DB = "db"


class BackendConfig(BaseModel):
    """MusicBrainz backend configuration.

    Supports two modes:
    - api: Uses MusicBrainz REST API (default, rate-limited)
    - db: Direct PostgreSQL access (faster for batch operations)
    """

    mode: BackendMode = Field(default=BackendMode.API)
    db_url: str | None = Field(default=None)
    db_echo: bool = Field(default=False)


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


class SearxNGConfig(BaseModel):
    """SearxNG search configuration."""

    url: str = Field(default="http://localhost:8080")
    timeout_s: float = Field(default=10.0, ge=1.0)
    enabled: bool = Field(default=False)


class LLMConfig(BaseModel):
    """LLM adjudication configuration (Epic 13).

    Uses LangChain agent with native tool calling and structured output.
    Supports multiple providers (Ollama, OpenAI, Anthropic) with provider-agnostic
    model factory and Pydantic schema validation.
    """

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
    max_tokens: int = Field(default=4096, ge=1)  # Increased for tool calling workflows
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    # Confidence thresholds
    auto_accept_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # Prompt versioning for A/B testing
    prompt_template_version: str = Field(default="v1")

    # Review queue database path
    review_queue_path: Path = Field(default=Path("review_queue.sqlite"))

    # SearxNG integration
    searxng: SearxNGConfig = Field(default_factory=SearxNGConfig)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="WARNING")  # DEBUG, INFO, WARNING, ERROR
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    hash_paths: bool = Field(default=False)


class Config(BaseModel):
    """
    Main configuration for chart-binder.

    Loads from TOML file with optional environment variable overrides.
    """

    backend: BackendConfig = Field(default_factory=BackendConfig)
    http_cache: HttpCacheConfig = Field(default_factory=HttpCacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    live_sources: LiveSourcesConfig = Field(default_factory=LiveSourcesConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
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
        if cls._should_load_dotenv():
            load_dotenv(override=False)
        cls._apply_env_aliases()

        config_dict: dict[str, object] = {}

        if config_path and config_path.exists():
            config_dict = tomllib.loads(config_path.read_text())

        config_dict = cls._merge_env_overrides(config_dict)
        return cls.model_validate(config_dict)

    @classmethod
    def _apply_env_aliases(cls) -> None:
        def set_if_missing(target: str, value: str | None) -> None:
            if value is None:
                return
            if os.getenv(target) is None:
                os.environ[target] = value

        aliases: dict[str, tuple[str, ...]] = {
            "CHART_BINDER_LLM_ENABLED": ("LLM_ENABLED",),
            "CHART_BINDER_LLM_PROVIDER": ("LLM_PROVIDER",),
            "CHART_BINDER_LLM_MODEL_ID": ("LLM_MODEL_ID",),
            "CHART_BINDER_LLM_TEMPERATURE": ("LLM_TEMPERATURE",),
            "CHART_BINDER_LLM_TIMEOUT_S": ("LLM_TIMEOUT_S",),
            "CHART_BINDER_LLM_MAX_TOKENS": ("LLM_MAX_TOKENS",),
            "CHART_BINDER_LLM_AUTO_ACCEPT_THRESHOLD": ("LLM_AUTO_ACCEPT_THRESHOLD",),
            "CHART_BINDER_LLM_REVIEW_THRESHOLD": ("LLM_REVIEW_THRESHOLD",),
            "CHART_BINDER_LLM_PROMPT_TEMPLATE_VERSION": ("LLM_PROMPT_TEMPLATE_VERSION",),
            "CHART_BINDER_LLM_REVIEW_QUEUE_PATH": ("LLM_REVIEW_QUEUE_PATH",),
            "CHART_BINDER_LLM_OLLAMA_BASE_URL": ("OLLAMA_HOST", "OLLAMA_BASE_URL"),
            "CHART_BINDER_LLM_OPENAI_BASE_URL": ("OPENAI_BASE_URL", "OPENAI_API_BASE"),
            "CHART_BINDER_SEARXNG_URL": ("SEARXNG_URL", "SEARX_URL"),
            "CHART_BINDER_SEARXNG_TIMEOUT_S": ("SEARXNG_TIMEOUT_S", "SEARX_TIMEOUT_S"),
            "CHART_BINDER_SEARXNG_ENABLED": ("SEARXNG_ENABLED", "SEARX_ENABLED"),
        }

        for target, sources in aliases.items():
            for source in sources:
                if value := os.getenv(source):
                    set_if_missing(target, value)
                    break

        if os.getenv("CHART_BINDER_LLM_API_KEY_ENV") is None:
            for source in ("OPENAI_API_KEY", "OPEN_AI_KEY", "OPEN_AI_API_KEY"):
                if os.getenv(source):
                    os.environ["CHART_BINDER_LLM_API_KEY_ENV"] = source
                    break

    @classmethod
    def _should_load_dotenv(cls) -> bool:
        if os.getenv("PYTEST_CURRENT_TEST") is not None:
            return False
        if dotenv_enabled := os.getenv("CHART_BINDER_DOTENV"):
            return dotenv_enabled.lower() in ("true", "1", "yes")
        return True

    @classmethod
    def _merge_env_overrides(cls, config_dict: dict[str, object]) -> dict[str, object]:
        """
        Merge environment variable overrides into config dictionary.

        Returns a new dictionary with env vars applied, ready for Pydantic validation.
        """
        env_prefix = "CHART_BINDER_"

        if offline := os.getenv(f"{env_prefix}OFFLINE_MODE"):
            config_dict["offline_mode"] = offline.lower() in ("true", "1", "yes")

        # Backend config
        backend = config_dict.setdefault("backend", {})
        if not isinstance(backend, dict):
            backend = {}
            config_dict["backend"] = backend

        if backend_mode := os.getenv(f"{env_prefix}BACKEND_MODE"):
            backend["mode"] = backend_mode
        if backend_db_url := os.getenv(f"{env_prefix}BACKEND_DB_URL"):
            backend["db_url"] = backend_db_url
        if backend_db_echo := os.getenv(f"{env_prefix}BACKEND_DB_ECHO"):
            backend["db_echo"] = backend_db_echo.lower() in ("true", "1", "yes")

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

        # Cache TTLs
        if cache_ttl_mb := os.getenv(f"{env_prefix}LIVE_SOURCES_CACHE_TTL_MUSICBRAINZ"):
            live_sources["cache_ttl_musicbrainz"] = cache_ttl_mb
        if cache_ttl_discogs := os.getenv(f"{env_prefix}LIVE_SOURCES_CACHE_TTL_DISCOGS"):
            live_sources["cache_ttl_discogs"] = cache_ttl_discogs
        if cache_ttl_spotify := os.getenv(f"{env_prefix}LIVE_SOURCES_CACHE_TTL_SPOTIFY"):
            live_sources["cache_ttl_spotify"] = cache_ttl_spotify
        if cache_ttl_wikidata := os.getenv(f"{env_prefix}LIVE_SOURCES_CACHE_TTL_WIKIDATA"):
            live_sources["cache_ttl_wikidata"] = cache_ttl_wikidata
        if cache_ttl_acoustid := os.getenv(f"{env_prefix}LIVE_SOURCES_CACHE_TTL_ACOUSTID"):
            live_sources["cache_ttl_acoustid"] = cache_ttl_acoustid

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
        if llm_openai_url := os.getenv(f"{env_prefix}LLM_OPENAI_BASE_URL"):
            llm["openai_base_url"] = llm_openai_url
        if llm_timeout := os.getenv(f"{env_prefix}LLM_TIMEOUT_S"):
            llm["timeout_s"] = llm_timeout
        if llm_max_tokens := os.getenv(f"{env_prefix}LLM_MAX_TOKENS"):
            llm["max_tokens"] = llm_max_tokens
        if llm_temperature := os.getenv(f"{env_prefix}LLM_TEMPERATURE"):
            llm["temperature"] = llm_temperature
        if llm_auto_accept := os.getenv(f"{env_prefix}LLM_AUTO_ACCEPT_THRESHOLD"):
            llm["auto_accept_threshold"] = llm_auto_accept
        if llm_review := os.getenv(f"{env_prefix}LLM_REVIEW_THRESHOLD"):
            llm["review_threshold"] = llm_review
        if llm_prompt_version := os.getenv(f"{env_prefix}LLM_PROMPT_TEMPLATE_VERSION"):
            llm["prompt_template_version"] = llm_prompt_version
        if llm_review_queue := os.getenv(f"{env_prefix}LLM_REVIEW_QUEUE_PATH"):
            llm["review_queue_path"] = llm_review_queue

        # SearxNG config (nested under llm)
        searxng = llm.setdefault("searxng", {})
        if not isinstance(searxng, dict):
            searxng = {}
            llm["searxng"] = searxng

        if searxng_url := os.getenv(f"{env_prefix}SEARXNG_URL"):
            searxng["url"] = searxng_url
        if searxng_timeout := os.getenv(f"{env_prefix}SEARXNG_TIMEOUT_S"):
            searxng["timeout_s"] = searxng_timeout
        if searxng_enabled := os.getenv(f"{env_prefix}SEARXNG_ENABLED"):
            searxng["enabled"] = searxng_enabled.lower() in ("true", "1", "yes")

        # Logging config
        logging_config = config_dict.setdefault("logging", {})
        if not isinstance(logging_config, dict):
            logging_config = {}
            config_dict["logging"] = logging_config

        if log_level := os.getenv(f"{env_prefix}LOGGING_LEVEL"):
            logging_config["level"] = log_level
        if log_format := os.getenv(f"{env_prefix}LOGGING_FORMAT"):
            logging_config["format"] = log_format
        if log_hash_paths := os.getenv(f"{env_prefix}LOGGING_HASH_PATHS"):
            logging_config["hash_paths"] = log_hash_paths.lower() in ("true", "1", "yes")

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


def test_config_searxng_defaults():
    """Test SearxNG configuration defaults."""
    config = Config()
    assert config.llm.searxng.enabled is False
    assert config.llm.searxng.url == "http://localhost:8080"
    assert config.llm.searxng.timeout_s == 10.0


def test_config_searxng_from_dict():
    """Test SearxNG configuration from dict."""
    config = Config.model_validate(
        {
            "llm": {
                "searxng": {
                    "enabled": True,
                    "url": "http://searxng.example.com",
                    "timeout_s": 15.0,
                }
            }
        }
    )
    assert config.llm.searxng.enabled is True
    assert config.llm.searxng.url == "http://searxng.example.com"
    assert config.llm.searxng.timeout_s == 15.0


def test_config_searxng_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test SearxNG configuration from environment variables."""
    monkeypatch.setenv("CHART_BINDER_SEARXNG_ENABLED", "true")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_SEARXNG_URL", "http://search.local")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_SEARXNG_TIMEOUT_S", "20.0")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()
    assert config.llm.searxng.enabled is True
    assert config.llm.searxng.url == "http://search.local"
    assert config.llm.searxng.timeout_s == 20.0


def test_config_all_llm_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test all LLM configuration options from environment variables."""
    monkeypatch.setenv("CHART_BINDER_LLM_ENABLED", "true")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_PROVIDER", "openai")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_MODEL_ID", "gpt-4o-mini")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_OPENAI_BASE_URL", "https://api.custom.com/v1")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_TEMPERATURE", "0.7")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_PROMPT_TEMPLATE_VERSION", "v2")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LLM_REVIEW_QUEUE_PATH", "/tmp/reviews.db")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()
    assert config.llm.enabled is True
    assert config.llm.provider == "openai"
    assert config.llm.model_id == "gpt-4o-mini"
    assert config.llm.openai_base_url == "https://api.custom.com/v1"
    assert config.llm.temperature == 0.7
    assert config.llm.prompt_template_version == "v2"
    assert config.llm.review_queue_path == Path("/tmp/reviews.db")


def test_config_live_sources_cache_ttl_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test live sources cache TTL configuration from environment variables."""
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_MUSICBRAINZ", "7200")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_DISCOGS", "172800")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_SPOTIFY", "14400")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_WIKIDATA", "1209600")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_LIVE_SOURCES_CACHE_TTL_ACOUSTID", "43200")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()
    assert config.live_sources.cache_ttl_musicbrainz == 7200
    assert config.live_sources.cache_ttl_discogs == 172800
    assert config.live_sources.cache_ttl_spotify == 14400
    assert config.live_sources.cache_ttl_wikidata == 1209600
    assert config.live_sources.cache_ttl_acoustid == 43200


def test_config_backend_defaults():
    """Test backend configuration defaults."""
    config = Config()
    assert config.backend.mode == BackendMode.API
    assert config.backend.db_url is None
    assert config.backend.db_echo is False


def test_config_backend_mode_enum():
    """Test BackendMode enum values."""
    assert BackendMode.API == "api"
    assert BackendMode.DB == "db"


def test_config_backend_env_overrides(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test backend configuration from environment variables."""
    monkeypatch.setenv("CHART_BINDER_BACKEND_MODE", "db")  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv(
        "CHART_BINDER_BACKEND_DB_URL", "postgresql://user:pass@localhost/musicbrainz"
    )  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setenv("CHART_BINDER_BACKEND_DB_ECHO", "true")  # pyright: ignore[reportUnknownMemberType]

    config = Config.load()
    assert config.backend.mode == BackendMode.DB
    assert config.backend.db_url == "postgresql://user:pass@localhost/musicbrainz"
    assert config.backend.db_echo is True


def test_config_backend_db_echo_false_values(
    monkeypatch,  # pyright: ignore[reportMissingParameterType,reportUnknownParameterType]
):
    """Test db_echo env var with various false values."""
    # Test that only specific values are truthy
    monkeypatch.setenv("CHART_BINDER_BACKEND_DB_ECHO", "false")  # pyright: ignore[reportUnknownMemberType]
    config = Config.load()
    assert config.backend.db_echo is False

    monkeypatch.setenv("CHART_BINDER_BACKEND_DB_ECHO", "0")  # pyright: ignore[reportUnknownMemberType]
    config = Config.load()
    assert config.backend.db_echo is False

    monkeypatch.setenv("CHART_BINDER_BACKEND_DB_ECHO", "no")  # pyright: ignore[reportUnknownMemberType]
    config = Config.load()
    assert config.backend.db_echo is False
