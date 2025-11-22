from __future__ import annotations

import os
import tomllib
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


class Config(BaseModel):
    """
    Main configuration for chart-binder.

    Loads from TOML file with optional environment variable overrides.
    """

    http_cache: HttpCacheConfig = Field(default_factory=HttpCacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    live_sources: LiveSourcesConfig = Field(default_factory=LiveSourcesConfig)
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
