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


class Config(BaseModel):
    """
    Main configuration for chart-binder.

    Loads from TOML file with optional environment variable overrides.
    """

    http_cache: HttpCacheConfig = Field(default_factory=HttpCacheConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    offline_mode: bool = Field(default=False)

    @classmethod
    def load(cls, config_path: Path | None = None) -> Config:
        """
        Load configuration from TOML file with environment variable overrides.

        Environment variables take precedence and follow the pattern:
        CHART_BINDER_<SECTION>_<KEY> (e.g., CHART_BINDER_HTTP_CACHE_TTL_SECONDS)
        """
        config_dict: dict[str, object] = {}

        if config_path and config_path.exists():
            config_dict = tomllib.loads(config_path.read_text())

        config = cls.model_validate(config_dict)
        return cls._apply_env_overrides(config)

    @classmethod
    def _apply_env_overrides(cls, config: Config) -> Config:
        """Apply environment variable overrides to config."""
        env_prefix = "CHART_BINDER_"

        if offline := os.getenv(f"{env_prefix}OFFLINE_MODE"):
            config.offline_mode = offline.lower() in ("true", "1", "yes")

        if cache_dir := os.getenv(f"{env_prefix}HTTP_CACHE_DIRECTORY"):
            config.http_cache.directory = Path(cache_dir)

        if cache_ttl := os.getenv(f"{env_prefix}HTTP_CACHE_TTL_SECONDS"):
            config.http_cache.ttl_seconds = int(cache_ttl)

        if cache_enabled := os.getenv(f"{env_prefix}HTTP_CACHE_ENABLED"):
            config.http_cache.enabled = cache_enabled.lower() in ("true", "1", "yes")

        if db_music_path := os.getenv(f"{env_prefix}DATABASE_MUSIC_GRAPH_PATH"):
            config.database.music_graph_path = Path(db_music_path)

        if db_charts_path := os.getenv(f"{env_prefix}DATABASE_CHARTS_PATH"):
            config.database.charts_path = Path(db_charts_path)

        return config


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
