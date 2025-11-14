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
