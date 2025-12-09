"""
Factory function for creating MusicBrainz backends.

Provides a unified interface for getting the appropriate backend
based on configuration.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from chart_binder.backends.base import MusicBrainzBackend

if TYPE_CHECKING:
    from chart_binder.http_cache import HttpCache


class BackendMode(StrEnum):
    """Backend mode selection."""

    API = "api"
    DB = "db"


def get_backend(
    mode: BackendMode | str = BackendMode.API,
    *,
    # API backend options
    cache: HttpCache | None = None,
    rate_limit_per_sec: float = 1.0,
    # DB backend options
    db_url: str | None = None,
    db_echo: bool = False,
) -> MusicBrainzBackend:
    """
    Get a MusicBrainz backend based on mode.

    Args:
        mode: Backend mode ("api" or "db")
        cache: HTTP cache for API backend
        rate_limit_per_sec: Rate limit for API backend
        db_url: PostgreSQL URL for DB backend
        db_echo: Log SQL queries for DB backend

    Returns:
        Configured MusicBrainzBackend instance

    Raises:
        ValueError: If DB mode selected but db_url not provided
    """
    mode_enum = BackendMode(mode) if isinstance(mode, str) else mode

    if mode_enum == BackendMode.API:
        from chart_binder.backends.api_backend import APIBackend

        return APIBackend(cache=cache, rate_limit_per_sec=rate_limit_per_sec)

    elif mode_enum == BackendMode.DB:
        if not db_url:
            raise ValueError("db_url is required for DB backend mode")

        from chart_binder.backends.db_backend import DBBackend

        return DBBackend(db_url=db_url, echo=db_echo)

    else:
        raise ValueError(f"Unknown backend mode: {mode}")


def get_backend_from_config(config: object) -> MusicBrainzBackend:
    """
    Get a MusicBrainz backend from a Config object.

    Args:
        config: Config object with backend settings

    Returns:
        Configured MusicBrainzBackend instance
    """
    # Import here to avoid circular imports
    from chart_binder.config import Config

    if not isinstance(config, Config):
        raise TypeError(f"Expected Config, got {type(config)}")

    # Get backend settings
    backend_config = config.backend

    if backend_config.mode == BackendMode.DB:
        if not backend_config.db_url:
            raise ValueError("backend.db_url is required when mode is 'db'")

        return get_backend(
            mode=BackendMode.DB,
            db_url=backend_config.db_url,
            db_echo=backend_config.db_echo,
        )
    else:
        # Default to API mode
        from chart_binder.http_cache import HttpCache

        cache: HttpCache | None = None
        if config.http_cache.enabled:
            cache = HttpCache(
                cache_dir=config.http_cache.directory,
                ttl_seconds=config.live_sources.cache_ttl_musicbrainz,
            )

        return get_backend(
            mode=BackendMode.API,
            cache=cache,
            rate_limit_per_sec=config.live_sources.musicbrainz_rate_limit,
        )


## Tests


def test_get_backend_api_mode():
    """Test API backend creation."""
    backend = get_backend(mode="api")
    from chart_binder.backends.api_backend import APIBackend

    assert isinstance(backend, APIBackend)


def test_get_backend_db_mode_requires_url():
    """Test DB backend requires URL."""
    try:
        get_backend(mode="db")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "db_url is required" in str(e)


def test_backend_mode_enum():
    """Test BackendMode enum values."""
    assert BackendMode.API == "api"
    assert BackendMode.DB == "db"
