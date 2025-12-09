"""
Backend abstraction layer for MusicBrainz data access.

Supports two modes:
- API mode: Uses MusicBrainz REST API (default)
- DB mode: Direct PostgreSQL queries (faster for batch operations)
"""

from __future__ import annotations

from chart_binder.backends.base import (
    BackendArtist,
    BackendRecording,
    BackendReleaseGroup,
    BackendWork,
    MusicBrainzBackend,
)
from chart_binder.backends.factory import BackendMode, get_backend

__all__ = [
    "MusicBrainzBackend",
    "BackendRecording",
    "BackendReleaseGroup",
    "BackendWork",
    "BackendArtist",
    "get_backend",
    "BackendMode",
]
