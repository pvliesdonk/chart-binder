"""
Unified fetcher interface for live music metadata sources.

Coordinates AcoustID, MusicBrainz, Discogs, Spotify, and Wikidata clients
with fallback chains, cache-aware fetching, and entity hydration into musicgraph.sqlite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from chart_binder.acoustid import AcoustIDClient
from chart_binder.discogs import DiscogsClient
from chart_binder.http_cache import HttpCache
from chart_binder.musicbrainz import MusicBrainzClient
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.spotify import SpotifyClient
from chart_binder.wikidata import WikidataClient

# Confidence scores for different search methods (0.0-1.0)
# Higher values indicate more reliable matching methods
CONFIDENCE_ISRC_MUSICBRAINZ = 0.95  # ISRC is unique and MusicBrainz is canonical
CONFIDENCE_ISRC_SPOTIFY = 0.90  # ISRC is unique but Spotify metadata less canonical
CONFIDENCE_TEXT_SEARCH = 0.70  # Text search is fuzzy and may have false positives
# Note: AcoustID confidence comes directly from the API response


class FetchMode(Enum):
    """Fetch mode for controlling network access."""

    NORMAL = "normal"  # Use cache, fetch if needed
    OFFLINE = "offline"  # Cache only, never fetch
    REFRESH = "refresh"  # Always fetch, update cache
    FROZEN = "frozen"  # Use cache, error if miss


@dataclass
class FetcherConfig:
    """Configuration for unified fetcher."""

    cache_dir: Path
    db_path: Path
    mode: FetchMode = FetchMode.NORMAL

    # API credentials (optional, read from env if not provided)
    acoustid_api_key: str | None = None
    discogs_token: str | None = None
    spotify_client_id: str | None = None
    spotify_client_secret: str | None = None

    # Rate limits
    musicbrainz_rate_limit: float = 1.0  # req/sec
    acoustid_rate_limit: float = 3.0  # req/sec
    discogs_rate_limit: int = 25  # req/min (unauth)

    # Cache TTLs (seconds)
    cache_ttl_musicbrainz: int = 3600  # 1 hour
    cache_ttl_discogs: int = 86400  # 24 hours
    cache_ttl_spotify: int = 7200  # 2 hours
    cache_ttl_wikidata: int = 604800  # 7 days
    cache_ttl_acoustid: int = 86400  # 24 hours


class UnifiedFetcher:
    """
    Unified fetcher for music metadata from multiple sources.

    Coordinates:
    - MusicBrainz: Primary source for canonical music data
    - AcoustID: Fingerprint-based identification
    - Discogs: Additional release metadata
    - Spotify: Popularity and preview data
    - Wikidata: Artist country/origin data

    Provides:
    - fetch_recording(mbid): Get recording by MusicBrainz ID
    - search_recordings(isrc|title_artist): Search with fallback chain
    - Cache-aware with TTL/ETag respect
    - Offline/refresh/frozen modes
    - Entity hydration into musicgraph.sqlite
    """

    def __init__(self, config: FetcherConfig):
        """Initialize unified fetcher with all clients."""
        self.config = config
        self.db = MusicGraphDB(config.db_path)

        # Initialize caches per source
        self._caches = {
            "musicbrainz": HttpCache(
                config.cache_dir / "musicbrainz",
                ttl_seconds=config.cache_ttl_musicbrainz,
            ),
            "discogs": HttpCache(
                config.cache_dir / "discogs",
                ttl_seconds=config.cache_ttl_discogs,
            ),
            "spotify": HttpCache(
                config.cache_dir / "spotify",
                ttl_seconds=config.cache_ttl_spotify,
            ),
            "wikidata": HttpCache(
                config.cache_dir / "wikidata",
                ttl_seconds=config.cache_ttl_wikidata,
            ),
            "acoustid": HttpCache(
                config.cache_dir / "acoustid",
                ttl_seconds=config.cache_ttl_acoustid,
            ),
        }

        # Initialize clients
        self.mb_client = MusicBrainzClient(
            cache=self._caches["musicbrainz"],
            rate_limit_per_sec=config.musicbrainz_rate_limit,
        )

        # AcoustID (optional - requires API key)
        try:
            self.acoustid_client = AcoustIDClient(
                api_key=config.acoustid_api_key,
                cache=self._caches["acoustid"],
                rate_limit_per_sec=config.acoustid_rate_limit,
            )
        except ValueError:
            self.acoustid_client = None

        # Discogs (optional - works without token but lower rate limit)
        self.discogs_client = DiscogsClient(
            token=config.discogs_token,
            cache=self._caches["discogs"],
            rate_limit_per_min=config.discogs_rate_limit,
        )

        # Spotify (optional - requires client credentials)
        try:
            self.spotify_client = SpotifyClient(
                client_id=config.spotify_client_id,
                client_secret=config.spotify_client_secret,
                cache=self._caches["spotify"],
            )
        except ValueError:
            self.spotify_client = None

        # Wikidata (always available)
        self.wikidata_client = WikidataClient(
            cache=self._caches["wikidata"],
        )

    def fetch_recording(self, mbid: str) -> dict[str, Any]:
        """
        Fetch recording by MusicBrainz ID and hydrate into database.

        Args:
            mbid: MusicBrainz recording ID

        Returns:
            Dict with recording data and related entities

        Raises:
            ValueError: If recording not found in database when in OFFLINE mode
        """
        if self.config.mode == FetchMode.OFFLINE:
            # Return from database only, no network access
            recording_data = self.db.get_recording(mbid)
            if not recording_data:
                raise ValueError(
                    f"Recording {mbid} not found in database (OFFLINE mode - no network access)"
                )
            return {"recording": recording_data}

        # Fetch from MusicBrainz
        recording = self.mb_client.get_recording(mbid)

        # Hydrate artist if present
        if recording.artist_mbid:
            artist = self.mb_client.get_artist(recording.artist_mbid)
            self.db.upsert_artist(
                mbid=artist.mbid,
                name=artist.name,
                sort_name=artist.sort_name,
                begin_area_country=artist.begin_area_country,
                disambiguation=artist.disambiguation,
            )

            # Enrich with Wikidata if available
            # TODO: Link MusicBrainz to Wikidata QID
            # For now, skip Wikidata enrichment

        # Hydrate recording
        self.db.upsert_recording(
            mbid=recording.mbid,
            title=recording.title,
            artist_mbid=recording.artist_mbid,
            length_ms=recording.length_ms,
            isrcs_json=json.dumps(recording.isrcs) if recording.isrcs else None,
            disambiguation=recording.disambiguation,
        )

        return {
            "recording": recording,
        }

    def search_recordings(
        self,
        isrc: str | None = None,
        title: str | None = None,
        artist: str | None = None,
        fingerprint: str | None = None,
        duration_sec: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for recordings with fallback chain.

        Tries in order:
        1. ISRC lookup (MB, then Spotify)
        2. Fingerprint (AcoustID)
        3. Title + artist search (MB, then Spotify)

        Args:
            isrc: ISRC code
            title: Recording title
            artist: Artist name
            fingerprint: AcoustID fingerprint
            duration_sec: Track duration in seconds (for fingerprint)

        Returns:
            List of recording matches with confidence scores
        """
        results: list[dict[str, Any]] = []

        # Try ISRC first (most reliable)
        if isrc:
            mb_results = self.mb_client.search_recordings(isrc=isrc)
            for rec in mb_results:
                results.append(
                    {
                        "recording_mbid": rec.mbid,
                        "title": rec.title,
                        "artist_name": rec.artist_name,
                        "source": "musicbrainz_isrc",
                        "confidence": CONFIDENCE_ISRC_MUSICBRAINZ,
                    }
                )

            # Also try Spotify
            if self.spotify_client:
                sp_results = self.spotify_client.search_tracks(isrc=isrc)
                for track in sp_results:
                    results.append(
                        {
                            "spotify_id": track.id,
                            "title": track.name,
                            "artist_name": track.artist_name,
                            "isrc": track.isrc,
                            "source": "spotify_isrc",
                            "confidence": CONFIDENCE_ISRC_SPOTIFY,
                        }
                    )

        # Try fingerprint (if available)
        if fingerprint and duration_sec and self.acoustid_client:
            acoustid_results = self.acoustid_client.lookup(fingerprint, duration_sec)
            for result in acoustid_results:
                results.append(
                    {
                        "recording_mbid": result.recording_mbid,
                        "source": "acoustid",
                        "confidence": result.confidence,
                    }
                )

        # Try title + artist search
        if title and artist:
            mb_results = self.mb_client.search_recordings(artist=artist, title=title, limit=10)
            for rec in mb_results:
                results.append(
                    {
                        "recording_mbid": rec.mbid,
                        "title": rec.title,
                        "artist_name": rec.artist_name,
                        "source": "musicbrainz_search",
                        "confidence": CONFIDENCE_TEXT_SEARCH,
                    }
                )

        # Sort by confidence
        results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

        return results

    def close(self) -> None:
        """Close all clients."""
        self.mb_client.close()
        if self.acoustid_client:
            self.acoustid_client.close()
        self.discogs_client.close()
        if self.spotify_client:
            self.spotify_client.close()
        self.wikidata_client.close()

    def __enter__(self) -> UnifiedFetcher:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


## Tests


def test_fetcher_config():
    """Test FetcherConfig dataclass."""
    config = FetcherConfig(
        cache_dir=Path("/tmp/cache"),
        db_path=Path("/tmp/musicgraph.sqlite"),
        mode=FetchMode.OFFLINE,
    )
    assert config.mode == FetchMode.OFFLINE
    assert config.cache_dir == Path("/tmp/cache")


def test_fetch_mode_enum():
    """Test FetchMode enum."""
    assert FetchMode.NORMAL.value == "normal"
    assert FetchMode.OFFLINE.value == "offline"
    assert FetchMode.REFRESH.value == "refresh"
    assert FetchMode.FROZEN.value == "frozen"
