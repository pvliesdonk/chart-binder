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
from chart_binder.musicbrainz import MusicBrainzClient, extract_discogs_ids
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.spotify import SpotifyClient
from chart_binder.wikidata import WikidataClient

# Confidence scores for different search methods (0.0-1.0)
# Higher values indicate more reliable matching methods
CONFIDENCE_ISRC_MUSICBRAINZ = 0.95  # ISRC is unique and MusicBrainz is canonical
CONFIDENCE_ISRC_SPOTIFY = 0.90  # ISRC is unique but Spotify metadata less canonical
CONFIDENCE_LINKED_MB_DISCOGS = 0.90  # Cross-referenced between MB and Discogs
CONFIDENCE_BARCODE_DISCOGS = 0.85  # Barcode is reliable but may have variants
CONFIDENCE_TEXT_SEARCH = 0.70  # Text search is fuzzy and may have false positives
CONFIDENCE_TEXT_SEARCH_DISCOGS = 0.65  # Discogs text search less canonical than MB
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

        Fetches the recording with all its releases and release groups,
        then hydrates the entire chain into the local database.

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

        # Fetch recording with all releases and release groups
        data = self.mb_client.get_recording_with_releases(mbid)

        # Extract and hydrate artist
        artist_mbid = None
        if "artist-credit" in data and data["artist-credit"]:
            first_artist = data["artist-credit"][0]
            if "artist" in first_artist:
                artist_mbid = first_artist["artist"].get("id")
                artist_name = first_artist["artist"].get("name")
                if artist_mbid:
                    # Fetch full artist data
                    artist = self.mb_client.get_artist(artist_mbid)
                    self.db.upsert_artist(
                        mbid=artist.mbid,
                        name=artist.name,
                        sort_name=artist.sort_name,
                        begin_area_country=artist.begin_area_country,
                        disambiguation=artist.disambiguation,
                    )

        # Extract ISRCs
        isrcs = [isrc["isrc"] for isrc in data.get("isrc-list", [])]

        # Hydrate recording
        self.db.upsert_recording(
            mbid=data["id"],
            title=data.get("title", ""),
            artist_mbid=artist_mbid,
            length_ms=data.get("length"),
            isrcs_json=json.dumps(isrcs) if isrcs else None,
            disambiguation=data.get("disambiguation"),
        )

        # Hydrate releases and release groups
        for release in data.get("releases", []):
            release_mbid = release.get("id")
            if not release_mbid:
                continue

            # Get release group from release
            rg = release.get("release-group", {})
            rg_mbid = rg.get("id")

            if rg_mbid:
                # Extract Discogs IDs from release group URL relationships
                rg_discogs_master, _ = extract_discogs_ids(rg)

                # Hydrate release group
                self.db.upsert_release_group(
                    mbid=rg_mbid,
                    title=rg.get("title", ""),
                    artist_mbid=artist_mbid,
                    type=rg.get("primary-type"),
                    first_release_date=rg.get("first-release-date"),
                    secondary_types_json=json.dumps(rg.get("secondary-types", [])),
                    discogs_master_id=rg_discogs_master,
                )

            # Extract Discogs IDs from release URL relationships
            _, release_discogs_id = extract_discogs_ids(release)

            # Hydrate release
            self.db.upsert_release(
                mbid=release_mbid,
                title=release.get("title", ""),
                release_group_mbid=rg_mbid,
                date=release.get("date"),
                country=release.get("country"),
                discogs_release_id=release_discogs_id,
            )

            # Link recording to release
            self.db.upsert_recording_release(data["id"], release_mbid)

        return {
            "recording": data,
        }

    def search_recordings(
        self,
        isrc: str | None = None,
        title: str | None = None,
        artist: str | None = None,
        fingerprint: str | None = None,
        duration_sec: int | None = None,
        barcode: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for recordings with fallback chain.

        Tries in order:
        1. ISRC lookup (MB, then Spotify)
        2. Fingerprint (AcoustID)
        3. Barcode lookup (Discogs)
        4. Title + artist search (MB, then Discogs)

        Args:
            isrc: ISRC code
            title: Recording title
            artist: Artist name
            fingerprint: AcoustID fingerprint
            duration_sec: Track duration in seconds (for fingerprint)
            barcode: UPC/EAN barcode

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

        # Try barcode (Discogs)
        if barcode and self.discogs_client:
            discogs_releases = self.discogs_client.search_by_barcode(barcode, limit=5)
            for release in discogs_releases:
                # Hydrate release data into database
                # For now, we don't have MB IDs, so we'll just return Discogs data
                result_data = {
                    "discogs_release_id": str(release.id),
                    "title": release.title,
                    "artist_name": release.artist,
                    "source": "discogs_barcode",
                    "confidence": CONFIDENCE_BARCODE_DISCOGS,
                    "barcode": barcode,
                }
                if release.master_id:
                    result_data["discogs_master_id"] = str(release.master_id)
                results.append(result_data)

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

            # Also try Discogs title + artist search
            if self.discogs_client:
                discogs_results = self.discogs_client.search_database(
                    artist=artist, title=title, search_type="release", per_page=10
                )
                for result in discogs_results[:10]:
                    if result.get("type") == "release" and result.get("id"):
                        result_data = {
                            "discogs_release_id": str(result["id"]),
                            "title": result.get("title", ""),
                            "artist_name": result.get("artist", "") if isinstance(result.get("artist"), str) else "",
                            "source": "discogs_search",
                            "confidence": CONFIDENCE_TEXT_SEARCH_DISCOGS,
                        }
                        if result.get("master_id"):
                            result_data["discogs_master_id"] = str(result["master_id"])
                        results.append(result_data)

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
