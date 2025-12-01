"""
Unified fetcher interface for live music metadata sources.

Coordinates AcoustID, MusicBrainz, Discogs, Spotify, and Wikidata clients
with fallback chains, cache-aware fetching, and entity hydration into musicgraph.sqlite.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from chart_binder.acoustid import AcoustIDClient
from chart_binder.discogs import DiscogsClient
from chart_binder.http_cache import HttpCache
from chart_binder.musicbrainz import MusicBrainzClient, extract_discogs_ids
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer
from chart_binder.spotify import SpotifyClient
from chart_binder.wikidata import WikidataClient

# Confidence scores for different search methods (0.0-1.0)
# Higher values indicate more reliable matching methods
CONFIDENCE_ISRC_MUSICBRAINZ = 0.95  # ISRC is unique and MusicBrainz is canonical
CONFIDENCE_ISRC_SPOTIFY = 0.90  # ISRC is unique but Spotify metadata less canonical
CONFIDENCE_LINKED_MB_DISCOGS = 0.90  # Cross-referenced between MB and Discogs
CONFIDENCE_BARCODE_DISCOGS = 0.85  # Barcode is reliable but may have variants
CONFIDENCE_TEXT_SEARCH = 0.70  # Text search is fuzzy and may have false positives
CONFIDENCE_TEXT_SEARCH_SPOTIFY = 0.68  # Spotify text search, middle ground
CONFIDENCE_TEXT_SEARCH_DISCOGS = 0.65  # Discogs text search less canonical than MB
# Note: AcoustID confidence comes directly from the API response

# Cross-source confidence boosts (applied when entity found in multiple sources)
CONFIDENCE_BOOST_MULTI_SOURCE = 0.10  # Boost when found in 2+ sources
CONFIDENCE_BOOST_THREE_SOURCES = 0.15  # Boost when found in 3+ sources

# Popularity-based confidence adjustments
CONFIDENCE_BOOST_VERY_POPULAR = 0.05  # Spotify popularity >= 70
CONFIDENCE_BOOST_POPULAR = 0.03  # Spotify popularity >= 50
CONFIDENCE_BOOST_MODERATE = 0.01  # Spotify popularity >= 30

# Date validation thresholds
DATE_MISMATCH_PENALTY = 0.10  # Penalty when dates differ by > 1 year
DATE_MATCH_BOOST = 0.05  # Boost when dates match closely (same year)

# Label/format validation
LABEL_MATCH_BOOST = 0.03  # Boost when labels match across sources

log = logging.getLogger(__name__)


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

        # Normalizer for storing normalized artist/title fields
        self.normalizer = Normalizer()

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
        # Check if recording already exists in database (avoids redundant API calls)
        existing = self.db.get_recording(mbid)
        if existing:
            log.debug(f"Recording {mbid} already in database, skipping fetch")
            return {"recording": existing}

        if self.config.mode == FetchMode.OFFLINE:
            # Not in database and in OFFLINE mode
            raise ValueError(
                f"Recording {mbid} not found in database (OFFLINE mode - no network access)"
            )

        # Fetch recording with all releases and release groups
        data = self.mb_client.get_recording_with_releases(mbid)

        # Extract and hydrate artist
        artist_mbid = None
        if "artist-credit" in data and data["artist-credit"]:
            first_artist = data["artist-credit"][0]
            if "artist" in first_artist:
                artist_mbid = first_artist["artist"].get("id")
                if artist_mbid:
                    # Fetch full artist data with URL relationships
                    params = {"inc": "area-rels+url-rels"}
                    artist_data = self.mb_client._request(f"artist/{artist_mbid}", params)

                    # Extract Wikidata QID from URL relationships
                    wikidata_qid = self._extract_wikidata_qid(artist_data)

                    # Extract begin area country
                    begin_area_country = None
                    if "begin-area" in artist_data and artist_data["begin-area"]:
                        if "country" in artist_data["begin-area"]:
                            begin_area_country = artist_data["begin-area"]["country"]
                        elif "iso-3166-1-codes" in artist_data["begin-area"]:
                            codes = artist_data["begin-area"]["iso-3166-1-codes"]
                            if codes:
                                begin_area_country = codes[0]

                    # Enrich with Wikidata country if available and MB doesn't have it
                    if wikidata_qid and not begin_area_country:
                        # Use Wikidata to find artist country if MB doesn't have it
                        try:
                            countries = self.wikidata_client.get_artist_countries(wikidata_qid)
                            if countries:
                                # Prefer P27 (citizenship), then P740 (formation), then P495 (origin)
                                for country in countries:
                                    if country.property_type == "P27":
                                        begin_area_country = country.country_code
                                        break
                                if not begin_area_country:
                                    for country in countries:
                                        if country.property_type == "P740":
                                            begin_area_country = country.country_code
                                            break
                                if not begin_area_country and countries:
                                    begin_area_country = countries[0].country_code
                        except Exception:
                            # Wikidata lookup can fail, skip silently
                            pass

                    artist_name = artist_data.get("name", "")
                    artist_norm_result = self.normalizer.normalize_artist(artist_name)

                    self.db.upsert_artist(
                        mbid=artist_data["id"],
                        name=artist_name,
                        sort_name=artist_data.get("sort-name"),
                        begin_area_country=begin_area_country,
                        wikidata_qid=wikidata_qid,
                        disambiguation=artist_data.get("disambiguation"),
                        name_normalized=artist_norm_result.normalized,
                    )

        # Extract ISRCs
        isrcs = [isrc["isrc"] for isrc in data.get("isrc-list", [])]

        # Hydrate recording
        title = data.get("title", "")
        title_norm_result = self.normalizer.normalize_title(title)

        self.db.upsert_recording(
            mbid=data["id"],
            title=title,
            artist_mbid=artist_mbid,
            length_ms=data.get("length"),
            isrcs_json=json.dumps(isrcs) if isrcs else None,
            disambiguation=data.get("disambiguation"),
            title_normalized=title_norm_result.normalized,
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

    def fetch_discogs_release(self, discogs_release_id: str) -> dict[str, Any]:
        """
        Fetch Discogs release by ID and hydrate into database.

        For Discogs-only data (no MB ID), uses synthetic IDs:
        - Release: discogs-release-{id}
        - Master: discogs-master-{id}
        - Artist: discogs-artist-{name_normalized}

        Args:
            discogs_release_id: Discogs release ID (as string)

        Returns:
            Dict with release data
        """
        # Fetch full release data from Discogs (convert ID to int)
        release = self.discogs_client.get_release(int(discogs_release_id))

        # Create synthetic IDs for Discogs-only entities
        synthetic_release_id = f"discogs-release-{release.id}"
        synthetic_master_id = f"discogs-master-{release.master_id}" if release.master_id else None

        # Normalize artist name for synthetic artist ID
        artist_name = release.artist or "Unknown"
        artist_normalized = artist_name.lower().replace(" ", "-").replace("&", "and")
        synthetic_artist_id = f"discogs-artist-{artist_normalized}"

        # Hydrate artist (Discogs-only)
        artist_norm_result = self.normalizer.normalize_artist(artist_name)
        self.db.upsert_artist(
            mbid=synthetic_artist_id,
            name=artist_name,
            sort_name=artist_name,
            name_normalized=artist_norm_result.normalized,
        )

        # Hydrate master (release group equivalent)
        if synthetic_master_id:
            self.db.upsert_release_group(
                mbid=synthetic_master_id,
                title=release.title,
                artist_mbid=synthetic_artist_id,
                first_release_date=str(release.year) if release.year else None,
                discogs_master_id=str(release.master_id) if release.master_id else None,
            )

        # Hydrate release
        # Note: Discogs has lists for labels/formats, convert to single value for now
        label = release.labels[0] if release.labels else None
        format_str = release.formats[0] if release.formats else None

        self.db.upsert_release(
            mbid=synthetic_release_id,
            title=release.title,
            release_group_mbid=synthetic_master_id,
            artist_mbid=synthetic_artist_id,
            date=str(release.year) if release.year else None,
            country=release.country,
            label=label,
            format=format_str,
            barcode=release.barcode,
            discogs_release_id=str(release.id),
        )

        # Create synthetic recording ID (Discogs doesn't have recording concept)
        # Use release ID as recording ID since we don't have track-level data
        synthetic_recording_id = f"discogs-recording-{release.id}"

        title = release.title
        title_norm_result = self.normalizer.normalize_title(title)
        self.db.upsert_recording(
            mbid=synthetic_recording_id,
            title=title,
            artist_mbid=synthetic_artist_id,
            title_normalized=title_norm_result.normalized,
        )

        # Link recording to release
        self.db.upsert_recording_release(synthetic_recording_id, synthetic_release_id)

        return {
            "release": release,
            "synthetic_ids": {
                "recording": synthetic_recording_id,
                "release": synthetic_release_id,
                "release_group": synthetic_master_id,
                "artist": synthetic_artist_id,
            },
        }

    def fetch_spotify_track(self, spotify_track_id: str) -> dict[str, Any]:
        """
        Fetch Spotify track by ID and hydrate into database.

        For Spotify-only data (no MB ID), uses synthetic IDs:
        - Track: spotify-track-{id}
        - Album: spotify-album-{id}
        - Artist: spotify-artist-{name_normalized}

        Args:
            spotify_track_id: Spotify track ID

        Returns:
            Dict with track data
        """
        if not self.spotify_client:
            raise ValueError("Spotify client not available (missing credentials)")

        # Fetch full track data from Spotify
        track = self.spotify_client.get_track(spotify_track_id)

        # Create synthetic IDs for Spotify-only entities
        synthetic_track_id = f"spotify-track-{track.id}"
        synthetic_album_id = f"spotify-album-{track.album_id}" if track.album_id else None

        # Normalize artist name for synthetic artist ID
        artist_name = track.artist_name or "Unknown"
        artist_normalized = artist_name.lower().replace(" ", "-").replace("&", "and")
        synthetic_artist_id = f"spotify-artist-{artist_normalized}"

        # Hydrate artist (Spotify-only)
        artist_norm_result = self.normalizer.normalize_artist(artist_name)
        self.db.upsert_artist(
            mbid=synthetic_artist_id,
            name=artist_name,
            sort_name=artist_name,
            name_normalized=artist_norm_result.normalized,
        )

        # Hydrate album (release group equivalent)
        if synthetic_album_id and track.album_name:
            self.db.upsert_release_group(
                mbid=synthetic_album_id,
                title=track.album_name,
                artist_mbid=synthetic_artist_id,
                spotify_album_id=track.album_id,
            )

        # Hydrate release (same as album in Spotify's model)
        if synthetic_album_id and track.album_name:
            self.db.upsert_release(
                mbid=synthetic_album_id,
                title=track.album_name,
                release_group_mbid=synthetic_album_id,
                artist_mbid=synthetic_artist_id,
            )

        # Hydrate recording (track)
        title = track.name
        title_norm_result = self.normalizer.normalize_title(title)
        self.db.upsert_recording(
            mbid=synthetic_track_id,
            title=title,
            artist_mbid=synthetic_artist_id,
            length_ms=track.duration_ms,
            isrcs_json=json.dumps([track.isrc]) if track.isrc else None,
            title_normalized=title_norm_result.normalized,
        )

        # Link recording to release
        if synthetic_album_id:
            self.db.upsert_recording_release(synthetic_track_id, synthetic_album_id)

        return {
            "track": track,
            "synthetic_ids": {
                "recording": synthetic_track_id,
                "release": synthetic_album_id,
                "release_group": synthetic_album_id,
                "artist": synthetic_artist_id,
            },
        }

    def discover_siblings_via_work(
        self, recording_mbid: str, artist_mbid: str | None = None
    ) -> list[str]:
        """
        Discover sibling recordings via the Work entity.

        This provides DETERMINISTIC discovery:
        1. Get the work linked to this recording
        2. Browse ALL recordings linked to that work
        3. Filter to same artist (to exclude covers)

        Unlike search (which returns different subsets), browse is deterministic.

        Note: Work entity has limitations:
        - Covers link to same work (filtered by artist_mbid)
        - Medleys may link to multiple works
        - Live versions link to same work (may or may not be desired)
        - Not all recordings have work relationships

        Args:
            recording_mbid: A recording MBID to use as starting point
            artist_mbid: If provided, filter to recordings by this artist only

        Returns:
            List of sibling recording MBIDs by the same artist
        """
        if self.config.mode == FetchMode.OFFLINE:
            return [recording_mbid]  # Can't discover in offline mode

        try:
            # Get work relationship from the recording
            data = self.mb_client.get_recording_with_work(recording_mbid)

            # Extract artist MBID if not provided
            if not artist_mbid and "artist-credit" in data:
                for credit in data.get("artist-credit", []):
                    if isinstance(credit, dict) and "artist" in credit:
                        artist_mbid = credit["artist"].get("id")
                        break

            # Find work MBID from relations
            work_mbid = None
            relations = data.get("relations", [])
            for rel in relations:
                if rel.get("type") == "performance" and "work" in rel:
                    work_mbid = rel["work"].get("id")
                    break

            if not work_mbid:
                # No work linked - can't discover siblings
                return [recording_mbid]

            # Browse ALL recordings of this work (deterministic!)
            sibling_recordings = self.mb_client.browse_all_recordings_by_work(
                work_mbid,
                max_recordings=200,  # Safety limit
            )

            # Filter by artist to exclude covers
            sibling_mbids = []
            for rec in sibling_recordings:
                rec_id = rec.get("id")
                if not rec_id:
                    continue

                # Check if this recording is by the same artist
                if artist_mbid:
                    rec_artist_mbid = None
                    for credit in rec.get("artist-credit", []):
                        if isinstance(credit, dict) and "artist" in credit:
                            rec_artist_mbid = credit["artist"].get("id")
                            break
                    if rec_artist_mbid != artist_mbid:
                        continue  # Skip - different artist (likely a cover)

                sibling_mbids.append(rec_id)

            # Ensure the original is included
            if recording_mbid not in sibling_mbids:
                sibling_mbids.insert(0, recording_mbid)

            return sibling_mbids

        except Exception as e:
            import logging

            logging.warning(f"Work-based discovery failed for {recording_mbid}: {e}")
            return [recording_mbid]

    def hydrate_recordings_via_work(
        self, seed_recording_mbid: str, artist_mbid: str | None = None, max_hydrate: int = 20
    ) -> list[str]:
        """
        Discover and hydrate sibling recordings via Work entity.

        This is the main entry point for deterministic discovery:
        1. Use seed recording to find the Work
        2. Browse all recordings of the Work (same artist only)
        3. Hydrate top N recordings

        Args:
            seed_recording_mbid: Starting recording MBID
            artist_mbid: Filter to this artist only (excludes covers)
            max_hydrate: Maximum recordings to hydrate (default 20)

        Returns:
            List of hydrated recording MBIDs
        """
        # Discover all siblings by same artist
        sibling_mbids = self.discover_siblings_via_work(seed_recording_mbid, artist_mbid)

        # Hydrate up to max_hydrate
        hydrated = []
        for mbid in sibling_mbids[:max_hydrate]:
            try:
                self.fetch_recording(mbid)
                hydrated.append(mbid)
            except Exception as e:
                import logging

                logging.debug(f"Failed to hydrate sibling recording {mbid}: {e}")

        return hydrated

    def _extract_wikidata_qid(self, entity_data: dict[str, Any]) -> str | None:
        """
        Extract Wikidata QID from MusicBrainz entity URL relationships.

        Args:
            entity_data: MusicBrainz entity data with relations

        Returns:
            Wikidata QID (e.g., "Q1299") or None
        """
        import re

        relations = entity_data.get("relations", [])
        for relation in relations:
            if relation.get("type") == "wikidata":
                url_resource = relation.get("url", {}).get("resource", "")

                # Match wikidata.org/wiki/Q12345
                match = re.search(r"wikidata\.org/wiki/(Q\d+)", url_resource)
                if match:
                    return match.group(1)

        return None

    def _apply_popularity_boost(self, result: dict[str, Any], track_data: Any) -> None:
        """
        Apply confidence boost based on Spotify popularity score.

        High popularity indicates widespread recognition and validation,
        which can help disambiguate between originals and covers.

        Args:
            result: Result dict to modify
            track_data: SpotifyTrack object with popularity field
        """
        popularity = getattr(track_data, "popularity", None)
        if popularity is None:
            return

        if popularity >= 70:
            result["confidence"] = min(
                0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_VERY_POPULAR
            )
            result["popularity_tier"] = "very_popular"
        elif popularity >= 50:
            result["confidence"] = min(
                0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_POPULAR
            )
            result["popularity_tier"] = "popular"
        elif popularity >= 30:
            result["confidence"] = min(
                0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_MODERATE
            )
            result["popularity_tier"] = "moderate"

        result["popularity"] = popularity

    def _extract_year(self, date_str: str | None) -> int | None:
        """Extract year from various date formats."""
        if not date_str:
            return None

        # Handle formats: YYYY, YYYY-MM, YYYY-MM-DD
        parts = str(date_str).split("-")
        if parts and parts[0].isdigit():
            return int(parts[0])

        return None

    def _apply_date_validation(self, results: list[dict[str, Any]]) -> None:
        """
        Cross-reference release dates across sources.

        When dates match closely (same year), boost confidence.
        When dates differ significantly (>1 year), penalize confidence.

        Args:
            results: List of results to validate
        """
        from collections import defaultdict

        # Group by normalized title+artist to compare dates across sources
        def normalize_key(result: dict[str, Any]) -> str:
            title = result.get("title", "").lower().strip()
            artist = result.get("artist_name", "").lower().strip()
            return f"{artist}|{title}"

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            key = normalize_key(result)
            if key and key != "|":
                groups[key].append(result)

        # For each group, extract dates and validate
        for group_results in groups.values():
            if len(group_results) < 2:
                continue

            # Extract years from various sources
            years = []
            for result in group_results:
                year = result.get("year") or result.get("release_date")
                if year:
                    extracted = self._extract_year(str(year))
                    if extracted:
                        years.append(extracted)

            if not years:
                continue

            # Check for consensus or conflict
            min_year = min(years)
            max_year = max(years)

            if max_year - min_year <= 1:
                # Dates match closely - boost confidence
                for result in group_results:
                    result["confidence"] = min(
                        0.99, result.get("confidence", 0.0) + DATE_MATCH_BOOST
                    )
                    result["date_validated"] = True
            elif max_year - min_year > 1:
                # Significant date mismatch - penalize confidence
                for result in group_results:
                    result["confidence"] = max(
                        0.0, result.get("confidence", 0.0) - DATE_MISMATCH_PENALTY
                    )
                    result["date_conflict"] = True
                    result["date_range"] = f"{min_year}-{max_year}"

    def _apply_label_validation(self, results: list[dict[str, Any]]) -> None:
        """
        Compare label names across sources for consistency.

        Matching labels boost confidence. Uses fuzzy matching to handle
        label name variations.

        Args:
            results: List of results to validate
        """
        from collections import defaultdict

        def normalize_key(result: dict[str, Any]) -> str:
            title = result.get("title", "").lower().strip()
            artist = result.get("artist_name", "").lower().strip()
            return f"{artist}|{title}"

        def normalize_label(label: str | None) -> str:
            """Normalize label name for comparison."""
            if not label:
                return ""

            # Remove common suffixes and normalize spacing
            normalized = label.lower().strip()
            for suffix in [" records", " music", " entertainment", " inc.", " ltd."]:
                normalized = normalized.replace(suffix, "")
            return normalized.strip()

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            key = normalize_key(result)
            if key and key != "|":
                groups[key].append(result)

        # For each group, compare labels
        for group_results in groups.values():
            if len(group_results) < 2:
                continue

            # Extract normalized labels from all results
            labels_by_source = {}
            for result in group_results:
                label = result.get("label")
                if label:
                    source = result.get("source", "").split("_")[0]
                    labels_by_source[source] = normalize_label(label)

            if len(labels_by_source) < 2:
                continue

            # Check if labels match across sources
            unique_labels = set(labels_by_source.values())
            if len(unique_labels) == 1:
                # All labels match - boost confidence
                for result in group_results:
                    if result.get("label"):
                        result["confidence"] = min(
                            0.99, result.get("confidence", 0.0) + LABEL_MATCH_BOOST
                        )
                        result["label_validated"] = True

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
            # Use larger limit to find original recordings (can be deep in results)
            mb_results = self.mb_client.search_recordings(artist=artist, title=title, limit=100)
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
                            "artist_name": result.get("artist", "")
                            if isinstance(result.get("artist"), str)
                            else "",
                            "source": "discogs_search",
                            "confidence": CONFIDENCE_TEXT_SEARCH_DISCOGS,
                        }
                        if result.get("master_id"):
                            result_data["discogs_master_id"] = str(result["master_id"])
                        if result.get("year"):
                            result_data["year"] = result["year"]
                        if (
                            result.get("label")
                            and isinstance(result.get("label"), list)
                            and result["label"]
                        ):
                            result_data["label"] = result["label"][0]  # Take first label
                        elif result.get("label") and isinstance(result.get("label"), str):
                            result_data["label"] = result["label"]
                        results.append(result_data)

            # Also try Spotify title + artist search
            if self.spotify_client:
                try:
                    spotify_results = self.spotify_client.search_tracks(
                        artist=artist, track=title, limit=10
                    )
                    for track in spotify_results[:10]:
                        result_data = {
                            "spotify_track_id": track.id,
                            "title": track.name,
                            "artist_name": track.artist_name,
                            "source": "spotify_search",
                            "confidence": CONFIDENCE_TEXT_SEARCH_SPOTIFY,
                        }
                        if track.isrc:
                            result_data["isrc"] = track.isrc
                        if track.album_id:
                            result_data["spotify_album_id"] = track.album_id

                        # Apply popularity-weighted confidence boost
                        self._apply_popularity_boost(result_data, track)

                        results.append(result_data)

                        # ISRC cross-linking: If Spotify track has ISRC, search MB for it
                        if track.isrc and not any(
                            r.get("isrc") == track.isrc
                            and r.get("source", "").startswith("musicbrainz")
                            for r in results
                        ):
                            # Search MusicBrainz by this ISRC to create cross-link
                            try:
                                mb_isrc_results = self.mb_client.search_recordings(
                                    isrc=track.isrc, limit=1
                                )
                                if mb_isrc_results:
                                    mb_rec = mb_isrc_results[0]
                                    results.append(
                                        {
                                            "recording_mbid": mb_rec.mbid,
                                            "title": mb_rec.title,
                                            "artist_name": mb_rec.artist_name,
                                            "source": "musicbrainz_isrc_from_spotify",
                                            "confidence": CONFIDENCE_ISRC_MUSICBRAINZ,
                                            "isrc": track.isrc,
                                            "cross_linked": True,
                                        }
                                    )
                            except Exception:
                                # ISRC lookup can fail, skip silently
                                pass
                except Exception:
                    # Spotify search can fail due to missing credentials, skip silently
                    pass

        # Apply enhanced cross-source intelligence validations
        self._apply_date_validation(results)
        self._apply_label_validation(results)

        # Apply cross-source confidence boosts
        # Group results by normalized title+artist to identify multi-source entities
        from collections import defaultdict

        def normalize_key(result: dict[str, Any]) -> str:
            """Create normalized key for grouping similar results."""
            title = result.get("title", "").lower().strip()
            artist = result.get("artist_name", "").lower().strip()
            return f"{artist}|{title}"

        # Group results by normalized key
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            key = normalize_key(result)
            if key and key != "|":  # Skip empty keys
                groups[key].append(result)

        # Count unique sources per group and apply boosts
        for _key, group_results in groups.items():
            sources = {
                r.get("source", "").split("_")[0] for r in group_results
            }  # e.g., "musicbrainz_search" → "musicbrainz"
            source_count = len(sources)

            if source_count >= 3:
                # Found in 3+ sources (MB, Discogs, Spotify) - strong validation
                for result in group_results:
                    result["confidence"] = min(
                        0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_THREE_SOURCES
                    )
                    result["multi_source"] = True
                    result["source_count"] = source_count
            elif source_count >= 2:
                # Found in 2+ sources - moderate validation
                for result in group_results:
                    result["confidence"] = min(
                        0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_MULTI_SOURCE
                    )
                    result["multi_source"] = True
                    result["source_count"] = source_count

        # Sort by confidence (after boosts applied)
        results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

        return results

    async def asearch_recordings(
        self,
        isrc: str | None = None,
        title: str | None = None,
        artist: str | None = None,
        fingerprint: str | None = None,
        duration_sec: int | None = None,
        barcode: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Async version of search_recordings with parallel API calls.

        Runs independent searches in parallel using asyncio.gather() for
        significant performance improvements on I/O-bound operations.

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
        import asyncio

        results: list[dict[str, Any]] = []

        # Gather all search tasks that can run in parallel
        search_tasks = []

        # ISRC searches (MB is async, Spotify not yet for POC)
        if isrc:
            search_tasks.append(self._asearch_mb_isrc(isrc))
            # TODO: Add async Spotify search when client is converted

        # Title + artist searches (run MB in parallel)
        if title and artist:
            search_tasks.append(self._asearch_mb_text(artist, title))
            # TODO: Add async Discogs/Spotify when converted

        # Execute all searches in parallel
        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result_batch in search_results:
                if isinstance(result_batch, Exception):
                    log.warning(f"Search task failed: {result_batch}")
                    continue
                if result_batch:
                    results.extend(result_batch)

        # Non-async fallbacks for POC (TODO: convert these to async)
        # Fingerprint (AcoustID - not yet async)
        if fingerprint and duration_sec and self.acoustid_client:
            try:
                acoustid_results = self.acoustid_client.lookup(fingerprint, duration_sec)
                for result in acoustid_results:
                    results.append(
                        {
                            "recording_mbid": result.recording_mbid,
                            "source": "acoustid",
                            "confidence": result.confidence,
                        }
                    )
            except Exception as e:
                log.warning(f"AcoustID search failed: {e}")

        # Barcode (Discogs - not yet async)
        if barcode and self.discogs_client:
            try:
                discogs_releases = self.discogs_client.search_by_barcode(barcode, limit=5)
                for release in discogs_releases:
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
            except Exception as e:
                log.warning(f"Discogs barcode search failed: {e}")

        # Apply enhanced cross-source intelligence validations
        self._apply_date_validation(results)
        self._apply_label_validation(results)

        # Apply cross-source confidence boosts
        from collections import defaultdict

        def normalize_key(result: dict[str, Any]) -> str:
            """Create normalized key for grouping similar results."""
            title_str = result.get("title", "").lower().strip()
            artist_str = result.get("artist_name", "").lower().strip()
            return f"{artist_str}|{title_str}"

        # Group results by normalized key
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            key = normalize_key(result)
            if key and key != "|":
                groups[key].append(result)

        # Count unique sources per group and apply boosts
        for _key, group_results in groups.items():
            sources = {r.get("source", "").split("_")[0] for r in group_results}
            source_count = len(sources)

            if source_count >= 3:
                for result in group_results:
                    result["confidence"] = min(
                        0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_THREE_SOURCES
                    )
                    result["multi_source"] = True
                    result["source_count"] = source_count
            elif source_count >= 2:
                for result in group_results:
                    result["confidence"] = min(
                        0.99, result.get("confidence", 0.0) + CONFIDENCE_BOOST_MULTI_SOURCE
                    )
                    result["multi_source"] = True
                    result["source_count"] = source_count

        # Sort by confidence
        results.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

        return results

    async def _asearch_mb_isrc(self, isrc: str) -> list[dict[str, Any]]:
        """Helper: async ISRC search on MusicBrainz."""
        results = []
        try:
            mb_results = await self.mb_client.search_recordings(isrc=isrc)
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
        except Exception as e:
            log.warning(f"MB ISRC search failed: {e}")
        return results

    async def _asearch_mb_text(self, artist: str, title: str) -> list[dict[str, Any]]:
        """Helper: async text search on MusicBrainz."""
        results = []
        try:
            mb_results = await self.mb_client.search_recordings(
                artist=artist, title=title, limit=100
            )
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
        except Exception as e:
            log.warning(f"MB text search failed: {e}")
        return results

    async def close_async(self) -> None:
        """Close all clients (async version)."""
        await self.mb_client.close()
        if self.acoustid_client:
            self.acoustid_client.close()
        self.discogs_client.close()
        if self.spotify_client:
            self.spotify_client.close()
        self.wikidata_client.close()

    def close(self) -> None:
        """Close all clients (sync version - for backwards compatibility)."""
        import asyncio

        # Run async close in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use get_event_loop().run_until_complete() in running loop
                # Create task to close later
                asyncio.create_task(self.close_async())
            else:
                loop.run_until_complete(self.close_async())
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(self.close_async())

    async def __aenter__(self) -> UnifiedFetcher:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close_async()

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
