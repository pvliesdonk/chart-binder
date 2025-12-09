"""
API backend implementation using MusicBrainz REST API.

This backend wraps the existing MusicBrainzClient and provides the
standard MusicBrainzBackend interface for API-based access.
"""

from __future__ import annotations

import logging

from chart_binder.backends.base import (
    BackendArtist,
    BackendRecording,
    BackendReleaseGroup,
    BackendWork,
    MusicBrainzBackend,
)
from chart_binder.http_cache import HttpCache
from chart_binder.musicbrainz import MusicBrainzClient

log = logging.getLogger(__name__)


class APIBackend(MusicBrainzBackend):
    """
    MusicBrainz backend using REST API.

    Wraps the existing MusicBrainzClient to provide the standard
    MusicBrainzBackend interface. Rate-limited to 1 req/sec by default.
    """

    def __init__(
        self,
        cache: HttpCache | None = None,
        rate_limit_per_sec: float = 1.0,
    ):
        """
        Initialize API backend.

        Args:
            cache: Optional HTTP cache for responses
            rate_limit_per_sec: Max requests per second (default 1.0 per ToS)
        """
        self._client = MusicBrainzClient(
            cache=cache,
            rate_limit_per_sec=rate_limit_per_sec,
        )

    # --- Recording Operations ---

    async def get_recording(self, mbid: str) -> BackendRecording | None:
        """Get recording by MBID."""
        try:
            rec = await self._client.get_recording(mbid, include_isrcs=True)
            return BackendRecording(
                mbid=rec.mbid,
                title=rec.title,
                artist_mbid=rec.artist_mbid,
                artist_name=rec.artist_name,
                length_ms=rec.length_ms,
                isrcs=rec.isrcs,
                disambiguation=rec.disambiguation,
            )
        except Exception as e:
            log.warning(f"Failed to get recording {mbid}: {e}")
            return None

    async def search_recordings(
        self,
        artist: str,
        title: str,
        *,
        strict: bool = True,
        limit: int = 25,
    ) -> list[BackendRecording]:
        """Search recordings by artist and title."""
        # Build query based on strictness
        if strict:
            # Exact phrase matching
            query = f'artist:"{artist}" AND recording:"{title}"'
        else:
            # Fuzzy matching without quotes
            query = f"artist:{artist} AND recording:{title}"

        try:
            results = await self._client.search_recordings(query=query, limit=limit)
            return [
                BackendRecording(
                    mbid=rec.mbid,
                    title=rec.title,
                    artist_mbid=rec.artist_mbid,
                    artist_name=rec.artist_name,
                    length_ms=rec.length_ms,
                    isrcs=rec.isrcs,
                    disambiguation=rec.disambiguation,
                )
                for rec in results
            ]
        except Exception as e:
            log.warning(f"Recording search failed for '{artist}' - '{title}': {e}")
            return []

    async def search_recordings_by_isrc(self, isrc: str) -> list[BackendRecording]:
        """Search recordings by ISRC code."""
        try:
            results = await self._client.search_recordings(isrc=isrc)
            return [
                BackendRecording(
                    mbid=rec.mbid,
                    title=rec.title,
                    artist_mbid=rec.artist_mbid,
                    artist_name=rec.artist_name,
                    length_ms=rec.length_ms,
                    isrcs=rec.isrcs,
                    disambiguation=rec.disambiguation,
                )
                for rec in results
            ]
        except Exception as e:
            log.warning(f"ISRC search failed for {isrc}: {e}")
            return []

    # --- Work Operations ---

    async def get_work(self, mbid: str) -> BackendWork | None:
        """Get work by MBID."""
        try:
            params = {"inc": "iswcs"}
            data = await self._client._request(f"work/{mbid}", params)
            return BackendWork(
                mbid=data["id"],
                name=data.get("title", ""),
                disambiguation=data.get("disambiguation"),
                iswcs=[iswc for iswc in data.get("iswcs", [])],
            )
        except Exception as e:
            log.warning(f"Failed to get work {mbid}: {e}")
            return None

    async def get_work_for_recording(self, recording_mbid: str) -> BackendWork | None:
        """Get the work linked to a recording."""
        try:
            data = await self._client.get_recording_with_work(recording_mbid)

            # Find work in relations
            for relation in data.get("relations", []):
                if relation.get("type") == "performance" and "work" in relation:
                    work_data = relation["work"]
                    return BackendWork(
                        mbid=work_data["id"],
                        name=work_data.get("title", ""),
                        disambiguation=work_data.get("disambiguation"),
                    )

            return None
        except Exception as e:
            log.warning(f"Failed to get work for recording {recording_mbid}: {e}")
            return None

    async def get_recordings_for_work(
        self,
        work_mbid: str,
        *,
        artist_mbid: str | None = None,
        limit: int = 500,
    ) -> list[BackendRecording]:
        """Get all recordings linked to a work."""
        try:
            raw_recordings = await self._client.browse_all_recordings_by_work(
                work_mbid, max_recordings=limit
            )

            results: list[BackendRecording] = []
            for rec in raw_recordings:
                rec_id = rec.get("id")
                if not rec_id:
                    continue

                # Extract artist info
                rec_artist_mbid: str | None = None
                rec_artist_name: str | None = None
                for credit in rec.get("artist-credit", []):
                    if isinstance(credit, dict) and "artist" in credit:
                        rec_artist_mbid = credit["artist"].get("id")
                        rec_artist_name = credit["artist"].get("name")
                        break

                # Filter by artist if specified
                if artist_mbid and rec_artist_mbid != artist_mbid:
                    continue

                results.append(
                    BackendRecording(
                        mbid=rec_id,
                        title=rec.get("title", ""),
                        artist_mbid=rec_artist_mbid,
                        artist_name=rec_artist_name,
                        length_ms=rec.get("length"),
                        disambiguation=rec.get("disambiguation"),
                    )
                )

            return results
        except Exception as e:
            log.warning(f"Failed to get recordings for work {work_mbid}: {e}")
            return []

    # --- Release Group Operations ---

    async def get_release_group(self, mbid: str) -> BackendReleaseGroup | None:
        """Get release group by MBID."""
        try:
            rg = await self._client.get_release_group(mbid)
            return BackendReleaseGroup(
                mbid=rg.mbid,
                title=rg.title,
                artist_mbid=rg.artist_mbid,
                artist_name=rg.artist_name,
                primary_type=rg.type,
                secondary_types=rg.secondary_types,
                first_release_date=rg.first_release_date,
                disambiguation=rg.disambiguation,
            )
        except Exception as e:
            log.warning(f"Failed to get release group {mbid}: {e}")
            return None

    async def get_release_groups_for_recording(
        self,
        recording_mbid: str,
    ) -> list[BackendReleaseGroup]:
        """Get all release groups containing a recording."""
        try:
            # Get recording with releases and release groups
            data = await self._client.get_recording_with_releases(recording_mbid)

            seen_rg_ids: set[str] = set()
            results: list[BackendReleaseGroup] = []

            for release in data.get("releases", []):
                rg = release.get("release-group", {})
                rg_id = rg.get("id")
                if not rg_id or rg_id in seen_rg_ids:
                    continue
                seen_rg_ids.add(rg_id)

                # Extract artist from release or release group
                artist_mbid: str | None = None
                artist_name: str | None = None
                for credit in rg.get("artist-credit", []) or release.get("artist-credit", []):
                    if isinstance(credit, dict) and "artist" in credit:
                        artist_mbid = credit["artist"].get("id")
                        artist_name = credit["artist"].get("name")
                        break

                results.append(
                    BackendReleaseGroup(
                        mbid=rg_id,
                        title=rg.get("title", ""),
                        artist_mbid=artist_mbid,
                        artist_name=artist_name,
                        primary_type=rg.get("primary-type"),
                        secondary_types=rg.get("secondary-types", []),
                        first_release_date=rg.get("first-release-date"),
                        disambiguation=rg.get("disambiguation"),
                    )
                )

            return results
        except Exception as e:
            log.warning(f"Failed to get release groups for recording {recording_mbid}: {e}")
            return []

    async def get_release_groups_for_artist(
        self,
        artist_mbid: str,
        *,
        primary_type: str | None = None,
        limit: int = 100,
    ) -> list[BackendReleaseGroup]:
        """Get all release groups for an artist."""
        try:
            # Build query params for browse
            params: dict[str, str] = {
                "artist": artist_mbid,
                "limit": str(min(limit, 100)),
                "inc": "artist-credits",
            }
            if primary_type:
                params["type"] = primary_type

            data = await self._client._request("release-group", params)

            results: list[BackendReleaseGroup] = []
            for rg in data.get("release-groups", []):
                # Extract artist
                artist_name: str | None = None
                for credit in rg.get("artist-credit", []):
                    if isinstance(credit, dict) and "artist" in credit:
                        artist_name = credit["artist"].get("name")
                        break

                results.append(
                    BackendReleaseGroup(
                        mbid=rg["id"],
                        title=rg.get("title", ""),
                        artist_mbid=artist_mbid,
                        artist_name=artist_name,
                        primary_type=rg.get("primary-type"),
                        secondary_types=rg.get("secondary-types", []),
                        first_release_date=rg.get("first-release-date"),
                        disambiguation=rg.get("disambiguation"),
                    )
                )

            return results
        except Exception as e:
            log.warning(f"Failed to get release groups for artist {artist_mbid}: {e}")
            return []

    # --- Artist Operations ---

    async def get_artist(self, mbid: str) -> BackendArtist | None:
        """Get artist by MBID."""
        try:
            artist = await self._client.get_artist(mbid)
            return BackendArtist(
                mbid=artist.mbid,
                name=artist.name,
                sort_name=artist.sort_name,
                country=artist.begin_area_country,
                disambiguation=artist.disambiguation,
            )
        except Exception as e:
            log.warning(f"Failed to get artist {mbid}: {e}")
            return None

    async def search_artists(
        self,
        name: str,
        *,
        limit: int = 10,
    ) -> list[BackendArtist]:
        """Search artists by name."""
        try:
            params = {"query": f'artist:"{name}"', "limit": str(limit)}
            data = await self._client._request("artist", params)

            results: list[BackendArtist] = []
            for artist in data.get("artists", []):
                # Extract country from begin-area
                country: str | None = None
                if "begin-area" in artist and artist["begin-area"]:
                    if "country" in artist["begin-area"]:
                        country = artist["begin-area"]["country"]
                    elif "iso-3166-1-codes" in artist["begin-area"]:
                        codes = artist["begin-area"]["iso-3166-1-codes"]
                        if codes:
                            country = codes[0]

                results.append(
                    BackendArtist(
                        mbid=artist["id"],
                        name=artist.get("name", ""),
                        sort_name=artist.get("sort-name"),
                        country=country,
                        disambiguation=artist.get("disambiguation"),
                    )
                )

            return results
        except Exception as e:
            log.warning(f"Artist search failed for '{name}': {e}")
            return []

    # --- Lifecycle ---

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()


## Tests


def test_api_backend_creation():
    """Test APIBackend can be instantiated."""
    backend = APIBackend()
    assert backend._client is not None
