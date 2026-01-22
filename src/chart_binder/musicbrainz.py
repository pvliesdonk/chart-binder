"""
MusicBrainz API client for music metadata lookups.

Supports recording, release-group, and release lookups by MBID or search query
with rate limiting (1 req/sec default) and URL relationships parsing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


def extract_discogs_ids(entity_data: dict[str, Any]) -> tuple[str | None, str | None]:
    """
    Extract Discogs master and release IDs from MusicBrainz URL relationships.

    Args:
        entity_data: MusicBrainz entity dict with URL relationships

    Returns:
        Tuple of (discogs_master_id, discogs_release_id)
    """
    import re

    discogs_master_id = None
    discogs_release_id = None

    relations = entity_data.get("relations", [])
    for relation in relations:
        if relation.get("type") == "discogs":
            url_resource = relation.get("url", {}).get("resource", "")

            # Extract master ID from URLs like https://www.discogs.com/master/12345
            master_match = re.search(r"discogs\.com/master/(\d+)", url_resource)
            if master_match:
                discogs_master_id = master_match.group(1)

            # Extract release ID from URLs like https://www.discogs.com/release/12345
            release_match = re.search(r"discogs\.com/release/(\d+)", url_resource)
            if release_match:
                discogs_release_id = release_match.group(1)

    return discogs_master_id, discogs_release_id


@dataclass
class MusicBrainzRecording:
    """MusicBrainz recording entity."""

    mbid: str
    title: str
    artist_mbid: str | None = None
    artist_name: str | None = None
    length_ms: int | None = None
    isrcs: list[str] = field(default_factory=list)
    disambiguation: str | None = None


@dataclass
class MusicBrainzReleaseGroup:
    """MusicBrainz release group entity."""

    mbid: str
    title: str
    artist_mbid: str | None = None
    artist_name: str | None = None
    type: str | None = None
    first_release_date: str | None = None
    secondary_types: list[str] = field(default_factory=list)
    disambiguation: str | None = None


@dataclass
class MusicBrainzRelease:
    """MusicBrainz release entity."""

    mbid: str
    title: str
    release_group_mbid: str | None = None
    artist_mbid: str | None = None
    artist_name: str | None = None
    date: str | None = None
    country: str | None = None
    label: str | None = None
    barcode: str | None = None
    disambiguation: str | None = None


@dataclass
class MusicBrainzArtist:
    """MusicBrainz artist entity."""

    mbid: str
    name: str
    sort_name: str | None = None
    begin_area_country: str | None = None
    disambiguation: str | None = None


class MusicBrainzClient:
    """
    MusicBrainz API client for metadata lookups.

    Provides recording, release-group, and release lookups by MBID or search
    with rate limiting and URL relationships parsing.
    """

    BASE_URL = "https://musicbrainz.org/ws/2"
    USER_AGENT = "chart-binder/0.1.0 ( https://github.com/pvliesdonk/chart-binder )"

    def __init__(
        self,
        cache: HttpCache | None = None,
        rate_limit_per_sec: float = 1.0,
    ):
        """
        Initialize MusicBrainz client.

        Args:
            cache: Optional HTTP cache for responses
            rate_limit_per_sec: Max requests per second (default 1.0 per ToS)
        """
        self.cache = cache
        self.rate_limit_per_sec = rate_limit_per_sec
        self._last_request_time = 0.0
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        )
        self._rate_limit_lock = asyncio.Lock()

    async def _rate_limit(self) -> None:
        """Enforce rate limiting (1 req/sec by default per MusicBrainz ToS)."""
        if self.rate_limit_per_sec <= 0:
            return

        async with self._rate_limit_lock:
            min_interval = 1.0 / self.rate_limit_per_sec
            elapsed = time.time() - self._last_request_time

            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            self._last_request_time = time.time()

    async def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        """Make rate-limited request to MusicBrainz API."""
        await self._rate_limit()

        # Ensure JSON format
        params["fmt"] = "json"

        url = f"{self.BASE_URL}/{endpoint}"
        cache_key = f"{url}?{self._make_cache_key(params)}"

        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached.json()

        # Make live request
        response = await self._client.get(url, params=params)
        response.raise_for_status()

        # Cache response
        if self.cache:
            self.cache.put(cache_key, response)

        return response.json()

    def _make_cache_key(self, params: dict[str, str]) -> str:
        """Generate stable cache key from params."""
        sorted_items = sorted(params.items())
        return "&".join(f"{k}={v}" for k, v in sorted_items)

    async def get_recording(
        self,
        mbid: str,
        include_isrcs: bool = True,
        include_releases: bool = False,
    ) -> MusicBrainzRecording:
        """
        Get recording by MBID.

        Args:
            mbid: MusicBrainz recording ID
            include_isrcs: Whether to include ISRC codes
            include_releases: Whether to include releases (and their release groups)

        Returns:
            MusicBrainzRecording object
        """
        inc = ["artists"]
        if include_isrcs:
            inc.append("isrcs")
        if include_releases:
            inc.append("releases")
            inc.append("release-groups")

        params = {"inc": "+".join(inc)}
        data = await self._request(f"recording/{mbid}", params)

        # Extract artist info
        artist_mbid = None
        artist_name = None
        if "artist-credit" in data and data["artist-credit"]:
            first_artist = data["artist-credit"][0]
            if "artist" in first_artist:
                artist_mbid = first_artist["artist"].get("id")
                artist_name = first_artist["artist"].get("name")

        # Extract ISRCs (API returns simple list of strings)
        isrcs = data.get("isrcs", [])

        return MusicBrainzRecording(
            mbid=data["id"],
            title=data.get("title", ""),
            artist_mbid=artist_mbid,
            artist_name=artist_name,
            length_ms=data.get("length"),
            isrcs=isrcs,
            disambiguation=data.get("disambiguation"),
        )

    async def get_recording_with_releases(self, mbid: str) -> dict[str, Any]:
        """
        Get recording with all releases and release groups.

        Returns raw API response dict for full hydration.
        Includes URL relationships to extract Discogs IDs.
        """
        params = {"inc": "artists+isrcs+releases+release-groups+url-rels"}
        return await self._request(f"recording/{mbid}", params)

    async def get_recording_with_work(self, mbid: str) -> dict[str, Any]:
        """
        Get recording with work relationships.

        Returns raw API response including work-rels which links
        the recording to its abstract composition (work).
        """
        params = {"inc": "artists+work-rels"}
        return await self._request(f"recording/{mbid}", params)

    async def browse_recordings_by_work(
        self, work_mbid: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """
        Browse all recordings linked to a work.

        This is deterministic - returns ALL recordings of a composition,
        unlike search which returns relevance-sorted subsets.

        Args:
            work_mbid: MusicBrainz work ID
            limit: Max results per page (max 100)
            offset: Pagination offset

        Returns:
            List of recording dicts with basic info
        """
        params = {
            "work": work_mbid,
            "limit": str(min(limit, 100)),
            "offset": str(offset),
        }
        data = await self._request("recording", params)
        return data.get("recordings", [])

    async def browse_all_recordings_by_work(
        self, work_mbid: str, max_recordings: int = 500
    ) -> list[dict[str, Any]]:
        """
        Browse ALL recordings linked to a work, handling pagination.

        Args:
            work_mbid: MusicBrainz work ID
            max_recordings: Safety limit to prevent runaway queries

        Returns:
            List of all recording dicts
        """
        all_recordings: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while len(all_recordings) < max_recordings:
            batch = await self.browse_recordings_by_work(work_mbid, limit=limit, offset=offset)
            if not batch:
                break
            all_recordings.extend(batch)
            if len(batch) < limit:
                break  # Last page
            offset += limit

        return all_recordings[:max_recordings]

    async def get_release_group(self, mbid: str) -> MusicBrainzReleaseGroup:
        """
        Get release group by MBID.

        Args:
            mbid: MusicBrainz release group ID

        Returns:
            MusicBrainzReleaseGroup object
        """
        params = {"inc": "artists"}
        data = await self._request(f"release-group/{mbid}", params)

        # Extract artist info
        artist_mbid = None
        artist_name = None
        if "artist-credit" in data and data["artist-credit"]:
            first_artist = data["artist-credit"][0]
            if "artist" in first_artist:
                artist_mbid = first_artist["artist"].get("id")
                artist_name = first_artist["artist"].get("name")

        return MusicBrainzReleaseGroup(
            mbid=data["id"],
            title=data.get("title", ""),
            artist_mbid=artist_mbid,
            artist_name=artist_name,
            type=data.get("primary-type"),
            first_release_date=data.get("first-release-date"),
            secondary_types=data.get("secondary-types", []),
            disambiguation=data.get("disambiguation"),
        )

    async def get_release(self, mbid: str) -> MusicBrainzRelease:
        """
        Get release by MBID.

        Args:
            mbid: MusicBrainz release ID

        Returns:
            MusicBrainzRelease object
        """
        params = {"inc": "artists+labels+release-groups"}
        data = await self._request(f"release/{mbid}", params)

        # Extract artist info
        artist_mbid = None
        artist_name = None
        if "artist-credit" in data and data["artist-credit"]:
            first_artist = data["artist-credit"][0]
            if "artist" in first_artist:
                artist_mbid = first_artist["artist"].get("id")
                artist_name = first_artist["artist"].get("name")

        # Extract release group
        rg_mbid = None
        if "release-group" in data:
            rg_mbid = data["release-group"].get("id")

        # Extract label info
        label = None
        if "label-info" in data and data["label-info"]:
            first_label = data["label-info"][0]
            if "label" in first_label and first_label["label"]:
                label = first_label["label"].get("name")

        return MusicBrainzRelease(
            mbid=data["id"],
            title=data.get("title", ""),
            release_group_mbid=rg_mbid,
            artist_mbid=artist_mbid,
            artist_name=artist_name,
            date=data.get("date"),
            country=data.get("country"),
            label=label,
            barcode=data.get("barcode"),
            disambiguation=data.get("disambiguation"),
        )

    async def get_artist(self, mbid: str) -> MusicBrainzArtist:
        """
        Get artist by MBID.

        Args:
            mbid: MusicBrainz artist ID

        Returns:
            MusicBrainzArtist object
        """
        params = {"inc": "area-rels"}
        data = await self._request(f"artist/{mbid}", params)

        # Extract begin area country
        begin_area_country = None
        if "begin-area" in data and data["begin-area"]:
            # Try to get country code
            if "country" in data["begin-area"]:
                begin_area_country = data["begin-area"]["country"]
            elif "iso-3166-1-codes" in data["begin-area"]:
                codes = data["begin-area"]["iso-3166-1-codes"]
                if codes:
                    begin_area_country = codes[0]

        return MusicBrainzArtist(
            mbid=data["id"],
            name=data.get("name", ""),
            sort_name=data.get("sort-name"),
            begin_area_country=begin_area_country,
            disambiguation=data.get("disambiguation"),
        )

    async def search_recordings(
        self,
        query: str | None = None,
        isrc: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        limit: int = 25,
    ) -> list[MusicBrainzRecording]:
        """
        Search recordings by query or structured fields.

        Args:
            query: Lucene-style query string
            isrc: ISRC code to search
            artist: Artist name to filter
            title: Recording title to filter
            limit: Max results (default 25)

        Returns:
            List of MusicBrainzRecording objects
        """
        # Build query string
        if not query:
            query_parts = []
            if isrc:
                query_parts.append(f'isrc:"{isrc}"')
            if artist:
                query_parts.append(f'artist:"{artist}"')
            if title:
                query_parts.append(f'recording:"{title}"')
            query = " AND ".join(query_parts) if query_parts else "*"

        params = {"query": query, "limit": str(limit)}
        data = await self._request("recording", params)

        results = []
        for rec in data.get("recordings", []):
            # Extract artist
            artist_mbid = None
            artist_name = None
            if "artist-credit" in rec and rec["artist-credit"]:
                first_artist = rec["artist-credit"][0]
                if "artist" in first_artist:
                    artist_mbid = first_artist["artist"].get("id")
                    artist_name = first_artist["artist"].get("name")

            # Extract ISRCs (API returns simple list of strings)
            isrcs = rec.get("isrcs", [])

            results.append(
                MusicBrainzRecording(
                    mbid=rec["id"],
                    title=rec.get("title", ""),
                    artist_mbid=artist_mbid,
                    artist_name=artist_name,
                    length_ms=rec.get("length"),
                    isrcs=isrcs,
                    disambiguation=rec.get("disambiguation"),
                )
            )

        return results

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> MusicBrainzClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class SyncMusicBrainzClient:
    """Sync wrapper for MusicBrainzClient's async methods.

    Many parts of the codebase expect sync methods, but MusicBrainzClient
    uses async for rate limiting and HTTP requests. This wrapper runs async
    methods in an event loop for sync access.

    Usage:
        async_client = MusicBrainzClient(cache=cache)
        sync_client = SyncMusicBrainzClient(async_client)
        results = sync_client.search_recordings(title="Yesterday", artist="Beatles")
    """

    def __init__(self, client: MusicBrainzClient):
        self._client = client

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists
            return asyncio.run(coro)

    def get_recording(
        self,
        mbid: str,
        include_isrcs: bool = True,
        include_releases: bool = False,
    ) -> MusicBrainzRecording:
        """Sync wrapper for get_recording."""
        return self._run_async(
            self._client.get_recording(mbid, include_isrcs, include_releases)
        )

    def get_recording_with_releases(self, mbid: str) -> dict[str, Any]:
        """Sync wrapper for get_recording_with_releases."""
        return self._run_async(self._client.get_recording_with_releases(mbid))

    def get_recording_with_work(self, mbid: str) -> dict[str, Any]:
        """Sync wrapper for get_recording_with_work."""
        return self._run_async(self._client.get_recording_with_work(mbid))

    def browse_recordings_by_work(
        self, work_mbid: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Sync wrapper for browse_recordings_by_work."""
        return self._run_async(
            self._client.browse_recordings_by_work(work_mbid, limit, offset)
        )

    def browse_all_recordings_by_work(
        self, work_mbid: str, max_recordings: int = 500
    ) -> list[dict[str, Any]]:
        """Sync wrapper for browse_all_recordings_by_work."""
        return self._run_async(
            self._client.browse_all_recordings_by_work(work_mbid, max_recordings)
        )

    def get_release_group(self, mbid: str) -> MusicBrainzReleaseGroup:
        """Sync wrapper for get_release_group."""
        return self._run_async(self._client.get_release_group(mbid))

    def get_release(self, mbid: str) -> MusicBrainzRelease:
        """Sync wrapper for get_release."""
        return self._run_async(self._client.get_release(mbid))

    def get_artist(self, mbid: str) -> MusicBrainzArtist:
        """Sync wrapper for get_artist."""
        return self._run_async(self._client.get_artist(mbid))

    def search_recordings(
        self,
        query: str | None = None,
        isrc: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        limit: int = 25,
    ) -> list[MusicBrainzRecording]:
        """Sync wrapper for search_recordings."""
        return self._run_async(
            self._client.search_recordings(query, isrc, artist, title, limit)
        )

    def search_artists(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        """Search for artists by name (sync).

        Note: MusicBrainzClient doesn't have a dedicated search_artists method,
        so we implement it using the generic search endpoint.
        """
        return self._run_async(self._search_artists_async(query, limit))

    async def _search_artists_async(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search for artists by name."""
        params = {"query": f'artist:"{query}"', "limit": str(limit), "fmt": "json"}
        await self._client._rate_limit()

        url = f"{self._client.BASE_URL}/artist"

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            cached = self._client.cache.get(cache_key)
            if cached:
                return cached.json().get("artists", [])

        response = await self._client._client.get(url, params=params)
        response.raise_for_status()

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            self._client.cache.put(cache_key, response)

        return response.json().get("artists", [])

    def search_release_groups(
        self, title: str, artist: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Search for release groups by title and optional artist (sync).

        Note: MusicBrainzClient doesn't have a dedicated search_release_groups method,
        so we implement it using the generic search endpoint.
        """
        return self._run_async(self._search_release_groups_async(title, artist, limit))

    async def _search_release_groups_async(
        self, title: str, artist: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Search for release groups by title and optional artist."""
        query_parts = [f'releasegroup:"{title}"']
        if artist:
            query_parts.append(f'artist:"{artist}"')
        query = " AND ".join(query_parts)

        params = {"query": query, "limit": str(limit), "fmt": "json"}
        await self._client._rate_limit()

        url = f"{self._client.BASE_URL}/release-group"

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            cached = self._client.cache.get(cache_key)
            if cached:
                return cached.json().get("release-groups", [])

        response = await self._client._client.get(url, params=params)
        response.raise_for_status()

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            self._client.cache.put(cache_key, response)

        return response.json().get("release-groups", [])

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        """Sync wrapper for _request (low-level API access)."""
        return self._run_async(self._client._request(endpoint, params))

    def close(self) -> None:
        """Close the underlying async client."""
        self._run_async(self._client.close())


## Tests


def test_musicbrainz_recording_dataclass():
    """Test MusicBrainzRecording dataclass."""
    rec = MusicBrainzRecording(
        mbid="test-mbid",
        title="Test Song",
        artist_mbid="artist-mbid",
        artist_name="Test Artist",
        length_ms=180000,
        isrcs=["USTEST1234567"],
    )
    assert rec.mbid == "test-mbid"
    assert rec.title == "Test Song"
    assert len(rec.isrcs) == 1


def test_musicbrainz_cache_key():
    """Test cache key generation."""
    client = MusicBrainzClient()
    params = {"inc": "artists", "fmt": "json"}
    key = client._make_cache_key(params)
    assert "fmt=json" in key
    assert "inc=artists" in key
