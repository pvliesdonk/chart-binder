"""
MusicBrainz API client for music metadata lookups.

Supports recording, release-group, and release lookups by MBID or search query
with rate limiting (1 req/sec default) and URL relationships parsing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


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
        self._client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": self.USER_AGENT},
        )

    def _rate_limit(self) -> None:
        """Enforce rate limiting (1 req/sec by default per MusicBrainz ToS)."""
        if self.rate_limit_per_sec <= 0:
            return

        min_interval = 1.0 / self.rate_limit_per_sec
        elapsed = time.time() - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        """Make rate-limited request to MusicBrainz API."""
        self._rate_limit()

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
        response = self._client.get(url, params=params)
        response.raise_for_status()

        # Cache response
        if self.cache:
            self.cache.put(cache_key, response)

        return response.json()

    def _make_cache_key(self, params: dict[str, str]) -> str:
        """Generate stable cache key from params."""
        sorted_items = sorted(params.items())
        return "&".join(f"{k}={v}" for k, v in sorted_items)

    def get_recording(
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
        data = self._request(f"recording/{mbid}", params)

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

    def get_recording_with_releases(self, mbid: str) -> dict[str, Any]:
        """
        Get recording with all releases and release groups.

        Returns raw API response dict for full hydration.
        """
        params = {"inc": "artists+isrcs+releases+release-groups"}
        return self._request(f"recording/{mbid}", params)

    def get_release_group(self, mbid: str) -> MusicBrainzReleaseGroup:
        """
        Get release group by MBID.

        Args:
            mbid: MusicBrainz release group ID

        Returns:
            MusicBrainzReleaseGroup object
        """
        params = {"inc": "artists"}
        data = self._request(f"release-group/{mbid}", params)

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

    def get_release(self, mbid: str) -> MusicBrainzRelease:
        """
        Get release by MBID.

        Args:
            mbid: MusicBrainz release ID

        Returns:
            MusicBrainzRelease object
        """
        params = {"inc": "artists+labels+release-groups"}
        data = self._request(f"release/{mbid}", params)

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

    def get_artist(self, mbid: str) -> MusicBrainzArtist:
        """
        Get artist by MBID.

        Args:
            mbid: MusicBrainz artist ID

        Returns:
            MusicBrainzArtist object
        """
        params = {"inc": "area-rels"}
        data = self._request(f"artist/{mbid}", params)

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

    def search_recordings(
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
        data = self._request("recording", params)

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

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> MusicBrainzClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


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
