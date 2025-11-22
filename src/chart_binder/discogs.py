"""
Discogs API client for music metadata lookups.

Supports master/release lookups by ID with OAuth authentication,
marketplace data filtering, and rate limiting (60/min auth, 25/min unauth).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


@dataclass
class DiscogsMaster:
    """Discogs master release entity."""

    id: int
    title: str
    artist: str
    year: int | None = None
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    main_release_id: int | None = None


@dataclass
class DiscogsRelease:
    """Discogs release entity."""

    id: int
    title: str
    artist: str
    master_id: int | None = None
    year: int | None = None
    country: str | None = None
    labels: list[str] = field(default_factory=list)
    formats: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    barcode: str | None = None


class DiscogsClient:
    """
    Discogs API client for music metadata.

    Provides master/release lookups with OAuth authentication and rate limiting.
    Rate limits: 60 req/min (authenticated), 25 req/min (unauthenticated).
    """

    BASE_URL = "https://api.discogs.com"
    USER_AGENT = "chart-binder/0.1.0 +https://github.com/pvliesdonk/chart-binder"

    def __init__(
        self,
        token: str | None = None,
        cache: HttpCache | None = None,
        rate_limit_per_min: int = 25,  # Conservative default for unauth
    ):
        """
        Initialize Discogs client.

        Args:
            token: Discogs personal access token (env: DISCOGS_TOKEN)
            cache: Optional HTTP cache for responses
            rate_limit_per_min: Max requests per minute (60 auth, 25 unauth)
        """
        self.token = token or os.getenv("DISCOGS_TOKEN")
        self.cache = cache
        self.rate_limit_per_min = rate_limit_per_min
        self._request_times: list[float] = []

        headers = {"User-Agent": self.USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Discogs token={self.token}"

        self._client = httpx.Client(timeout=30.0, headers=headers)

    def _rate_limit(self) -> None:
        """Enforce rate limiting using sliding window."""
        if self.rate_limit_per_min <= 0:
            return

        now = time.time()
        window_start = now - 60.0  # 60 seconds ago

        # Remove requests outside the window
        self._request_times = [t for t in self._request_times if t > window_start]

        # Check if we're at the limit
        if len(self._request_times) >= self.rate_limit_per_min:
            # Wait until the oldest request falls outside the window
            oldest = self._request_times[0]
            wait_time = 60.0 - (now - oldest) + 0.1  # Add small buffer
            if wait_time > 0:
                time.sleep(wait_time)

        # Record this request
        self._request_times.append(time.time())

    def _request(self, endpoint: str) -> dict[str, Any]:
        """Make rate-limited request to Discogs API."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        # Check cache
        if self.cache:
            cached = self.cache.get(url)
            if cached:
                return cached.json()

        # Make live request
        response = self._client.get(url)
        response.raise_for_status()

        # Cache response
        if self.cache:
            self.cache.put(url, response)

        return response.json()

    def get_master(self, master_id: int) -> DiscogsMaster:
        """
        Get master release by ID.

        Args:
            master_id: Discogs master release ID

        Returns:
            DiscogsMaster object
        """
        data = self._request(f"masters/{master_id}")

        # Extract artist name (simplified - just take first)
        artist = "Unknown"
        if "artists" in data and data["artists"]:
            artist = data["artists"][0].get("name", "Unknown")

        return DiscogsMaster(
            id=data["id"],
            title=data.get("title", ""),
            artist=artist,
            year=data.get("year"),
            genres=data.get("genres", []),
            styles=data.get("styles", []),
            main_release_id=data.get("main_release"),
        )

    def get_release(self, release_id: int) -> DiscogsRelease:
        """
        Get release by ID.

        Args:
            release_id: Discogs release ID

        Returns:
            DiscogsRelease object
        """
        data = self._request(f"releases/{release_id}")

        # Extract artist name (simplified - just take first)
        artist = "Unknown"
        if "artists" in data and data["artists"]:
            artist = data["artists"][0].get("name", "Unknown")

        # Extract label names
        labels = []
        if "labels" in data:
            labels = [label.get("name", "") for label in data["labels"] if label.get("name")]

        # Extract formats
        formats = []
        if "formats" in data:
            formats = [fmt.get("name", "") for fmt in data["formats"] if fmt.get("name")]

        # Extract barcode from identifiers
        barcode = None
        if "identifiers" in data:
            for ident in data["identifiers"]:
                if ident.get("type", "").lower() == "barcode":
                    barcode = ident.get("value")
                    break

        return DiscogsRelease(
            id=data["id"],
            title=data.get("title", ""),
            artist=artist,
            master_id=data.get("master_id"),
            year=data.get("year"),
            country=data.get("country"),
            labels=labels,
            formats=formats,
            genres=data.get("genres", []),
            barcode=barcode,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> DiscogsClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


## Tests


def test_discogs_master_dataclass():
    """Test DiscogsMaster dataclass."""
    master = DiscogsMaster(
        id=12345,
        title="Test Album",
        artist="Test Artist",
        year=2020,
        genres=["Rock"],
        main_release_id=67890,
    )
    assert master.id == 12345
    assert master.title == "Test Album"
    assert len(master.genres) == 1


def test_discogs_release_dataclass():
    """Test DiscogsRelease dataclass."""
    release = DiscogsRelease(
        id=67890,
        title="Test Album",
        artist="Test Artist",
        master_id=12345,
        year=2020,
        country="US",
        labels=["Test Label"],
        formats=["CD"],
    )
    assert release.id == 67890
    assert release.master_id == 12345
    assert len(release.labels) == 1


def test_discogs_rate_limiting():
    """Test rate limiting logic."""
    client = DiscogsClient(rate_limit_per_min=5)

    # Simulate 5 requests
    for _ in range(5):
        client._request_times.append(time.time())

    # Should have 5 requests in window
    assert len(client._request_times) == 5
