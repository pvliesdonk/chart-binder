"""
Spotify Web API client for music metadata lookups.

Supports track/album metadata by ID or search with client credentials flow,
preview URL and popularity extraction, and rate limiting per ToS.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


@dataclass
class SpotifyTrack:
    """Spotify track entity."""

    id: str
    name: str
    artist_name: str
    album_id: str | None = None
    album_name: str | None = None
    duration_ms: int | None = None
    isrc: str | None = None
    popularity: int | None = None
    preview_url: str | None = None


@dataclass
class SpotifyAlbum:
    """Spotify album entity."""

    id: str
    name: str
    artist_name: str
    release_date: str | None = None
    total_tracks: int | None = None
    label: str | None = None
    genres: list[str] = field(default_factory=list)
    popularity: int | None = None


class SpotifyClient:
    """
    Spotify Web API client for music metadata.

    Uses client credentials flow for authentication.
    Extracts preview URLs and popularity metrics.
    """

    BASE_URL = "https://api.spotify.com/v1"
    AUTH_URL = "https://accounts.spotify.com/api/token"

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache: HttpCache | None = None,
    ):
        """
        Initialize Spotify client with client credentials flow.

        Args:
            client_id: Spotify client ID (env: SPOTIFY_CLIENT_ID)
            client_secret: Spotify client secret (env: SPOTIFY_CLIENT_SECRET)
            cache: Optional HTTP cache for responses
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.cache = cache

        # Validate that both credentials are provided together or both are None
        if bool(self.client_id) != bool(self.client_secret):
            raise ValueError(
                "Both SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be provided together. "
                f"Got: client_id={'set' if self.client_id else 'missing'}, "
                f"client_secret={'set' if self.client_secret else 'missing'}"
            )

        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._client = httpx.Client(timeout=30.0)

    def _get_access_token(self) -> str:
        """
        Get access token using client credentials flow.

        Caches token until expiration.
        """
        # Return cached token if still valid
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token

        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify client_id and client_secret required "
                "(set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET env vars)"
            )

        # Encode credentials
        credentials = f"{self.client_id}:{self.client_secret}"
        b64_credentials = base64.b64encode(credentials.encode()).decode()

        # Request token
        response = self._client.post(
            self.AUTH_URL,
            headers={
                "Authorization": f"Basic {b64_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()

        data = response.json()
        token = data["access_token"]
        self._access_token = token
        expires_in = data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60  # 60s buffer

        return token

    def _request(self, endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        """Make authenticated request to Spotify API."""
        token = self._get_access_token()
        url = f"{self.BASE_URL}/{endpoint}"

        # Build cache key
        cache_key = url
        if params:
            sorted_params = sorted(params.items())
            cache_key += "?" + "&".join(f"{k}={v}" for k, v in sorted_params)

        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached.json()

        # Make live request
        response = self._client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()

        # Cache response
        if self.cache:
            # Create mock response for caching
            mock_response = httpx.Response(
                status_code=200,
                content=response.content,
                headers=response.headers,
                request=httpx.Request("GET", cache_key),
            )
            self.cache.put(cache_key, mock_response)

        return response.json()

    def get_track(self, track_id: str) -> SpotifyTrack:
        """
        Get track metadata by Spotify ID.

        Args:
            track_id: Spotify track ID

        Returns:
            SpotifyTrack object
        """
        data = self._request(f"tracks/{track_id}")

        # Extract artist (simplified - first artist)
        artist_name = "Unknown"
        if "artists" in data and data["artists"]:
            artist_name = data["artists"][0].get("name", "Unknown")

        # Extract album info
        album_id = None
        album_name = None
        if "album" in data:
            album_id = data["album"].get("id")
            album_name = data["album"].get("name")

        # Extract ISRC
        isrc = None
        if "external_ids" in data:
            isrc = data["external_ids"].get("isrc")

        return SpotifyTrack(
            id=data["id"],
            name=data.get("name", ""),
            artist_name=artist_name,
            album_id=album_id,
            album_name=album_name,
            duration_ms=data.get("duration_ms"),
            isrc=isrc,
            popularity=data.get("popularity"),
            preview_url=data.get("preview_url"),
        )

    def get_album(self, album_id: str) -> SpotifyAlbum:
        """
        Get album metadata by Spotify ID.

        Args:
            album_id: Spotify album ID

        Returns:
            SpotifyAlbum object
        """
        data = self._request(f"albums/{album_id}")

        # Extract artist (simplified - first artist)
        artist_name = "Unknown"
        if "artists" in data and data["artists"]:
            artist_name = data["artists"][0].get("name", "Unknown")

        return SpotifyAlbum(
            id=data["id"],
            name=data.get("name", ""),
            artist_name=artist_name,
            release_date=data.get("release_date"),
            total_tracks=data.get("total_tracks"),
            label=data.get("label"),
            genres=data.get("genres", []),
            popularity=data.get("popularity"),
        )

    def search_tracks(
        self,
        query: str | None = None,
        artist: str | None = None,
        track: str | None = None,
        isrc: str | None = None,
        limit: int = 20,
    ) -> list[SpotifyTrack]:
        """
        Search for tracks.

        Args:
            query: Free-form query string
            artist: Artist name to filter
            track: Track name to filter
            isrc: ISRC code to search
            limit: Max results (default 20)

        Returns:
            List of SpotifyTrack objects
        """
        # Build query
        if not query:
            query_parts = []
            if isrc:
                query_parts.append(f"isrc:{isrc}")
            if artist:
                query_parts.append(f'artist:"{artist}"')
            if track:
                query_parts.append(f'track:"{track}"')
            query = " ".join(query_parts) if query_parts else "*"

        params = {
            "q": query,
            "type": "track",
            "limit": str(limit),
        }

        data = self._request("search", params)

        results = []
        for item in data.get("tracks", {}).get("items", []):
            # Extract artist
            artist_name = "Unknown"
            if "artists" in item and item["artists"]:
                artist_name = item["artists"][0].get("name", "Unknown")

            # Extract album info
            album_id = None
            album_name = None
            if "album" in item:
                album_id = item["album"].get("id")
                album_name = item["album"].get("name")

            # Extract ISRC
            isrc_code = None
            if "external_ids" in item:
                isrc_code = item["external_ids"].get("isrc")

            results.append(
                SpotifyTrack(
                    id=item["id"],
                    name=item.get("name", ""),
                    artist_name=artist_name,
                    album_id=album_id,
                    album_name=album_name,
                    duration_ms=item.get("duration_ms"),
                    isrc=isrc_code,
                    popularity=item.get("popularity"),
                    preview_url=item.get("preview_url"),
                )
            )

        return results

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> SpotifyClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


## Tests


def test_spotify_track_dataclass():
    """Test SpotifyTrack dataclass."""
    track = SpotifyTrack(
        id="abc123",
        name="Test Song",
        artist_name="Test Artist",
        album_id="xyz789",
        duration_ms=180000,
        isrc="USTEST1234567",
        popularity=75,
    )
    assert track.id == "abc123"
    assert track.name == "Test Song"
    assert track.popularity == 75


def test_spotify_album_dataclass():
    """Test SpotifyAlbum dataclass."""
    album = SpotifyAlbum(
        id="xyz789",
        name="Test Album",
        artist_name="Test Artist",
        release_date="2020-01-01",
        total_tracks=12,
        label="Test Label",
    )
    assert album.id == "xyz789"
    assert album.total_tracks == 12
