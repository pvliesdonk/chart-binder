"""
AcoustID API client for fingerprint-based music identification.

Supports fingerprint submission and lookup with duration+fingerprint corroboration,
confidence thresholds, and rate limiting per ToS.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from chart_binder.http_cache import HttpCache


@dataclass
class AcoustIDResult:
    """Result from AcoustID fingerprint lookup."""

    recording_mbid: str
    confidence: float
    duration_ms: int | None = None


class AcoustIDClient:
    """
    AcoustID API client for fingerprint-based lookups.

    Provides fingerprint submission and lookup with confidence thresholds
    and duration corroboration.
    """

    BASE_URL = "https://api.acoustid.org/v2"

    def __init__(
        self,
        api_key: str | None = None,
        cache: HttpCache | None = None,
        rate_limit_per_sec: float = 3.0,
        min_confidence: float = 0.5,
    ):
        """
        Initialize AcoustID client.

        Args:
            api_key: AcoustID API key (defaults to ACOUSTID_API_KEY env var)
            cache: Optional HTTP cache for responses
            rate_limit_per_sec: Max requests per second (default 3.0)
            min_confidence: Minimum confidence threshold (default 0.5)
        """
        self.api_key = api_key or os.getenv("ACOUSTID_API_KEY")
        if not self.api_key:
            raise ValueError("AcoustID API key required (set ACOUSTID_API_KEY env var)")

        self.cache = cache
        self.rate_limit_per_sec = rate_limit_per_sec
        self.min_confidence = min_confidence
        self._last_request_time = 0.0
        self._client = httpx.Client(timeout=30.0)

    def _rate_limit(self) -> None:
        """Enforce rate limiting using token bucket approach."""
        if self.rate_limit_per_sec <= 0:
            return

        min_interval = 1.0 / self.rate_limit_per_sec
        elapsed = time.time() - self._last_request_time

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_time = time.time()

    def lookup(
        self,
        fingerprint: str,
        duration_sec: int,
        max_duration_diff_sec: int = 2,
    ) -> list[AcoustIDResult]:
        """
        Look up recording by audio fingerprint and duration.

        Args:
            fingerprint: Chromaprint fingerprint string
            duration_sec: Track duration in seconds
            max_duration_diff_sec: Max allowed duration difference for corroboration

        Returns:
            List of AcoustIDResult objects sorted by confidence (highest first)
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/lookup"
        params = {
            "client": self.api_key,
            "meta": "recordings",
            "duration": str(duration_sec),
            "fingerprint": fingerprint,
        }

        # Check cache first
        cache_key = f"{url}?{self._make_cache_key(params)}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return self._parse_response(cached.json(), duration_sec, max_duration_diff_sec)

        # Make live request
        response = self._client.get(url, params=params)
        response.raise_for_status()

        # Cache response
        if self.cache:
            self.cache.put(cache_key, response)

        return self._parse_response(response.json(), duration_sec, max_duration_diff_sec)

    def _make_cache_key(self, params: dict[str, str | None]) -> str:
        """Generate stable cache key from params."""
        sorted_items = sorted(params.items())
        return "&".join(f"{k}={v}" for k, v in sorted_items if v is not None)

    def _parse_response(
        self,
        data: dict[str, Any],
        duration_sec: int,
        max_duration_diff_sec: int,
    ) -> list[AcoustIDResult]:
        """
        Parse AcoustID API response and filter by confidence and duration.

        Args:
            data: JSON response from AcoustID API
            duration_sec: Expected duration for corroboration
            max_duration_diff_sec: Max allowed duration difference

        Returns:
            Filtered and sorted list of results
        """
        if data.get("status") != "ok":
            error_msg = data.get("error", {}).get("message", "Unknown error")
            raise ValueError(f"AcoustID API error: {error_msg}")

        results: list[AcoustIDResult] = []

        for result in data.get("results", []):
            confidence = result.get("score", 0.0)

            # Filter by minimum confidence
            if confidence < self.min_confidence:
                continue

            recordings = result.get("recordings", [])
            for recording in recordings:
                rec_mbid = recording.get("id")
                if not rec_mbid:
                    continue

                # Extract duration if available
                rec_duration_ms: int | None = None
                if "duration" in recording:
                    rec_duration_ms = int(recording["duration"] * 1000)

                    # Corroborate duration if available
                    if rec_duration_ms:
                        duration_diff_sec = abs(rec_duration_ms / 1000 - duration_sec)
                        if duration_diff_sec > max_duration_diff_sec:
                            # Skip if duration mismatch is too large
                            continue

                results.append(
                    AcoustIDResult(
                        recording_mbid=rec_mbid,
                        confidence=confidence,
                        duration_ms=rec_duration_ms,
                    )
                )

        # Sort by confidence (highest first)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> AcoustIDClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


## Tests


def test_acoustid_result():
    """Test AcoustIDResult dataclass."""
    result = AcoustIDResult(
        recording_mbid="a1b2c3d4-1234-5678-90ab-cdef12345678",
        confidence=0.95,
        duration_ms=180000,
    )
    assert result.recording_mbid == "a1b2c3d4-1234-5678-90ab-cdef12345678"
    assert result.confidence == 0.95
    assert result.duration_ms == 180000


def test_acoustid_parse_response():
    """Test parsing AcoustID API response."""
    client = AcoustIDClient(api_key="test_key")

    response_data = {
        "status": "ok",
        "results": [
            {
                "score": 0.95,
                "recordings": [
                    {
                        "id": "rec1",
                        "duration": 180.5,
                    }
                ],
            },
            {
                "score": 0.85,
                "recordings": [
                    {
                        "id": "rec2",
                        "duration": 181.0,
                    }
                ],
            },
            {
                "score": 0.45,  # Below min_confidence
                "recordings": [{"id": "rec3"}],
            },
        ],
    }

    results = client._parse_response(response_data, duration_sec=180, max_duration_diff_sec=2)

    assert len(results) == 2  # Third filtered out by confidence
    assert results[0].recording_mbid == "rec1"
    assert results[0].confidence == 0.95
    assert results[1].recording_mbid == "rec2"
    assert results[1].confidence == 0.85


def test_acoustid_duration_corroboration():
    """Test duration corroboration filtering."""
    client = AcoustIDClient(api_key="test_key")

    response_data = {
        "status": "ok",
        "results": [
            {
                "score": 0.95,
                "recordings": [
                    {
                        "id": "rec_good",
                        "duration": 180.5,  # Within 2 sec of 180
                    }
                ],
            },
            {
                "score": 0.90,
                "recordings": [
                    {
                        "id": "rec_bad",
                        "duration": 200.0,  # More than 2 sec diff
                    }
                ],
            },
        ],
    }

    results = client._parse_response(response_data, duration_sec=180, max_duration_diff_sec=2)

    # Only the first recording should pass duration check
    assert len(results) == 1
    assert results[0].recording_mbid == "rec_good"
