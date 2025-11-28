"""Token-bucket rate limiter for Chart-Binder (Epic 14).

Provides per-source rate limiting with configurable token-bucket algorithm.
Supports multiple independent rate limits for different API sources.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenBucket:
    """Token bucket rate limiter.

    Implements the token bucket algorithm for rate limiting:
    - Tokens are added at a fixed rate (refill_rate per second)
    - Requests consume tokens from the bucket
    - If no tokens are available, requests block until tokens refill
    - Maximum tokens in bucket is capped at capacity
    """

    capacity: float
    refill_rate: float  # tokens per second
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.refill_rate
        self._tokens = min(self.capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            blocking: If True, wait until tokens are available

        Returns:
            True if tokens were acquired, False if non-blocking and tokens unavailable
        """
        with self._lock:
            self._refill()

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            if not blocking:
                return False

            # Calculate wait time
            deficit = tokens - self._tokens
            wait_time = deficit / self.refill_rate

        # Wait outside lock to avoid blocking other callers
        time.sleep(wait_time)

        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens were acquired, False otherwise
        """
        return self.acquire(tokens, blocking=False)

    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self._tokens

    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate wait time for specified tokens.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds (0 if tokens are available)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0
            deficit = tokens - self._tokens
            return deficit / self.refill_rate


class RateLimiterRegistry:
    """Registry for per-source rate limiters.

    Manages multiple token bucket rate limiters for different API sources.
    Thread-safe for concurrent access.
    """

    # Default rate limits per source (requests per minute)
    DEFAULT_LIMITS: dict[str, tuple[float, float]] = {
        "musicbrainz": (1.0, 1.0),  # 1 req/sec, capacity 1
        "discogs_auth": (60.0, 60.0 / 60),  # 60 req/min
        "discogs_unauth": (25.0, 25.0 / 60),  # 25 req/min
        "spotify": (30.0, 30.0 / 60),  # ~30 req/min (conservative)
        "wikidata": (10.0, 10.0 / 60),  # 10 req/min (SPARQL endpoint)
        "acoustid": (3.0, 3.0),  # 3 req/sec
    }

    def __init__(self) -> None:
        self._limiters: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get_limiter(self, source: str) -> TokenBucket:
        """Get or create rate limiter for source.

        Args:
            source: API source name (e.g., 'musicbrainz', 'discogs')

        Returns:
            TokenBucket rate limiter for the source
        """
        with self._lock:
            if source not in self._limiters:
                capacity, refill_rate = self.DEFAULT_LIMITS.get(source, (10.0, 10.0 / 60))
                self._limiters[source] = TokenBucket(
                    capacity=capacity,
                    refill_rate=refill_rate,
                )
            return self._limiters[source]

    def configure(self, source: str, capacity: float, refill_rate: float) -> None:
        """Configure rate limiter for a source.

        Args:
            source: API source name
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        with self._lock:
            self._limiters[source] = TokenBucket(
                capacity=capacity,
                refill_rate=refill_rate,
            )

    def acquire(self, source: str, tokens: float = 1.0, blocking: bool = True) -> bool:
        """Acquire tokens from source's rate limiter.

        Args:
            source: API source name
            tokens: Number of tokens to acquire
            blocking: Whether to block until tokens available

        Returns:
            True if tokens were acquired
        """
        limiter = self.get_limiter(source)
        return limiter.acquire(tokens, blocking)

    def status(self) -> dict[str, dict[str, Any]]:
        """Get status of all rate limiters.

        Returns:
            Dict with source names as keys and status dicts as values
        """
        with self._lock:
            return {
                source: {
                    "capacity": limiter.capacity,
                    "refill_rate": limiter.refill_rate,
                    "available_tokens": limiter.available_tokens,
                }
                for source, limiter in self._limiters.items()
            }


# Global registry instance
_registry: RateLimiterRegistry | None = None
_registry_lock = threading.Lock()


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get the global rate limiter registry."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = RateLimiterRegistry()
        return _registry


def rate_limit(source: str, tokens: float = 1.0) -> None:
    """Apply rate limiting for a source.

    Convenience function that blocks until tokens are available.

    Args:
        source: API source name
        tokens: Number of tokens to consume
    """
    registry = get_rate_limiter_registry()
    registry.acquire(source, tokens, blocking=True)


## Tests


def test_token_bucket_basic():
    """Test basic token bucket operations."""
    bucket = TokenBucket(capacity=5.0, refill_rate=1.0)
    # Use approximate comparison due to time-based refill
    assert abs(bucket.available_tokens - 5.0) < 0.1

    # Consume tokens
    assert bucket.acquire(3.0) is True
    assert abs(bucket.available_tokens - 2.0) < 0.1

    # Consume remaining
    assert bucket.acquire(2.0) is True
    assert bucket.available_tokens < 0.1


def test_token_bucket_try_acquire():
    """Test non-blocking token acquisition."""
    bucket = TokenBucket(capacity=2.0, refill_rate=1.0)

    assert bucket.try_acquire(1.0) is True
    assert bucket.try_acquire(1.0) is True
    assert bucket.try_acquire(1.0) is False  # No tokens left


def test_token_bucket_refill():
    """Test token refill over time."""
    bucket = TokenBucket(capacity=2.0, refill_rate=100.0)  # Fast refill for testing

    # Empty bucket
    bucket.acquire(2.0)
    assert bucket.available_tokens < 0.1  # Approximately 0

    # Wait for refill
    time.sleep(0.02)  # 20ms should refill ~2 tokens at 100/sec
    assert bucket.available_tokens >= 1.5


def test_token_bucket_wait_time():
    """Test wait time calculation."""
    bucket = TokenBucket(capacity=1.0, refill_rate=1.0)

    # Empty the bucket
    bucket.acquire(1.0)

    # Should need to wait ~1 second for 1 token
    wait = bucket.wait_time(1.0)
    assert 0.9 <= wait <= 1.1

    # No wait if tokens available
    bucket._tokens = 1.0
    assert bucket.wait_time(1.0) == 0.0


def test_registry_default_limits():
    """Test registry with default limits."""
    registry = RateLimiterRegistry()

    mb_limiter = registry.get_limiter("musicbrainz")
    assert mb_limiter.capacity == 1.0
    assert mb_limiter.refill_rate == 1.0

    discogs_limiter = registry.get_limiter("discogs_unauth")
    assert discogs_limiter.capacity == 25.0


def test_registry_configure():
    """Test custom rate limit configuration."""
    registry = RateLimiterRegistry()

    registry.configure("custom_api", capacity=100.0, refill_rate=10.0)
    limiter = registry.get_limiter("custom_api")

    assert limiter.capacity == 100.0
    assert limiter.refill_rate == 10.0


def test_registry_status():
    """Test status reporting."""
    registry = RateLimiterRegistry()

    # Access a few limiters
    registry.get_limiter("musicbrainz")
    registry.get_limiter("spotify")

    status = registry.status()
    assert "musicbrainz" in status
    assert "spotify" in status
    assert "capacity" in status["musicbrainz"]
    assert "available_tokens" in status["musicbrainz"]


def test_global_registry():
    """Test global registry access."""
    registry1 = get_rate_limiter_registry()
    registry2 = get_rate_limiter_registry()
    assert registry1 is registry2


def test_rate_limit_function():
    """Test convenience rate_limit function."""
    # Reset global registry
    global _registry
    _registry = RateLimiterRegistry()

    # Should not raise, just consume tokens
    rate_limit("musicbrainz", 1.0)

    # Verify registry has the limiter
    registry = get_rate_limiter_registry()
    assert "musicbrainz" in registry.status()
