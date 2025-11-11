from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import httpx


class HttpCache:
    """
    File-backed HTTP cache with SQLite index tracking ETag, Last-Modified, and TTL.

    Stores response bodies as files and metadata in SQLite for efficient cache validation.
    """

    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache_index.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                url TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL,
                etag TEXT,
                last_modified TEXT,
                cached_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                status_code INTEGER NOT NULL,
                content_type TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)")
        conn.commit()
        conn.close()

    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached content."""
        return self.cache_dir / f"{cache_key}.cache"

    def get(self, url: str) -> httpx.Response | None:
        """
        Get cached response if valid, otherwise return None.

        Checks TTL and returns cached response only if not expired.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM cache_entries WHERE url = ? AND expires_at > ?",
            (url, time.time()),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        cache_path = self._get_cache_path(row["cache_key"])
        if not cache_path.exists():
            return None

        content = cache_path.read_bytes()
        headers = {}
        if row["etag"]:
            headers["etag"] = row["etag"]
        if row["last_modified"]:
            headers["last-modified"] = row["last_modified"]
        if row["content_type"]:
            headers["content-type"] = row["content_type"]

        return httpx.Response(
            status_code=row["status_code"],
            headers=headers,
            content=content,
            request=httpx.Request("GET", url),
        )

    def put(self, url: str, response: httpx.Response) -> None:
        """
        Store response in cache with metadata.

        Extracts ETag and Last-Modified headers for cache validation.
        """
        cache_key = self._get_cache_key(url)
        cache_path = self._get_cache_path(cache_key)

        cache_path.write_bytes(response.content)

        cached_at = time.time()
        expires_at = cached_at + self.ttl_seconds

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries
            (url, cache_key, etag, last_modified, cached_at, expires_at, status_code, content_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                cache_key,
                response.headers.get("etag"),
                response.headers.get("last-modified"),
                cached_at,
                expires_at,
                response.status_code,
                response.headers.get("content-type"),
            ),
        )
        conn.commit()
        conn.close()

    def invalidate(self, url: str) -> None:
        """Remove cache entry for specific URL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT cache_key FROM cache_entries WHERE url = ?", (url,))
        row = cursor.fetchone()

        if row:
            cache_path = self._get_cache_path(row[0])
            cache_path.unlink(missing_ok=True)
            conn.execute("DELETE FROM cache_entries WHERE url = ?", (url,))
            conn.commit()

        conn.close()

    def purge_expired(self) -> int:
        """Remove expired cache entries and return count of removed entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT cache_key FROM cache_entries WHERE expires_at <= ?", (time.time(),))
        expired_keys = [row[0] for row in cursor.fetchall()]

        for cache_key in expired_keys:
            cache_path = self._get_cache_path(cache_key)
            cache_path.unlink(missing_ok=True)

        cursor.execute("DELETE FROM cache_entries WHERE expires_at <= ?", (time.time(),))
        removed_count = cursor.rowcount
        conn.commit()
        conn.close()

        return removed_count

    def clear(self) -> None:
        """Clear all cache entries."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT cache_key FROM cache_entries")
        all_keys = [row[0] for row in cursor.fetchall()]

        for cache_key in all_keys:
            cache_path = self._get_cache_path(cache_key)
            cache_path.unlink(missing_ok=True)

        conn.execute("DELETE FROM cache_entries")
        conn.commit()
        conn.close()


## Tests


def test_http_cache_basic(tmp_path):
    cache = HttpCache(tmp_path / "cache", ttl_seconds=3600)

    url = "https://example.com/test"
    response = httpx.Response(
        status_code=200,
        content=b"test content",
        headers={"etag": "abc123", "content-type": "text/plain"},
        request=httpx.Request("GET", url),
    )

    cache.put(url, response)

    cached = cache.get(url)
    assert cached is not None
    assert cached.status_code == 200
    assert cached.content == b"test content"
    assert cached.headers.get("etag") == "abc123"


def test_http_cache_ttl_expired(tmp_path):
    cache = HttpCache(tmp_path / "cache", ttl_seconds=1)

    url = "https://example.com/test"
    response = httpx.Response(
        status_code=200,
        content=b"test content",
        request=httpx.Request("GET", url),
    )

    cache.put(url, response)
    time.sleep(1.1)

    cached = cache.get(url)
    assert cached is None


def test_http_cache_invalidate(tmp_path):
    cache = HttpCache(tmp_path / "cache")

    url = "https://example.com/test"
    response = httpx.Response(
        status_code=200,
        content=b"test content",
        request=httpx.Request("GET", url),
    )

    cache.put(url, response)
    assert cache.get(url) is not None

    cache.invalidate(url)
    assert cache.get(url) is None


def test_http_cache_purge_expired(tmp_path):
    cache = HttpCache(tmp_path / "cache", ttl_seconds=1)

    url1 = "https://example.com/test1"
    url2 = "https://example.com/test2"

    for url in [url1, url2]:
        response = httpx.Response(
            status_code=200,
            content=b"test",
            request=httpx.Request("GET", url),
        )
        cache.put(url, response)

    time.sleep(1.1)
    removed = cache.purge_expired()
    assert removed == 2


def test_http_cache_clear(tmp_path):
    cache = HttpCache(tmp_path / "cache")

    for i in range(3):
        url = f"https://example.com/test{i}"
        response = httpx.Response(
            status_code=200,
            content=b"test",
            request=httpx.Request("GET", url),
        )
        cache.put(url, response)

    cache.clear()
    assert cache.get("https://example.com/test0") is None
    assert cache.get("https://example.com/test1") is None
    assert cache.get("https://example.com/test2") is None
