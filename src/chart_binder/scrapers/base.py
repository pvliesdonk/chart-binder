from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod

import httpx

from chart_binder.http_cache import HttpCache

logger = logging.getLogger(__name__)


class ChartScraper(ABC):
    """
    Base class for chart scrapers.

    Provides common infrastructure for fetching and parsing chart data.
    All requests go through `HttpCache` for offline-first operation.
    """

    def __init__(self, chart_id: str, cache: HttpCache):
        self.chart_id = chart_id
        self.cache = cache
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialized HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "chart-binder/1.0"},
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> ChartScraper:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        self.close()

    @abstractmethod
    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        """
        Scrape chart data for a given period.

        Args:
            period: Period identifier (format varies by scraper, e.g., YYYY-Www for weekly)

        Returns:
            List of (rank, artist, title) tuples
        """
        ...

    def _fetch_url(self, url: str) -> str | None:
        """
        Fetch URL content with caching.

        Returns None on 404 or other errors.
        """
        cached = self.cache.get(url)
        if cached is not None:
            if cached.status_code == 404:
                return None
            return cached.text

        try:
            response = self.client.get(url)

            self.cache.put(url, response)

            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching {url}: {e}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Request error fetching {url}: {e}")
            return None

    def _fetch_json(self, url: str) -> dict[str, object] | list[object] | None:
        """
        Fetch URL and parse as JSON with caching.

        Returns None on 404 or other errors.
        """
        cached = self.cache.get(url)
        if cached is not None:
            if cached.status_code == 404:
                return None
            try:
                return cached.json()
            except Exception:
                return None

        try:
            response = self.client.get(url)

            self.cache.put(url, response)

            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching {url}: {e}")
            return None
        except httpx.RequestError as e:
            logger.warning(f"Request error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.warning(f"JSON parse error for {url}: {e}")
            return None

    def _parse_period(self, period: str) -> tuple[int, int]:
        """
        Parse ISO week period string (YYYY-Www).

        Returns (year, week) tuple.
        """
        match = re.match(r"(\d{4})-W(\d{1,2})", period, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid period format: {period}, expected YYYY-Www")
        return int(match.group(1)), int(match.group(2))

    def _parse_year_period(self, period: str) -> int:
        """
        Parse year-only period string.

        Returns year as integer.
        """
        match = re.match(r"(\d{4})", period)
        if not match:
            raise ValueError(f"Invalid period format: {period}, expected YYYY")
        return int(match.group(1))

    @staticmethod
    def _generate_hash_id(prefix: str, *parts: str) -> str:
        """Generate deterministic hash ID from parts."""
        combined = "_".join([prefix] + list(parts))
        return hashlib.md5(combined.encode()).hexdigest()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean whitespace and normalize text."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _remove_double_parens(text: str) -> str:
        """Remove text inside double parentheses ((...)) or brackets [...]."""
        text = re.sub(r"\(\([^)]*\)\)", "", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        return text.strip()

    def _validate_entry(
        self, rank: int, artist: str, title: str, strict: bool = False
    ) -> list[str]:
        """
        Validate a scraped entry for suspicious patterns.

        Args:
            rank: Chart rank
            artist: Artist name
            title: Song title
            strict: If True, raise exception on errors; if False, just return warnings

        Returns:
            List of warning messages (empty if no issues)

        Raises:
            ValueError: If strict=True and critical issues are found
        """
        warnings: list[str] = []

        # Check for HTML tags (critical error)
        html_pattern = r"<[^>]+>"
        if re.search(html_pattern, title):
            msg = f"HTML tags found in title: {title}"
            warnings.append(msg)
            if strict:
                raise ValueError(msg)

        if re.search(html_pattern, artist):
            msg = f"HTML tags found in artist: {artist}"
            warnings.append(msg)
            if strict:
                raise ValueError(msg)

        # Check for excessively long fields
        if len(title) > 200:
            warnings.append(
                f"Unusually long title ({len(title)} chars): {title[:50]}..."
            )

        if len(artist) > 150:
            warnings.append(
                f"Unusually long artist ({len(artist)} chars): {artist[:50]}..."
            )

        # Check for excessive slashes (might indicate parsing issue)
        slash_count = title.count("/") + artist.count("/")
        if slash_count > 5:
            warnings.append(
                f"Excessive slashes detected ({slash_count}) - possible parsing issue"
            )

        # Check for numeric-only title
        if title.strip() and title.strip().isdigit():
            warnings.append(f"Title is numeric-only: '{title}' - likely parsing error")

        # Check for empty required fields
        if not artist.strip():
            warnings.append("Artist is empty")

        if not title.strip():
            warnings.append("Title is empty")

        # Log warnings if any found
        if warnings:
            logger.warning(
                f"Validation warnings for [{self.chart_id}] rank {rank}: {'; '.join(warnings)}"
            )

        return warnings

    def _validate_entries(
        self, entries: list[tuple[int, str, str]], strict: bool = False
    ) -> dict[str, list]:
        """
        Validate all entries in a scraped result.

        Args:
            entries: List of (rank, artist, title) tuples
            strict: If True, raise exception on critical errors

        Returns:
            Dict mapping issue types to lists of problematic entries
        """
        issues: dict[str, list] = {
            "html_in_text": [],
            "long_titles": [],
            "long_artists": [],
            "excessive_slashes": [],
            "numeric_titles": [],
            "empty_fields": [],
            "duplicates": [],
        }

        seen = set()

        for rank, artist, title in entries:
            # Check for duplicates
            key = (rank, artist, title)
            if key in seen:
                issues["duplicates"].append((rank, artist, title))
            seen.add(key)

            # Validate entry
            warnings = self._validate_entry(rank, artist, title, strict=strict)

            # Categorize issues
            for warning in warnings:
                if "HTML tags" in warning:
                    issues["html_in_text"].append((rank, artist, title))
                elif "long title" in warning:
                    issues["long_titles"].append((rank, artist, title))
                elif "long artist" in warning:
                    issues["long_artists"].append((rank, artist, title))
                elif "slashes" in warning:
                    issues["excessive_slashes"].append((rank, artist, title))
                elif "numeric-only" in warning:
                    issues["numeric_titles"].append((rank, artist, title))
                elif "empty" in warning.lower():
                    issues["empty_fields"].append((rank, artist, title))

        # Log summary if issues found
        total_issues = sum(len(v) for v in issues.values())
        if total_issues > 0:
            logger.warning(
                f"Validation found {total_issues} potential issues in {len(entries)} entries for {self.chart_id}"
            )

        return issues


## Tests


def test_parse_period():
    class DummyScraper(ChartScraper):
        def scrape(self, period: str) -> list[tuple[int, str, str]]:
            return []

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = DummyScraper("test", cache)

        assert scraper._parse_period("2024-W01") == (2024, 1)
        assert scraper._parse_period("1967-W07") == (1967, 7)
        assert scraper._parse_period("2024-w52") == (2024, 52)


def test_parse_year_period():
    class DummyScraper(ChartScraper):
        def scrape(self, period: str) -> list[tuple[int, str, str]]:
            return []

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = DummyScraper("test", cache)

        assert scraper._parse_year_period("2024") == 2024
        assert scraper._parse_year_period("1999") == 1999


def test_generate_hash_id():
    id1 = ChartScraper._generate_hash_id("test", "artist", "title")
    id2 = ChartScraper._generate_hash_id("test", "artist", "title")
    id3 = ChartScraper._generate_hash_id("test", "artist2", "title")

    assert id1 == id2
    assert id1 != id3


def test_clean_text():
    assert ChartScraper._clean_text("  hello   world  ") == "hello world"
    assert ChartScraper._clean_text("foo\n\tbar") == "foo bar"


def test_remove_double_parens():
    assert ChartScraper._remove_double_parens("Title ((metadata))") == "Title"
    assert ChartScraper._remove_double_parens("Title [info]") == "Title"
    assert ChartScraper._remove_double_parens("Title (normal parens)") == "Title (normal parens)"
