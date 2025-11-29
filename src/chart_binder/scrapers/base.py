from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from chart_binder.http_cache import HttpCache

logger = logging.getLogger(__name__)


@dataclass
class ScrapedEntry:
    """A single scraped chart entry with optional metadata."""

    rank: int
    artist: str
    title: str
    previous_position: int | None = None  # Position in previous week (from website)
    weeks_on_chart: int | None = None  # How long on chart (from website)

    def as_tuple(self) -> tuple[int, str, str]:
        """Convert to simple tuple format for backwards compatibility."""
        return (self.rank, self.artist, self.title)


@dataclass
class ScrapeResult:
    """Result of a scrape operation with validation info."""

    entries: list[tuple[int, str, str]]
    expected_count: int
    chart_type: str
    period: str
    warnings: list[str]
    # Rich entry data (if available)
    rich_entries: list[ScrapedEntry] | None = None
    # Continuity validation
    continuity_overlap: float | None = None  # Overlap % with reference run
    continuity_reference: str | None = None  # Reference period used
    # Cross-reference validation
    position_mismatches: list[tuple[str, str, int, int]] | None = (
        None  # (artist, title, claimed, actual)
    )

    @property
    def actual_count(self) -> int:
        return len(self.entries)

    @property
    def is_valid(self) -> bool:
        """Check if scrape result passes sanity checks."""
        if self.actual_count == 0:
            return False
        # Allow 10% tolerance for expected count
        min_expected = int(self.expected_count * 0.9)
        return self.actual_count >= min_expected

    @property
    def continuity_valid(self) -> bool:
        """Check if continuity with previous run is acceptable."""
        if self.continuity_overlap is None:
            return True  # No reference to compare
        # For weekly charts, expect at least 50% overlap
        # For yearly charts, this doesn't apply
        return self.continuity_overlap >= 0.5

    @property
    def shortage(self) -> int:
        """How many entries short of expected (0 if at or above expected)."""
        return max(0, self.expected_count - self.actual_count)

    def __str__(self) -> str:
        status = "✔︎" if self.is_valid else "✘"
        base = f"{status} {self.chart_type} {self.period}: {self.actual_count}/{self.expected_count} entries"
        if self.continuity_overlap is not None:
            cont_status = "✔︎" if self.continuity_valid else "⚠"
            base += f" [{cont_status} {self.continuity_overlap:.0%} overlap]"
        return base


def calculate_overlap(
    current_entries: list[tuple[int, str, str]],
    reference_entries: list[tuple[str, str]],
) -> float:
    """
    Calculate overlap percentage between current scrape and reference entries.

    Args:
        current_entries: List of (rank, artist, title) from current scrape
        reference_entries: List of (artist, title) from reference run

    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    if not reference_entries:
        return 1.0  # No reference, assume OK

    # Normalize for comparison (lowercase, strip whitespace)
    def normalize(s: str) -> str:
        return s.lower().strip()

    current_set = {(normalize(artist), normalize(title)) for _, artist, title in current_entries}
    reference_set = {(normalize(artist), normalize(title)) for artist, title in reference_entries}

    if not reference_set:
        return 1.0

    overlap = len(current_set & reference_set)
    return overlap / len(reference_set)


# Expected entry counts per chart type
EXPECTED_ENTRY_COUNTS: dict[str, int] = {
    "nl_top40": 40,  # Weekly Top 40
    "nl_top40_jaar": 100,  # Year-end Top 100
    "nl_top2000": 2000,  # NPO Top 2000
    "nl_538_zwaarste": 150,  # 538 De Zwaarste Lijst (typically 150)
}


def cross_reference_previous_positions(
    scraped_entries: list[ScrapedEntry],
    db_previous_ranks: dict[tuple[str, str], int],
) -> list[tuple[str, str, int, int]]:
    """
    Cross-reference scraped previous_position against actual database positions.

    Args:
        scraped_entries: List of ScrapedEntry objects with previous_position from website
        db_previous_ranks: Dict mapping (normalized_artist, normalized_title) to actual rank
                          from the previous period in our database

    Returns:
        List of mismatches as (artist, title, claimed_position, actual_position) tuples.
        Empty list means all positions match or no previous_position data available.
    """
    mismatches: list[tuple[str, str, int, int]] = []

    for entry in scraped_entries:
        if entry.previous_position is None:
            # No previous position claimed, skip
            continue

        # Normalize for matching
        key = (entry.artist.lower().strip(), entry.title.lower().strip())

        actual_rank = db_previous_ranks.get(key)
        if actual_rank is None:
            # Entry not found in previous period - might be new or naming mismatch
            # Not necessarily a mismatch, could be genuinely new entry
            continue

        if actual_rank != entry.previous_position:
            mismatches.append((entry.artist, entry.title, entry.previous_position, actual_rank))

    return mismatches


class ChartScraper(ABC):
    """
    Base class for chart scrapers.

    Provides common infrastructure for fetching and parsing chart data.
    All requests go through `HttpCache` for offline-first operation.
    """

    # Subclasses should override these
    chart_db_id: str = ""  # ID used in charts.sqlite (e.g., "nl_top40")
    expected_entry_count: int = 0  # Expected number of entries

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
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
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

    def scrape_with_validation(
        self,
        period: str,
        db_previous_ranks: dict[tuple[str, str], int] | None = None,
    ) -> ScrapeResult:
        """
        Scrape chart data and return result with validation info.

        Performs the scrape and validates the result against expected entry counts.
        If db_previous_ranks is provided, also performs cross-reference validation
        against the previous period's positions from the database.

        Args:
            period: Period to scrape
            db_previous_ranks: Optional dict mapping (artist, title) to rank from
                              the previous period in the database, for cross-reference
                              validation
        """
        entries = self.scrape(period)
        warnings: list[str] = []
        rich_entries: list[ScrapedEntry] | None = None
        position_mismatches: list[tuple[str, str, int, int]] | None = None

        # Try to get rich entries if scraper supports it
        if db_previous_ranks is not None:
            rich_entries = self.scrape_rich(period)
            if rich_entries:
                position_mismatches = cross_reference_previous_positions(
                    rich_entries, db_previous_ranks
                )

        # Validate entries for suspicious patterns
        issues = self._validate_entries(entries)
        for issue_type, issue_entries in issues.items():
            if issue_entries:
                warnings.append(f"{issue_type}: {len(issue_entries)} entries")

        result = ScrapeResult(
            entries=entries,
            expected_count=self.expected_entry_count,
            chart_type=self.chart_db_id,
            period=period,
            warnings=warnings,
            rich_entries=rich_entries,
            position_mismatches=position_mismatches,
        )

        if not result.is_valid:
            logger.warning(
                f"Scrape validation failed for {self.chart_db_id} {period}: "
                f"got {result.actual_count}, expected ~{result.expected_count}"
            )

        return result

    def scrape_rich(self, period: str) -> list[ScrapedEntry]:
        """
        Scrape chart with full metadata (previous_position, weeks_on_chart).

        Override in subclasses that support rich scraping.
        Returns empty list by default.
        """
        return []

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
            warnings.append(f"Unusually long title ({len(title)} chars): {title[:50]}...")

        if len(artist) > 150:
            warnings.append(f"Unusually long artist ({len(artist)} chars): {artist[:50]}...")

        # Check for excessive slashes (might indicate parsing issue)
        slash_count = title.count("/") + artist.count("/")
        if slash_count > 5:
            warnings.append(f"Excessive slashes detected ({slash_count}) - possible parsing issue")

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


def test_cross_reference_previous_positions():
    # Scraped entries with claimed previous positions
    scraped = [
        ScrapedEntry(rank=1, artist="Artist A", title="Song A", previous_position=3),
        ScrapedEntry(rank=2, artist="Artist B", title="Song B", previous_position=1),
        ScrapedEntry(
            rank=3, artist="Artist C", title="Song C", previous_position=None
        ),  # New entry
        ScrapedEntry(rank=4, artist="Artist D", title="Song D", previous_position=5),
    ]

    # Database records from previous period
    db_previous = {
        ("artist a", "song a"): 3,  # Correct
        ("artist b", "song b"): 2,  # Mismatch! Website claims 1, db says 2
        ("artist d", "song d"): 5,  # Correct
    }

    mismatches = cross_reference_previous_positions(scraped, db_previous)

    # Only Artist B should be flagged (claimed 1, actual 2)
    assert len(mismatches) == 1
    assert mismatches[0] == ("Artist B", "Song B", 1, 2)


def test_cross_reference_previous_positions_empty():
    # No previous positions claimed
    scraped = [
        ScrapedEntry(rank=1, artist="Artist A", title="Song A", previous_position=None),
    ]
    db_previous: dict[tuple[str, str], int] = {}

    mismatches = cross_reference_previous_positions(scraped, db_previous)
    assert len(mismatches) == 0
