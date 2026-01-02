from __future__ import annotations

import logging
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

from chart_binder.http_cache import HttpCache
from chart_binder.scrapers.base import ChartScraper, ScrapedEntry
from chart_binder.scrapers.wikipedia_parser import Top2000WikipediaParser

logger = logging.getLogger(__name__)


class Top2000FutureYearError(ValueError):
    """Raised when attempting to scrape a Top 2000 edition that doesn't exist yet."""

    pass


class Top2000Scraper(ChartScraper):
    """
    Scraper for NPO Radio 2 Top 2000 (yearly chart).

    Primary: NPO API
    Fallback: Wikipedia tables (for older years)

    Note: The Top 2000 is broadcast in late December (around Christmas).
    The {year} edition becomes available around December 25 of that year.
    Attempting to scrape future editions will raise Top2000FutureYearError.
    """

    API_NEW_PATTERN = "https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-{year}-12-25"
    API_OLD_PATTERN = "https://www.nporadio2.nl/api/charts/top-2000-van-{year}-12-25"
    chart_db_id = "nl_top2000"
    expected_entry_count = 2000

    # Top 2000 is typically available around December 25
    AVAILABILITY_MONTH = 12
    AVAILABILITY_DAY = 25

    TITLE_CORRECTIONS = {
        "peaceful easy feeling": (
            "Peaceful Easy Feelin'",
            "top2000_672351ad-d2e8-4f8c-b2c2-52e4a6f2e8ad",
        ),
    }

    def __init__(self, cache: HttpCache, *, enrich_wikipedia: bool = False):
        super().__init__("t2000", cache)
        self.enrich_wikipedia = enrich_wikipedia
        self._wikipedia_parser: Top2000WikipediaParser | None = None

    @property
    def wikipedia_parser(self) -> Top2000WikipediaParser:
        """Lazily initialize Wikipedia parser only when needed."""
        if self._wikipedia_parser is None:
            self._wikipedia_parser = Top2000WikipediaParser(self.cache, self.client)
        return self._wikipedia_parser

    def _validate_year_available(self, year: int) -> None:
        """
        Check if the Top 2000 for the given year is likely available.

        The Top 2000 is broadcast in late December, so:
        - Past years are always available
        - Current year is available if we're past December 25
        - Future years are never available

        Raises:
            Top2000FutureYearError: If the requested year's list isn't available yet
        """
        now = datetime.now()
        current_year = now.year

        if year > current_year:
            raise Top2000FutureYearError(
                f"Top 2000 {year} doesn't exist yet. "
                f"The Top 2000 for year {year} won't be available until December {year}."
            )

        if year == current_year:
            # Check if we're past the broadcast date
            availability_date = datetime(year, self.AVAILABILITY_MONTH, self.AVAILABILITY_DAY)
            if now < availability_date:
                raise Top2000FutureYearError(
                    f"Top 2000 {year} isn't available yet. "
                    f"It will be broadcast around December 25, {year}."
                )

    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        """
        Scrape Top 2000 chart for a specific year.

        This is a convenience wrapper around scrape_rich() that returns
        the simple tuple format for backward compatibility.

        Args:
            period: Year in format YYYY (e.g., "2024", "1999")

        Returns:
            List of (rank, artist, title) tuples

        Raises:
            Top2000FutureYearError: If the requested year's list isn't available yet
        """
        return [entry.as_tuple() for entry in self.scrape_rich(period)]

    def scrape_rich(self, period: str) -> list[ScrapedEntry]:
        """
        Scrape Top 2000 chart with full metadata.

        This is the primary scrape method. It returns ScrapedEntry objects
        which include previous_position and weeks_on_chart when available.

        Args:
            period: Year in format YYYY (e.g., "2024", "1999")

        Returns:
            List of ScrapedEntry objects with full metadata

        Raises:
            Top2000FutureYearError: If the requested year's list isn't available yet
        """
        year = self._parse_year_period(period)

        # Guard against scraping future editions
        self._validate_year_available(year)

        # Try NPO API (captures previous_position)
        entries = self._try_api_rich(year)
        if entries:
            if self.enrich_wikipedia:
                entries = self._apply_wikipedia_enrichment(entries, year)
            return entries

        # Fallback to Wikipedia (no previous_position available)
        basic_entries = self._try_wikipedia_fallback(year)
        if basic_entries:
            entries = [
                ScrapedEntry(rank=rank, artist=artist, title=title)
                for rank, artist, title in basic_entries
            ]
            if self.enrich_wikipedia:
                entries = self._apply_wikipedia_enrichment(entries, year)
            return entries

        logger.warning(f"No data found for Top 2000 {year}")
        return []

    def _apply_wikipedia_enrichment(
        self, entries: list[ScrapedEntry], year: int
    ) -> list[ScrapedEntry]:
        """
        Enrich entries with Wikipedia links in-place.

        Adds artist_url, title_url, and history_url from the Dutch Wikipedia
        Top 2000 tables page when available.
        """
        for entry in entries:
            wiki_data = self.wikipedia_parser.get_enrichment(year, entry.rank)
            if wiki_data:
                entry.wikipedia_artist = wiki_data.artist_url
                entry.wikipedia_title = wiki_data.title_url
                entry.history_url = wiki_data.history_url
        return entries

    def _try_api_rich(self, year: int) -> list[ScrapedEntry]:
        """Try to fetch from NPO API and capture previous position if available."""
        # URL pattern changed in 2024
        if year < 2024:
            url = self.API_OLD_PATTERN.format(year=year)
        else:
            url = self.API_NEW_PATTERN.format(year=year)

        data = self._fetch_json(url)

        # Fallback to other pattern if primary fails
        if data is None:
            fallback_url = (
                self.API_NEW_PATTERN.format(year=year)
                if year < 2024
                else self.API_OLD_PATTERN.format(year=year)
            )
            data = self._fetch_json(fallback_url)

        if data is None:
            return []

        return self._parse_api_response_rich(data)

    def _parse_api_response_rich(self, data: Any) -> list[ScrapedEntry]:
        """Parse NPO API JSON response with full metadata."""
        entries: list[ScrapedEntry] = []

        if isinstance(data, dict):
            # NPO API uses "positions" key
            items = data.get(
                "positions", data.get("data", data.get("items", data.get("chart", [])))
            )
            if isinstance(items, dict):
                items = items.get("items", [])
        else:
            items = data

        if not isinstance(items, list):
            return []

        for item in items:
            if not isinstance(item, dict):
                continue

            # Handle nested structure: position.current and track.artist/track.title
            position_data = item.get("position", {})
            track_data = item.get("track", {})

            if position_data and track_data:
                # New NPO API format
                rank = position_data.get("current")
                artist = track_data.get("artist", "")
                title = track_data.get("title", "")
                prev_pos = position_data.get("previous")
            else:
                # Legacy format
                rank = item.get("position") or item.get("rank") or item.get("pos")
                artist = item.get("artist") or item.get("artistName") or ""
                title = item.get("title") or item.get("trackTitle") or item.get("name") or ""
                prev_pos = (
                    item.get("lastYearPosition")
                    or item.get("previousPosition")
                    or item.get("prev_position")
                    or item.get("lastYear")
                )

            if rank is None or not artist or not title:
                continue

            try:
                rank = int(rank)
            except (ValueError, TypeError):
                continue

            prev_pos_int: int | None = None
            if prev_pos is not None:
                try:
                    prev_pos_int = int(prev_pos)
                except (ValueError, TypeError):
                    pass

            artist = self._clean_text(str(artist))
            title = self._clean_text(str(title))
            title = self._apply_corrections(title)

            entries.append(
                ScrapedEntry(
                    rank=rank,
                    artist=artist,
                    title=title,
                    previous_position=prev_pos_int,
                )
            )

        return entries

    def _apply_corrections(self, title: str) -> str:
        """Apply hardcoded title corrections for known data issues."""
        title_lower = title.lower().strip()
        if title_lower in self.TITLE_CORRECTIONS:
            corrected_title, _ = self.TITLE_CORRECTIONS[title_lower]
            return corrected_title
        return title

    def get_corrected_uuid(self, title: str) -> str | None:
        """Get overridden UUID for titles with known data issues."""
        title_lower = title.lower().strip()
        if title_lower in self.TITLE_CORRECTIONS:
            _, uuid = self.TITLE_CORRECTIONS[title_lower]
            return uuid
        return None

    def _try_wikipedia_fallback(self, year: int) -> list[tuple[int, str, str]]:
        """
        Try Wikipedia fallback for older years.

        Uses the Lijst van Radio 2-Top 2000's page.
        """
        url = "https://nl.wikipedia.org/wiki/Lijst_van_Radio_2-Top_2000%27s"
        html = self._fetch_url(url)
        if html is None:
            return []

        return self._parse_wikipedia_html(html, year)

    def _parse_wikipedia_html(self, html: str, year: int) -> list[tuple[int, str, str]]:
        """
        Parse Wikipedia table for a specific year.

        This is a simplified parser - full implementation would need more
        robust table parsing.
        """
        entries: list[tuple[int, str, str]] = []

        try:

            class WikiTableParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.in_table = False
                    self.in_row = False
                    self.in_cell = False
                    self.current_row: list[str] = []
                    self.rows: list[list[str]] = []
                    self.cell_text = ""
                    self.year_column: int | None = None
                    self.headers: list[str] = []
                    self.in_header = False

                def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                    if tag == "table":
                        self.in_table = True
                    elif tag == "tr" and self.in_table:
                        self.in_row = True
                        self.current_row = []
                    elif tag in ("td", "th") and self.in_row:
                        self.in_cell = True
                        self.in_header = tag == "th"
                        self.cell_text = ""

                def handle_endtag(self, tag: str) -> None:
                    if tag == "table":
                        self.in_table = False
                    elif tag == "tr" and self.in_row:
                        self.in_row = False
                        if self.current_row:
                            self.rows.append(self.current_row)
                    elif tag in ("td", "th") and self.in_cell:
                        self.in_cell = False
                        text = self.cell_text.strip()
                        self.current_row.append(text)
                        if self.in_header:
                            self.headers.append(text)

                def handle_data(self, data: str) -> None:
                    if self.in_cell:
                        self.cell_text += data

            parser = WikiTableParser()
            parser.feed(html)

            year_col_idx = None
            for idx, header in enumerate(parser.headers):
                if str(year) in header:
                    year_col_idx = idx
                    break

            if year_col_idx is None:
                return []

            for row in parser.rows[1:]:
                if len(row) <= year_col_idx:
                    continue

                cell = row[year_col_idx]

                match = re.match(r"(\d+)\.\s*(.+?)\s*[-–]\s*(.+)", cell)
                if match:
                    rank = int(match.group(1))
                    artist = match.group(2).strip()
                    title = match.group(3).strip()
                    entries.append((rank, artist, title))

        except Exception as e:
            logger.error(f"Error parsing Wikipedia HTML: {e}")

        return entries


## Tests


def test_top2000_scraper_period_parsing():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        assert scraper._parse_year_period("2024") == 2024
        assert scraper._parse_year_period("1999") == 1999


def test_top2000_title_correction():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        corrected = scraper._apply_corrections("Peaceful Easy Feeling")
        assert corrected == "Peaceful Easy Feelin'"


def test_top2000_get_corrected_uuid():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        uuid = scraper.get_corrected_uuid("Peaceful Easy Feeling")
        assert uuid == "top2000_672351ad-d2e8-4f8c-b2c2-52e4a6f2e8ad"

        uuid = scraper.get_corrected_uuid("Some Other Title")
        assert uuid is None


def test_top2000_parse_api_response_rich():
    """Test parsing NPO API response returns ScrapedEntry objects."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        data = {
            "data": [
                {"position": 1, "artist": "Queen", "title": "Bohemian Rhapsody"},
                {"position": 2, "artist": "Eagles", "title": "Hotel California"},
            ]
        }
        entries = scraper._parse_api_response_rich(data)
        assert len(entries) == 2
        assert entries[0].rank == 1
        assert entries[0].artist == "Queen"
        assert entries[0].title == "Bohemian Rhapsody"
        assert entries[1].rank == 2
        assert entries[1].title == "Hotel California"


def test_top2000_parse_api_response_alternative_keys():
    """Test parsing with alternative key names (legacy API format)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        data = [
            {"rank": 1, "artistName": "Queen", "trackTitle": "Bohemian Rhapsody"},
        ]
        entries = scraper._parse_api_response_rich(data)
        assert len(entries) == 1
        assert entries[0].rank == 1
        assert entries[0].artist == "Queen"
        assert entries[0].title == "Bohemian Rhapsody"


def test_top2000_future_year_guard():
    """Test that scraping future years raises an error."""
    import tempfile
    from pathlib import Path

    import pytest

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        # Far future year should always fail
        with pytest.raises(Top2000FutureYearError, match="doesn't exist yet"):
            scraper.scrape("2099")

        # Past year should not raise (the validation step)
        # Note: This doesn't test the actual scrape, just the validation
        scraper._validate_year_available(2020)  # Should not raise


def test_top2000_validate_year_available_past():
    """Test that past years are always considered available."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        # Historical years should never raise
        scraper._validate_year_available(1999)
        scraper._validate_year_available(2010)
        scraper._validate_year_available(2020)
        scraper._validate_year_available(2023)
        scraper._validate_year_available(2024)


def test_top2000_enrich_wikipedia_flag():
    """Test that enrich_wikipedia flag initializes correctly."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")

        # Default is no enrichment
        scraper1 = Top2000Scraper(cache)
        assert scraper1.enrich_wikipedia is False

        # Explicit enrichment
        scraper2 = Top2000Scraper(cache, enrich_wikipedia=True)
        assert scraper2.enrich_wikipedia is True


def test_top2000_wikipedia_parser_lazy_init():
    """Test that Wikipedia parser is lazily initialized."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache, enrich_wikipedia=True)

        # Parser not initialized until accessed
        assert scraper._wikipedia_parser is None

        # Accessing property initializes it
        parser = scraper.wikipedia_parser
        assert parser is not None
        assert scraper._wikipedia_parser is parser

        # Second access returns same instance
        assert scraper.wikipedia_parser is parser


def test_top2000_apply_wikipedia_enrichment():
    """Test _apply_wikipedia_enrichment method."""
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock

    from chart_binder.scrapers.wikipedia_parser import WikipediaEnrichment

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache, enrich_wikipedia=True)

        # Mock the Wikipedia parser
        mock_parser = MagicMock()
        scraper._wikipedia_parser = mock_parser

        # Configure mock to return enrichment for rank 1, None for rank 2
        mock_parser.get_enrichment.side_effect = lambda year, rank: (
            WikipediaEnrichment(
                artist_url="https://nl.wikipedia.org/wiki/Queen_(band)",
                title_url="https://nl.wikipedia.org/wiki/Bohemian_Rhapsody",
                history_url="https://nl.wikipedia.org/wiki/Top_2000",
            )
            if rank == 1
            else None
        )

        entries = [
            ScrapedEntry(rank=1, artist="Queen", title="Bohemian Rhapsody"),
            ScrapedEntry(rank=2, artist="Eagles", title="Hotel California"),
        ]

        enriched = scraper._apply_wikipedia_enrichment(entries, 2024)

        assert len(enriched) == 2

        # First entry should have Wikipedia data
        assert enriched[0].wikipedia_artist == "https://nl.wikipedia.org/wiki/Queen_(band)"
        assert enriched[0].wikipedia_title == "https://nl.wikipedia.org/wiki/Bohemian_Rhapsody"
        assert enriched[0].history_url == "https://nl.wikipedia.org/wiki/Top_2000"

        # Second entry should not have Wikipedia data
        assert enriched[1].wikipedia_artist is None
        assert enriched[1].wikipedia_title is None
        assert enriched[1].history_url is None
