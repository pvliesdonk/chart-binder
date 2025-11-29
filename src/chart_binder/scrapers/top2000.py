from __future__ import annotations

import logging
import re
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

from chart_binder.http_cache import HttpCache
from chart_binder.scrapers.base import ChartScraper, ScrapedEntry

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

    def __init__(self, cache: HttpCache):
        super().__init__("t2000", cache)

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

        Args:
            period: Year in format YYYY (e.g., "2024", "1999")

        Returns:
            List of (rank, artist, title) tuples

        Raises:
            Top2000FutureYearError: If the requested year's list isn't available yet
        """
        year = self._parse_year_period(period)

        # Guard against scraping future editions
        self._validate_year_available(year)

        entries = self._try_api(year)
        if entries:
            return entries

        entries = self._try_wikipedia_fallback(year)
        if entries:
            return entries

        logger.warning(f"No data found for Top 2000 {year}")
        return []

    def scrape_rich(self, period: str) -> list[ScrapedEntry]:
        """
        Scrape Top 2000 chart with full metadata.

        The NPO API may include lastYearPosition for cross-reference.

        Raises:
            Top2000FutureYearError: If the requested year's list isn't available yet
        """
        year = self._parse_year_period(period)

        # Guard against scraping future editions
        self._validate_year_available(year)

        # Try API with rich data
        entries = self._try_api_rich(year)
        if entries:
            return entries

        # Fallback to basic entries
        basic = self.scrape(period)
        return [
            ScrapedEntry(rank=rank, artist=artist, title=title) for rank, artist, title in basic
        ]

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

    def _try_api(self, year: int) -> list[tuple[int, str, str]]:
        """Try to fetch from NPO API."""
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

        return self._parse_api_response(data)

    def _parse_api_response(self, data: Any) -> list[tuple[int, str, str]]:
        """Parse NPO API JSON response."""
        entries: list[tuple[int, str, str]] = []

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
            else:
                # Legacy format
                rank = item.get("position") or item.get("rank") or item.get("pos")
                artist = item.get("artist") or item.get("artistName") or ""
                title = item.get("title") or item.get("trackTitle") or item.get("name") or ""

            if rank is None or not artist or not title:
                continue

            try:
                rank = int(rank)
            except (ValueError, TypeError):
                continue

            artist = self._clean_text(str(artist))
            title = self._clean_text(str(title))

            title = self._apply_corrections(title)

            entries.append((rank, artist, title))

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


def test_top2000_parse_api_response():
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
        entries = scraper._parse_api_response(data)
        assert len(entries) == 2
        assert entries[0] == (1, "Queen", "Bohemian Rhapsody")
        assert entries[1] == (2, "Eagles", "Hotel California")


def test_top2000_parse_api_response_alternative_keys():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top2000Scraper(cache)

        data = [
            {"rank": 1, "artistName": "Queen", "trackTitle": "Bohemian Rhapsody"},
        ]
        entries = scraper._parse_api_response(data)
        assert len(entries) == 1
        assert entries[0] == (1, "Queen", "Bohemian Rhapsody")


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
