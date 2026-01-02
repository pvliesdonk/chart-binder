from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from html.parser import HTMLParser

from chart_binder.http_cache import HttpCache
from chart_binder.scrapers.base import ChartScraper, ScrapedEntry

logger = logging.getLogger(__name__)


class ZwaarsteScraper(ChartScraper):
    """
    Scraper for De Zwaarste Lijst (Studio Brussel).

    URL changes every year (blog posts), so this scraper requires
    a config-driven URL map.
    """

    DEFAULT_URL_MAP: dict[int, str] = {
        2024: "https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
    }
    chart_db_id = "nl_538_zwaarste"
    expected_entry_count = 150

    def __init__(self, cache: HttpCache, url_map: Mapping[int, str] | None = None):
        """
        Initialize scraper with optional URL override map.

        Args:
            cache: HTTP cache instance
            url_map: Override mapping of year -> URL. Merged with defaults.
        """
        super().__init__("zwaar", cache)
        self.url_map = dict(self.DEFAULT_URL_MAP)
        if url_map:
            self.url_map.update(url_map)

    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        """
        Scrape De Zwaarste Lijst for a specific year.

        Args:
            period: Year in format YYYY (e.g., "2024")

        Returns:
            List of (rank, artist, title) tuples
        """
        year = self._parse_year_period(period)

        if year not in self.url_map:
            logger.warning(f"No URL configured for De Zwaarste Lijst {year}")
            return []

        url = self.url_map[year]
        html = self._fetch_url(url)
        if html is None:
            logger.warning(f"Failed to fetch De Zwaarste Lijst {year} from {url}")
            return []

        return self._parse_html(html, year)

    def scrape_rich(self, period: str) -> list[ScrapedEntry]:
        """
        Scrape De Zwaarste Lijst with full metadata.

        For table format (2009+), captures previous_position when available.
        For list format (blog posts), returns basic ScrapedEntry.
        """
        year = self._parse_year_period(period)

        if year not in self.url_map:
            logger.warning(f"No URL configured for De Zwaarste Lijst {year}")
            return []

        url = self.url_map[year]
        html = self._fetch_url(url)
        if html is None:
            logger.warning(f"Failed to fetch De Zwaarste Lijst {year} from {url}")
            return []

        return self._parse_html_rich(html, year)

    def _parse_html(self, html: str, year: int) -> list[tuple[int, str, str]]:
        """Parse De Zwaarste Lijst HTML page."""
        rich_entries = self._parse_html_rich(html, year)
        return [(e.rank, e.artist, e.title) for e in rich_entries]

    def _parse_html_rich(self, html: str, year: int) -> list[ScrapedEntry]:
        """Parse De Zwaarste Lijst HTML page with full metadata."""
        # Try table format first (captures previous_position for 2010+)
        entries = self._try_parse_table_rich(html, year)
        if entries:
            return entries

        # Fall back to ordered list format (no previous_position)
        tuples = self._try_parse_ordered_list(html)
        if tuples:
            return [
                ScrapedEntry(rank=rank, artist=artist, title=title)
                for rank, artist, title in tuples
            ]

        # Fall back to text lines (no previous_position)
        tuples = self._try_parse_text_lines(html)
        if tuples:
            return [
                ScrapedEntry(rank=rank, artist=artist, title=title)
                for rank, artist, title in tuples
            ]

        logger.warning("Could not parse De Zwaarste Lijst HTML with any strategy")
        return []

    def _try_parse_ordered_list(self, html: str) -> list[tuple[int, str, str]]:
        """Try parsing as ordered list (<ol> with <li> items)."""

        class OLParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_ol = False
                self.in_li = False
                self.in_article = False
                self.in_text_div = False
                self.items: list[str] = []
                self.current_text = ""

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                attr_dict = {k: v for k, v in attrs if v is not None}
                class_attr = attr_dict.get("class", "")

                if tag == "article":
                    self.in_article = True
                if tag == "div" and "text" in class_attr:
                    self.in_text_div = True

                if tag == "ol" and (self.in_article or self.in_text_div or not self.in_ol):
                    self.in_ol = True
                elif tag == "li" and self.in_ol:
                    self.in_li = True
                    self.current_text = ""

            def handle_endtag(self, tag: str) -> None:
                if tag == "article":
                    self.in_article = False
                if tag == "div":
                    self.in_text_div = False
                if tag == "ol":
                    self.in_ol = False
                elif tag == "li" and self.in_li:
                    self.in_li = False
                    if self.current_text.strip():
                        self.items.append(self.current_text.strip())

            def handle_data(self, data: str) -> None:
                if self.in_li:
                    self.current_text += data

        parser = OLParser()
        parser.feed(html)

        return self._parse_list_items(parser.items)

    def _try_parse_table(self, html: str) -> list[tuple[int, str, str]]:
        """Try parsing as table."""

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_table = False
                self.in_row = False
                self.in_cell = False
                self.rows: list[list[str]] = []
                self.current_row: list[str] = []
                self.cell_text = ""

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "table":
                    self.in_table = True
                elif tag == "tr" and self.in_table:
                    self.in_row = True
                    self.current_row = []
                elif tag in ("td", "th") and self.in_row:
                    self.in_cell = True
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
                    self.current_row.append(self.cell_text.strip())

            def handle_data(self, data: str) -> None:
                if self.in_cell:
                    self.cell_text += data

        parser = TableParser()
        parser.feed(html)

        entries: list[tuple[int, str, str]] = []
        for row in parser.rows:
            if len(row) >= 3:
                try:
                    rank = int(re.sub(r"\D", "", row[0]))
                    artist = self._clean_text(row[1])
                    title = self._clean_text(row[2])
                    if artist and title:
                        entries.append((rank, artist, title))
                except ValueError:
                    continue

        return entries

    def _try_parse_table_rich(self, html: str, year: int) -> list[ScrapedEntry]:
        """
        Try parsing as table with year-specific format detection.

        Format variations:
        - 2009: 3 columns (Position | Artist | Title)
        - 2010+: 4 columns (Position | Previous | Artist | Title)

        Previous position indicators:
        - (123) → Previous position was 123
        - (-) → New entry
        - (re) → Re-entry
        """

        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_table = False
                self.in_row = False
                self.in_cell = False
                self.in_header = False
                self.rows: list[list[str]] = []
                self.current_row: list[str] = []
                self.cell_text = ""
                self._row_has_data = False

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag == "table":
                    self.in_table = True
                elif tag == "tr" and self.in_table:
                    self.in_row = True
                    self.current_row = []
                    self._row_has_data = False
                elif tag in ("td", "th") and self.in_row:
                    self.in_cell = True
                    self.in_header = tag == "th"
                    if tag == "td":
                        self._row_has_data = True
                    self.cell_text = ""

            def handle_endtag(self, tag: str) -> None:
                if tag == "table":
                    self.in_table = False
                elif tag == "tr" and self.in_row:
                    self.in_row = False
                    # Only add rows with data cells (skip header-only rows)
                    if self.current_row and self._row_has_data:
                        self.rows.append(self.current_row)
                elif tag in ("td", "th") and self.in_cell:
                    self.in_cell = False
                    self.current_row.append(self.cell_text.strip())

            def handle_data(self, data: str) -> None:
                if self.in_cell:
                    self.cell_text += data

        parser = TableParser()
        parser.feed(html)

        if not parser.rows:
            return []

        entries: list[ScrapedEntry] = []

        # Detect format based on column count of first data row
        # 2009 format: 3 columns (Position | Artist | Title)
        # 2010+ format: 4 columns (Position | Previous | Artist | Title)
        first_row_cols = len(parser.rows[0]) if parser.rows else 0
        is_four_column = first_row_cols >= 4 or year >= 2010

        for row in parser.rows:
            try:
                if is_four_column and len(row) >= 4:
                    # 4-column format: Position | Previous | Artist | Title
                    rank = int(re.sub(r"\D", "", row[0]))
                    prev_pos = self._parse_previous_position(row[1])
                    artist = self._clean_text(row[2])
                    title = self._clean_text(row[3])
                elif len(row) >= 3:
                    # 3-column format: Position | Artist | Title
                    rank = int(re.sub(r"\D", "", row[0]))
                    prev_pos = None
                    artist = self._clean_text(row[1])
                    title = self._clean_text(row[2])
                else:
                    continue

                if artist and title:
                    entries.append(
                        ScrapedEntry(
                            rank=rank,
                            artist=artist,
                            title=title,
                            previous_position=prev_pos,
                        )
                    )
            except ValueError:
                continue

        return entries

    def _parse_previous_position(self, text: str) -> int | None:
        """
        Parse previous position indicator.

        Formats:
        - (123) or 123 → Previous position was 123
        - (-) or - → New entry (returns None)
        - (re) or re → Re-entry (returns None)
        - (nieuw) → New entry (returns None)
        - empty → No data (returns None)
        """
        text = text.strip().lower()

        # Handle empty or dash indicators
        if not text or text in ("-", "(-)", "nieuw", "(nieuw)", "new", "(new)"):
            return None

        # Handle re-entry indicator
        if text in ("re", "(re)", "re-entry", "(re-entry)"):
            return None

        # Try to extract numeric position
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())

        return None

    def _try_parse_text_lines(self, html: str) -> list[tuple[int, str, str]]:
        """Try parsing raw text lines with regex."""

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.lines: list[str] = []
                self.in_content = False
                self.in_article = False
                self.in_p = False
                self.current_text = ""

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                attr_dict = {k: v for k, v in attrs if v is not None}
                class_attr = attr_dict.get("class", "")

                if tag == "article":
                    self.in_article = True
                if tag == "div" and ("text" in class_attr or "content" in class_attr):
                    self.in_content = True
                if tag == "p" and (self.in_article or self.in_content):
                    self.in_p = True
                    self.current_text = ""

            def handle_endtag(self, tag: str) -> None:
                if tag == "article":
                    self.in_article = False
                if tag == "div":
                    self.in_content = False
                if tag == "p" and self.in_p:
                    self.in_p = False
                    if self.current_text.strip():
                        self.lines.append(self.current_text.strip())

            def handle_data(self, data: str) -> None:
                if self.in_p:
                    self.current_text += data

        parser = TextExtractor()
        parser.feed(html)

        return self._parse_list_items(parser.lines)

    def _parse_list_items(self, items: list[str]) -> list[tuple[int, str, str]]:
        """Parse list items into (rank, artist, title) tuples."""
        entries: list[tuple[int, str, str]] = []

        pattern = re.compile(r"^(\d+)\.?\s+(.+?)\s+[-–—]\s+(.+)$")

        for item in items:
            item = self._clean_text(item)
            match = pattern.match(item)
            if match:
                rank = int(match.group(1))
                artist = match.group(2).strip()
                title = match.group(3).strip()
                entries.append((rank, artist, title))

        return entries

    def generate_entry_id(self, year: int, rank: int, artist: str, title: str) -> str:
        """
        Generate deterministic ID for an entry.

        Since De Zwaarste Lijst has no stable IDs, we generate a hash.
        """
        return self._generate_hash_id("zwaar", str(year), str(rank), artist, title)


## Tests


def test_zwaarste_scraper_period_parsing():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = ZwaarsteScraper(cache)

        assert scraper._parse_year_period("2024") == 2024


def test_zwaarste_url_map_override():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        custom_urls = {
            2023: "https://example.com/zwaarste-2023",
        }
        scraper = ZwaarsteScraper(cache, url_map=custom_urls)

        assert 2024 in scraper.url_map
        assert 2023 in scraper.url_map
        assert scraper.url_map[2023] == "https://example.com/zwaarste-2023"


def test_zwaarste_parse_list_items():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = ZwaarsteScraper(cache)

        items = [
            "1. Metallica - Master Of Puppets",
            "2. Iron Maiden - The Trooper",
            "3  Slayer – Raining Blood",
        ]
        entries = scraper._parse_list_items(items)
        assert len(entries) == 3
        assert entries[0] == (1, "Metallica", "Master Of Puppets")
        assert entries[1] == (2, "Iron Maiden", "The Trooper")
        assert entries[2] == (3, "Slayer", "Raining Blood")


def test_zwaarste_generate_entry_id():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = ZwaarsteScraper(cache)

        id1 = scraper.generate_entry_id(2024, 1, "Metallica", "Master Of Puppets")
        id2 = scraper.generate_entry_id(2024, 1, "Metallica", "Master Of Puppets")
        id3 = scraper.generate_entry_id(2024, 2, "Metallica", "Master Of Puppets")

        assert id1 == id2
        assert id1 != id3


def test_zwaarste_missing_year():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = ZwaarsteScraper(cache)

        entries = scraper.scrape("1990")
        assert entries == []
