from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from chart_binder.http_cache import HttpCache
from chart_binder.scrapers.base import ChartScraper

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

    def _parse_html(self, html: str, year: int) -> list[tuple[int, str, str]]:
        """Parse De Zwaarste Lijst HTML page."""
        entries: list[tuple[int, str, str]] = []

        entries = self._try_parse_ordered_list(html)
        if entries:
            return entries

        entries = self._try_parse_table(html)
        if entries:
            return entries

        entries = self._try_parse_text_lines(html)
        if entries:
            return entries

        logger.warning("Could not parse De Zwaarste Lijst HTML with any strategy")
        return []

    def _try_parse_ordered_list(self, html: str) -> list[tuple[int, str, str]]:
        """Try parsing as ordered list (<ol> with <li> items)."""
        from html.parser import HTMLParser

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
        from html.parser import HTMLParser

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

    def _try_parse_text_lines(self, html: str) -> list[tuple[int, str, str]]:
        """Try parsing raw text lines with regex."""
        from html.parser import HTMLParser

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
