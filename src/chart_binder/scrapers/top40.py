from __future__ import annotations

import logging
import re

from chart_binder.http_cache import HttpCache
from chart_binder.scrapers.base import ChartScraper

logger = logging.getLogger(__name__)


class Top40Scraper(ChartScraper):
    """
    Scraper for Nederlandse Top 40 (weekly chart).

    Target: https://www.top40.nl
    """

    BASE_URL = "https://www.top40.nl"

    def __init__(self, cache: HttpCache):
        super().__init__("t40", cache)

    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        """
        Scrape Top 40 chart for a specific week.

        Args:
            period: Week in format YYYY-Www (e.g., "2024-W01", "1967-W07")

        Returns:
            List of (rank, artist, title) tuples
        """
        year, week = self._parse_period(period)
        url = f"{self.BASE_URL}/top40/{year}/week-{week:02d}"

        html = self._fetch_url(url)
        if html is None:
            logger.warning(f"Week {period} not found")
            return []

        return self._parse_html(html)

    def _parse_html(self, html: str) -> list[tuple[int, str, str]]:
        """Parse Top40 HTML page and extract chart entries."""
        try:
            entries: list[tuple[int, str, str]] = []
            parser = Top40Parser()
            parser.feed(html)
            raw_entries = parser.entries

            for entry in raw_entries:
                rank_val = entry.get("rank")
                artist_val = entry.get("artist", "")
                title_val = entry.get("title", "")

                if rank_val is None or not artist_val or not title_val:
                    continue

                if not isinstance(rank_val, int):
                    continue
                if not isinstance(artist_val, str):
                    continue
                if not isinstance(title_val, str):
                    continue

                split_entries = self._handle_split_entries(rank_val, artist_val, title_val)
                entries.extend(split_entries)

            return entries

        except Exception as e:
            logger.error(f"Error parsing Top40 HTML: {e}")
            return []

    def _handle_split_entries(
        self, rank: int, artist: str, title: str
    ) -> list[tuple[int, str, str]]:
        """
        Handle legacy split entries (double A-sides with multiple songs).

        Older charts often had multiple songs in one entry with `/` or `;` separators.
        """
        artist = self._clean_text(artist)
        title = self._clean_text(title)
        title = self._remove_double_parens(title)

        title = title.replace(";", "/")

        artists = [a.strip() for a in artist.split("/") if a.strip()]
        titles = [t.strip() for t in title.split("/") if t.strip()]

        if not artists or not titles:
            return [(rank, artist, title)]

        if len(artists) == 1 and len(titles) > 1:
            return [(rank, artists[0], t) for t in titles]

        if len(artists) > 1 and len(titles) > 1:
            if len(artists) == len(titles):
                return [(rank, a, t) for a, t in zip(artists, titles, strict=True)]
            else:
                return [(rank, artist, title)]

        return [(rank, artists[0], titles[0])]

    def extract_track_id(self, url: str) -> str | None:
        """
        Extract stable track ID from Top40 track URL.

        Pattern: `.*-(\\d+)` (e.g., `.../bad-guy-12345` -> ID `12345`)
        """
        match = re.search(r"-(\d+)$", url)
        return match.group(1) if match else None


class Top40Parser:
    """
    Simple HTML parser for Top40 chart pages.

    Uses standard library to avoid BeautifulSoup dependency.
    """

    def __init__(self):
        self.entries: list[dict[str, int | str]] = []
        self._current_entry: dict[str, int | str] = {}
        self._in_item = False
        self._in_position = False
        self._in_title = False
        self._in_artist = False
        self._parser = _Top40HTMLParser(self)

    def feed(self, html: str) -> None:
        """Feed HTML to parser."""
        self._parser.feed(html)


class _Top40HTMLParser:
    """Internal HTML parser implementation."""

    def __init__(self, parent: Top40Parser):
        self.parent = parent
        self._in_item = False
        self._in_position = False
        self._in_title = False
        self._in_artist = False
        self._current_entry: dict[str, int | str] = {}
        self._text_buffer = ""

    def feed(self, html: str) -> None:
        """Parse HTML and extract chart data."""
        from html.parser import HTMLParser

        class Parser(HTMLParser):
            def __init__(parser_self):  # pyright: ignore[reportSelfClsParameterName]
                super().__init__()
                parser_self.outer = self

            def handle_starttag(
                parser_self,  # pyright: ignore[reportSelfClsParameterName]
                tag: str,
                attrs: list[tuple[str, str | None]],
            ) -> None:
                self._handle_starttag(tag, attrs)

            def handle_endtag(parser_self, tag: str) -> None:  # pyright: ignore[reportSelfClsParameterName]
                self._handle_endtag(tag)

            def handle_data(parser_self, data: str) -> None:  # pyright: ignore[reportSelfClsParameterName]
                self._handle_data(data)

        parser = Parser()
        parser.feed(html)

    def _handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: v for k, v in attrs if v is not None}
        class_attr = attr_dict.get("class", "")

        if "top40-list__item" in class_attr or (tag == "li" and "list-item" in class_attr):
            self._in_item = True
            self._current_entry = {}

        if self._in_item:
            if "top40-list__position" in class_attr or "position" in class_attr:
                self._in_position = True
                self._text_buffer = ""
            elif "top40-list__title" in class_attr or "title" in class_attr:
                self._in_title = True
                self._text_buffer = ""
            elif "top40-list__artist" in class_attr or "artist" in class_attr:
                self._in_artist = True
                self._text_buffer = ""

    def _handle_endtag(self, tag: str) -> None:
        if self._in_position:
            self._in_position = False
            try:
                rank = int(re.sub(r"\D", "", self._text_buffer))
                self._current_entry["rank"] = rank
            except ValueError:
                pass

        if self._in_title:
            self._in_title = False
            self._current_entry["title"] = self._text_buffer.strip()

        if self._in_artist:
            self._in_artist = False
            self._current_entry["artist"] = self._text_buffer.strip()

        if self._in_item and tag in ("li", "div"):
            if "rank" in self._current_entry:
                self.parent.entries.append(self._current_entry.copy())
            self._in_item = False
            self._current_entry = {}

    def _handle_data(self, data: str) -> None:
        if self._in_position or self._in_title or self._in_artist:
            self._text_buffer += data


## Tests


def test_top40_scraper_period_parsing():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        year, week = scraper._parse_period("1967-W07")
        assert year == 1967
        assert week == 7


def test_top40_split_entries_single_artist_multiple_titles():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        entries = scraper._handle_split_entries(1, "The Beatles", "Penny Lane / Strawberry Fields")
        assert len(entries) == 2
        assert entries[0] == (1, "The Beatles", "Penny Lane")
        assert entries[1] == (1, "The Beatles", "Strawberry Fields")


def test_top40_split_entries_multiple_artists_multiple_titles():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        entries = scraper._handle_split_entries(1, "Artist A / Artist B", "Song A / Song B")
        assert len(entries) == 2
        assert entries[0] == (1, "Artist A", "Song A")
        assert entries[1] == (1, "Artist B", "Song B")


def test_top40_split_entries_semicolon_delimiter():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        entries = scraper._handle_split_entries(
            1, "Queen", "We Will Rock You; We Are The Champions"
        )
        assert len(entries) == 2


def test_top40_extract_track_id():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        assert scraper.extract_track_id("/song/bad-guy-12345") == "12345"
        assert scraper.extract_track_id("/song/title") is None


def test_top40_remove_double_parens():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = HttpCache(Path(tmpdir) / "cache")
        scraper = Top40Scraper(cache)

        entries = scraper._handle_split_entries(1, "Artist", "Title ((metadata))")
        assert entries[0][2] == "Title"
