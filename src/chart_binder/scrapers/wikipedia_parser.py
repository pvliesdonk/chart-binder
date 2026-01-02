"""Wikipedia parser for Top 2000 enrichment data.

Parses the Dutch Wikipedia page "Lijst van Radio 2-Top 2000's" to extract
artist and song Wikipedia links for chart entries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx

from chart_binder.http_cache import HttpCache

logger = logging.getLogger(__name__)

WIKIPEDIA_BASE = "https://nl.wikipedia.org"
TOP2000_WIKI_URL = "https://nl.wikipedia.org/wiki/Lijst_van_Radio_2-Top_2000%27s"


@dataclass
class WikipediaEnrichment:
    """Wikipedia enrichment data for a chart entry."""

    artist_url: str | None = None
    title_url: str | None = None
    history_url: str | None = None


class Top2000WikipediaParser:
    """
    Parser for Top 2000 Wikipedia tables.

    Extracts artist and song Wikipedia links from the Dutch Wikipedia page
    that lists historical Top 2000 positions.
    """

    def __init__(self, cache: HttpCache, client: httpx.Client):
        self.cache = cache
        self.client = client
        self._parsed_data: dict[int, dict[int, WikipediaEnrichment]] = {}

    def get_enrichment(self, year: int, rank: int) -> WikipediaEnrichment | None:
        """
        Get Wikipedia enrichment data for a specific entry.

        Args:
            year: The Top 2000 year (e.g., 2024)
            rank: The chart position (1-2000)

        Returns:
            WikipediaEnrichment with URLs, or None if not found
        """
        if year not in self._parsed_data:
            self._parse_wikipedia_page(year)

        return self._parsed_data.get(year, {}).get(rank)

    def _parse_wikipedia_page(self, year: int) -> None:
        """Parse the Wikipedia page and cache results for the given year."""
        html = self._fetch_page()
        if html is None:
            self._parsed_data[year] = {}
            return

        enrichments = self._extract_enrichments(html, year)
        self._parsed_data[year] = enrichments

    def _fetch_page(self) -> str | None:
        """Fetch the Wikipedia page HTML, using cache first."""
        cached_response = self.cache.get(TOP2000_WIKI_URL)
        if cached_response is not None:
            if cached_response.status_code == 404:
                return None
            return cached_response.text

        try:
            response = self.client.get(TOP2000_WIKI_URL)
            self.cache.put(TOP2000_WIKI_URL, response)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Wikipedia page: {e}")
            return None
        except httpx.RequestError as e:
            logger.error(f"Request error fetching Wikipedia page: {e}")
            return None

    def _extract_enrichments(
        self, html: str, year: int
    ) -> dict[int, WikipediaEnrichment]:
        """Extract enrichment data from Wikipedia HTML for a specific year."""
        enrichments: dict[int, WikipediaEnrichment] = {}

        try:
            parser = _WikiTableParser()
            parser.feed(html)

            # Find the year column index
            year_col_idx = None
            for idx, header in enumerate(parser.headers):
                if str(year) in header:
                    year_col_idx = idx
                    break

            if year_col_idx is None:
                logger.warning(f"Year {year} not found in Wikipedia table headers")
                return enrichments

            # Process rows to extract links
            for row in parser.rows:
                if len(row) <= year_col_idx:
                    continue

                cell = row[year_col_idx]
                if not cell.get("text"):
                    continue

                # Parse rank from cell text (format: "1. Artist - Title" or just "1")
                text = cell["text"].strip()
                rank_match = re.match(r"(\d+)", text)
                if not rank_match:
                    continue

                rank = int(rank_match.group(1))
                if rank < 1 or rank > 2000:
                    continue

                # Extract links from cell
                links = cell.get("links", [])
                artist_url = None
                title_url = None

                # First link is typically artist, second is title
                for i, link in enumerate(links[:2]):
                    if link.startswith("/wiki/"):
                        full_url = WIKIPEDIA_BASE + link
                        if i == 0:
                            artist_url = full_url
                        else:
                            title_url = full_url

                if artist_url or title_url:
                    enrichments[rank] = WikipediaEnrichment(
                        artist_url=artist_url,
                        title_url=title_url,
                        history_url=TOP2000_WIKI_URL,
                    )

        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Wikipedia HTML: {e}")

        return enrichments


class _WikiTableParser(HTMLParser):
    """Internal HTML parser for Wikipedia tables."""

    def __init__(self):
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.in_header = False
        self.current_row: list[dict] = []
        self.rows: list[list[dict]] = []
        self.headers: list[str] = []
        self.current_cell: dict = {}
        self.cell_text = ""
        self.current_links: list[str] = []
        self._row_has_data_cells = False  # Track if row has <td> cells

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)

        if tag == "table":
            # Look for wikitable class
            class_attr = attrs_dict.get("class", "")
            if "wikitable" in class_attr:
                self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_row = []
            self._row_has_data_cells = False
        elif tag in ("td", "th") and self.in_row:
            if tag == "td":
                self._row_has_data_cells = True
            self.in_cell = True
            self.in_header = tag == "th"
            self.cell_text = ""
            self.current_links = []
        elif tag == "a" and self.in_cell:
            href = attrs_dict.get("href", "")
            if href and href.startswith("/wiki/") and ":" not in href:
                # Skip Wikipedia special pages (File:, Category:, etc.)
                self.current_links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self.in_table:
            self.in_table = False
        elif tag == "tr" and self.in_row:
            self.in_row = False
            # Only add rows that contain at least one <td> cell (not header-only rows)
            if self.current_row and self._row_has_data_cells:
                self.rows.append(self.current_row)
        elif tag in ("td", "th") and self.in_cell:
            self.in_cell = False
            text = self.cell_text.strip()
            cell_data = {"text": text, "links": self.current_links.copy()}
            self.current_row.append(cell_data)
            if self.in_header:
                self.headers.append(text)
            self.in_header = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell_text += data


## Tests


def test_wikipedia_enrichment_dataclass():
    """Test WikipediaEnrichment dataclass defaults."""
    enrichment = WikipediaEnrichment()
    assert enrichment.artist_url is None
    assert enrichment.title_url is None
    assert enrichment.history_url is None

    enrichment = WikipediaEnrichment(
        artist_url="https://nl.wikipedia.org/wiki/Queen_(band)",
        title_url="https://nl.wikipedia.org/wiki/Bohemian_Rhapsody",
        history_url=TOP2000_WIKI_URL,
    )
    assert "Queen" in enrichment.artist_url
    assert "Bohemian" in enrichment.title_url


def test_wiki_table_parser_basic():
    """Test basic HTML table parsing."""
    html = """
    <table class="wikitable">
        <tr><th>2024</th><th>2023</th></tr>
        <tr>
            <td>1. <a href="/wiki/Queen_(band)">Queen</a> -
                <a href="/wiki/Bohemian_Rhapsody">Bohemian Rhapsody</a></td>
            <td>2</td>
        </tr>
    </table>
    """
    parser = _WikiTableParser()
    parser.feed(html)

    assert "2024" in parser.headers
    assert len(parser.rows) == 1
    assert parser.rows[0][0]["text"].startswith("1.")
    assert len(parser.rows[0][0]["links"]) == 2
    assert "/wiki/Queen_(band)" in parser.rows[0][0]["links"]
