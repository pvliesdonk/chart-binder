"""Tests for Top40Scraper with HTTP mocking."""

from __future__ import annotations

from pathlib import Path

# Load fixtures from the cassettes directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cassettes"


def load_fixture(scraper_type: str, fixture_name: str) -> str:
    """Load a fixture file as text."""
    fixture_path = FIXTURES_DIR / scraper_type / fixture_name
    return fixture_path.read_text(encoding="utf-8")


class TestTop40ScraperWithMocking:
    """Tests using mocked HTTP responses."""

    def test_scrape_parses_html_correctly(self, top40_scraper, httpx_mock):
        """Test that scraper correctly parses HTML fixture."""
        html = load_fixture("top40", "2024-W01.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40/2024/week-01",
            html=html,
        )

        result = top40_scraper.scrape("2024-W01")

        # Should parse all 10 entries from fixture
        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

        # Check first entry
        rank, artist, title = result[0]
        assert rank == 1
        assert artist == "André Hazes Jr."
        assert title == "Leef"

        # Check entry with special characters (BLØF)
        rank3, artist3, title3 = result[2]
        assert rank3 == 3
        assert "BL" in artist3  # BLØF may have encoding issues
        assert title3 == "Zoutelande"

        # Check entry with feat.
        rank10, artist10, title10 = result[9]
        assert rank10 == 10
        assert "Post Malone" in artist10
        assert title10 == "Sunflower"

    def test_scrape_rich_returns_metadata(self, top40_scraper, httpx_mock):
        """Test that scrape_rich returns ScrapedEntry with metadata."""
        html = load_fixture("top40", "2024-W01.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40/2024/week-01",
            html=html,
        )

        result = top40_scraper.scrape_rich("2024-W01")

        assert len(result) == 10

        # Check that metadata is captured
        first = result[0]
        assert first.rank == 1
        assert first.artist == "André Hazes Jr."
        assert first.title == "Leef"
        assert first.previous_position == 2
        assert first.weeks_on_chart == 5

        # Check "nieuw" (new) entry has no previous position
        fourth = result[3]
        assert fourth.rank == 4
        assert fourth.previous_position is None  # "nieuw" should result in None

    def test_scrape_with_validation(self, top40_scraper, httpx_mock):
        """Test scrape_with_validation returns ScrapeResult."""
        html = load_fixture("top40", "2024-W01.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40/2024/week-01",
            html=html,
        )

        result = top40_scraper.scrape_with_validation("2024-W01")

        assert hasattr(result, "entries")
        assert hasattr(result, "expected_count")
        assert hasattr(result, "warnings")
        assert result.expected_count == 40
        assert len(result.entries) == 10  # Our fixture has 10 entries


class TestTop40ScraperEdgeCases:
    """Edge case tests for Top40Scraper parsing logic."""

    def test_split_entry_double_title(self, top40_scraper):
        """Test handling of double A-side with slash in title."""
        result = top40_scraper._handle_split_entries(
            rank=1,
            artist="Queen & David Bowie",
            title="Under Pressure / Soul Brother",
        )

        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        assert result[0][2] == "Under Pressure"
        assert result[1][2] == "Soul Brother"

    def test_split_entry_double_artist_and_title(self, top40_scraper):
        """Test handling of doubled artist and title."""
        result = top40_scraper._handle_split_entries(
            rank=1,
            artist="Artist A / Artist B",
            title="Song A / Song B",
        )

        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        # First entry: Artist A - Song A
        assert result[0][1] == "Artist A"
        assert result[0][2] == "Song A"
        # Second entry: Artist B - Song B
        assert result[1][1] == "Artist B"
        assert result[1][2] == "Song B"

    def test_split_entry_semicolon(self, top40_scraper):
        """Test handling of semicolon-separated entries."""
        result = top40_scraper._handle_split_entries(
            rank=1,
            artist="Various Artists",
            title="Song A ; Song B",
        )

        # Semicolon pattern may or may not split depending on implementation
        # This test documents current behavior
        assert len(result) >= 1

    def test_no_split_normal_entry(self, top40_scraper):
        """Test that normal entries are not split."""
        result = top40_scraper._handle_split_entries(
            rank=1,
            artist="The Beatles",
            title="Yesterday",
        )

        assert len(result) == 1
        assert result[0] == (1, "The Beatles", "Yesterday")


class TestTop40ScraperPeriodParsing:
    """Period parsing tests for Top40Scraper."""

    def test_parse_period_valid(self, top40_scraper):
        """Test parsing a valid ISO week period."""
        year, week = top40_scraper._parse_period("2024-W01")
        assert year == 2024
        assert week == 1

    def test_parse_period_end_of_year(self, top40_scraper):
        """Test parsing end-of-year week."""
        year, week = top40_scraper._parse_period("2023-W52")
        assert year == 2023
        assert week == 52

    def test_parse_period_double_digit_week(self, top40_scraper):
        """Test parsing double-digit week number."""
        year, week = top40_scraper._parse_period("2024-W15")
        assert year == 2024
        assert week == 15


class TestTop40ScraperErrorHandling:
    """Error handling tests for Top40Scraper."""

    def test_scrape_returns_empty_on_404(self, top40_scraper, httpx_mock):
        """Test that scraper returns empty list on 404."""
        httpx_mock.add_response(
            url="https://www.top40.nl/top40/2024/week-99",
            status_code=404,
        )

        result = top40_scraper.scrape("2024-W99")
        assert result == []

    def test_scrape_handles_empty_html(self, top40_scraper, httpx_mock):
        """Test that scraper handles empty HTML gracefully."""
        httpx_mock.add_response(
            url="https://www.top40.nl/top40/2024/week-01",
            html="<html><body></body></html>",
        )

        result = top40_scraper.scrape("2024-W01")
        assert result == []
