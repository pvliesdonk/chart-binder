"""Tests for Top2000Scraper with HTTP mocking."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from freezegun import freeze_time

from chart_binder.scrapers.top2000 import Top2000FutureYearError

# Load fixtures from the cassettes directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cassettes"


def load_fixture(scraper_type: str, fixture_name: str) -> str:
    """Load a fixture file as text."""
    fixture_path = FIXTURES_DIR / scraper_type / fixture_name
    return fixture_path.read_text(encoding="utf-8")


class TestTop2000ScraperWithMocking:
    """Tests using mocked HTTP responses."""

    @freeze_time("2025-01-01")
    def test_scrape_parses_json_correctly(self, top2000_scraper, httpx_mock):
        """Test that scraper correctly parses JSON fixture."""
        json_data = load_fixture("top2000", "2024.json")
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2024-12-25",
            json=json.loads(json_data),
        )

        result = top2000_scraper.scrape("2024")

        # Should parse all 10 entries from fixture
        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

        # Check first entry
        rank, artist, title = result[0]
        assert rank == 1
        assert artist == "Queen"
        assert title == "Bohemian Rhapsody"

        # Check last entry
        rank10, artist10, title10 = result[9]
        assert rank10 == 10
        assert artist10 == "Eagles"

    @freeze_time("2025-01-01")
    def test_scrape_rich_returns_metadata(self, top2000_scraper, httpx_mock):
        """Test that scrape_rich returns ScrapedEntry with metadata."""
        json_data = load_fixture("top2000", "2024.json")
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2024-12-25",
            json=json.loads(json_data),
        )

        result = top2000_scraper.scrape_rich("2024")

        assert len(result) == 10

        # Check that metadata is captured
        first = result[0]
        assert first.rank == 1
        assert first.artist == "Queen"
        assert first.title == "Bohemian Rhapsody"
        assert first.previous_position == 2  # From fixture

        # Check entry with null previous position
        sixth = result[5]
        assert sixth.rank == 6
        assert sixth.previous_position is None  # Was null in fixture

    @freeze_time("2025-01-01")
    def test_scrape_with_validation(self, top2000_scraper, httpx_mock):
        """Test scrape_with_validation returns ScrapeResult."""
        json_data = load_fixture("top2000", "2024.json")
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2024-12-25",
            json=json.loads(json_data),
        )

        result = top2000_scraper.scrape_with_validation("2024")

        assert hasattr(result, "entries")
        assert hasattr(result, "expected_count")
        assert hasattr(result, "warnings")
        assert result.expected_count == 2000  # Top 2000 expected entries
        assert len(result.entries) == 10  # Our fixture has 10 entries


class TestTop2000TitleCorrection:
    """Tests for title correction functionality."""

    def test_apply_corrections(self, top2000_scraper):
        """Test that known title corrections are applied."""
        corrected = top2000_scraper._apply_corrections("Peaceful Easy Feeling")
        assert corrected == "Peaceful Easy Feelin'"

    def test_no_correction_for_unknown_title(self, top2000_scraper):
        """Test that unknown titles are not modified."""
        original = "Some Unknown Title"
        corrected = top2000_scraper._apply_corrections(original)
        assert corrected == original

    def test_get_corrected_uuid(self, top2000_scraper):
        """Test getting corrected UUID for known titles."""
        uuid = top2000_scraper.get_corrected_uuid("Peaceful Easy Feeling")
        assert uuid == "top2000_672351ad-d2e8-4f8c-b2c2-52e4a6f2e8ad"

        uuid = top2000_scraper.get_corrected_uuid("Some Unknown Title")
        assert uuid is None

    @freeze_time("2025-01-01")
    def test_correction_applied_during_scrape(self, top2000_scraper, httpx_mock):
        """Test that title corrections are applied during scraping."""
        json_data = load_fixture("top2000", "2024.json")
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2024-12-25",
            json=json.loads(json_data),
        )

        result = top2000_scraper.scrape("2024")

        # Find the Eagles entry with the corrected title
        eagles_entries = [(r, a, t) for r, a, t in result if "Eagles" in a]
        # Original was "Peaceful Easy Feeling", should be corrected to "Peaceful Easy Feelin'"
        corrected = [t for _, _, t in eagles_entries if "Feelin'" in t]
        assert len(corrected) == 1


class TestTop2000FutureYearGuard:
    """Tests for future year validation."""

    @freeze_time("2025-06-15")
    def test_future_year_raises_error(self, top2000_scraper):
        """Test that scraping future years raises an error."""
        with pytest.raises(Top2000FutureYearError, match="doesn't exist yet"):
            top2000_scraper.scrape("2030")

    @freeze_time("2025-06-15")
    def test_current_year_before_broadcast_raises_error(self, top2000_scraper):
        """Test that current year before December 25 raises error."""
        with pytest.raises(Top2000FutureYearError, match="isn't available yet"):
            top2000_scraper.scrape("2025")

    @freeze_time("2025-12-26")
    def test_current_year_after_broadcast_succeeds(self, top2000_scraper, httpx_mock):
        """Test that current year after December 25 works."""
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2025-12-25",
            json={"positions": []},
        )
        # Mock the Wikipedia fallback (called when API returns empty)
        httpx_mock.add_response(
            url="https://nl.wikipedia.org/wiki/Lijst_van_Radio_2-Top_2000%27s",
            html="<html><body></body></html>",
        )

        # Should not raise, even if API returns empty
        result = top2000_scraper.scrape("2025")
        assert result == []

    @freeze_time("2025-01-01")
    def test_past_year_always_available(self, top2000_scraper):
        """Test that past years are always considered available."""
        # Should not raise - just validates the year
        top2000_scraper._validate_year_available(1999)
        top2000_scraper._validate_year_available(2020)
        top2000_scraper._validate_year_available(2024)


class TestTop2000ScraperPeriodParsing:
    """Period parsing tests for Top2000Scraper."""

    def test_parse_period_valid(self, top2000_scraper):
        """Test parsing a valid year period."""
        year = top2000_scraper._parse_year_period("2024")
        assert year == 2024

    def test_parse_period_early_year(self, top2000_scraper):
        """Test parsing an early year."""
        year = top2000_scraper._parse_year_period("1999")
        assert year == 1999


class TestTop2000APIResponseParsing:
    """Tests for API response parsing."""

    def test_parse_new_api_format(self, top2000_scraper):
        """Test parsing new NPO API format with nested structure."""
        data = {
            "positions": [
                {
                    "position": {"current": 1, "previous": 2},
                    "track": {"artist": "Queen", "title": "Bohemian Rhapsody"},
                },
                {
                    "position": {"current": 2, "previous": None},
                    "track": {"artist": "Eagles", "title": "Hotel California"},
                },
            ]
        }
        entries = top2000_scraper._parse_api_response_rich(data)

        assert len(entries) == 2
        assert entries[0].rank == 1
        assert entries[0].artist == "Queen"
        assert entries[0].previous_position == 2
        assert entries[1].previous_position is None

    def test_parse_legacy_api_format(self, top2000_scraper):
        """Test parsing legacy API format with flat structure."""
        data = [
            {"position": 1, "artist": "Queen", "title": "Bohemian Rhapsody"},
            {"rank": 2, "artistName": "Eagles", "trackTitle": "Hotel California"},
        ]
        entries = top2000_scraper._parse_api_response_rich(data)

        assert len(entries) == 2
        assert entries[0].rank == 1
        assert entries[1].rank == 2
        assert entries[1].artist == "Eagles"

    def test_parse_with_last_year_position(self, top2000_scraper):
        """Test parsing entries with lastYearPosition field."""
        data = {
            "data": [
                {"position": 1, "artist": "Queen", "title": "Test", "lastYearPosition": 5},
            ]
        }
        entries = top2000_scraper._parse_api_response_rich(data)

        assert len(entries) == 1
        assert entries[0].previous_position == 5


class TestTop2000ScraperErrorHandling:
    """Error handling tests for Top2000Scraper."""

    @freeze_time("2025-01-01")
    def test_scrape_returns_empty_on_api_failure(self, top2000_scraper, httpx_mock):
        """Test that scraper returns empty list when API fails."""
        # Mock both API patterns to fail
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/npo-radio-2-top-2000-van-2024-12-25",
            status_code=404,
        )
        httpx_mock.add_response(
            url="https://www.nporadio2.nl/api/charts/top-2000-van-2024-12-25",
            status_code=404,
        )
        # Also mock Wikipedia fallback to fail
        httpx_mock.add_response(
            url="https://nl.wikipedia.org/wiki/Lijst_van_Radio_2-Top_2000%27s",
            status_code=404,
        )

        result = top2000_scraper.scrape("2024")
        assert result == []
