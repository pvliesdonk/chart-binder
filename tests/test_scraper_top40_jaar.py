"""Tests for Top40JaarScraper with HTTP mocking."""

from __future__ import annotations


class TestTop40JaarScraperWithMocking:
    """Tests using mocked HTTP responses."""

    def test_scrape_parses_html_correctly(self, top40jaar_scraper, httpx_mock, top40jaar_fixture):
        """Test that scraper correctly parses HTML fixture."""
        html = top40jaar_fixture("2023.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/2023",
            html=html,
        )

        result = top40jaar_scraper.scrape("2023")

        # Should parse entries from fixture (including split entries)
        # Position 9 has double title (Penny Lane / Strawberry Fields -> 2 entries)
        # Position 10 has double artist+title (Artist A/B - Song A/B -> 2 entries)
        assert len(result) == 12, f"Expected 12 entries after splits, got {len(result)}"

        # Check first entry
        rank, artist, title = result[0]
        assert rank == 1
        assert artist == "Miley Cyrus"
        assert title == "Flowers"

        # Check entry with special characters (&)
        rank6, artist6, title6 = result[5]
        assert rank6 == 6
        assert artist6 == "Taylor Swift"
        assert title6 == "Anti-Hero"

    def test_scrape_handles_split_entries(self, top40jaar_scraper, httpx_mock, top40jaar_fixture):
        """Test that split entries (double A-sides) are correctly handled."""
        html = top40jaar_fixture("2023.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/2023",
            html=html,
        )

        result = top40jaar_scraper.scrape("2023")

        # Find The Beatles entry which should be split into two
        beatles_entries = [(r, a, t) for r, a, t in result if "Beatles" in a]
        assert len(beatles_entries) == 2, f"Expected 2 Beatles entries, got {len(beatles_entries)}"
        assert beatles_entries[0][2] == "Penny Lane"
        assert beatles_entries[1][2] == "Strawberry Fields"

    def test_scrape_rich_returns_metadata(self, top40jaar_scraper, httpx_mock, top40jaar_fixture):
        """Test that scrape_rich returns ScrapedEntry objects."""
        html = top40jaar_fixture("2023.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/2023",
            html=html,
        )

        result = top40jaar_scraper.scrape_rich("2023")

        assert len(result) == 12

        # Check that metadata is captured
        first = result[0]
        assert first.rank == 1
        assert first.artist == "Miley Cyrus"
        assert first.title == "Flowers"
        # Year-end charts don't have previous_position
        assert first.previous_position is None

    def test_scrape_with_validation(self, top40jaar_scraper, httpx_mock, top40jaar_fixture):
        """Test scrape_with_validation returns ScrapeResult."""
        html = top40jaar_fixture("2023.html")
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/2023",
            html=html,
        )

        result = top40jaar_scraper.scrape_with_validation("2023")

        assert hasattr(result, "entries")
        assert hasattr(result, "expected_count")
        assert hasattr(result, "warnings")
        assert result.expected_count == 100  # Year-end charts have 100 entries


class TestTop40JaarScraperPeriodParsing:
    """Period parsing tests for Top40JaarScraper."""

    def test_parse_period_valid(self, top40jaar_scraper):
        """Test parsing a valid year period."""
        year = top40jaar_scraper._parse_year_period("2023")
        assert year == 2023

    def test_parse_period_early_year(self, top40jaar_scraper):
        """Test parsing an early year."""
        year = top40jaar_scraper._parse_year_period("1965")
        assert year == 1965

    def test_url_construction(self, top40jaar_scraper):
        """Test URL is constructed correctly."""
        year = top40jaar_scraper._parse_year_period("2023")
        expected_url = f"{top40jaar_scraper.BASE_URL}/top40-jaarlijsten/{year}"
        assert expected_url == "https://www.top40.nl/top40-jaarlijsten/2023"


class TestTop40JaarScraperEdgeCases:
    """Edge case tests for Top40JaarScraper."""

    def test_split_entry_double_title(self, top40jaar_scraper):
        """Test handling of double A-side with slash in title."""
        result = top40jaar_scraper._handle_split_entries(
            rank=1,
            artist="Queen & David Bowie",
            title="Under Pressure / Soul Brother",
        )

        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        assert result[0][2] == "Under Pressure"
        assert result[1][2] == "Soul Brother"

    def test_split_entry_double_artist_and_title(self, top40jaar_scraper):
        """Test handling of doubled artist and title."""
        result = top40jaar_scraper._handle_split_entries(
            rank=1,
            artist="Artist A / Artist B",
            title="Song A / Song B",
        )

        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        assert result[0][1] == "Artist A"
        assert result[0][2] == "Song A"
        assert result[1][1] == "Artist B"
        assert result[1][2] == "Song B"

    def test_split_entry_semicolon(self, top40jaar_scraper):
        """Test handling of semicolon-separated entries."""
        result = top40jaar_scraper._handle_split_entries(
            rank=1,
            artist="Queen",
            title="We Will Rock You ; We Are The Champions",
        )

        assert len(result) == 2
        assert result[0][2] == "We Will Rock You"
        assert result[1][2] == "We Are The Champions"

    def test_no_split_normal_entry(self, top40jaar_scraper):
        """Test that normal entries are not split."""
        result = top40jaar_scraper._handle_split_entries(
            rank=1,
            artist="The Beatles",
            title="Yesterday",
        )

        assert len(result) == 1
        assert result[0] == (1, "The Beatles", "Yesterday")

    def test_double_parens_removed(self, top40jaar_scraper):
        """Test that double parentheses are removed from title."""
        result = top40jaar_scraper._handle_split_entries(
            rank=1,
            artist="Artist",
            title="Title ((metadata))",
        )

        assert result[0][2] == "Title"


class TestTop40JaarScraperErrorHandling:
    """Error handling tests for Top40JaarScraper."""

    def test_scrape_returns_empty_on_404(self, top40jaar_scraper, httpx_mock):
        """Test that scraper returns empty list on 404."""
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/1900",
            status_code=404,
        )

        result = top40jaar_scraper.scrape("1900")
        assert result == []

    def test_scrape_handles_empty_html(self, top40jaar_scraper, httpx_mock):
        """Test that scraper handles empty HTML gracefully."""
        httpx_mock.add_response(
            url="https://www.top40.nl/top40-jaarlijsten/2023",
            html="<html><body></body></html>",
        )

        result = top40jaar_scraper.scrape("2023")
        assert result == []


class TestTop40JaarScraperSideDesignation:
    """Tests for side designation in split entries."""

    def test_split_entry_has_side_designation(self, top40jaar_scraper):
        """Test that split entries get side designations A, B, etc."""
        result = top40jaar_scraper._handle_split_entries_rich(
            rank=1,
            artist="The Beatles",
            title="Penny Lane / Strawberry Fields Forever",
        )

        assert len(result) == 2
        assert result[0].side == "A"
        assert result[0].title == "Penny Lane"
        assert result[1].side == "B"
        assert result[1].title == "Strawberry Fields Forever"

    def test_split_multiple_artists_has_side_designation(self, top40jaar_scraper):
        """Test that split entries with multiple artists get side designations."""
        result = top40jaar_scraper._handle_split_entries_rich(
            rank=1,
            artist="Artist A / Artist B",
            title="Song A / Song B",
        )

        assert len(result) == 2
        assert result[0].side == "A"
        assert result[0].artist == "Artist A"
        assert result[1].side == "B"
        assert result[1].artist == "Artist B"

    def test_non_split_entry_has_no_side_designation(self, top40jaar_scraper):
        """Test that non-split entries don't have side designation."""
        result = top40jaar_scraper._handle_split_entries_rich(
            rank=1,
            artist="The Beatles",
            title="Yesterday",
        )

        assert len(result) == 1
        assert result[0].side is None
