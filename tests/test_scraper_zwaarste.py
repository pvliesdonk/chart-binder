"""Tests for ZwaarsteScraper with HTTP mocking."""

from __future__ import annotations


class TestZwaarsteScraperWithMocking:
    """Tests using mocked HTTP responses."""

    def test_scrape_parses_html_correctly(self, zwaarste_scraper, httpx_mock, zwaarste_fixture):
        """Test that scraper correctly parses HTML fixture."""
        html = zwaarste_fixture("2024.html")
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            html=html,
        )

        result = zwaarste_scraper.scrape("2024")

        # Should parse all 10 entries from fixture
        assert len(result) == 10, f"Expected 10 entries, got {len(result)}"

        # Check first entry
        rank, artist, title = result[0]
        assert rank == 1
        assert artist == "Metallica"
        assert title == "Master Of Puppets"

        # Check entry with special characters
        rank9, artist9, title9 = result[8]
        assert rank9 == 9
        assert artist9 == "System Of A Down"
        assert title9 == "Chop Suey!"

    def test_scrape_rich_returns_metadata(self, zwaarste_scraper, httpx_mock, zwaarste_fixture):
        """Test that scrape_rich returns ScrapedEntry objects."""
        html = zwaarste_fixture("2024.html")
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            html=html,
        )

        result = zwaarste_scraper.scrape_rich("2024")

        assert len(result) == 10

        # Check that metadata is captured
        first = result[0]
        assert first.rank == 1
        assert first.artist == "Metallica"
        assert first.title == "Master Of Puppets"
        # Zwaarste Lijst doesn't have previous_position in blog format
        assert first.previous_position is None

    def test_scrape_with_validation(self, zwaarste_scraper, httpx_mock, zwaarste_fixture):
        """Test scrape_with_validation returns ScrapeResult."""
        html = zwaarste_fixture("2024.html")
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            html=html,
        )

        result = zwaarste_scraper.scrape_with_validation("2024")

        assert hasattr(result, "entries")
        assert hasattr(result, "expected_count")
        assert hasattr(result, "warnings")
        assert result.expected_count == 150  # Zwaarste Lijst expected entries
        assert len(result.entries) == 10  # Our fixture has 10 entries


class TestZwaarsteScraperURLMap:
    """Tests for URL map functionality."""

    def test_default_url_map(self, tmp_cache):
        """Test that default URL map is loaded."""
        from chart_binder.scrapers import ZwaarsteScraper

        scraper = ZwaarsteScraper(tmp_cache)
        assert 2024 in scraper.url_map

    def test_custom_url_map_override(self, tmp_cache):
        """Test that custom URL map overrides defaults."""
        from chart_binder.scrapers import ZwaarsteScraper

        custom_urls = {
            2023: "https://example.com/zwaarste-2023",
            2022: "https://example.com/zwaarste-2022",
        }
        scraper = ZwaarsteScraper(tmp_cache, url_map=custom_urls)

        # Should have both defaults and custom
        assert 2024 in scraper.url_map
        assert 2023 in scraper.url_map
        assert 2022 in scraper.url_map
        assert scraper.url_map[2023] == "https://example.com/zwaarste-2023"

    def test_custom_url_map_can_override_default(self, tmp_cache):
        """Test that custom URL can override default."""
        from chart_binder.scrapers import ZwaarsteScraper

        custom_urls = {
            2024: "https://example.com/custom-2024",
        }
        scraper = ZwaarsteScraper(tmp_cache, url_map=custom_urls)

        assert scraper.url_map[2024] == "https://example.com/custom-2024"


class TestZwaarsteScraperParsing:
    """Tests for parsing strategies."""

    def test_parse_list_items(self, zwaarste_scraper):
        """Test parsing list items into tuples."""
        items = [
            "1. Metallica - Master Of Puppets",
            "2. Iron Maiden - The Trooper",
            "3  Slayer – Raining Blood",  # Different dash and spacing
        ]
        entries = zwaarste_scraper._parse_list_items(items)

        assert len(entries) == 3
        assert entries[0] == (1, "Metallica", "Master Of Puppets")
        assert entries[1] == (2, "Iron Maiden", "The Trooper")
        assert entries[2] == (3, "Slayer", "Raining Blood")

    def test_parse_list_items_with_em_dash(self, zwaarste_scraper):
        """Test parsing with em dash separator."""
        items = [
            "1 Tool — Lateralus",
        ]
        entries = zwaarste_scraper._parse_list_items(items)

        assert len(entries) == 1
        assert entries[0][1] == "Tool"
        assert entries[0][2] == "Lateralus"

    def test_parse_list_items_skips_invalid(self, zwaarste_scraper):
        """Test that invalid items are skipped."""
        items = [
            "1. Metallica - Master Of Puppets",
            "This is not a valid entry",
            "Also not valid",
            "2. Iron Maiden - The Trooper",
        ]
        entries = zwaarste_scraper._parse_list_items(items)

        assert len(entries) == 2
        assert entries[0][0] == 1
        assert entries[1][0] == 2


class TestZwaarsteScraperEntryId:
    """Tests for entry ID generation."""

    def test_generate_entry_id_deterministic(self, zwaarste_scraper):
        """Test that entry ID generation is deterministic."""
        id1 = zwaarste_scraper.generate_entry_id(2024, 1, "Metallica", "Master Of Puppets")
        id2 = zwaarste_scraper.generate_entry_id(2024, 1, "Metallica", "Master Of Puppets")

        assert id1 == id2

    def test_generate_entry_id_unique_for_different_entries(self, zwaarste_scraper):
        """Test that different entries get different IDs."""
        id1 = zwaarste_scraper.generate_entry_id(2024, 1, "Metallica", "Master Of Puppets")
        id2 = zwaarste_scraper.generate_entry_id(2024, 2, "Metallica", "Master Of Puppets")
        id3 = zwaarste_scraper.generate_entry_id(2023, 1, "Metallica", "Master Of Puppets")

        assert id1 != id2
        assert id1 != id3


class TestZwaarsteScraperPeriodParsing:
    """Period parsing tests for ZwaarsteScraper."""

    def test_parse_period_valid(self, zwaarste_scraper):
        """Test parsing a valid year period."""
        year = zwaarste_scraper._parse_year_period("2024")
        assert year == 2024


class TestZwaarsteScraperErrorHandling:
    """Error handling tests for ZwaarsteScraper."""

    def test_scrape_returns_empty_for_unknown_year(self, zwaarste_scraper):
        """Test that scraping unknown year returns empty list."""
        result = zwaarste_scraper.scrape("1990")
        assert result == []

    def test_scrape_returns_empty_on_404(self, zwaarste_scraper, httpx_mock):
        """Test that scraper returns empty list on 404."""
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            status_code=404,
        )

        result = zwaarste_scraper.scrape("2024")
        assert result == []

    def test_scrape_handles_empty_html(self, zwaarste_scraper, httpx_mock):
        """Test that scraper handles empty HTML gracefully."""
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            html="<html><body></body></html>",
        )

        result = zwaarste_scraper.scrape("2024")
        assert result == []


class TestZwaarsteScraperTableParsing:
    """Tests for table parsing strategy."""

    def test_parse_table_format(self, zwaarste_scraper, httpx_mock):
        """Test parsing table format HTML."""
        html = """
        <html>
        <body>
        <table>
            <tr><th>Rank</th><th>Artist</th><th>Title</th></tr>
            <tr><td>1</td><td>Metallica</td><td>Master Of Puppets</td></tr>
            <tr><td>2</td><td>Iron Maiden</td><td>The Trooper</td></tr>
        </table>
        </body>
        </html>
        """
        httpx_mock.add_response(
            url="https://communication.studiobrussel.be/de-zwaarste-lijst-2024-de-volledige-lijst",
            html=html,
        )

        result = zwaarste_scraper.scrape("2024")

        assert len(result) == 2
        assert result[0] == (1, "Metallica", "Master Of Puppets")
        assert result[1] == (2, "Iron Maiden", "The Trooper")
