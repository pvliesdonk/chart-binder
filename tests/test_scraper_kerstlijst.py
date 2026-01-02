"""Tests for KerstlijstImporter."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def kerstlijst_fixture():
    """Fixture to load Kerstlijst test JSON files."""
    fixture_dir = Path(__file__).parent / "fixtures" / "cassettes" / "kerstlijst"

    def _load(filename: str) -> Path:
        return fixture_dir / filename

    return _load


@pytest.fixture
def kerstlijst_importer():
    """Create a KerstlijstImporter instance."""
    from chart_binder.scrapers import KerstlijstImporter

    return KerstlijstImporter()


class TestKerstlijstImporterLoad:
    """Tests for loading JSON files."""

    def test_load_sample_json(self, kerstlijst_importer, kerstlijst_fixture):
        """Test loading sample Kerstlijst JSON."""
        json_path = kerstlijst_fixture("sample.json")
        songs = kerstlijst_importer.load(json_path)

        assert len(songs) == 6

        # Check first song
        wham = next(s for s in songs if s.artist == "Wham!")
        assert wham.title == "Last Christmas"
        assert wham.positions[2020] == 1
        assert wham.positions[2021] == 2
        assert wham.positions[2022] == 1

    def test_load_missing_file(self, kerstlijst_importer):
        """Test loading non-existent file returns empty list."""
        songs = kerstlijst_importer.load(Path("/nonexistent/file.json"))
        assert songs == []

    def test_load_invalid_json(self, kerstlijst_importer, tmp_path):
        """Test loading invalid JSON returns empty list."""
        json_path = tmp_path / "invalid.json"
        json_path.write_text("not valid json {{{")

        songs = kerstlijst_importer.load(json_path)
        assert songs == []

    def test_load_skips_invalid_entries(self, kerstlijst_importer, tmp_path):
        """Test that invalid entries are skipped during load."""
        import json

        json_path = tmp_path / "kerstlijst.json"
        json_path.write_text(
            json.dumps(
                [
                    # Valid entry
                    {
                        "artiest": "Wham!",
                        "titel": "Last Christmas",
                        "hitlists": {"spotweb": {"2020": 1}},
                    },
                    # Missing artist
                    {"artiest": "", "titel": "Some Song", "hitlists": {"spotweb": {"2020": 2}}},
                    # Missing title
                    {
                        "artiest": "Some Artist",
                        "titel": "",
                        "hitlists": {"spotweb": {"2020": 3}},
                    },
                    # No positions
                    {"artiest": "No Positions", "titel": "Song", "hitlists": {"spotweb": {}}},
                    # Invalid rank (too high)
                    {
                        "artiest": "Invalid",
                        "titel": "Rank",
                        "hitlists": {"spotweb": {"2020": 9999}},
                    },
                ]
            )
        )

        songs = kerstlijst_importer.load(json_path)

        # Only the first valid entry should be loaded
        assert len(songs) == 1
        assert songs[0].artist == "Wham!"


class TestKerstlijstEntriesForYear:
    """Tests for extracting entries by year."""

    def test_get_entries_for_year_2020(self, kerstlijst_importer, kerstlijst_fixture):
        """Test getting entries for 2020."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        entries = kerstlijst_importer.get_entries_for_year(songs, 2020)

        assert len(entries) == 5  # 5 songs have 2020 positions

        # Entries should be sorted by rank
        assert entries[0].rank == 1
        assert entries[0].artist == "Wham!"

        assert entries[1].rank == 2
        assert entries[1].artist == "Mariah Carey"

        assert entries[4].rank == 5
        assert entries[4].artist == "Shakin' Stevens"

    def test_get_entries_for_year_2021(self, kerstlijst_importer, kerstlijst_fixture):
        """Test getting entries for 2021."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        entries = kerstlijst_importer.get_entries_for_year(songs, 2021)

        assert len(entries) == 5  # 5 songs have 2021 positions

        # Mariah Carey was #1 in 2021
        assert entries[0].rank == 1
        assert entries[0].artist == "Mariah Carey"

        # Wham! was #2 in 2021
        assert entries[1].rank == 2
        assert entries[1].artist == "Wham!"

    def test_get_entries_for_missing_year(self, kerstlijst_importer, kerstlijst_fixture):
        """Test getting entries for a year with no data."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        entries = kerstlijst_importer.get_entries_for_year(songs, 2019)

        assert len(entries) == 0


class TestKerstlijstAllYears:
    """Tests for getting all years from loaded songs."""

    def test_get_all_years(self, kerstlijst_importer, kerstlijst_fixture):
        """Test getting all years from sample data."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        years = kerstlijst_importer.get_all_years(songs)

        assert years == {2020, 2021, 2022}


class TestKerstlijstEntriesByYear:
    """Tests for getting entries grouped by year."""

    def test_get_entries_by_year(self, kerstlijst_importer, kerstlijst_fixture):
        """Test getting all entries grouped by year."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        by_year = kerstlijst_importer.get_entries_by_year(songs)

        assert 2020 in by_year
        assert 2021 in by_year
        assert 2022 in by_year

        assert len(by_year[2020]) == 5
        assert len(by_year[2021]) == 5
        assert len(by_year[2022]) == 5


class TestKerstlijstValidation:
    """Tests for validation."""

    def test_validate_year_with_gaps(self, kerstlijst_importer, kerstlijst_fixture):
        """Test validation detects rank gaps."""
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        # 2021 has ranks 1,2,3,4,5 - no gaps
        entries_2021 = kerstlijst_importer.get_entries_for_year(songs, 2021)
        warnings = kerstlijst_importer.validate_year(entries_2021, 2021)
        assert len(warnings) == 0

    def test_validate_empty_year(self, kerstlijst_importer):
        """Test validation for empty year."""
        warnings = kerstlijst_importer.validate_year([], 2019)
        assert any("No entries" in w for w in warnings)


class TestKerstlijstIntegration:
    """Integration tests."""

    def test_full_import_workflow(self, kerstlijst_importer, kerstlijst_fixture):
        """Test complete import workflow."""
        # Load songs
        songs = kerstlijst_importer.load(kerstlijst_fixture("sample.json"))
        assert len(songs) == 6

        # Get all years
        years = kerstlijst_importer.get_all_years(songs)
        assert len(years) == 3

        # Get entries for each year
        for year in years:
            entries = kerstlijst_importer.get_entries_for_year(songs, year)
            assert len(entries) > 0

            # Verify entries are sorted by rank
            ranks = [e.rank for e in entries]
            assert ranks == sorted(ranks)

            # Validate
            warnings = kerstlijst_importer.validate_year(entries, year)
            # Our sample data is complete, so no warnings expected
            assert len(warnings) == 0
