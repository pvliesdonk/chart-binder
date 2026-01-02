"""Tests for cross-chart analytics module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from chart_binder.analytics import ChartAnalytics, ChartAppearance, ChartComparison
from chart_binder.charts_db import ChartsDB


@pytest.fixture
def temp_db(tmp_path: Path) -> ChartsDB:
    """Create a temporary charts database with test data."""
    db = ChartsDB(tmp_path / "charts.sqlite")
    return db


@pytest.fixture
def populated_db(temp_db: ChartsDB) -> ChartsDB:
    """Create a charts database with sample data for testing."""
    db = temp_db

    # Create charts
    db.upsert_chart("nl_top2000", "NPO Radio 2 Top 2000", "yearly", "NL")
    db.upsert_chart("nl_top40", "Nederlandse Top 40", "weekly", "NL")

    # Create songs
    conn = db._get_connection()
    now = time.time()

    # Insert songs directly
    songs = [
        ("song-1", "Queen", "Bohemian Rhapsody", "Queen", "queen_bohemian_rhapsody"),
        ("song-2", "The Beatles", "Hey Jude", "Beatles, The", "beatles_hey_jude"),
        ("song-3", "Eagles", "Hotel California", "Eagles", "eagles_hotel_california"),
        ("song-4", "Led Zeppelin", "Stairway To Heaven", "Led Zeppelin", "led_zeppelin_stairway_to_heaven"),
        ("song-5", "Pink Floyd", "Wish You Were Here", "Pink Floyd", "pink_floyd_wish_you_were_here"),
    ]
    for song_id, artist, title, artist_sort, work_key in songs:
        conn.execute(
            """
            INSERT INTO song (song_id, artist_canonical, title_canonical, artist_sort, work_key, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (song_id, artist, title, artist_sort, work_key, now),
        )

    # Create chart runs
    runs = [
        ("run-2024", "nl_top2000", "2024"),
        ("run-2023", "nl_top2000", "2023"),
        ("run-t40-w01", "nl_top40", "2024-W01"),
        ("run-t40-w02", "nl_top40", "2024-W02"),
    ]
    for run_id, chart_id, period in runs:
        conn.execute(
            """
            INSERT INTO chart_run (run_id, chart_id, period, scraped_at, source_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, chart_id, period, now, "hash123"),
        )

    # Create chart entries and links
    entries = [
        # Top 2000 2024
        ("entry-2024-1", "run-2024", 1, "Queen", "Bohemian Rhapsody", "song-1"),
        ("entry-2024-2", "run-2024", 2, "Eagles", "Hotel California", "song-3"),
        ("entry-2024-3", "run-2024", 3, "Led Zeppelin", "Stairway To Heaven", "song-4"),
        ("entry-2024-4", "run-2024", 4, "The Beatles", "Hey Jude", "song-2"),
        # Top 2000 2023 (different order)
        ("entry-2023-1", "run-2023", 1, "Queen", "Bohemian Rhapsody", "song-1"),
        ("entry-2023-2", "run-2023", 2, "Led Zeppelin", "Stairway To Heaven", "song-4"),
        ("entry-2023-3", "run-2023", 3, "Eagles", "Hotel California", "song-3"),
        ("entry-2023-4", "run-2023", 4, "Pink Floyd", "Wish You Were Here", "song-5"),  # Not in 2024
        # Top 40 2024-W01
        ("entry-w01-1", "run-t40-w01", 1, "Queen", "Bohemian Rhapsody", "song-1"),
        ("entry-w01-2", "run-t40-w01", 2, "The Beatles", "Hey Jude", "song-2"),
        # Top 40 2024-W02
        ("entry-w02-1", "run-t40-w02", 1, "The Beatles", "Hey Jude", "song-2"),
        ("entry-w02-2", "run-t40-w02", 2, "Queen", "Bohemian Rhapsody", "song-1"),
    ]
    for entry_id, run_id, rank, artist, title, song_id in entries:
        conn.execute(
            """
            INSERT INTO chart_entry (entry_id, run_id, rank, artist_raw, title_raw, entry_unit, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, run_id, rank, artist, title, "recording", now),
        )
        conn.execute(
            """
            INSERT INTO chart_entry_song (id, entry_id, song_idx, song_id, link_method, link_confidence)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (f"link-{entry_id}", entry_id, 0, song_id, "auto", 1.0),
        )

    conn.commit()
    conn.close()

    return db


class TestChartAnalyticsSongHistory:
    """Tests for get_song_chart_history."""

    def test_song_with_multiple_appearances(self, populated_db: ChartsDB):
        """Test getting history for a song with multiple chart appearances."""
        analytics = ChartAnalytics(populated_db)
        history = analytics.get_song_chart_history("song-1")

        # Queen - Bohemian Rhapsody appears in Top 2000 2024, 2023, and Top 40 W01, W02
        assert len(history) == 4

        # All should be ChartAppearance objects
        assert all(isinstance(h, ChartAppearance) for h in history)

        # Check specific appearances
        chart_periods = [(h.chart_id, h.period, h.rank) for h in history]
        assert ("nl_top2000", "2023", 1) in chart_periods
        assert ("nl_top2000", "2024", 1) in chart_periods
        assert ("nl_top40", "2024-W01", 1) in chart_periods
        assert ("nl_top40", "2024-W02", 2) in chart_periods

    def test_song_with_no_appearances(self, populated_db: ChartsDB):
        """Test getting history for a song not in any chart."""
        analytics = ChartAnalytics(populated_db)
        history = analytics.get_song_chart_history("nonexistent-song")

        assert history == []


class TestChartAnalyticsFuzzySearch:
    """Tests for get_song_by_artist_title fuzzy matching."""

    def test_exact_match(self, populated_db: ChartsDB):
        """Test exact match returns correct song."""
        analytics = ChartAnalytics(populated_db)
        song = analytics.get_song_by_artist_title("Queen", "Bohemian Rhapsody")

        assert song is not None
        assert song.artist_canonical == "Queen"
        assert song.title_canonical == "Bohemian Rhapsody"

    def test_fuzzy_match_case_insensitive(self, populated_db: ChartsDB):
        """Test fuzzy matching is case insensitive."""
        analytics = ChartAnalytics(populated_db)
        song = analytics.get_song_by_artist_title("QUEEN", "BOHEMIAN RHAPSODY")

        assert song is not None
        assert song.artist_canonical == "Queen"

    def test_fuzzy_match_slight_variation(self, populated_db: ChartsDB):
        """Test fuzzy matching with slight spelling variations."""
        analytics = ChartAnalytics(populated_db)
        # "Beatles" instead of "The Beatles"
        song = analytics.get_song_by_artist_title("Beatles", "Hey Jude")

        assert song is not None
        assert song.title_canonical == "Hey Jude"

    def test_no_match_below_threshold(self, populated_db: ChartsDB):
        """Test that matches below threshold return None."""
        analytics = ChartAnalytics(populated_db)
        # Completely different artist/title
        song = analytics.get_song_by_artist_title("Xyz Artist", "Abc Song", threshold=0.9)

        assert song is None

    def test_threshold_controls_matching(self, populated_db: ChartsDB):
        """Test that threshold controls match sensitivity."""
        analytics = ChartAnalytics(populated_db)

        # With high threshold, partial match fails
        song_strict = analytics.get_song_by_artist_title("Quen", "Bohemian Rhap", threshold=0.95)
        assert song_strict is None

        # With lower threshold, it matches
        song_loose = analytics.get_song_by_artist_title("Quen", "Bohemian Rhap", threshold=0.5)
        assert song_loose is not None


class TestChartAnalyticsCompare:
    """Tests for compare_charts."""

    def test_compare_top2000_years(self, populated_db: ChartsDB):
        """Test comparing two Top 2000 years."""
        analytics = ChartAnalytics(populated_db)
        comparison = analytics.compare_charts("nl_top2000:2024", "nl_top2000:2023")

        assert isinstance(comparison, ChartComparison)
        assert comparison.run1_period == "2024"
        assert comparison.run2_period == "2023"

        # Common songs: Queen, Eagles, Led Zeppelin (The Beatles not in 2023, Pink Floyd not in 2024)
        assert len(comparison.common_songs) == 3

        # The Beatles only in 2024
        assert len(comparison.only_in_run1) == 1
        assert any("Beatles" in entry[0] for entry in comparison.only_in_run1)

        # Pink Floyd only in 2023
        assert len(comparison.only_in_run2) == 1
        assert any("Pink Floyd" in entry[0] for entry in comparison.only_in_run2)

    def test_compare_detects_movers(self, populated_db: ChartsDB):
        """Test that compare_charts detects position changes."""
        analytics = ChartAnalytics(populated_db)
        comparison = analytics.compare_charts("nl_top2000:2024", "nl_top2000:2023")

        # Eagles moved from #3 to #2, Led Zeppelin from #2 to #3
        assert len(comparison.movers) >= 2

        # Movers should be sorted by absolute delta
        for i in range(len(comparison.movers) - 1):
            assert abs(comparison.movers[i][4]) >= abs(comparison.movers[i + 1][4])

    def test_compare_weekly_charts(self, populated_db: ChartsDB):
        """Test comparing weekly chart runs."""
        analytics = ChartAnalytics(populated_db)
        comparison = analytics.compare_charts("nl_top40:2024-W01", "nl_top40:2024-W02")

        # Both songs appear in both weeks
        assert len(comparison.common_songs) == 2
        assert len(comparison.only_in_run1) == 0
        assert len(comparison.only_in_run2) == 0

        # Both songs moved
        assert len(comparison.movers) == 2

    def test_compare_nonexistent_runs(self, populated_db: ChartsDB):
        """Test comparing with nonexistent run returns empty comparison."""
        analytics = ChartAnalytics(populated_db)
        comparison = analytics.compare_charts("fake:2024", "nl_top2000:2024")

        assert comparison.run1_period == ""
        assert len(comparison.common_songs) == 0

    def test_overlap_percentage(self, populated_db: ChartsDB):
        """Test overlap percentage calculation."""
        analytics = ChartAnalytics(populated_db)
        comparison = analytics.compare_charts("nl_top2000:2024", "nl_top2000:2023")

        # 3 common out of 5 unique songs (3 common + 1 only in 2024 + 1 only in 2023)
        expected_overlap = 3 / 5 * 100  # 60%
        assert abs(comparison.overlap_pct - expected_overlap) < 0.1


class TestChartAnalyticsHelpers:
    """Tests for helper methods."""

    def test_get_biggest_movers(self, populated_db: ChartsDB):
        """Test getting biggest movers between periods."""
        analytics = ChartAnalytics(populated_db)
        movers = analytics.get_biggest_movers("nl_top2000", "2024", "2023", limit=5)

        assert isinstance(movers, list)
        # Eagles and Led Zeppelin both moved by 1 position
        assert len(movers) >= 2

    def test_get_new_entries(self, populated_db: ChartsDB):
        """Test getting new entries in a chart run."""
        analytics = ChartAnalytics(populated_db)
        new = analytics.get_new_entries("nl_top2000", "2024", "2023")

        # The Beatles is new in 2024
        assert len(new) == 1
        assert any("Beatles" in entry[0] for entry in new)

    def test_get_dropped_entries(self, populated_db: ChartsDB):
        """Test getting dropped entries from a chart run."""
        analytics = ChartAnalytics(populated_db)
        dropped = analytics.get_dropped_entries("nl_top2000", "2024", "2023")

        # Pink Floyd dropped from 2024
        assert len(dropped) == 1
        assert any("Pink Floyd" in entry[0] for entry in dropped)
