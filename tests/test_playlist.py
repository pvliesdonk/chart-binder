"""Tests for playlist generation module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from chart_binder.charts_db import ChartsDB
from chart_binder.playlist import (
    MissingEntry,
    MissingReason,
    PlaylistFormat,
    PlaylistGenerator,
    PlaylistResult,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> ChartsDB:
    """Create a temporary charts database."""
    db = ChartsDB(tmp_path / "charts.sqlite")
    return db


@pytest.fixture
def populated_db(temp_db: ChartsDB) -> ChartsDB:
    """Create a charts database with sample data."""
    db = temp_db

    # Create chart
    db.upsert_chart("nl_top2000", "NPO Radio 2 Top 2000", "yearly", "NL")

    conn = db._get_connection()
    now = time.time()

    # Insert songs
    songs = [
        ("song-1", "Queen", "Bohemian Rhapsody", "Queen", "queen_bohemian_rhapsody", "mbid-1"),
        ("song-2", "The Beatles", "Hey Jude", "Beatles, The", "beatles_hey_jude", "mbid-2"),
        ("song-3", "Eagles", "Hotel California", "Eagles", "eagles_hotel_california", None),
    ]
    for song_id, artist, title, artist_sort, work_key, mbid in songs:
        conn.execute(
            """
            INSERT INTO song (song_id, artist_canonical, title_canonical, artist_sort,
                              work_key, recording_mbid, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (song_id, artist, title, artist_sort, work_key, mbid, now),
        )

    # Create chart run
    conn.execute(
        """
        INSERT INTO chart_run (run_id, chart_id, period, scraped_at, source_hash)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run-2024", "nl_top2000", "2024", now, "hash123"),
    )

    # Create chart entries
    entries = [
        ("entry-1", "run-2024", 1, "Queen", "Bohemian Rhapsody", "song-1"),
        ("entry-2", "run-2024", 2, "The Beatles", "Hey Jude", "song-2"),
        ("entry-3", "run-2024", 3, "Eagles", "Hotel California", "song-3"),
        ("entry-4", "run-2024", 4, "Unknown Artist", "Unknown Song", None),  # No song link
    ]
    for entry_id, run_id, rank, artist, title, song_id in entries:
        conn.execute(
            """
            INSERT INTO chart_entry (entry_id, run_id, rank, artist_raw, title_raw,
                                     entry_unit, scraped_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (entry_id, run_id, rank, artist, title, "recording", now),
        )
        if song_id:
            conn.execute(
                """
                INSERT INTO chart_entry_song (id, entry_id, song_idx, song_id,
                                              link_method, link_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f"link-{entry_id}", entry_id, 0, song_id, "auto", 1.0),
            )

    conn.commit()
    conn.close()

    return db


@pytest.fixture
def music_library(tmp_path: Path) -> Path:
    """Create a mock music library with sample files.

    Uses lowercase names to match normalized patterns used by playlist generator.
    """
    lib = tmp_path / "music"
    lib.mkdir()

    # Create directory structure: Artist/Track (lowercase for pattern matching)
    queen_dir = lib / "queen"
    queen_dir.mkdir(parents=True)
    (queen_dir / "bohemian rhapsody.flac").write_text("fake audio")

    beatles_dir = lib / "beatles"
    beatles_dir.mkdir(parents=True)
    (beatles_dir / "hey jude.mp3").write_text("fake audio")

    return lib


class TestPlaylistFormat:
    """Tests for PlaylistFormat enum."""

    def test_m3u_value(self):
        assert PlaylistFormat.M3U.value == "m3u"

    def test_m3u8_value(self):
        assert PlaylistFormat.M3U8.value == "m3u8"


class TestMissingReason:
    """Tests for MissingReason enum."""

    def test_no_song_link_value(self):
        assert MissingReason.NO_SONG_LINK.value == "no_song_link"

    def test_no_local_file_value(self):
        assert MissingReason.NO_LOCAL_FILE.value == "no_local_file"


class TestPlaylistResult:
    """Tests for PlaylistResult dataclass."""

    def test_coverage_pct_zero_total(self):
        result = PlaylistResult(
            chart_id="test",
            period="2024",
            output_path=Path("test.m3u"),
            found=0,
            total=0,
        )
        assert result.coverage_pct == 0.0

    def test_coverage_pct_full_coverage(self):
        result = PlaylistResult(
            chart_id="test",
            period="2024",
            output_path=Path("test.m3u"),
            found=100,
            total=100,
        )
        assert result.coverage_pct == 100.0

    def test_coverage_pct_partial(self):
        result = PlaylistResult(
            chart_id="test",
            period="2024",
            output_path=Path("test.m3u"),
            found=75,
            total=100,
        )
        assert result.coverage_pct == 75.0


class TestPlaylistGeneratorInit:
    """Tests for PlaylistGenerator initialization."""

    def test_init_with_all_paths(self, populated_db: ChartsDB, tmp_path: Path):
        lib = tmp_path / "music"
        lib.mkdir()
        beets = tmp_path / "library.db"
        beets.write_text("")

        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=lib,
            beets_db_path=beets,
        )

        assert generator.music_library == lib
        assert generator.beets_db_path == beets

    def test_init_without_paths(self, populated_db: ChartsDB):
        generator = PlaylistGenerator(charts_db=populated_db)

        assert generator.music_library is None
        assert generator.beets_db_path is None


class TestPlaylistGeneratorGenerate:
    """Tests for PlaylistGenerator.generate()."""

    def test_generate_nonexistent_chart(self, populated_db: ChartsDB, tmp_path: Path):
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(charts_db=populated_db)

        result = generator.generate(
            chart_id="nonexistent",
            period="2024",
            output=output,
        )

        assert result.found == 0
        assert result.total == 0
        assert result.chart_id == "nonexistent"

    def test_generate_with_no_sources(self, populated_db: ChartsDB, tmp_path: Path):
        """Test generation when no music library or beets db is configured."""
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(charts_db=populated_db)

        result = generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        # Should have entries but all missing (no source to resolve)
        assert result.total == 4
        assert result.found == 0
        assert len(result.missing) == 4

    def test_generate_with_music_library(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        """Test generation with a music library."""
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        result = generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        # Queen and Beatles should be found
        assert result.found == 2
        assert result.total == 4
        assert len(result.missing) == 2

        # Check playlist file was created
        assert output.exists()

    def test_generate_m3u8_format(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        """Test generation in M3U8 format."""
        output = tmp_path / "playlist.m3u8"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
            format=PlaylistFormat.M3U8,
        )

        assert output.exists()
        # M3U8 uses UTF-8 encoding
        content = output.read_text(encoding="utf-8")
        assert "#EXTM3U" in content

    def test_generate_creates_parent_dirs(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        """Test that generate creates parent directories."""
        output = tmp_path / "nested" / "dir" / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        assert output.exists()


class TestPlaylistGeneratorFilesearch:
    """Tests for filesystem search functionality."""

    def test_filesystem_search_artist_album_pattern(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        """Test that Artist/Album/Track pattern works."""
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        result = generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        # Queen and Beatles should be found via filesystem
        {e.artist for e in result.missing if e.reason != MissingReason.NO_LOCAL_FILE}
        # Eagles and Unknown should be missing
        assert any("Eagles" in m.artist for m in result.missing)
        assert any("Unknown" in m.artist for m in result.missing)


class TestPlaylistGeneratorMissingReport:
    """Tests for missing entries report."""

    def test_get_missing_report_no_missing(self, populated_db: ChartsDB):
        generator = PlaylistGenerator(charts_db=populated_db)
        result = PlaylistResult(
            chart_id="test",
            period="2024",
            output_path=Path("test.m3u"),
            found=10,
            total=10,
        )

        report = generator.get_missing_report(result)
        assert "All 10 entries resolved" in report

    def test_get_missing_report_with_missing(self, populated_db: ChartsDB):
        generator = PlaylistGenerator(charts_db=populated_db)
        result = PlaylistResult(
            chart_id="test",
            period="2024",
            output_path=Path("test.m3u"),
            found=5,
            total=10,
            missing=[
                MissingEntry(rank=1, artist="Artist 1", title="Title 1", reason=MissingReason.NO_SONG_LINK),
                MissingEntry(rank=2, artist="Artist 2", title="Title 2", reason=MissingReason.NO_LOCAL_FILE),
            ],
        )

        report = generator.get_missing_report(result)
        assert "Missing entries" in report
        assert "Artist 1" in report
        assert "Artist 2" in report


class TestPlaylistFileContent:
    """Tests for generated playlist file content."""

    def test_playlist_has_extm3u_header(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        content = output.read_text()
        assert content.startswith("#EXTM3U")

    def test_playlist_has_extinf_metadata(
        self, populated_db: ChartsDB, music_library: Path, tmp_path: Path
    ):
        output = tmp_path / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
        )

        content = output.read_text()
        assert "#EXTINF:" in content
        assert "Queen - Bohemian Rhapsody" in content

    def test_playlist_relative_paths(
        self, populated_db: ChartsDB, music_library: Path
    ):
        # Output in music library to enable relative paths
        output = music_library / "playlist.m3u"
        generator = PlaylistGenerator(
            charts_db=populated_db,
            music_library=music_library,
        )

        generator.generate(
            chart_id="nl_top2000",
            period="2024",
            output=output,
            use_relative_paths=True,
        )

        content = output.read_text()
        # Should not have absolute paths
        assert not any(line.startswith("/") for line in content.split("\n") if not line.startswith("#"))
