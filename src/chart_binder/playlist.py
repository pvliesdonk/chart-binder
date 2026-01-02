"""Playlist generation from chart runs.

Generates M3U/M3U8 playlists from chart data, resolving entries
to local audio files via beets database or filesystem search.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chart_binder.normalize import Normalizer

if TYPE_CHECKING:
    from chart_binder.charts_db import ChartsDB

logger = logging.getLogger(__name__)


class PlaylistFormat(StrEnum):
    """Supported playlist formats."""

    M3U = "m3u"
    M3U8 = "m3u8"


class MissingReason(StrEnum):
    """Reason why a chart entry couldn't be resolved to a file."""

    NO_SONG_LINK = "no_song_link"
    NO_LOCAL_FILE = "no_local_file"
    BEETS_NOT_CONFIGURED = "beets_not_configured"


@dataclass
class MissingEntry:
    """A chart entry that couldn't be resolved to a local file."""

    rank: int
    artist: str
    title: str
    reason: MissingReason
    song_id: str | None = None


@dataclass
class PlaylistResult:
    """Result of playlist generation."""

    chart_id: str
    period: str
    output_path: Path
    found: int
    missing: list[MissingEntry] = field(default_factory=list)
    total: int = 0

    @property
    def coverage_pct(self) -> float:
        """Calculate coverage percentage."""
        if self.total == 0:
            return 0.0
        return (self.found / self.total) * 100


@dataclass
class ResolvedEntry:
    """A chart entry resolved to a local file."""

    rank: int
    artist: str
    title: str
    file_path: Path
    duration_seconds: int | None = None


class PlaylistGenerator:
    """Generate M3U playlists from chart runs."""

    def __init__(
        self,
        charts_db: ChartsDB,
        music_library: Path | None = None,
        beets_db_path: Path | None = None,
    ):
        """
        Initialize playlist generator.

        Args:
            charts_db: Charts database instance
            music_library: Path to music library root (for filesystem search)
            beets_db_path: Path to beets database (for beets lookup)
        """
        self.db = charts_db
        self.music_library = music_library
        self.beets_db_path = beets_db_path
        self._normalizer = Normalizer()

        # Audio file extensions to search for
        self.audio_extensions = {".flac", ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".aiff"}

    def generate(
        self,
        chart_id: str,
        period: str,
        output: Path,
        format: PlaylistFormat = PlaylistFormat.M3U,
        use_relative_paths: bool = False,
    ) -> PlaylistResult:
        """
        Generate a playlist from a chart run.

        Args:
            chart_id: Chart identifier (e.g., "nl_top2000")
            period: Chart period (e.g., "2024")
            output: Output file path
            format: Playlist format (m3u or m3u8)
            use_relative_paths: Use paths relative to output file

        Returns:
            PlaylistResult with generation statistics
        """
        # Get run and entries
        run = self.db.get_run_by_period(chart_id, period)
        if not run:
            logger.warning(f"No chart run found for {chart_id}:{period}")
            return PlaylistResult(
                chart_id=chart_id,
                period=period,
                output_path=output,
                found=0,
                total=0,
            )

        entries = self._get_entries_with_songs(run["run_id"])
        total = len(entries)

        resolved: list[ResolvedEntry] = []
        missing: list[MissingEntry] = []

        for entry in entries:
            result = self._resolve_entry(entry)
            if isinstance(result, ResolvedEntry):
                resolved.append(result)
            else:
                missing.append(result)

        # Write playlist
        self._write_playlist(output, resolved, format, use_relative_paths)

        return PlaylistResult(
            chart_id=chart_id,
            period=period,
            output_path=output,
            found=len(resolved),
            missing=missing,
            total=total,
        )

    def _get_entries_with_songs(self, run_id: str) -> list[dict[str, Any]]:
        """Get chart entries with linked song information."""
        conn = self.db._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    e.entry_id,
                    e.rank,
                    e.artist_raw,
                    e.title_raw,
                    s.song_id,
                    s.artist_canonical,
                    s.title_canonical,
                    s.recording_mbid
                FROM chart_entry e
                LEFT JOIN chart_entry_song es ON e.entry_id = es.entry_id AND es.song_idx = 0
                LEFT JOIN song s ON es.song_id = s.song_id
                WHERE e.run_id = ?
                ORDER BY e.rank
                """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _resolve_entry(self, entry: dict[str, Any]) -> ResolvedEntry | MissingEntry:
        """Resolve a chart entry to a local file."""
        rank = entry["rank"]
        artist = entry["artist_raw"]
        title = entry["title_raw"]
        song_id = entry.get("song_id")

        # If no song link, try direct filesystem search
        if not song_id:
            file_path = self._search_filesystem(artist, title)
            if file_path:
                return ResolvedEntry(
                    rank=rank,
                    artist=artist,
                    title=title,
                    file_path=file_path,
                )
            return MissingEntry(
                rank=rank,
                artist=artist,
                title=title,
                reason=MissingReason.NO_SONG_LINK,
            )

        # Try beets database first
        if self.beets_db_path and self.beets_db_path.exists():
            file_path = self._lookup_beets(entry)
            if file_path:
                return ResolvedEntry(
                    rank=rank,
                    artist=artist,
                    title=title,
                    file_path=file_path,
                    song_id=song_id,
                )

        # Fall back to filesystem search
        canonical_artist = entry.get("artist_canonical") or artist
        canonical_title = entry.get("title_canonical") or title

        file_path = self._search_filesystem(canonical_artist, canonical_title)
        if file_path:
            return ResolvedEntry(
                rank=rank,
                artist=artist,
                title=title,
                file_path=file_path,
            )

        # Also try raw artist/title if canonical didn't match
        if canonical_artist != artist or canonical_title != title:
            file_path = self._search_filesystem(artist, title)
            if file_path:
                return ResolvedEntry(
                    rank=rank,
                    artist=artist,
                    title=title,
                    file_path=file_path,
                )

        return MissingEntry(
            rank=rank,
            artist=artist,
            title=title,
            reason=MissingReason.NO_LOCAL_FILE,
            song_id=song_id,
        )

    def _lookup_beets(self, entry: dict[str, Any]) -> Path | None:
        """Look up file path in beets database."""
        if not self.beets_db_path:
            return None

        try:
            with sqlite3.connect(self.beets_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Try MBID lookup first
                recording_mbid = entry.get("recording_mbid")
                if recording_mbid:
                    cursor.execute(
                        "SELECT path FROM items WHERE mb_trackid = ?",
                        (recording_mbid,),
                    )
                    row = cursor.fetchone()
                    if row:
                        path = Path(row["path"].decode() if isinstance(row["path"], bytes) else row["path"])
                        if path.exists():
                            return path

                # Fall back to artist/title search
                artist = entry.get("artist_canonical") or entry["artist_raw"]
                title = entry.get("title_canonical") or entry["title_raw"]

                cursor.execute(
                    """
                    SELECT path FROM items
                    WHERE LOWER(artist) = LOWER(?) AND LOWER(title) = LOWER(?)
                    LIMIT 1
                    """,
                    (artist, title),
                )
                row = cursor.fetchone()
                if row:
                    path = Path(row["path"].decode() if isinstance(row["path"], bytes) else row["path"])
                    if path.exists():
                        return path
        except sqlite3.Error as e:
            logger.warning(f"Beets database error: {e}")

        return None

    def _search_filesystem(self, artist: str, title: str) -> Path | None:
        """Search music library filesystem for a matching file."""
        if not self.music_library or not self.music_library.exists():
            return None

        # Normalize for matching
        artist_norm = self._normalize_for_filename(artist)
        title_norm = self._normalize_for_filename(title)

        # Build search patterns
        patterns = [
            # Artist/Album/Track pattern
            f"*{artist_norm}*/*{title_norm}*",
            # Flat structure: Artist - Title
            f"*{artist_norm}*-*{title_norm}*",
            f"*{artist_norm}*{title_norm}*",
            # Just title (in case artist folder is different)
            f"**/*{title_norm}*",
        ]

        for pattern in patterns:
            for ext in self.audio_extensions:
                full_pattern = f"{pattern}{ext}"
                for match in self.music_library.glob(full_pattern):
                    # Return first match (could be improved with scoring)
                    return match

        return None

    def _normalize_for_filename(self, text: str) -> str:
        """Normalize text for filename matching."""
        # Use normalizer to get base form
        normalized = self._normalizer.normalize_title(text).normalized

        # Remove characters that are typically removed from filenames
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            normalized = normalized.replace(char, '')

        # Replace spaces with wildcards for glob
        return normalized.replace(' ', '*')

    def _write_playlist(
        self,
        output: Path,
        entries: list[ResolvedEntry],
        format: PlaylistFormat,
        use_relative_paths: bool,
    ) -> None:
        """Write playlist file in specified format."""
        output.parent.mkdir(parents=True, exist_ok=True)

        # Determine encoding based on format
        encoding = "utf-8" if format == PlaylistFormat.M3U8 else "latin-1"

        lines = ["#EXTM3U"]

        for entry in entries:
            # Get file path (relative or absolute)
            if use_relative_paths:
                try:
                    file_path = entry.file_path.relative_to(output.parent)
                except ValueError:
                    # Can't make relative, use absolute
                    file_path = entry.file_path
            else:
                file_path = entry.file_path

            # Duration (-1 if unknown)
            duration = entry.duration_seconds if entry.duration_seconds else -1

            # EXTINF line
            lines.append(f"#EXTINF:{duration},{entry.artist} - {entry.title}")

            # File path line
            lines.append(str(file_path))

        content = "\n".join(lines) + "\n"

        try:
            output.write_text(content, encoding=encoding)
        except UnicodeEncodeError:
            # Fall back to UTF-8 if latin-1 fails
            output.write_text(content, encoding="utf-8")

        logger.info(f"Wrote playlist with {len(entries)} entries to {output}")

    def get_missing_report(self, result: PlaylistResult) -> str:
        """Generate a human-readable missing entries report."""
        if not result.missing:
            return f"All {result.total} entries resolved successfully!"

        lines = [
            f"Missing entries for {result.chart_id}:{result.period}:",
            "",
        ]

        # Group by reason
        by_reason: dict[MissingReason, list[MissingEntry]] = {}
        for entry in result.missing:
            if entry.reason not in by_reason:
                by_reason[entry.reason] = []
            by_reason[entry.reason].append(entry)

        for reason, entries in by_reason.items():
            reason_desc = {
                MissingReason.NO_SONG_LINK: "No song link in database",
                MissingReason.NO_LOCAL_FILE: "No local file found",
                MissingReason.BEETS_NOT_CONFIGURED: "Beets not configured",
            }.get(reason, str(reason))

            lines.append(f"  {reason_desc}: {len(entries)} entries")
            for entry in entries[:10]:  # Show first 10
                lines.append(f"    #{entry.rank}: {entry.artist} - {entry.title}")
            if len(entries) > 10:
                lines.append(f"    ... and {len(entries) - 10} more")
            lines.append("")

        lines.append(
            f"Total: {len(result.missing)} missing out of {result.total} "
            f"({result.coverage_pct:.1f}% coverage)"
        )

        return "\n".join(lines)


def get_music_library_path() -> Path | None:
    """Get music library path from environment."""
    path = os.getenv("MUSIC_LIBRARY")
    if path:
        return Path(path)
    return None


def get_beets_db_path() -> Path | None:
    """Get beets database path from environment or default location."""
    # Check explicit config
    beets_config = os.getenv("BEETS_CONFIG")
    if beets_config:
        config_path = Path(beets_config)
        if config_path.exists():
            # TODO: Parse beets config YAML to find library path.
            # For now, fall back to checking common locations below.
            pass

    # Check default beets database locations
    default_paths = [
        Path.home() / ".config" / "beets" / "library.db",
        Path.home() / ".beets" / "library.db",
        Path("/var/lib/beets/library.db"),
    ]

    for path in default_paths:
        if path.exists():
            return path

    return None
