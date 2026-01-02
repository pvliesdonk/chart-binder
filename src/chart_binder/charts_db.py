from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from chart_binder.normalize import Normalizer


class EntryUnit(StrEnum):
    """Type of chart entry."""

    RECORDING = "recording"
    SINGLE_RELEASE = "single_release"
    MEDLEY = "medley"
    UNKNOWN = "unknown"


class LinkMethod(StrEnum):
    """Method used to link chart entry to work."""

    ISRC = "isrc"
    TITLE_ARTIST_YEAR = "title_artist_year"
    BUNDLE_RELEASE = "bundle_release"
    MULTI_SOURCE = "multi_source"  # Multi-source search (MB, Discogs, Spotify)
    MANUAL = "manual"
    AUTO = "auto"  # Automatic normalization match
    ALIAS = "alias"  # Matched via alias_norm


@dataclass
class ChartEntry:
    """A single entry in a chart run (Layer 1: RAW)."""

    rank: int
    artist_raw: str
    title_raw: str
    entry_unit: EntryUnit = EntryUnit.RECORDING
    extra_raw: str | None = None
    previous_position: int | None = None
    weeks_on_chart: int | None = None
    side_designation: str | None = None  # Side for split entries ('A', 'B', 'AA', etc.)
    # Wikipedia enrichment fields
    wikipedia_artist: str | None = None  # Wikipedia URL for artist
    wikipedia_title: str | None = None  # Wikipedia URL for song
    history_url: str | None = None  # Chart history page URL
    entry_id: str | None = None  # Deterministic ID (hash of run_id + rank)

    # Normalized fields (computed, for legacy compatibility)
    artist_normalized: str = ""
    title_normalized: str = ""
    title_tags: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChartLink:
    """Link between chart entry and work_key with canonical IDs."""

    run_id: str
    rank: int
    work_key: str | None
    link_method: LinkMethod
    confidence: float
    release_anchor_id: str | None = None
    side_designation: str | None = None  # A, B, AA, or None
    recording_mbid: str | None = None
    discogs_release_id: str | None = None
    discogs_master_id: str | None = None
    spotify_track_id: str | None = None
    source: str | None = None  # Which source provided the match


@dataclass
class CoverageReport:
    """Coverage report for a chart run."""

    run_id: str
    total_entries: int
    linked_entries: int
    unlinked_entries: int
    coverage_pct: float
    by_method: dict[str, int] = field(default_factory=dict)
    by_confidence: dict[str, int] = field(default_factory=dict)


@dataclass
class Song:
    """Canonical song entity (Layer 3: SONG)."""

    song_id: str
    artist_canonical: str
    title_canonical: str
    artist_sort: str | None = None
    work_key: str | None = None
    recording_mbid: str | None = None
    release_group_mbid: str | None = None
    spotify_id: str | None = None
    isrc: str | None = None
    created_at: float | None = None


@dataclass
class ChartEntrySong:
    """Link between chart entry and song (Layer 2: LINK)."""

    id: str
    entry_id: str
    song_idx: int
    song_id: str
    link_method: LinkMethod | None = None
    link_confidence: float | None = None


class ChartsDB:
    """
    SQLite database for charts data and normalization aliases.

    Stores chart runs, entries, links, and the alias_norm registry.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema for charts and aliases."""
        conn = self._get_connection()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chart (
                chart_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                frequency TEXT NOT NULL,
                jurisdiction TEXT,
                source_url TEXT,
                license TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chart_run (
                run_id TEXT PRIMARY KEY,
                chart_id TEXT NOT NULL REFERENCES chart(chart_id),
                period TEXT NOT NULL,
                scraped_at REAL NOT NULL,
                source_hash TEXT NOT NULL,
                notes TEXT,
                UNIQUE(chart_id, period)
            );

            -- Layer 1: RAW - immutable after scrape
            CREATE TABLE IF NOT EXISTS chart_entry (
                entry_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES chart_run(run_id),
                rank INTEGER NOT NULL,
                artist_raw TEXT NOT NULL,
                title_raw TEXT NOT NULL,
                previous_position INTEGER,
                weeks_on_chart INTEGER,
                side_designation TEXT,
                wikipedia_artist TEXT,
                wikipedia_title TEXT,
                history_url TEXT,
                entry_unit TEXT NOT NULL,
                extra_raw TEXT,
                scraped_at REAL NOT NULL,
                UNIQUE(run_id, rank)
            );

            -- Layer 3: SONG - canonical registry, stored ONCE
            CREATE TABLE IF NOT EXISTS song (
                song_id TEXT PRIMARY KEY,
                artist_canonical TEXT NOT NULL,
                title_canonical TEXT NOT NULL,
                artist_sort TEXT,
                work_key TEXT,
                recording_mbid TEXT,
                release_group_mbid TEXT,
                spotify_id TEXT,
                isrc TEXT,
                created_at REAL NOT NULL,
                UNIQUE(artist_canonical, title_canonical)
            );

            -- Layer 2: LINK - editable join table (no text duplication)
            CREATE TABLE IF NOT EXISTS chart_entry_song (
                id TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL REFERENCES chart_entry(entry_id),
                song_idx INTEGER NOT NULL,
                song_id TEXT NOT NULL REFERENCES song(song_id),
                link_method TEXT,
                link_confidence REAL,
                UNIQUE(entry_id, song_idx)
            );

            -- Legacy compatibility table
            CREATE TABLE IF NOT EXISTS chart_link (
                run_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                work_key TEXT,
                link_method TEXT NOT NULL,
                confidence REAL NOT NULL,
                release_anchor_id TEXT,
                side_designation TEXT,
                recording_mbid TEXT,
                discogs_release_id TEXT,
                discogs_master_id TEXT,
                spotify_track_id TEXT,
                source TEXT,
                PRIMARY KEY (run_id, rank)
            );

            CREATE TABLE IF NOT EXISTS alias_norm (
                alias_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                raw TEXT NOT NULL,
                normalized TEXT NOT NULL,
                ruleset_version TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_alias_type_raw ON alias_norm(type, raw);
            CREATE INDEX IF NOT EXISTS idx_alias_normalized ON alias_norm(normalized);
            CREATE INDEX IF NOT EXISTS idx_entry_run ON chart_entry(run_id);
            CREATE INDEX IF NOT EXISTS idx_entry_artist_title ON chart_entry(artist_raw, title_raw);
            CREATE INDEX IF NOT EXISTS idx_link_work ON chart_link(work_key);
            CREATE INDEX IF NOT EXISTS idx_link_confidence ON chart_link(confidence);
            CREATE INDEX IF NOT EXISTS idx_run_chart ON chart_run(chart_id);
            CREATE INDEX IF NOT EXISTS idx_song_canonical ON song(artist_canonical, title_canonical);
            CREATE INDEX IF NOT EXISTS idx_song_mbid ON song(recording_mbid);
            CREATE INDEX IF NOT EXISTS idx_entry_song_song ON chart_entry_song(song_id);
            CREATE INDEX IF NOT EXISTS idx_entry_song_entry ON chart_entry_song(entry_id);
            """
        )

        # Additive migrations: Add canonical ID columns if they don't exist
        # This handles migration from older schema versions
        try:
            conn.execute("SELECT recording_mbid FROM chart_link LIMIT 1")
        except sqlite3.OperationalError:
            # Columns don't exist, add them
            conn.executescript(
                """
                ALTER TABLE chart_link ADD COLUMN recording_mbid TEXT;
                ALTER TABLE chart_link ADD COLUMN discogs_release_id TEXT;
                ALTER TABLE chart_link ADD COLUMN discogs_master_id TEXT;
                ALTER TABLE chart_link ADD COLUMN spotify_track_id TEXT;
                ALTER TABLE chart_link ADD COLUMN source TEXT;
                """
            )

        # Migration: Add side_designation column to chart_entry if it doesn't exist
        try:
            conn.execute("SELECT side_designation FROM chart_entry LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE chart_entry ADD COLUMN side_designation TEXT")

        # Migration: Add Wikipedia enrichment columns if they don't exist
        try:
            conn.execute("SELECT wikipedia_artist FROM chart_entry LIMIT 1")
        except sqlite3.OperationalError:
            conn.executescript(
                """
                ALTER TABLE chart_entry ADD COLUMN wikipedia_artist TEXT;
                ALTER TABLE chart_entry ADD COLUMN wikipedia_title TEXT;
                ALTER TABLE chart_entry ADD COLUMN history_url TEXT;
                """
            )

        conn.commit()
        conn.close()

    # Chart management

    def upsert_chart(
        self,
        chart_id: str,
        name: str,
        frequency: str,
        jurisdiction: str | None = None,
        source_url: str | None = None,
        license_info: str | None = None,
    ) -> None:
        """Upsert chart definition."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart (chart_id, name, frequency, jurisdiction, source_url, license, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chart_id) DO UPDATE SET
                    name = excluded.name,
                    frequency = excluded.frequency,
                    jurisdiction = excluded.jurisdiction,
                    source_url = excluded.source_url,
                    license = excluded.license
                """,
                (chart_id, name, frequency, jurisdiction, source_url, license_info, time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_chart(self, chart_id: str) -> dict[str, Any] | None:
        """Get chart definition by ID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart WHERE chart_id = ?", (chart_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_charts(self) -> list[dict[str, Any]]:
        """List all chart definitions."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart ORDER BY chart_id")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # Chart run management

    def create_run(
        self,
        chart_id: str,
        period: str,
        source_hash: str,
        notes: str | None = None,
    ) -> str:
        """
        Create a new chart run.

        Returns the run_id.
        """
        run_id = str(uuid.uuid4())
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart_run (run_id, chart_id, period, scraped_at, source_hash, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, chart_id, period, time.time(), source_hash, notes),
            )
            conn.commit()
            return run_id
        finally:
            conn.close()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get chart run by ID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart_run WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_run_by_period(self, chart_id: str, period: str) -> dict[str, Any] | None:
        """Get chart run by chart_id and period."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chart_run WHERE chart_id = ? AND period = ?", (chart_id, period)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_runs(self, chart_id: str | None = None) -> list[dict[str, Any]]:
        """List chart runs, optionally filtered by chart_id."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if chart_id:
                cursor.execute(
                    "SELECT * FROM chart_run WHERE chart_id = ? ORDER BY period DESC", (chart_id,)
                )
            else:
                cursor.execute("SELECT * FROM chart_run ORDER BY chart_id, period DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # Chart entry management

    @staticmethod
    def generate_entry_id(run_id: str, rank: int, side: str | None = None) -> str:
        """Generate deterministic entry ID from run_id, rank, and optional side.

        For split entries (double A-sides), include the side designation to
        ensure unique IDs when multiple entries share the same rank.
        """
        if side:
            combined = f"{run_id}:{rank}:{side}"
        else:
            combined = f"{run_id}:{rank}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def add_entry(
        self,
        run_id: str,
        rank: int,
        artist_raw: str,
        title_raw: str,
        entry_unit: EntryUnit = EntryUnit.RECORDING,
        extra_raw: str | None = None,
        previous_position: int | None = None,
        weeks_on_chart: int | None = None,
        side_designation: str | None = None,
        wikipedia_artist: str | None = None,
        wikipedia_title: str | None = None,
        history_url: str | None = None,
        scraped_at: float | None = None,
    ) -> str:
        """
        Add a chart entry to a run (Layer 1: RAW).

        Returns the entry_id.
        """
        entry_id = self.generate_entry_id(run_id, rank, side_designation)
        if scraped_at is None:
            scraped_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart_entry (entry_id, run_id, rank, artist_raw, title_raw,
                    previous_position, weeks_on_chart, side_designation,
                    wikipedia_artist, wikipedia_title, history_url,
                    entry_unit, extra_raw, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    artist_raw = excluded.artist_raw,
                    title_raw = excluded.title_raw,
                    previous_position = excluded.previous_position,
                    weeks_on_chart = excluded.weeks_on_chart,
                    side_designation = excluded.side_designation,
                    wikipedia_artist = excluded.wikipedia_artist,
                    wikipedia_title = excluded.wikipedia_title,
                    history_url = excluded.history_url,
                    entry_unit = excluded.entry_unit,
                    extra_raw = excluded.extra_raw
                """,
                (
                    entry_id,
                    run_id,
                    rank,
                    artist_raw,
                    title_raw,
                    previous_position,
                    weeks_on_chart,
                    side_designation,
                    wikipedia_artist,
                    wikipedia_title,
                    history_url,
                    entry_unit.value,
                    extra_raw,
                    scraped_at,
                ),
            )
            conn.commit()
            return entry_id
        finally:
            conn.close()

    def add_entries_batch(self, run_id: str, entries: list[ChartEntry]) -> list[str]:
        """
        Add multiple chart entries in a single transaction (Layer 1: RAW).

        Returns list of entry_ids.
        """
        conn = self._get_connection()
        entry_ids = []
        scraped_at = time.time()
        try:
            cursor = conn.cursor()
            for entry in entries:
                entry_id = self.generate_entry_id(run_id, entry.rank, entry.side_designation)
                entry_ids.append(entry_id)
                cursor.execute(
                    """
                    INSERT INTO chart_entry (entry_id, run_id, rank, artist_raw, title_raw,
                        previous_position, weeks_on_chart, side_designation,
                        wikipedia_artist, wikipedia_title, history_url,
                        entry_unit, extra_raw, scraped_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entry_id) DO UPDATE SET
                        artist_raw = excluded.artist_raw,
                        title_raw = excluded.title_raw,
                        previous_position = excluded.previous_position,
                        weeks_on_chart = excluded.weeks_on_chart,
                        side_designation = excluded.side_designation,
                        wikipedia_artist = excluded.wikipedia_artist,
                        wikipedia_title = excluded.wikipedia_title,
                        history_url = excluded.history_url,
                        entry_unit = excluded.entry_unit,
                        extra_raw = excluded.extra_raw
                    """,
                    (
                        entry_id,
                        run_id,
                        entry.rank,
                        entry.artist_raw,
                        entry.title_raw,
                        entry.previous_position,
                        entry.weeks_on_chart,
                        entry.side_designation,
                        entry.wikipedia_artist,
                        entry.wikipedia_title,
                        entry.history_url,
                        entry.entry_unit.value,
                        entry.extra_raw,
                        scraped_at,
                    ),
                )
            conn.commit()
            return entry_ids
        finally:
            conn.close()

    def get_entry(self, run_id: str, rank: int) -> dict[str, Any] | None:
        """Get a chart entry by run_id and rank."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chart_entry WHERE run_id = ? AND rank = ?", (run_id, rank)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_entries(self, run_id: str) -> list[dict[str, Any]]:
        """List all entries for a chart run."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart_entry WHERE run_id = ? ORDER BY rank", (run_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_entries_by_period(self, chart_id: str, period: str) -> list[tuple[str, str]] | None:
        """
        Get entries for a chart by period.

        Returns list of (artist_raw, title_raw) tuples, or None if run doesn't exist.
        Useful for continuity validation.
        """
        run = self.get_run_by_period(chart_id, period)
        if not run:
            return None

        entries = self.list_entries(run["run_id"])
        return [(e["artist_raw"], e["title_raw"]) for e in entries]

    def get_entries_with_ranks_by_period(
        self, chart_id: str, period: str
    ) -> dict[tuple[str, str], int] | None:
        """
        Get entries with ranks for cross-reference validation.

        Returns dict mapping (normalized_artist, normalized_title) to rank,
        or None if run doesn't exist.
        """
        run = self.get_run_by_period(chart_id, period)
        if not run:
            return None

        entries = self.list_entries(run["run_id"])
        # Return dict keyed by (artist, title) normalized for matching
        return {
            (
                self._normalize_for_match(e["artist_raw"]),
                self._normalize_for_match(e["title_raw"]),
            ): e["rank"]
            for e in entries
        }

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Normalize text for matching (lowercase, strip whitespace)."""
        return text.lower().strip()

    def get_adjacent_period(self, chart_id: str, period: str, direction: int = -1) -> str | None:
        """
        Get the adjacent existing period (previous or next).

        Args:
            chart_id: Chart ID
            period: Current period
            direction: -1 for previous, +1 for next

        Returns:
            Adjacent period string or None if not found
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if direction < 0:
                cursor.execute(
                    """
                    SELECT period FROM chart_run
                    WHERE chart_id = ? AND period < ?
                    ORDER BY period DESC LIMIT 1
                    """,
                    (chart_id, period),
                )
            else:
                cursor.execute(
                    """
                    SELECT period FROM chart_run
                    WHERE chart_id = ? AND period > ?
                    ORDER BY period ASC LIMIT 1
                    """,
                    (chart_id, period),
                )
            row = cursor.fetchone()
            return row["period"] if row else None
        finally:
            conn.close()

    # Chart link management

    def add_link(
        self,
        run_id: str,
        rank: int,
        work_key: str | None,
        link_method: LinkMethod,
        confidence: float,
        release_anchor_id: str | None = None,
        side_designation: str | None = None,
    ) -> None:
        """Add or update a chart link."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart_link (run_id, rank, work_key, link_method, confidence,
                    release_anchor_id, side_designation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, rank) DO UPDATE SET
                    work_key = excluded.work_key,
                    link_method = excluded.link_method,
                    confidence = excluded.confidence,
                    release_anchor_id = excluded.release_anchor_id,
                    side_designation = excluded.side_designation
                """,
                (
                    run_id,
                    rank,
                    work_key,
                    link_method.value,
                    confidence,
                    release_anchor_id,
                    side_designation,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def add_links_batch(self, links: list[ChartLink]) -> None:
        """Add multiple chart links in a single transaction."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for link in links:
                cursor.execute(
                    """
                    INSERT INTO chart_link (run_id, rank, work_key, link_method, confidence,
                        release_anchor_id, side_designation, recording_mbid, discogs_release_id,
                        discogs_master_id, spotify_track_id, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, rank) DO UPDATE SET
                        work_key = excluded.work_key,
                        link_method = excluded.link_method,
                        confidence = excluded.confidence,
                        release_anchor_id = excluded.release_anchor_id,
                        side_designation = excluded.side_designation,
                        recording_mbid = excluded.recording_mbid,
                        discogs_release_id = excluded.discogs_release_id,
                        discogs_master_id = excluded.discogs_master_id,
                        spotify_track_id = excluded.spotify_track_id,
                        source = excluded.source
                    """,
                    (
                        link.run_id,
                        link.rank,
                        link.work_key,
                        link.link_method.value,
                        link.confidence,
                        link.release_anchor_id,
                        link.side_designation,
                        link.recording_mbid,
                        link.discogs_release_id,
                        link.discogs_master_id,
                        link.spotify_track_id,
                        link.source,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_link(self, run_id: str, rank: int) -> dict[str, Any] | None:
        """Get a chart link by run_id and rank."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart_link WHERE run_id = ? AND rank = ?", (run_id, rank))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_links(self, run_id: str) -> list[dict[str, Any]]:
        """List all links for a chart run."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chart_link WHERE run_id = ? ORDER BY rank", (run_id,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_coverage_report(self, run_id: str) -> CoverageReport:
        """Generate coverage report for a chart run."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) as cnt FROM chart_entry WHERE run_id = ?", (run_id,))
            total = cursor.fetchone()["cnt"]

            # Linked entries (work_key not null)
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM chart_link WHERE run_id = ? AND work_key IS NOT NULL",
                (run_id,),
            )
            linked = cursor.fetchone()["cnt"]

            # By method
            cursor.execute(
                """
                SELECT link_method, COUNT(*) as cnt
                FROM chart_link WHERE run_id = ? AND work_key IS NOT NULL
                GROUP BY link_method
                """,
                (run_id,),
            )
            by_method = {row["link_method"]: row["cnt"] for row in cursor.fetchall()}

            # By confidence bucket
            cursor.execute(
                """
                SELECT
                    CASE
                        WHEN confidence >= 0.85 THEN 'auto'
                        WHEN confidence >= 0.60 THEN 'review'
                        ELSE 'reject'
                    END as bucket,
                    COUNT(*) as cnt
                FROM chart_link WHERE run_id = ?
                GROUP BY bucket
                """,
                (run_id,),
            )
            by_confidence = {row["bucket"]: row["cnt"] for row in cursor.fetchall()}

            coverage_pct = (linked / total * 100) if total > 0 else 0.0

            return CoverageReport(
                run_id=run_id,
                total_entries=total,
                linked_entries=linked,
                unlinked_entries=total - linked,
                coverage_pct=coverage_pct,
                by_method=by_method,
                by_confidence=by_confidence,
            )
        finally:
            conn.close()

    def calculate_entry_scores(
        self, chart_id: str, normalizer: Any | None = None
    ) -> dict[tuple[str, str], float]:
        """
        Calculate priority scores for chart entries across all runs.

        Score = sum of (total_entries - rank) for each appearance.
        Higher scores indicate more important/popular entries.

        Args:
            chart_id: Chart identifier (e.g., 't2000')
            normalizer: Optional Normalizer instance (will create if not provided)

        Returns:
            Dictionary mapping (artist_normalized, title_normalized) -> score
        """
        # Import here to avoid circular dependency
        if normalizer is None:
            from chart_binder.normalize import Normalizer

            normalizer = Normalizer()

        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all runs for this chart
            cursor.execute("SELECT run_id FROM chart_run WHERE chart_id = ?", (chart_id,))
            run_ids = [row["run_id"] for row in cursor.fetchall()]

            if not run_ids:
                return {}

            # Calculate total entries per run
            run_totals: dict[str, int] = {}
            for run_id in run_ids:
                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM chart_entry WHERE run_id = ?", (run_id,)
                )
                run_totals[run_id] = cursor.fetchone()["cnt"]

            # Calculate scores: sum of (total - rank) across all runs
            scores: dict[tuple[str, str], float] = {}

            for run_id in run_ids:
                total = run_totals[run_id]
                cursor.execute(
                    """
                    SELECT artist_raw, title_raw, rank
                    FROM chart_entry
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

                for row in cursor.fetchall():
                    artist_raw = row["artist_raw"]
                    title_raw = row["title_raw"]
                    rank = row["rank"]

                    # Normalize on the fly
                    artist_norm = normalizer.normalize_artist(artist_raw).core
                    title_norm = normalizer.normalize_title(title_raw).core

                    # Skip entries without normalized values
                    if not artist_norm or not title_norm:
                        continue

                    key = (artist_norm, title_norm)
                    # Spec: points = N - r + 1 (where N=total, r=rank)
                    contribution = total - rank + 1
                    scores[key] = scores.get(key, 0.0) + contribution

            return scores

        finally:
            conn.close()

    # Alias management

    def upsert_alias(
        self,
        alias_id: str,
        type: str,  # noqa: A002
        raw: str,
        normalized: str,
        ruleset_version: str = "norm-v1",
        created_at: float | None = None,
    ) -> None:
        """Upsert alias normalization record."""
        if created_at is None:
            created_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO alias_norm (alias_id, type, raw, normalized, ruleset_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(alias_id) DO UPDATE SET
                    type = excluded.type,
                    raw = excluded.raw,
                    normalized = excluded.normalized,
                    ruleset_version = excluded.ruleset_version
                """,
                (alias_id, type, raw, normalized, ruleset_version, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def get_alias(self, type: str, raw: str) -> dict[str, Any] | None:  # noqa: A002
        """Get alias normalization by type and raw text."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alias_norm WHERE type = ? AND raw = ?", (type, raw))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_aliases(self, type: str | None = None) -> list[dict[str, Any]]:  # noqa: A002
        """List all aliases, optionally filtered by type."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if type:
                cursor.execute("SELECT * FROM alias_norm WHERE type = ? ORDER BY raw", (type,))
            else:
                cursor.execute("SELECT * FROM alias_norm ORDER BY type, raw")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def verify_foreign_keys(self) -> bool:
        """Verify that foreign key constraints are enabled."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
        finally:
            conn.close()

    # Song registry management (Layer 3)

    def get_or_create_song(
        self,
        artist_canonical: str,
        title_canonical: str,
        artist_sort: str | None = None,
        work_key: str | None = None,
    ) -> str:
        """
        Get existing song or create new one.

        Returns song_id (existing or new).
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Try to find existing song
            cursor.execute(
                "SELECT song_id FROM song WHERE artist_canonical = ? AND title_canonical = ?",
                (artist_canonical, title_canonical),
            )
            row = cursor.fetchone()
            if row:
                return row["song_id"]

            # Create new song
            song_id = str(uuid.uuid4())
            cursor.execute(
                """
                INSERT INTO song (song_id, artist_canonical, title_canonical, artist_sort, work_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (song_id, artist_canonical, title_canonical, artist_sort, work_key, time.time()),
            )
            conn.commit()
            return song_id
        finally:
            conn.close()

    def get_song(self, song_id: str) -> Song | None:
        """Get song by ID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM song WHERE song_id = ?", (song_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return Song(
                song_id=row["song_id"],
                artist_canonical=row["artist_canonical"],
                title_canonical=row["title_canonical"],
                artist_sort=row["artist_sort"],
                work_key=row["work_key"],
                recording_mbid=row["recording_mbid"],
                release_group_mbid=row["release_group_mbid"],
                spotify_id=row["spotify_id"],
                isrc=row["isrc"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def get_song_by_canonical(self, artist_canonical: str, title_canonical: str) -> Song | None:
        """Get song by canonical artist and title."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM song WHERE artist_canonical = ? AND title_canonical = ?",
                (artist_canonical, title_canonical),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return Song(
                song_id=row["song_id"],
                artist_canonical=row["artist_canonical"],
                title_canonical=row["title_canonical"],
                artist_sort=row["artist_sort"],
                work_key=row["work_key"],
                recording_mbid=row["recording_mbid"],
                release_group_mbid=row["release_group_mbid"],
                spotify_id=row["spotify_id"],
                isrc=row["isrc"],
                created_at=row["created_at"],
            )
        finally:
            conn.close()

    def update_song(
        self,
        song_id: str,
        recording_mbid: str | None = None,
        release_group_mbid: str | None = None,
        spotify_id: str | None = None,
        isrc: str | None = None,
    ) -> None:
        """Update song with external IDs."""
        conn = self._get_connection()
        try:
            updates = []
            params = []
            if recording_mbid is not None:
                updates.append("recording_mbid = ?")
                params.append(recording_mbid)
            if release_group_mbid is not None:
                updates.append("release_group_mbid = ?")
                params.append(release_group_mbid)
            if spotify_id is not None:
                updates.append("spotify_id = ?")
                params.append(spotify_id)
            if isrc is not None:
                updates.append("isrc = ?")
                params.append(isrc)

            if not updates:
                return

            params.append(song_id)
            conn.execute(
                f"UPDATE song SET {', '.join(updates)} WHERE song_id = ?",
                params,
            )
            conn.commit()
        finally:
            conn.close()

    # Chart entry-song linking (Layer 2)

    def link_entry_to_song(
        self,
        entry_id: str,
        song_id: str,
        song_idx: int = 0,
        link_method: LinkMethod | None = None,
        link_confidence: float | None = None,
    ) -> str:
        """
        Link a chart entry to a song.

        Args:
            entry_id: Chart entry ID
            song_id: Song ID
            song_idx: Index for multi-song entries (0 for first/only, 1 for second, etc.)
            link_method: How the link was made
            link_confidence: Confidence score

        Returns:
            Link ID
        """
        link_id = str(uuid.uuid4())
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart_entry_song (id, entry_id, song_idx, song_id, link_method, link_confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id, song_idx) DO UPDATE SET
                    song_id = excluded.song_id,
                    link_method = excluded.link_method,
                    link_confidence = excluded.link_confidence
                """,
                (
                    link_id,
                    entry_id,
                    song_idx,
                    song_id,
                    link_method.value if link_method else None,
                    link_confidence,
                ),
            )
            conn.commit()
            return link_id
        finally:
            conn.close()

    def get_songs_for_entry(self, entry_id: str) -> list[tuple[Song, ChartEntrySong]]:
        """Get all songs linked to a chart entry."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT s.*, es.id as link_id, es.entry_id, es.song_idx, es.link_method, es.link_confidence
                FROM song s
                JOIN chart_entry_song es ON s.song_id = es.song_id
                WHERE es.entry_id = ?
                ORDER BY es.song_idx
                """,
                (entry_id,),
            )
            results = []
            for row in cursor.fetchall():
                song = Song(
                    song_id=row["song_id"],
                    artist_canonical=row["artist_canonical"],
                    title_canonical=row["title_canonical"],
                    artist_sort=row["artist_sort"],
                    work_key=row["work_key"],
                    recording_mbid=row["recording_mbid"],
                    release_group_mbid=row["release_group_mbid"],
                    spotify_id=row["spotify_id"],
                    isrc=row["isrc"],
                    created_at=row["created_at"],
                )
                link = ChartEntrySong(
                    id=row["link_id"],
                    entry_id=row["entry_id"],
                    song_idx=row["song_idx"],
                    song_id=row["song_id"],
                    link_method=LinkMethod(row["link_method"]) if row["link_method"] else None,
                    link_confidence=row["link_confidence"],
                )
                results.append((song, link))
            return results
        finally:
            conn.close()

    # Reverse query: chart history for a song

    def get_chart_history(self, song_id: str) -> list[dict[str, Any]]:
        """
        Get all chart appearances for a song (reverse query).

        This is the key query for beets tagging - given a song,
        find all its chart positions across all charts and periods.

        Returns list of dicts with:
            - chart_id, chart_name
            - period
            - rank
            - previous_position
            - weeks_on_chart
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    c.chart_id,
                    c.name as chart_name,
                    r.period,
                    e.rank,
                    e.previous_position,
                    e.weeks_on_chart,
                    r.scraped_at
                FROM song s
                JOIN chart_entry_song es ON s.song_id = es.song_id
                JOIN chart_entry e ON es.entry_id = e.entry_id
                JOIN chart_run r ON e.run_id = r.run_id
                JOIN chart c ON r.chart_id = c.chart_id
                WHERE s.song_id = ?
                ORDER BY c.chart_id, r.period
                """,
                (song_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_chart_history_by_mbid(self, recording_mbid: str) -> list[dict[str, Any]]:
        """
        Get chart history by MusicBrainz recording ID.

        Useful for beets plugin integration.
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    c.chart_id,
                    c.name as chart_name,
                    r.period,
                    e.rank,
                    e.previous_position,
                    e.weeks_on_chart,
                    s.song_id,
                    s.artist_canonical,
                    s.title_canonical
                FROM song s
                JOIN chart_entry_song es ON s.song_id = es.song_id
                JOIN chart_entry e ON es.entry_id = e.entry_id
                JOIN chart_run r ON e.run_id = r.run_id
                JOIN chart c ON r.chart_id = c.chart_id
                WHERE s.recording_mbid = ?
                ORDER BY c.chart_id, r.period
                """,
                (recording_mbid,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_unlinked_entries(self, run_id: str | None = None) -> list[dict[str, Any]]:
        """
        Get chart entries not yet linked to songs.

        Args:
            run_id: Optional run_id to filter by

        Returns:
            List of entries without song links
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if run_id:
                cursor.execute(
                    """
                    SELECT e.*, r.chart_id, r.period
                    FROM chart_entry e
                    JOIN chart_run r ON e.run_id = r.run_id
                    LEFT JOIN chart_entry_song es ON e.entry_id = es.entry_id
                    WHERE e.run_id = ? AND es.id IS NULL
                    ORDER BY e.rank
                    """,
                    (run_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT e.*, r.chart_id, r.period
                    FROM chart_entry e
                    JOIN chart_run r ON e.run_id = r.run_id
                    LEFT JOIN chart_entry_song es ON e.entry_id = es.entry_id
                    WHERE es.id IS NULL
                    ORDER BY r.period DESC, e.rank
                    """
                )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


class ChartsETL:
    """
    Charts ETL pipeline for ingesting, normalizing, and linking chart data.

    Implements the three-layer data model:
    - Layer 1: RAW (chart_entry) - immutable scrape data
    - Layer 2: LINK (chart_entry_song) - editable join table
    - Layer 3: SONG (song) - canonical registry
    """

    # Auto-link confidence threshold
    AUTO_THRESHOLD = 0.85
    REVIEW_THRESHOLD = 0.60

    def __init__(
        self,
        db: ChartsDB,
        normalizer: Normalizer | None = None,
        fetcher: Any | None = None,  # UnifiedFetcher, avoid circular import
        adjudicator: Any | None = None,  # ReActAdjudicator, avoid circular import
        config: Any | None = None,  # Config, avoid circular import
    ):
        self.db = db
        self.normalizer = normalizer or Normalizer()
        self.fetcher = fetcher  # Optional: enables multi-source canonicalization
        self.adjudicator = adjudicator  # Optional: enables LLM adjudication for ambiguous matches
        self.config = config  # Optional: enables shared resolver pipeline

    def ingest(
        self,
        chart_id: str,
        period: str,
        entries: list[tuple[int, str, str]],  # (rank, artist, title)
        entry_unit: EntryUnit = EntryUnit.RECORDING,
        notes: str | None = None,
        link_songs: bool = True,
    ) -> str:
        """
        Ingest chart data from fixture/scraped entries.

        Args:
            chart_id: Chart identifier (e.g., 't2000', 't40')
            period: Period identifier (e.g., '2024', '1991-W07')
            entries: List of (rank, artist_raw, title_raw) tuples
            entry_unit: Default entry unit type
            notes: Optional notes about the run
            link_songs: If True, also link entries to songs

        Returns:
            run_id of the created chart run
        """
        # Hash the source content for deduplication
        source_content = json.dumps(entries, sort_keys=True)
        source_hash = hashlib.sha256(source_content.encode()).hexdigest()[:16]

        # Create the run
        run_id = self.db.create_run(chart_id, period, source_hash, notes)

        # Create raw entries (Layer 1)
        chart_entries = []
        for rank, artist_raw, title_raw in entries:
            entry = ChartEntry(
                rank=rank,
                artist_raw=artist_raw,
                title_raw=title_raw,
                entry_unit=entry_unit,
            )
            chart_entries.append(entry)

        self.db.add_entries_batch(run_id, chart_entries)

        # Link to songs (Layer 2 + 3)
        if link_songs:
            self.link_to_songs(run_id)

        return run_id

    def link_to_songs(self, run_id: str) -> int:
        """
        Link chart entries to songs (Layer 2 + 3).

        For each raw entry:
        1. Normalize artist and title
        2. Find or create canonical song
        3. Link entry to song

        Returns number of entries linked.
        """
        entries = self.db.list_entries(run_id)
        linked_count = 0

        for entry in entries:
            entry_id = entry["entry_id"]
            artist_raw = entry["artist_raw"]
            title_raw = entry["title_raw"]

            # Normalize
            artist_canonical, title_canonical, link_method = self._normalize_for_song(
                artist_raw, title_raw
            )

            # Skip karaoke tracks
            if self._is_karaoke(title_raw):
                continue

            # Get or create song (Layer 3)
            song_id = self.db.get_or_create_song(artist_canonical, title_canonical)

            # Link entry to song (Layer 2)
            self.db.link_entry_to_song(
                entry_id=entry_id,
                song_id=song_id,
                song_idx=0,
                link_method=link_method,
                link_confidence=self.AUTO_THRESHOLD,
            )
            linked_count += 1

        return linked_count

    def _normalize_for_song(self, artist_raw: str, title_raw: str) -> tuple[str, str, LinkMethod]:
        """
        Normalize artist and title for song matching.

        Returns (artist_canonical, title_canonical, link_method).
        """
        # Check for alias overrides first
        artist_alias = self.db.get_alias("artist", artist_raw)
        title_alias = self.db.get_alias("title", title_raw)

        if artist_alias:
            artist_canonical = artist_alias["normalized"]
            link_method = LinkMethod.ALIAS
        else:
            artist_result = self.normalizer.normalize_artist(artist_raw)
            artist_canonical = artist_result.core
            link_method = LinkMethod.AUTO

        if title_alias:
            title_canonical = title_alias["normalized"]
            link_method = LinkMethod.ALIAS
        else:
            title_result = self.normalizer.normalize_title(title_raw)
            title_canonical = title_result.core

        return artist_canonical, title_canonical, link_method

    def _is_karaoke(self, title_raw: str) -> bool:
        """Check if title is a karaoke track."""
        title_result = self.normalizer.normalize_title(title_raw)
        for tag in title_result.tags:
            if tag.kind.value == "karaoke" or tag.sub == "karaoke":
                return True
        return False

    def _normalize_entry(
        self, rank: int, artist_raw: str, title_raw: str, entry_unit: EntryUnit
    ) -> ChartEntry:
        """Normalize a single chart entry (legacy method for backward compatibility)."""
        # Check for alias overrides first
        artist_alias = self.db.get_alias("artist", artist_raw)
        title_alias = self.db.get_alias("title", title_raw)

        if artist_alias:
            artist_normalized = artist_alias["normalized"]
        else:
            artist_result = self.normalizer.normalize_artist(artist_raw)
            artist_normalized = artist_result.core

        if title_alias:
            title_normalized = title_alias["normalized"]
            title_tags: list[dict[str, Any]] = []
        else:
            title_result = self.normalizer.normalize_title(title_raw)
            title_normalized = title_result.core
            title_tags = [
                {"kind": tag.kind.value, "sub": tag.sub, "value": tag.value, "note": tag.note}
                for tag in title_result.tags
            ]

        # Detect entry unit from tags
        detected_unit = entry_unit
        for tag in title_tags:
            if tag["kind"] == "medley":
                detected_unit = EntryUnit.MEDLEY
                break
            elif tag["kind"] == "karaoke":
                detected_unit = EntryUnit.UNKNOWN
                break

        return ChartEntry(
            rank=rank,
            artist_raw=artist_raw,
            title_raw=title_raw,
            entry_unit=detected_unit,
            artist_normalized=artist_normalized,
            title_normalized=title_normalized,
            title_tags=title_tags,
        )

    def link(
        self,
        run_id: str,
        strategy: str = "multi_source",
        missing_only: bool = False,
        min_confidence: float = 0.85,
        limit: int | None = None,
        start_rank: int | None = None,
        end_rank: int | None = None,
        prioritize_by_score: bool = False,
        chart_id: str | None = None,
        batch_size: int = 1,
        progress: bool = True,
    ) -> CoverageReport:
        """
        Link chart entries to work_keys and canonical IDs.

        Args:
            run_id: Chart run to link
            strategy: Linking strategy:
                - 'multi_source': Use UnifiedFetcher for multi-source search (default, requires fetcher)
                - 'title_artist_year': Legacy hash-based linking (no fetcher required)
                - 'bundle_release': Bundle-based linking
            missing_only: If True, skip entries that already have links with confidence >= min_confidence
            min_confidence: Minimum confidence threshold for "missing_only" filter (default 0.85)
            limit: Maximum number of entries to process (for testing)
            start_rank: Start processing from this rank (1-based, inclusive)
            end_rank: Stop processing at this rank (1-based, inclusive)
            prioritize_by_score: If True, process entries by score (requires chart_id)
            chart_id: Chart ID for score calculation (required if prioritize_by_score=True)
            batch_size: Commit every N entries for checkpoint/resume (default 1)
            progress: If True, display progress messages (default True)

        Returns:
            CoverageReport with linking results
        """
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

        entries = self.db.list_entries(run_id)
        total_entries = len(entries)

        # Filter by rank range if specified
        if start_rank is not None or end_rank is not None:
            start = start_rank if start_rank is not None else 1
            end = end_rank if end_rank is not None else total_entries
            entries = [e for e in entries if start <= e["rank"] <= end]

        # Get existing links for missing_only filter
        existing_links: dict[int, dict[str, Any]] = {}
        if missing_only:
            for link in self.db.list_links(run_id):
                existing_links[link["rank"]] = link

        # Filter missing entries
        if missing_only:
            entries = [
                e
                for e in entries
                if e["rank"] not in existing_links
                or existing_links[e["rank"]].get("confidence", 0.0) < min_confidence
            ]

        # Prioritize by score if requested
        if prioritize_by_score:
            if not chart_id:
                raise ValueError("chart_id required for score-based prioritization")

            scores = self.db.calculate_entry_scores(chart_id, normalizer=self.normalizer)

            # Sort entries by score (highest first)
            def get_score(entry: dict[str, Any]) -> float:
                # Normalize the entry to match score keys
                artist_raw = entry.get("artist_raw", "")
                title_raw = entry.get("title_raw", "")
                artist_norm = self.normalizer.normalize_artist(artist_raw).core
                title_norm = self.normalizer.normalize_title(title_raw).core
                key = (artist_norm, title_norm)
                return scores.get(key, 0.0)

            entries = sorted(entries, key=get_score, reverse=True)

        # Apply limit
        if limit is not None:
            entries = entries[:limit]

        # Process entries with batching and progress
        links = []
        processed = 0
        total_to_process = len(entries)

        # Set up Rich Progress if enabled
        progress_bar = None
        task_id = None
        if progress:
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=False,
            )
            progress_bar.start()
            task_id = progress_bar.add_task(
                f"Processing {total_to_process} entries...", total=total_to_process
            )

        for entry in entries:
            # Normalize entry fields for linking
            artist_raw = entry.get("artist_raw", "")
            title_raw = entry.get("title_raw", "")

            if artist_raw and title_raw:
                artist_result = self.normalizer.normalize_artist(artist_raw)
                title_result = self.normalizer.normalize_title(title_raw)
                entry["artist_normalized"] = artist_result.core
                entry["title_normalized"] = title_result.core
            else:
                entry["artist_normalized"] = ""
                entry["title_normalized"] = ""

            # Update progress with current entry
            rank = entry.get("rank", "?")
            if progress_bar and task_id is not None:
                progress_bar.update(
                    task_id,
                    description=f"Rank {rank}: {artist_raw} - {title_raw}",
                )

            link_result = self._compute_link(entry, strategy)

            # Update progress with match result
            if progress_bar and task_id is not None:
                work_key = link_result.get("work_key")
                confidence = link_result.get("confidence", 0.0)
                source = link_result.get("source", "unknown")

                if work_key:
                    if source == "llm_adjudicated":
                        match_info = f"🤖 LLM ({confidence:.0%})"
                    elif source == "resolver_indeterminate":
                        match_info = f"⚠ indeterminate ({confidence:.0%})"
                    else:
                        match_info = f"✓ {source} ({confidence:.0%})"
                else:
                    match_info = "✗ no match"

                progress_bar.update(
                    task_id,
                    description=f"Rank {rank}: {artist_raw} - {title_raw} → {match_info}",
                    advance=1,
                )

            link = ChartLink(
                run_id=run_id,
                rank=entry["rank"],
                work_key=link_result["work_key"],
                link_method=link_result["method"],
                confidence=link_result["confidence"],
                recording_mbid=link_result.get("recording_mbid"),
                discogs_release_id=link_result.get("discogs_release_id"),
                discogs_master_id=link_result.get("discogs_master_id"),
                spotify_track_id=link_result.get("spotify_track_id"),
                source=link_result.get("source"),
            )
            links.append(link)
            processed += 1

            # Batch commit for checkpoint/resume
            if len(links) >= batch_size:
                self.db.add_links_batch(links)
                links = []

        # Final commit for remaining links
        if links:
            self.db.add_links_batch(links)

        # Stop progress bar if it was started
        if progress_bar:
            progress_bar.stop()

        return self.db.get_coverage_report(run_id)

    def _compute_link(self, entry: dict[str, Any], strategy: str) -> dict[str, Any]:
        """
        Compute work_key link and canonical IDs for an entry.

        Returns dict with:
            - work_key: Hash-based work key (or None if below threshold)
            - confidence: Confidence score
            - method: LinkMethod enum
            - recording_mbid: MusicBrainz recording ID (if found)
            - discogs_release_id: Discogs release ID (if found)
            - discogs_master_id: Discogs master ID (if found)
            - spotify_track_id: Spotify track ID (if found)
            - source: Which source provided the match
        """
        artist_norm = entry.get("artist_normalized", "")
        title_norm = entry.get("title_normalized", "")

        if not artist_norm or not title_norm:
            return {
                "work_key": None,
                "confidence": 0.0,
                "method": LinkMethod.TITLE_ARTIST_YEAR,
            }

        # Generate work_key from normalized artist + title (always)
        work_key = self._generate_work_key(artist_norm, title_norm)

        # Multi-source search if fetcher is available and strategy allows
        if strategy == "multi_source" and self.fetcher:
            return self._compute_link_multi_source(entry, artist_norm, title_norm, work_key)

        # Fallback to legacy hash-based linking
        confidence = self._compute_confidence(entry)

        # Determine method based on strategy
        if strategy == "bundle_release":
            method = LinkMethod.BUNDLE_RELEASE
        else:
            method = LinkMethod.TITLE_ARTIST_YEAR

        # Apply thresholds
        if confidence < self.REVIEW_THRESHOLD:
            work_key = None  # Reject - don't link

        return {
            "work_key": work_key,
            "confidence": confidence,
            "method": method,
        }

    def _compute_link_multi_source(
        self, entry: dict[str, Any], artist_norm: str, title_norm: str, work_key: str
    ) -> dict[str, Any]:
        """
        Compute link using multi-source search and 7-rule CRG selection.

        Uses the shared resolve_artist_title() function which applies:
        - Multi-source search (MusicBrainz, Discogs, Spotify)
        - CandidateBuilder for evidence bundle construction
        - 7-rule CRG selection algorithm (Lead Window, Compilation Exclusion, etc.)
        - LLM adjudication for INDETERMINATE decisions

        Falls back to legacy fetcher-based linking if config is not available.
        """
        # If we have config, use the shared resolver pipeline (same as decide command)
        if self.config:
            return self._compute_link_with_resolver(entry, artist_norm, title_norm, work_key)

        # Legacy fallback: use fetcher directly without 7-rule algorithm
        return self._compute_link_legacy(entry, artist_norm, title_norm, work_key)

    def _compute_link_with_resolver(
        self, entry: dict[str, Any], artist_norm: str, title_norm: str, work_key: str
    ) -> dict[str, Any]:
        """
        Compute link using the shared resolver pipeline.

        This uses exactly the same code path as the 'decide' command, ensuring
        consistent application of the 7-rule CRG selection algorithm.
        """
        import logging

        from chart_binder.resolve import resolve_artist_title

        try:
            # Use raw artist/title if available for better search results
            artist = entry.get("artist_raw", artist_norm)
            title = entry.get("title_raw", title_norm)

            # Note: self.config is guaranteed to be non-None here since
            # _compute_link_with_resolver is only called when self.config is truthy
            result = resolve_artist_title(
                artist=artist,
                title=title,
                config=self.config,  # type: ignore[arg-type]
                adjudicator=self.adjudicator,
                fetcher=self.fetcher,  # Pass fetcher to avoid creating new instance per song
            )

            # Debug: Log what we got back
            logging.info(
                f"resolve_artist_title returned: state={result.state}, "
                f"llm_adjudicated={result.llm_adjudicated}, "
                f"confidence={result.confidence}, llm_confidence={result.llm_confidence}"
            )

            if result.state == "decided":
                link_result: dict[str, Any] = {
                    "work_key": work_key,
                    "confidence": result.confidence if result.confidence > 0 else 0.9,
                    "method": LinkMethod.MULTI_SOURCE,
                    "source": "resolver",
                }

                if result.recording_mbid:
                    link_result["recording_mbid"] = result.recording_mbid

                if result.llm_adjudicated:
                    link_result["source"] = "llm_adjudicated"
                    link_result["confidence"] = result.llm_confidence or 0.85

                return link_result

            # INDETERMINATE - return with lower confidence
            # Log warning if this looks like it should have been decided
            if result.llm_adjudicated or (result.confidence and result.confidence > 0.85):
                logging.warning(
                    f"State check failed for {artist_norm} - {title_norm}: "
                    f"state='{result.state}' (expected 'decided'), "
                    f"llm_adjudicated={result.llm_adjudicated}, confidence={result.confidence}"
                )
            else:
                logging.debug(f"Treating as indeterminate: {artist_norm} - {title_norm}")

            return {
                "work_key": work_key,
                "confidence": 0.5,
                "method": LinkMethod.TITLE_ARTIST_YEAR,
                "source": "resolver_indeterminate",
            }

        except Exception as e:
            logging.warning(f"Resolver failed for {artist_norm} - {title_norm}: {e}")
            return {
                "work_key": work_key,
                "confidence": 0.6,
                "method": LinkMethod.TITLE_ARTIST_YEAR,
            }

    def _compute_link_legacy(
        self, entry: dict[str, Any], artist_norm: str, title_norm: str, work_key: str
    ) -> dict[str, Any]:
        """
        Legacy link computation using fetcher directly.

        This is the old code path that doesn't use the 7-rule algorithm.
        Kept for backwards compatibility when config is not provided.
        """
        import logging

        if not self.fetcher:
            return {
                "work_key": work_key,
                "confidence": 0.5,
                "method": LinkMethod.TITLE_ARTIST_YEAR,
            }

        try:
            results = self.fetcher.search_recordings(title=title_norm, artist=artist_norm)

            if not results:
                return {
                    "work_key": work_key,
                    "confidence": 0.5,
                    "method": LinkMethod.TITLE_ARTIST_YEAR,
                }

            top_result = results[0]
            confidence = top_result.get("confidence", 0.7)

            if confidence < self.AUTO_THRESHOLD and self.adjudicator and len(results) > 1:
                logging.info(
                    f"Low confidence ({confidence:.2f}) for {artist_norm} - {title_norm}, "
                    f"invoking LLM adjudicator with {len(results)} candidates"
                )

                llm_result = self._adjudicate_chart_match(
                    artist_raw=entry.get("artist_raw", artist_norm),
                    title_raw=entry.get("title_raw", title_norm),
                    candidates=results[:5],
                )

                if llm_result and llm_result.get("confidence", 0) >= self.REVIEW_THRESHOLD:
                    return {
                        "work_key": work_key,
                        "confidence": llm_result["confidence"],
                        "method": LinkMethod.MULTI_SOURCE,
                        "recording_mbid": llm_result.get("recording_mbid"),
                        "source": "llm_adjudicated",
                    }

            result: dict[str, Any] = {
                "work_key": work_key,
                "confidence": confidence,
                "method": LinkMethod.MULTI_SOURCE,
                "source": top_result.get("source", "unknown"),
            }

            if top_result.get("recording_mbid"):
                result["recording_mbid"] = top_result["recording_mbid"]
            if top_result.get("discogs_release_id"):
                result["discogs_release_id"] = top_result["discogs_release_id"]
            if top_result.get("discogs_master_id"):
                result["discogs_master_id"] = top_result["discogs_master_id"]
            if top_result.get("spotify_track_id"):
                result["spotify_track_id"] = top_result["spotify_track_id"]

            return result

        except Exception as e:
            logging.warning(f"Multi-source search failed for {artist_norm} - {title_norm}: {e}")
            return {
                "work_key": work_key,
                "confidence": 0.6,
                "method": LinkMethod.TITLE_ARTIST_YEAR,
            }

    def _adjudicate_chart_match(
        self,
        artist_raw: str,
        title_raw: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Use LLM adjudicator to select best match from candidates.

        Args:
            artist_raw: Raw artist name from chart
            title_raw: Raw title from chart
            candidates: List of candidate recordings from UnifiedFetcher

        Returns:
            Dict with selected recording_mbid and confidence, or None if adjudication fails
        """
        if not self.adjudicator or not candidates:
            return None

        # Build simplified evidence bundle for chart matching
        # Format candidates as recording_candidates list
        recording_candidates = []
        for candidate in candidates:
            recording_candidates.append(
                {
                    "mb_recording_id": candidate.get("recording_mbid"),
                    "title": candidate.get("title", title_raw),
                    "artist": candidate.get("artist", artist_raw),
                    "confidence": candidate.get("confidence", 0.7),
                    "source": candidate.get("source", "unknown"),
                    "rg_candidates": [],  # Simplified for chart matching
                }
            )

        evidence_bundle = {
            "artist": {
                "name": artist_raw,
            },
            "recording_title": title_raw,
            "recording_candidates": recording_candidates,
            "context": "chart_linking",  # Flag to indicate this is chart linking, not audio file
        }

        try:
            # Invoke adjudicator
            adjudication_result = self.adjudicator.adjudicate(evidence_bundle)

            # Extract recording MBID from LLM response
            # Note: LLM returns crg_mbid (release group), but for charts we want recording_mbid
            # For now, try to match the LLM's choice back to one of the candidates
            if adjudication_result.outcome.value in ("accepted", "review"):
                # Use the first candidate as LLM-approved match
                # TODO: Improve matching logic to find exact candidate LLM selected
                # by comparing adjudication_result.rr_mbid with candidate IDs
                for candidate in candidates:
                    return {
                        "recording_mbid": candidate.get("recording_mbid"),
                        "confidence": adjudication_result.confidence,
                        "rationale": adjudication_result.rationale,
                    }

            return None

        except Exception as e:
            import logging

            logging.warning(f"LLM adjudication failed for {artist_raw} - {title_raw}: {e}")
            return None

    def _generate_work_key(self, artist_norm: str, title_norm: str) -> str:
        """Generate deterministic work_key from normalized artist + title."""
        combined = f"{artist_norm} // {title_norm}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _compute_confidence(self, entry: dict[str, Any]) -> float:
        """
        Compute confidence score for linking.

        Based on normalization quality and tag presence.
        """
        base_confidence = 0.85  # Start with high confidence

        # Reduce confidence for certain tag types
        tags_json = entry.get("title_tags_json")
        if tags_json:
            tags = json.loads(tags_json)
            for tag in tags:
                kind = tag.get("kind", "")
                sub = tag.get("sub", "")
                if kind == "karaoke" or sub == "karaoke":
                    return 0.0  # Never link karaoke
                elif kind == "medley":
                    base_confidence -= 0.1  # Medleys need careful handling
                elif kind == "live":
                    base_confidence -= 0.05  # Live versions may have multiple RGs
                elif kind == "re_recording":
                    base_confidence -= 0.1  # Re-recordings need version matching

        # Entry unit affects confidence
        entry_unit = entry.get("entry_unit", "recording")
        if entry_unit == "single_release":
            base_confidence -= 0.1  # Single bundles need side mapping
        elif entry_unit == "medley":
            base_confidence -= 0.15
        elif entry_unit == "unknown":
            base_confidence -= 0.2

        return max(0.0, min(1.0, base_confidence))

    def get_missing_entries(self, run_id: str, threshold: float = 0.60) -> list[dict[str, Any]]:
        """Get entries that failed to link above threshold."""
        conn = self.db._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT e.*, l.work_key, l.confidence, l.link_method
                FROM chart_entry e
                LEFT JOIN chart_link l ON e.run_id = l.run_id AND e.rank = l.rank
                WHERE e.run_id = ? AND (l.work_key IS NULL OR l.confidence < ?)
                ORDER BY e.rank
                """,
                (run_id, threshold),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


## Tests


def test_charts_db_init(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    assert db.db_path.exists()
    assert db.verify_foreign_keys()

    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    # Core tables
    assert "chart" in tables
    assert "chart_run" in tables
    assert "chart_entry" in tables
    assert "alias_norm" in tables

    # Three-layer model tables
    assert "song" in tables
    assert "chart_entry_song" in tables

    # Legacy compatibility
    assert "chart_link" in tables


def test_chart_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y", "NL")

    chart = db.get_chart("t2000")
    assert chart is not None
    assert chart["name"] == "NPO Radio 2 Top 2000"
    assert chart["frequency"] == "y"
    assert chart["jurisdiction"] == "NL"


def test_chart_run_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y", "NL")
    run_id = db.create_run("t2000", "2024", "abc123")

    run = db.get_run(run_id)
    assert run is not None
    assert run["chart_id"] == "t2000"
    assert run["period"] == "2024"

    run2 = db.get_run_by_period("t2000", "2024")
    assert run2 is not None
    assert run2["run_id"] == run_id


def test_chart_entry_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")
    run_id = db.create_run("t2000", "2024", "hash123")

    db.add_entry(run_id, 1, "Queen", "Bohemian Rhapsody")
    db.add_entry(run_id, 2, "Nirvana", "Smells Like Teen Spirit")

    entry = db.get_entry(run_id, 1)
    assert entry is not None
    assert entry["artist_raw"] == "Queen"
    assert entry["title_raw"] == "Bohemian Rhapsody"

    entries = db.list_entries(run_id)
    assert len(entries) == 2


def test_chart_link_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")
    run_id = db.create_run("t2000", "2024", "hash123")
    db.add_entry(run_id, 1, "Queen", "Bohemian Rhapsody")

    db.add_link(run_id, 1, "work_abc123", LinkMethod.TITLE_ARTIST_YEAR, 0.95)

    link = db.get_link(run_id, 1)
    assert link is not None
    assert link["work_key"] == "work_abc123"
    assert link["confidence"] == 0.95


def test_coverage_report(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")
    run_id = db.create_run("t2000", "2024", "hash123")

    # Add 5 entries
    for i in range(1, 6):
        db.add_entry(run_id, i, f"Artist {i}", f"Title {i}")

    # Link 3 of them
    db.add_link(run_id, 1, "work_1", LinkMethod.TITLE_ARTIST_YEAR, 0.95)
    db.add_link(run_id, 2, "work_2", LinkMethod.TITLE_ARTIST_YEAR, 0.75)
    db.add_link(run_id, 3, None, LinkMethod.TITLE_ARTIST_YEAR, 0.40)  # Rejected
    db.add_link(run_id, 4, "work_4", LinkMethod.BUNDLE_RELEASE, 0.90)

    report = db.get_coverage_report(run_id)
    assert report.total_entries == 5
    assert report.linked_entries == 3  # work_1, work_2, work_4
    assert report.coverage_pct == 60.0


def test_alias_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_alias("alias-1", "artist", "The Beatles", "beatles")

    alias = db.get_alias("artist", "The Beatles")
    assert alias is not None
    assert alias["normalized"] == "beatles"
    assert alias["type"] == "artist"

    db.upsert_alias("alias-1", "artist", "The Beatles", "the beatles")
    alias = db.get_alias("artist", "The Beatles")
    assert alias is not None
    assert alias["normalized"] == "the beatles"


def test_alias_list(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_alias("alias-1", "artist", "The Beatles", "beatles")
    db.upsert_alias("alias-2", "title", "Let It Be", "let it be")
    db.upsert_alias("alias-3", "artist", "Queen", "queen")

    all_aliases = db.list_aliases()
    assert len(all_aliases) == 3

    artist_aliases = db.list_aliases(type="artist")
    assert len(artist_aliases) == 2

    title_aliases = db.list_aliases(type="title")
    assert len(title_aliases) == 1


def test_charts_etl_ingest(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")

    entries = [
        (1, "Queen", "Bohemian Rhapsody"),
        (2, "Nirvana", "Smells Like Teen Spirit (Live at Reading 1992)"),
        (3, "The Beatles", "Let It Be"),
    ]

    run_id = etl.ingest("t2000", "2024", entries)

    run = db.get_run(run_id)
    assert run is not None
    assert run["period"] == "2024"

    db_entries = db.list_entries(run_id)
    assert len(db_entries) == 3

    # Check raw entry stored correctly (Layer 1)
    entry1 = db.get_entry(run_id, 1)
    assert entry1 is not None
    assert entry1["artist_raw"] == "Queen"
    assert entry1["title_raw"] == "Bohemian Rhapsody"
    assert entry1["entry_id"] is not None

    # Check song was created (Layer 3)
    song = db.get_song_by_canonical("queen", "bohemian rhapsody")
    assert song is not None

    # Check entry is linked to song (Layer 2)
    songs = db.get_songs_for_entry(entry1["entry_id"])
    assert len(songs) == 1
    assert songs[0][0].artist_canonical == "queen"


def test_charts_etl_link_to_songs(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")

    entries = [
        (1, "Queen", "Bohemian Rhapsody"),
        (2, "Nirvana", "Smells Like Teen Spirit"),
    ]

    # Ingest without linking first
    run_id = etl.ingest("t2000", "2024", entries, link_songs=False)

    # Manually link
    linked = etl.link_to_songs(run_id)
    assert linked == 2

    # Check songs were created
    song1 = db.get_song_by_canonical("queen", "bohemian rhapsody")
    assert song1 is not None

    song2 = db.get_song_by_canonical("nirvana", "smells like teen spirit")
    assert song2 is not None


def test_charts_etl_karaoke_rejection(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    db.upsert_chart("t40", "Top 40", "w")

    entries = [
        (1, "Queen", "Bohemian Rhapsody"),
        (2, "Unknown Artist", "Bohemian Rhapsody (Karaoke Version)"),
    ]

    run_id = etl.ingest("t40", "1999-W25", entries)

    # Both entries should be in chart_entry (Layer 1)
    db_entries = db.list_entries(run_id)
    assert len(db_entries) == 2

    # Only non-karaoke entry should be linked to song (Layer 2)
    entry1 = db.get_entry(run_id, 1)
    entry2 = db.get_entry(run_id, 2)
    assert entry1 is not None
    assert entry2 is not None

    songs1 = db.get_songs_for_entry(entry1["entry_id"])
    songs2 = db.get_songs_for_entry(entry2["entry_id"])

    assert len(songs1) == 1  # Queen linked
    assert len(songs2) == 0  # Karaoke not linked


def test_chart_history_query(tmp_path):
    """Test reverse query: get all chart appearances for a song."""
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    # Create charts
    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")
    db.upsert_chart("t40", "Top 40", "w")

    # Song appears in multiple charts/runs
    entries_t2000_2023 = [(1, "Queen", "Bohemian Rhapsody")]
    entries_t2000_2024 = [(5, "Queen", "Bohemian Rhapsody")]  # Different position
    entries_t40 = [(1, "Queen", "Bohemian Rhapsody")]

    etl.ingest("t2000", "2023", entries_t2000_2023)
    etl.ingest("t2000", "2024", entries_t2000_2024)
    etl.ingest("t40", "2024-W01", entries_t40)

    # Get the song
    song = db.get_song_by_canonical("queen", "bohemian rhapsody")
    assert song is not None

    # Query chart history
    history = db.get_chart_history(song.song_id)
    assert len(history) == 3

    # Should have entries from both charts
    chart_ids = {h["chart_id"] for h in history}
    assert "t2000" in chart_ids
    assert "t40" in chart_ids
