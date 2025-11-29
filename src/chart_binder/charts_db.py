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
    MANUAL = "manual"


@dataclass
class ChartEntry:
    """A single entry in a chart run."""

    rank: int
    artist_raw: str
    title_raw: str
    entry_unit: EntryUnit = EntryUnit.RECORDING
    extra_raw: str | None = None

    # Normalized fields (computed)
    artist_normalized: str = ""
    title_normalized: str = ""
    title_tags: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ChartLink:
    """Link between chart entry and work_key."""

    run_id: str
    rank: int
    work_key: str | None
    link_method: LinkMethod
    confidence: float
    release_anchor_id: str | None = None
    side_designation: str | None = None  # A, B, AA, or None


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

            CREATE TABLE IF NOT EXISTS chart_entry (
                run_id TEXT NOT NULL REFERENCES chart_run(run_id),
                rank INTEGER NOT NULL,
                artist_raw TEXT NOT NULL,
                title_raw TEXT NOT NULL,
                entry_unit TEXT NOT NULL,
                extra_raw TEXT,
                artist_normalized TEXT,
                title_normalized TEXT,
                title_tags_json TEXT,
                PRIMARY KEY (run_id, rank)
            );

            CREATE TABLE IF NOT EXISTS chart_link (
                run_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                work_key TEXT,
                link_method TEXT NOT NULL,
                confidence REAL NOT NULL,
                release_anchor_id TEXT,
                side_designation TEXT,
                PRIMARY KEY (run_id, rank),
                FOREIGN KEY (run_id, rank) REFERENCES chart_entry(run_id, rank)
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
            CREATE INDEX IF NOT EXISTS idx_entry_artist_title ON chart_entry(artist_raw, title_raw);
            CREATE INDEX IF NOT EXISTS idx_link_work ON chart_link(work_key);
            CREATE INDEX IF NOT EXISTS idx_link_confidence ON chart_link(confidence);
            CREATE INDEX IF NOT EXISTS idx_run_chart ON chart_run(chart_id);
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

    def add_entry(
        self,
        run_id: str,
        rank: int,
        artist_raw: str,
        title_raw: str,
        entry_unit: EntryUnit = EntryUnit.RECORDING,
        extra_raw: str | None = None,
        artist_normalized: str | None = None,
        title_normalized: str | None = None,
        title_tags_json: str | None = None,
    ) -> None:
        """Add a chart entry to a run."""
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chart_entry (run_id, rank, artist_raw, title_raw, entry_unit,
                    extra_raw, artist_normalized, title_normalized, title_tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, rank) DO UPDATE SET
                    artist_raw = excluded.artist_raw,
                    title_raw = excluded.title_raw,
                    entry_unit = excluded.entry_unit,
                    extra_raw = excluded.extra_raw,
                    artist_normalized = excluded.artist_normalized,
                    title_normalized = excluded.title_normalized,
                    title_tags_json = excluded.title_tags_json
                """,
                (
                    run_id,
                    rank,
                    artist_raw,
                    title_raw,
                    entry_unit.value,
                    extra_raw,
                    artist_normalized,
                    title_normalized,
                    title_tags_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def add_entries_batch(self, run_id: str, entries: list[ChartEntry]) -> None:
        """Add multiple chart entries in a single transaction."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for entry in entries:
                tags_json = json.dumps(entry.title_tags) if entry.title_tags else None
                cursor.execute(
                    """
                    INSERT INTO chart_entry (run_id, rank, artist_raw, title_raw, entry_unit,
                        extra_raw, artist_normalized, title_normalized, title_tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, rank) DO UPDATE SET
                        artist_raw = excluded.artist_raw,
                        title_raw = excluded.title_raw,
                        entry_unit = excluded.entry_unit,
                        extra_raw = excluded.extra_raw,
                        artist_normalized = excluded.artist_normalized,
                        title_normalized = excluded.title_normalized,
                        title_tags_json = excluded.title_tags_json
                    """,
                    (
                        run_id,
                        entry.rank,
                        entry.artist_raw,
                        entry.title_raw,
                        entry.entry_unit.value,
                        entry.extra_raw,
                        entry.artist_normalized,
                        entry.title_normalized,
                        tags_json,
                    ),
                )
            conn.commit()
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

    def get_entries_by_period(
        self, chart_id: str, period: str
    ) -> list[tuple[str, str]] | None:
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

    def get_adjacent_period(
        self, chart_id: str, period: str, direction: int = -1
    ) -> str | None:
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
                        link.run_id,
                        link.rank,
                        link.work_key,
                        link.link_method.value,
                        link.confidence,
                        link.release_anchor_id,
                        link.side_designation,
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


class ChartsETL:
    """
    Charts ETL pipeline for ingesting, normalizing, and linking chart data.

    Implements the Charts ETL Core from Epic 8.
    """

    # Auto-link confidence threshold
    AUTO_THRESHOLD = 0.85
    REVIEW_THRESHOLD = 0.60

    def __init__(self, db: ChartsDB, normalizer: Normalizer | None = None):
        self.db = db
        self.normalizer = normalizer or Normalizer()

    def ingest(
        self,
        chart_id: str,
        period: str,
        entries: list[tuple[int, str, str]],  # (rank, artist, title)
        entry_unit: EntryUnit = EntryUnit.RECORDING,
        notes: str | None = None,
    ) -> str:
        """
        Ingest chart data from fixture/scraped entries.

        Args:
            chart_id: Chart identifier (e.g., 't2000', 't40')
            period: Period identifier (e.g., '2024', '1991-W07')
            entries: List of (rank, artist_raw, title_raw) tuples
            entry_unit: Default entry unit type
            notes: Optional notes about the run

        Returns:
            run_id of the created chart run
        """
        # Hash the source content for deduplication
        source_content = json.dumps(entries, sort_keys=True)
        source_hash = hashlib.sha256(source_content.encode()).hexdigest()[:16]

        # Create the run
        run_id = self.db.create_run(chart_id, period, source_hash, notes)

        # Normalize and add entries
        normalized_entries = []
        for rank, artist_raw, title_raw in entries:
            entry = self._normalize_entry(rank, artist_raw, title_raw, entry_unit)
            normalized_entries.append(entry)

        self.db.add_entries_batch(run_id, normalized_entries)

        return run_id

    def _normalize_entry(
        self, rank: int, artist_raw: str, title_raw: str, entry_unit: EntryUnit
    ) -> ChartEntry:
        """Normalize a single chart entry using alias registry and normalizer."""
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

    def link(self, run_id: str, strategy: str = "title_artist_year") -> CoverageReport:
        """
        Link chart entries to work_keys.

        Args:
            run_id: Chart run to link
            strategy: Linking strategy ('title_artist_year', 'bundle_release', etc.)

        Returns:
            CoverageReport with linking results
        """
        entries = self.db.list_entries(run_id)
        links = []

        for entry in entries:
            work_key, confidence, method = self._compute_link(entry, strategy)

            link = ChartLink(
                run_id=run_id,
                rank=entry["rank"],
                work_key=work_key,
                link_method=method,
                confidence=confidence,
            )
            links.append(link)

        self.db.add_links_batch(links)

        return self.db.get_coverage_report(run_id)

    def _compute_link(
        self, entry: dict[str, Any], strategy: str
    ) -> tuple[str | None, float, LinkMethod]:
        """
        Compute work_key link for an entry.

        Returns (work_key, confidence, method).
        """
        artist_norm = entry.get("artist_normalized", "")
        title_norm = entry.get("title_normalized", "")

        if not artist_norm or not title_norm:
            return None, 0.0, LinkMethod.TITLE_ARTIST_YEAR

        # Generate work_key from normalized artist + title
        work_key = self._generate_work_key(artist_norm, title_norm)

        # Base confidence from normalization quality
        confidence = self._compute_confidence(entry)

        # Determine method based on strategy
        if strategy == "bundle_release":
            method = LinkMethod.BUNDLE_RELEASE
        else:
            method = LinkMethod.TITLE_ARTIST_YEAR

        # Apply thresholds
        if confidence < self.REVIEW_THRESHOLD:
            work_key = None  # Reject - don't link

        return work_key, confidence, method

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

    assert "chart" in tables
    assert "chart_run" in tables
    assert "chart_entry" in tables
    assert "chart_link" in tables
    assert "alias_norm" in tables


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

    # Check normalization
    entry1 = db.get_entry(run_id, 1)
    assert entry1 is not None
    assert entry1["artist_normalized"] == "queen"
    assert entry1["title_normalized"] == "bohemian rhapsody"


def test_charts_etl_link(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    db.upsert_chart("t2000", "NPO Radio 2 Top 2000", "y")

    entries = [
        (1, "Queen", "Bohemian Rhapsody"),
        (2, "Nirvana", "Smells Like Teen Spirit"),
    ]

    run_id = etl.ingest("t2000", "2024", entries)
    report = etl.link(run_id)

    assert report.total_entries == 2
    assert report.linked_entries == 2
    assert report.coverage_pct == 100.0


def test_charts_etl_karaoke_rejection(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    etl = ChartsETL(db)

    db.upsert_chart("t40", "Top 40", "w")

    entries = [
        (1, "Queen", "Bohemian Rhapsody"),
        (2, "Unknown Artist", "Bohemian Rhapsody (Karaoke Version)"),
    ]

    run_id = etl.ingest("t40", "1999-W25", entries)
    report = etl.link(run_id)

    # Only the first entry should be linked
    assert report.linked_entries == 1

    # Check that karaoke entry was rejected
    link2 = db.get_link(run_id, 2)
    assert link2 is not None
    assert link2["work_key"] is None
    assert link2["confidence"] == 0.0
