"""
Decisions database for storing canonicalization decisions and drift tracking.

Implements the decision store and state machine per Epic 6.
Tracks file artifacts, decisions, history, and overrides.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class DecisionState(StrEnum):
    """State of canonicalization decision per drift contract."""

    UNDECIDED = "undecided"
    DECIDED = "decided"
    STALE_EVIDENCE = "stale_evidence"
    STALE_RULES = "stale_rules"
    STALE_BOTH = "stale_both"
    STALE_NONDETERMINISTIC = "stale_nondeterministic"
    INDETERMINATE = "indeterminate"


class SupersededReason(StrEnum):
    """Reason for decision superseding."""

    REFRESH = "refresh"
    RULESET_CHANGE = "ruleset_change"
    MANUAL_OVERRIDE = "manual_override"
    PIN = "pin"


@dataclass
class FileArtifact:
    """File artifact signature and metadata."""

    file_id: str
    library_root: str
    relative_path: str
    duration_ms: int | None = None
    fp_id: str | None = None
    orig_tags_hash: str | None = None
    created_at: float | None = None


@dataclass
class Decision:
    """Canonicalization decision for a file."""

    decision_id: str
    file_id: str
    work_key: str
    mb_rg_id: str  # Canonical Release Group
    mb_release_id: str  # Representative Release
    mb_recording_id: str | None
    ruleset_version: str
    config_snapshot_json: str
    evidence_hash: str
    trace_compact: str
    state: DecisionState
    pinned: bool = False
    created_at: float | None = None
    updated_at: float | None = None


@dataclass
class DecisionHistory:
    """Archived decision history."""

    decision_id: str
    file_id: str
    work_key: str
    mb_rg_id: str
    mb_release_id: str
    mb_recording_id: str | None
    ruleset_version: str
    config_snapshot_json: str
    evidence_hash: str
    trace_compact: str
    state: DecisionState
    pinned: bool
    created_at: float
    updated_at: float
    superseded_at: float
    superseded_reason: SupersededReason


class DecisionsDB:
    """
    SQLite database for decision storage and drift tracking.

    Provides schema creation and CRUD operations for:
    - file_artifact: File signatures and metadata
    - decision: Current decisions with state machine
    - decision_history: Archived decisions
    - override_rule: Manual overrides (future)
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _db_connection(self, exclusive: bool = False) -> Iterator[sqlite3.Connection]:
        """
        Context manager for database connections.

        Automatically handles connection lifecycle and ensures cleanup.

        Args:
            exclusive: If True, sets isolation level to EXCLUSIVE for
                      atomic operations that prevent race conditions.
        """
        conn = self._get_connection()
        if exclusive:
            conn.isolation_level = "EXCLUSIVE"
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS file_artifact (
                file_id TEXT PRIMARY KEY,
                library_root TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                duration_ms INTEGER,
                fp_id TEXT,
                orig_tags_hash TEXT,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_file_path
                ON file_artifact(library_root, relative_path);

            CREATE TABLE IF NOT EXISTS decision (
                decision_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL UNIQUE,
                work_key TEXT NOT NULL,
                mb_rg_id TEXT NOT NULL,
                mb_release_id TEXT NOT NULL,
                mb_recording_id TEXT,
                ruleset_version TEXT NOT NULL,
                config_snapshot_json TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                trace_compact TEXT NOT NULL,
                state TEXT NOT NULL,
                pinned INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (file_id) REFERENCES file_artifact(file_id)
            );

            CREATE INDEX IF NOT EXISTS idx_decision_file ON decision(file_id);
            CREATE INDEX IF NOT EXISTS idx_decision_state ON decision(state);
            CREATE INDEX IF NOT EXISTS idx_decision_rg ON decision(mb_rg_id);

            CREATE TABLE IF NOT EXISTS decision_history (
                decision_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                work_key TEXT NOT NULL,
                mb_rg_id TEXT NOT NULL,
                mb_release_id TEXT NOT NULL,
                mb_recording_id TEXT,
                ruleset_version TEXT NOT NULL,
                config_snapshot_json TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                trace_compact TEXT NOT NULL,
                state TEXT NOT NULL,
                pinned INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                superseded_at REAL NOT NULL,
                superseded_reason TEXT NOT NULL,
                PRIMARY KEY (decision_id, superseded_at)
            );

            CREATE INDEX IF NOT EXISTS idx_history_file ON decision_history(file_id);
            CREATE INDEX IF NOT EXISTS idx_history_superseded
                ON decision_history(superseded_at);

            CREATE TABLE IF NOT EXISTS override_rule (
                override_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                directive TEXT,
                note TEXT,
                created_by TEXT,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_override_scope
                ON override_rule(scope, scope_id);

            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO schema_meta (key, value)
                VALUES ('db_version_decisions', '1');
            """
        )

        conn.commit()
        conn.close()
        self._ensure_override_columns()

    def _ensure_override_columns(self) -> None:
        """Ensure override_rule has structured override columns."""
        with self._db_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(override_rule)")
            columns = {row[1] for row in cursor.fetchall()}
            updates: list[str] = []

            if "override_type" not in columns:
                updates.append("ALTER TABLE override_rule ADD COLUMN override_type TEXT")
            if "target_crg_mbid" not in columns:
                updates.append("ALTER TABLE override_rule ADD COLUMN target_crg_mbid TEXT")
            if "target_rr_mbid" not in columns:
                updates.append("ALTER TABLE override_rule ADD COLUMN target_rr_mbid TEXT")
            if "target_label" not in columns:
                updates.append("ALTER TABLE override_rule ADD COLUMN target_label TEXT")
            if "override_behavior" not in columns:
                updates.append("ALTER TABLE override_rule ADD COLUMN override_behavior TEXT")

            for statement in updates:
                conn.execute(statement)
            if updates:
                conn.commit()

    @staticmethod
    def generate_file_id(fingerprint: str, duration_sec: int) -> str:
        """
        Generate stable file ID from audio fingerprint and duration.

        This creates an acoustic identity that survives tag edits,
        file renames, and file moves.

        Args:
            fingerprint: Chromaprint audio fingerprint
            duration_sec: Track duration in seconds

        Returns:
            SHA-256 hash of fingerprint:duration
        """
        signature = f"{fingerprint}:{duration_sec}"
        return hashlib.sha256(signature.encode()).hexdigest()

    @staticmethod
    def generate_file_id_fallback(path: Path, size: int, mtime: float) -> str:
        """
        Generate fallback file ID from path, size, and mtime.

        DEPRECATED: Use generate_file_id with fingerprint for stable identity.
        This method is fragile - changes when tags are modified.
        """
        signature = f"fallback:{path}:{size}:{int(mtime)}"
        return hashlib.sha256(signature.encode()).hexdigest()

    def upsert_file_artifact(
        self,
        file_id: str,
        library_root: str,
        relative_path: str,
        duration_ms: int | None = None,
        fp_id: str | None = None,
        orig_tags_hash: str | None = None,
        created_at: float | None = None,
    ) -> None:
        """Upsert file artifact record."""
        if created_at is None:
            created_at = time.time()

        with self._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO file_artifact
                    (file_id, library_root, relative_path, duration_ms, fp_id,
                     orig_tags_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    library_root = excluded.library_root,
                    relative_path = excluded.relative_path,
                    duration_ms = excluded.duration_ms,
                    fp_id = excluded.fp_id,
                    orig_tags_hash = excluded.orig_tags_hash
                """,
                (
                    file_id,
                    library_root,
                    relative_path,
                    duration_ms,
                    fp_id,
                    orig_tags_hash,
                    created_at,
                ),
            )
            conn.commit()

    def get_file_artifact(self, file_id: str) -> dict[str, Any] | None:
        """Get file artifact by ID."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM file_artifact WHERE file_id = ?", (file_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def upsert_decision(
        self,
        file_id: str,
        work_key: str,
        mb_rg_id: str,
        mb_release_id: str,
        mb_recording_id: str | None,
        ruleset_version: str,
        config_snapshot: dict[str, Any],
        evidence_hash: str,
        trace_compact: str,
        state: DecisionState = DecisionState.DECIDED,
        pinned: bool = False,
        decision_id: str | None = None,
    ) -> str:
        """
        Upsert decision record.

        If decision exists for this file_id, archives the old one to history
        and creates a new decision.

        Returns decision_id (new or existing).

        Uses exclusive transaction to prevent race conditions when concurrent
        processes attempt to upsert decisions for the same file_id.
        """
        if decision_id is None:
            decision_id = str(uuid.uuid4())

        config_json = json.dumps(config_snapshot, sort_keys=True)
        now = time.time()

        with self._db_connection(exclusive=True) as conn:
            # Check if decision exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM decision WHERE file_id = ?",
                (file_id,),
            )
            existing = cursor.fetchone()

            if existing:
                # Archive existing decision to history
                conn.execute(
                    """
                    INSERT INTO decision_history
                        (decision_id, file_id, work_key, mb_rg_id, mb_release_id,
                         mb_recording_id, ruleset_version, config_snapshot_json,
                         evidence_hash, trace_compact, state, pinned,
                         created_at, updated_at, superseded_at, superseded_reason)
                    SELECT decision_id, file_id, work_key, mb_rg_id, mb_release_id,
                           mb_recording_id, ruleset_version, config_snapshot_json,
                           evidence_hash, trace_compact, state, pinned,
                           created_at, updated_at, ?, ?
                    FROM decision
                    WHERE file_id = ?
                    """,
                    (now, SupersededReason.REFRESH, file_id),
                )

                # Update existing decision
                conn.execute(
                    """
                    UPDATE decision SET
                        decision_id = ?,
                        work_key = ?,
                        mb_rg_id = ?,
                        mb_release_id = ?,
                        mb_recording_id = ?,
                        ruleset_version = ?,
                        config_snapshot_json = ?,
                        evidence_hash = ?,
                        trace_compact = ?,
                        state = ?,
                        pinned = ?,
                        updated_at = ?
                    WHERE file_id = ?
                    """,
                    (
                        decision_id,
                        work_key,
                        mb_rg_id,
                        mb_release_id,
                        mb_recording_id,
                        ruleset_version,
                        config_json,
                        evidence_hash,
                        trace_compact,
                        state,
                        1 if pinned else 0,
                        now,
                        file_id,
                    ),
                )
            else:
                # Insert new decision
                conn.execute(
                    """
                    INSERT INTO decision
                        (decision_id, file_id, work_key, mb_rg_id, mb_release_id,
                         mb_recording_id, ruleset_version, config_snapshot_json,
                         evidence_hash, trace_compact, state, pinned,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision_id,
                        file_id,
                        work_key,
                        mb_rg_id,
                        mb_release_id,
                        mb_recording_id,
                        ruleset_version,
                        config_json,
                        evidence_hash,
                        trace_compact,
                        state,
                        1 if pinned else 0,
                        now,
                        now,
                    ),
                )

            conn.commit()
            return decision_id

    def get_decision(self, file_id: str) -> dict[str, Any] | None:
        """Get current decision for file."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM decision WHERE file_id = ?", (file_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stale_decisions(self) -> list[dict[str, Any]]:
        """Get all decisions in STALE-* states."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM decision
                WHERE state LIKE 'stale_%' OR state = 'indeterminate'
                ORDER BY updated_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_decision_history(self, file_id: str) -> list[dict[str, Any]]:
        """Get decision history for a file."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM decision_history
                WHERE file_id = ?
                ORDER BY superseded_at DESC
                """,
                (file_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def update_decision_state(self, file_id: str, new_state: DecisionState) -> None:
        """Update decision state (for drift detection)."""
        with self._db_connection() as conn:
            conn.execute(
                """
                UPDATE decision
                SET state = ?, updated_at = ?
                WHERE file_id = ?
                """,
                (new_state, time.time(), file_id),
            )
            conn.commit()

    def set_pinned(self, decision_id: str, pinned: bool) -> None:
        """Set the pinned status of a decision."""
        with self._db_connection() as conn:
            conn.execute(
                """
                UPDATE decision
                SET pinned = ?, updated_at = ?
                WHERE decision_id = ?
                """,
                (1 if pinned else 0, time.time(), decision_id),
            )
            conn.commit()

    def create_override_rule(
        self,
        scope: str,
        scope_id: str,
        directive: str,
        note: str | None = None,
        created_by: str | None = None,
    ) -> str:
        """Create an override rule."""
        override_id = str(uuid.uuid4())
        now = time.time()

        with self._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO override_rule
                    (override_id, scope, scope_id, directive, note, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (override_id, scope, scope_id, directive, note, created_by, now),
            )
            conn.commit()
            return override_id

    def create_override(
        self,
        scope: str,
        scope_id: str,
        override_type: str,
        target_crg_mbid: str | None = None,
        target_rr_mbid: str | None = None,
        target_label: str | None = None,
        override_behavior: str | None = None,
        note: str | None = None,
        created_by: str | None = None,
    ) -> str:
        """Create a structured override rule."""
        override_id = str(uuid.uuid4())
        now = time.time()

        with self._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO override_rule
                    (
                        override_id,
                        scope,
                        scope_id,
                        override_type,
                        target_crg_mbid,
                        target_rr_mbid,
                        target_label,
                        override_behavior,
                        note,
                        created_by,
                        created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    override_id,
                    scope,
                    scope_id,
                    override_type,
                    target_crg_mbid,
                    target_rr_mbid,
                    target_label,
                    override_behavior,
                    note,
                    created_by,
                    now,
                ),
            )
            conn.commit()
            return override_id

    def get_override_rules(self, scope: str, scope_id: str) -> list[dict[str, Any]]:
        """Get override rules for a given scope."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM override_rule
                WHERE scope = ? AND scope_id = ?
                ORDER BY created_at DESC
                """,
                (scope, scope_id),
            )
            return [dict(row) for row in cursor.fetchall()]

    def list_override_rules(
        self,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List override rules with optional filtering."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if scope and scope_id:
                cursor.execute(
                    """
                    SELECT * FROM override_rule
                    WHERE scope = ? AND scope_id = ?
                    ORDER BY created_at DESC
                    """,
                    (scope, scope_id),
                )
            elif scope:
                cursor.execute(
                    """
                    SELECT * FROM override_rule
                    WHERE scope = ?
                    ORDER BY created_at DESC
                    """,
                    (scope,),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM override_rule
                    ORDER BY created_at DESC
                    """
                )

            return [dict(row) for row in cursor.fetchall()]

    def delete_override_rule(self, override_id: str) -> bool:
        """Delete an override rule by ID."""
        with self._db_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM override_rule WHERE override_id = ?",
                (override_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_applicable_override(
        self,
        *,
        file_id: str | None,
        artist: str | None,
        title: str | None,
    ) -> dict[str, Any] | None:
        """Return the highest-priority override for the given context."""
        scopes: list[tuple[str, str]] = []

        if file_id:
            scopes.append(("file", file_id))

        if artist and title:
            work_key = f"{artist} // {title}".strip().lower()
            scopes.append(("track", work_key))

        if artist:
            scopes.append(("artist", artist.strip().lower()))

        for scope, scope_id in scopes:
            overrides = self.get_override_rules(scope, scope_id)
            if overrides:
                return overrides[0]

        return None

    @staticmethod
    def extract_override_targets(
        override: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Extract CRG/RR targets from a stored override row."""
        crg_mbid = override.get("target_crg_mbid")
        rr_mbid = override.get("target_rr_mbid")

        if crg_mbid or rr_mbid:
            return crg_mbid, rr_mbid

        directive = override.get("directive")
        if not directive:
            return None, None

        parts = [part.strip() for part in directive.split(",") if part.strip()]
        for part in parts:
            if part.startswith("prefer_rg="):
                crg_mbid = part.replace("prefer_rg=", "", 1).strip() or crg_mbid
            if part.startswith("prefer_release="):
                rr_mbid = part.replace("prefer_release=", "", 1).strip() or rr_mbid

        return crg_mbid, rr_mbid


## Tests


def test_decisions_db_schema(tmp_path):
    """Test database schema creation."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")
    assert db.db_path.exists()

    # Verify tables exist
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "file_artifact" in tables
    assert "decision" in tables
    assert "decision_history" in tables
    assert "override_rule" in tables
    assert "schema_meta" in tables


def test_override_schema_columns(tmp_path):
    """Test override_rule columns include structured override fields."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(override_rule)")
    columns = {row[1] for row in cursor.fetchall()}
    conn.close()

    assert "override_type" in columns
    assert "target_crg_mbid" in columns
    assert "target_rr_mbid" in columns
    assert "target_label" in columns
    assert "override_behavior" in columns


def test_create_override_and_lookup(tmp_path):
    """Test creating and retrieving structured overrides."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")

    override_id = db.create_override(
        scope="track",
        scope_id="artist // title",
        override_type="crg",
        target_crg_mbid="rg-123",
        target_rr_mbid="rel-456",
        note="manual test",
        created_by="test",
    )

    overrides = db.get_override_rules("track", "artist // title")
    assert len(overrides) == 1
    assert overrides[0]["override_id"] == override_id

    applicable = db.get_applicable_override(file_id=None, artist="artist", title="title")
    assert applicable is not None

    crg_mbid, rr_mbid = db.extract_override_targets(applicable)
    assert crg_mbid == "rg-123"
    assert rr_mbid == "rel-456"


def test_file_id_generation():
    """Test fingerprint-based file ID generation is deterministic."""
    fingerprint = "AQABz0qUkZK4oOfhL-CPc4e5C_wW2H2QH9uDL4cvoT8UNQ"
    duration_sec = 180

    file_id1 = DecisionsDB.generate_file_id(fingerprint, duration_sec)
    file_id2 = DecisionsDB.generate_file_id(fingerprint, duration_sec)
    assert file_id1 == file_id2

    # Different fingerprint = different ID
    file_id3 = DecisionsDB.generate_file_id("different_fingerprint", duration_sec)
    assert file_id1 != file_id3

    # Different duration = different ID
    file_id4 = DecisionsDB.generate_file_id(fingerprint, 200)
    assert file_id1 != file_id4


def test_file_id_fallback_generation():
    """Test fallback file ID generation is deterministic."""
    path = Path("/music/song.mp3")
    file_id1 = DecisionsDB.generate_file_id_fallback(path, 1024, 1234567890.0)
    file_id2 = DecisionsDB.generate_file_id_fallback(path, 1024, 1234567890.0)
    assert file_id1 == file_id2

    # Different params = different ID
    file_id3 = DecisionsDB.generate_file_id_fallback(path, 1025, 1234567890.0)
    assert file_id1 != file_id3


def test_file_artifact_crud(tmp_path):
    """Test file artifact CRUD operations."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")

    file_id = "test_file_123"
    db.upsert_file_artifact(
        file_id=file_id,
        library_root="/music",
        relative_path="artist/album/song.mp3",
        duration_ms=180000,
    )

    artifact = db.get_file_artifact(file_id)
    assert artifact is not None
    assert artifact["file_id"] == file_id
    assert artifact["library_root"] == "/music"
    assert artifact["duration_ms"] == 180000


def test_decision_crud(tmp_path):
    """Test decision CRUD operations."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")

    file_id = "test_file_123"
    db.upsert_file_artifact(
        file_id=file_id,
        library_root="/music",
        relative_path="song.mp3",
    )

    decision_id = db.upsert_decision(
        file_id=file_id,
        work_key="artist_song",
        mb_rg_id="rg123",
        mb_release_id="rel123",
        mb_recording_id="rec123",
        ruleset_version="canon-1.0",
        config_snapshot={"lead_window_days": 90},
        evidence_hash="abc123",
        trace_compact="CRG:ALBUM|RR:ORIGIN",
    )

    assert decision_id is not None

    decision = db.get_decision(file_id)
    assert decision is not None
    assert decision["file_id"] == file_id
    assert decision["mb_rg_id"] == "rg123"
    assert decision["state"] == DecisionState.DECIDED


def test_decision_history(tmp_path):
    """Test decision archival to history."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")

    file_id = "test_file_123"
    db.upsert_file_artifact(file_id=file_id, library_root="/music", relative_path="song.mp3")

    # Create initial decision
    db.upsert_decision(
        file_id=file_id,
        work_key="artist_song",
        mb_rg_id="rg123",
        mb_release_id="rel123",
        mb_recording_id="rec123",
        ruleset_version="canon-1.0",
        config_snapshot={},
        evidence_hash="abc123",
        trace_compact="TRACE1",
    )

    # Update decision (should archive previous)
    db.upsert_decision(
        file_id=file_id,
        work_key="artist_song",
        mb_rg_id="rg456",  # Changed
        mb_release_id="rel456",
        mb_recording_id="rec123",
        ruleset_version="canon-1.0",
        config_snapshot={},
        evidence_hash="def456",
        trace_compact="TRACE2",
    )

    # Check current decision is updated
    decision = db.get_decision(file_id)
    assert decision is not None
    assert decision["mb_rg_id"] == "rg456"

    # Check history has old decision
    history = db.get_decision_history(file_id)
    assert len(history) == 1
    assert history[0]["mb_rg_id"] == "rg123"
    assert history[0]["superseded_reason"] == SupersededReason.REFRESH
