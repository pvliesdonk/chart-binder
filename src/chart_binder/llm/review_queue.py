"""Human review queue for Chart-Binder.

Manages INDETERMINATE decisions and LLM suggestions that require
human review before acceptance.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReviewAction(StrEnum):
    """Actions that can be taken on a review item."""

    ACCEPT = "accept"  # Accept a specific CRG/RR
    ACCEPT_LLM = "accept_llm"  # Accept LLM suggestion
    KEEP_INDETERMINATE = "keep_indeterminate"  # Keep as indeterminate
    ADD_ALIAS = "add_alias"  # Add normalization alias
    SKIP = "skip"  # Skip / ignore file


class ReviewSource(StrEnum):
    """Source of the review item."""

    INDETERMINATE = "indeterminate"  # From resolver indeterminate
    LLM_REVIEW = "llm_review"  # LLM confidence 0.60-0.85
    CONFLICT = "conflict"  # MB vs Discogs conflict


@dataclass
class ReviewItem:
    """Item in the human review queue."""

    review_id: str
    file_id: str
    work_key: str
    source: ReviewSource
    evidence_bundle_json: str
    decision_trace_json: str | None = None
    llm_suggestion_json: str | None = None
    created_at: float = field(default_factory=time.time)
    reviewed_at: float | None = None
    reviewed_by: str | None = None
    action: ReviewAction | None = None
    action_data_json: str | None = None
    notes: str | None = None

    @property
    def evidence_bundle(self) -> dict[str, Any]:
        """Parse evidence bundle JSON."""
        return json.loads(self.evidence_bundle_json) if self.evidence_bundle_json else {}

    @property
    def decision_trace(self) -> dict[str, Any] | None:
        """Parse decision trace JSON."""
        return json.loads(self.decision_trace_json) if self.decision_trace_json else None

    @property
    def llm_suggestion(self) -> dict[str, Any] | None:
        """Parse LLM suggestion JSON."""
        return json.loads(self.llm_suggestion_json) if self.llm_suggestion_json else None

    def to_display(self) -> str:
        """Format for CLI display."""
        lines = [
            f"Review ID: {self.review_id[:8]}...",
            f"File ID: {self.file_id[:16]}...",
            f"Work Key: {self.work_key}",
            f"Source: {self.source.value}",
        ]

        evidence = self.evidence_bundle
        if artist := evidence.get("artist", {}):
            lines.append(f"Artist: {artist.get('name', 'Unknown')}")

        candidates = evidence.get("recording_candidates", [])
        if candidates:
            for rec in candidates[:1]:
                lines.append(f"Title: {rec.get('title', 'Unknown')}")
                rgs = rec.get("rg_candidates", [])
                if rgs:
                    lines.append(f"Release Groups: {len(rgs)} candidates")
                    for rg in rgs[:3]:
                        lines.append(
                            f"  - {rg.get('mb_rg_id', '?')[:8]}... "
                            f"({rg.get('primary_type', '?')}) "
                            f"{rg.get('first_release_date', '?')}"
                        )

        if llm := self.llm_suggestion:
            lines.append(f"LLM Suggestion: {llm.get('crg_mbid', '?')}")
            lines.append(f"LLM Confidence: {llm.get('confidence', 0):.2f}")
            lines.append(f"LLM Rationale: {llm.get('rationale', '')}")

        trace = self.decision_trace
        if trace and trace.get("missing_facts"):
            lines.append(f"Missing Facts: {', '.join(trace['missing_facts'])}")

        return "\n".join(lines)


class ReviewQueue:
    """SQLite-backed human review queue.

    Stores pending reviews and completed review actions
    for audit trail purposes.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def _db_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_queue (
                review_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                work_key TEXT NOT NULL,
                source TEXT NOT NULL,
                evidence_bundle_json TEXT NOT NULL,
                decision_trace_json TEXT,
                llm_suggestion_json TEXT,
                created_at REAL NOT NULL,
                reviewed_at REAL,
                reviewed_by TEXT,
                action TEXT,
                action_data_json TEXT,
                notes TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_review_file ON review_queue(file_id);
            CREATE INDEX IF NOT EXISTS idx_review_source ON review_queue(source);
            CREATE INDEX IF NOT EXISTS idx_review_pending ON review_queue(reviewed_at)
                WHERE reviewed_at IS NULL;

            CREATE TABLE IF NOT EXISTS review_history (
                history_id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                work_key TEXT NOT NULL,
                source TEXT NOT NULL,
                action TEXT NOT NULL,
                action_data_json TEXT,
                reviewed_by TEXT,
                reviewed_at REAL NOT NULL,
                notes TEXT,
                FOREIGN KEY (review_id) REFERENCES review_queue(review_id)
            );

            CREATE INDEX IF NOT EXISTS idx_history_file ON review_history(file_id);
            CREATE INDEX IF NOT EXISTS idx_history_reviewed ON review_history(reviewed_at);

            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO schema_meta (key, value)
                VALUES ('db_version_review', '1');
            """
        )
        conn.commit()
        conn.close()

    def add_item(
        self,
        file_id: str,
        work_key: str,
        source: ReviewSource,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
        llm_suggestion: dict[str, Any] | None = None,
    ) -> str:
        """Add an item to the review queue.

        Returns:
            The review_id of the created item
        """
        review_id = str(uuid.uuid4())

        with self._db_connection() as conn:
            conn.execute(
                """
                INSERT INTO review_queue
                    (review_id, file_id, work_key, source, evidence_bundle_json,
                     decision_trace_json, llm_suggestion_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    file_id,
                    work_key,
                    source.value,
                    json.dumps(evidence_bundle),
                    json.dumps(decision_trace) if decision_trace else None,
                    json.dumps(llm_suggestion) if llm_suggestion else None,
                    time.time(),
                ),
            )
            conn.commit()

        return review_id

    def get_pending(
        self,
        source: ReviewSource | None = None,
        limit: int = 100,
    ) -> list[ReviewItem]:
        """Get pending review items.

        Args:
            source: Optional filter by source
            limit: Maximum items to return

        Returns:
            List of pending ReviewItems
        """
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row

            if source:
                cursor = conn.execute(
                    """
                    SELECT * FROM review_queue
                    WHERE reviewed_at IS NULL AND source = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (source.value, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM review_queue
                    WHERE reviewed_at IS NULL
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (limit,),
                )

            return [self._row_to_item(row) for row in cursor.fetchall()]

    def get_item(self, review_id: str) -> ReviewItem | None:
        """Get a specific review item by ID."""
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM review_queue WHERE review_id = ?",
                (review_id,),
            )
            row = cursor.fetchone()
            return self._row_to_item(row) if row else None

    def find_items_by_prefix(self, review_id_prefix: str, limit: int = 20) -> list[ReviewItem]:
        """Find review items by ID prefix."""
        if not review_id_prefix:
            return []
        with self._db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM review_queue
                WHERE review_id LIKE ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (f"{review_id_prefix}%", limit),
            )
            return [self._row_to_item(row) for row in cursor.fetchall()]

    def complete_review(
        self,
        review_id: str,
        action: ReviewAction,
        action_data: dict[str, Any] | None = None,
        reviewed_by: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Complete a review with an action.

        Args:
            review_id: ID of the review item
            action: Action taken
            action_data: Data associated with the action (e.g., selected CRG)
            reviewed_by: User who performed the review
            notes: Optional notes

        Returns:
            True if successful, False if item not found
        """
        now = time.time()
        action_json = json.dumps(action_data) if action_data else None

        with self._db_connection() as conn:
            # Update the review item
            cursor = conn.execute(
                """
                UPDATE review_queue
                SET reviewed_at = ?, reviewed_by = ?, action = ?,
                    action_data_json = ?, notes = ?
                WHERE review_id = ? AND reviewed_at IS NULL
                """,
                (now, reviewed_by, action.value, action_json, notes, review_id),
            )

            if cursor.rowcount == 0:
                return False

            # Get the item for history
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM review_queue WHERE review_id = ?",
                (review_id,),
            )
            row = cursor.fetchone()

            if row:
                # Add to history
                history_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO review_history
                        (history_id, review_id, file_id, work_key, source,
                         action, action_data_json, reviewed_by, reviewed_at, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        history_id,
                        review_id,
                        row["file_id"],
                        row["work_key"],
                        row["source"],
                        action.value,
                        action_json,
                        reviewed_by,
                        now,
                        notes,
                    ),
                )

            conn.commit()
            return True

    def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Total pending
            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE reviewed_at IS NULL")
            pending = cursor.fetchone()[0]

            # By source
            cursor.execute(
                """
                SELECT source, COUNT(*) FROM review_queue
                WHERE reviewed_at IS NULL
                GROUP BY source
                """
            )
            by_source = dict(cursor.fetchall())

            # Total completed
            cursor.execute("SELECT COUNT(*) FROM review_queue WHERE reviewed_at IS NOT NULL")
            completed = cursor.fetchone()[0]

            # Completed by action
            cursor.execute(
                """
                SELECT action, COUNT(*) FROM review_queue
                WHERE reviewed_at IS NOT NULL
                GROUP BY action
                """
            )
            by_action = dict(cursor.fetchall())

            return {
                "pending": pending,
                "pending_by_source": by_source,
                "completed": completed,
                "completed_by_action": by_action,
            }

    def _row_to_item(self, row: sqlite3.Row) -> ReviewItem:
        """Convert database row to ReviewItem."""
        return ReviewItem(
            review_id=row["review_id"],
            file_id=row["file_id"],
            work_key=row["work_key"],
            source=ReviewSource(row["source"]),
            evidence_bundle_json=row["evidence_bundle_json"],
            decision_trace_json=row["decision_trace_json"],
            llm_suggestion_json=row["llm_suggestion_json"],
            created_at=row["created_at"],
            reviewed_at=row["reviewed_at"],
            reviewed_by=row["reviewed_by"],
            action=ReviewAction(row["action"]) if row["action"] else None,
            action_data_json=row["action_data_json"],
            notes=row["notes"],
        )


## Tests


def test_review_queue_schema(tmp_path):
    """Test database schema creation."""
    queue = ReviewQueue(tmp_path / "review.sqlite")
    assert queue.db_path.exists()

    # Verify tables exist
    conn = queue._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "review_queue" in tables
    assert "review_history" in tables
    assert "schema_meta" in tables


def test_add_and_get_item(tmp_path):
    """Test adding and retrieving review items."""
    queue = ReviewQueue(tmp_path / "review.sqlite")

    review_id = queue.add_item(
        file_id="file-123",
        work_key="artist // title",
        source=ReviewSource.INDETERMINATE,
        evidence_bundle={"artist": {"name": "Test"}},
        decision_trace={"missing_facts": ["no_date"]},
    )

    assert review_id is not None

    item = queue.get_item(review_id)
    assert item is not None
    assert item.file_id == "file-123"
    assert item.work_key == "artist // title"
    assert item.source == ReviewSource.INDETERMINATE
    assert item.evidence_bundle == {"artist": {"name": "Test"}}


def test_find_items_by_prefix(tmp_path):
    """Test finding items by review ID prefix."""
    queue = ReviewQueue(tmp_path / "review.sqlite")
    review_id = queue.add_item(
        file_id="file-1",
        work_key="key1",
        source=ReviewSource.LLM_REVIEW,
        evidence_bundle={},
    )
    prefix = review_id[:8]
    matches = queue.find_items_by_prefix(prefix)
    assert any(match.review_id == review_id for match in matches)


def test_get_pending(tmp_path):
    """Test getting pending items."""
    queue = ReviewQueue(tmp_path / "review.sqlite")

    queue.add_item(
        file_id="file-1",
        work_key="key1",
        source=ReviewSource.INDETERMINATE,
        evidence_bundle={},
    )
    queue.add_item(
        file_id="file-2",
        work_key="key2",
        source=ReviewSource.LLM_REVIEW,
        evidence_bundle={},
    )

    # All pending
    pending = queue.get_pending()
    assert len(pending) == 2

    # Filter by source
    pending = queue.get_pending(source=ReviewSource.LLM_REVIEW)
    assert len(pending) == 1
    assert pending[0].source == ReviewSource.LLM_REVIEW


def test_complete_review(tmp_path):
    """Test completing a review."""
    queue = ReviewQueue(tmp_path / "review.sqlite")

    review_id = queue.add_item(
        file_id="file-123",
        work_key="key",
        source=ReviewSource.INDETERMINATE,
        evidence_bundle={},
    )

    success = queue.complete_review(
        review_id,
        action=ReviewAction.ACCEPT,
        action_data={"crg_mbid": "rg-123"},
        reviewed_by="test_user",
        notes="Looks good",
    )
    assert success is True

    # Item should no longer be pending
    pending = queue.get_pending()
    assert len(pending) == 0

    # Item should have action set
    item = queue.get_item(review_id)
    assert item is not None
    assert item.action == ReviewAction.ACCEPT
    assert item.reviewed_by == "test_user"


def test_get_stats(tmp_path):
    """Test queue statistics."""
    queue = ReviewQueue(tmp_path / "review.sqlite")

    queue.add_item(
        file_id="file-1",
        work_key="key1",
        source=ReviewSource.INDETERMINATE,
        evidence_bundle={},
    )
    review_id = queue.add_item(
        file_id="file-2",
        work_key="key2",
        source=ReviewSource.LLM_REVIEW,
        evidence_bundle={},
    )
    queue.complete_review(review_id, ReviewAction.ACCEPT)

    stats = queue.get_stats()
    assert stats["pending"] == 1
    assert stats["completed"] == 1
    assert stats["pending_by_source"]["indeterminate"] == 1


def test_review_item_display():
    """Test ReviewItem display formatting."""
    item = ReviewItem(
        review_id="test-id",
        file_id="file-123",
        work_key="artist // title",
        source=ReviewSource.LLM_REVIEW,
        evidence_bundle_json='{"artist": {"name": "Test Artist"}}',
        llm_suggestion_json='{"crg_mbid": "rg-123", "confidence": 0.75, "rationale": "test"}',
    )

    display = item.to_display()
    assert "test-id" in display
    assert "artist // title" in display
    assert "llm_review" in display
    assert "Test Artist" in display
    assert "rg-123" in display
    assert "0.75" in display
