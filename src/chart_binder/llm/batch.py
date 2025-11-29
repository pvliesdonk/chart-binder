"""Batch LLM adjudication for INDETERMINATE decisions.

Processes multiple INDETERMINATE decisions with rate limiting,
progress tracking, and resume capability.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chart_binder.config import Config
    from chart_binder.decisions_db import DecisionsDB
    from chart_binder.llm.adjudicator import LLMAdjudicator
    from chart_binder.llm.review_queue import ReviewQueue
    from chart_binder.musicgraph import MusicGraphDB

log = logging.getLogger(__name__)


class BatchState(StrEnum):
    """State of a batch adjudication session."""

    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class BatchSession:
    """Batch adjudication session metadata."""

    session_id: str
    started_at: float
    total_count: int
    processed_count: int = 0
    accepted_count: int = 0
    reviewed_count: int = 0
    rejected_count: int = 0
    error_count: int = 0
    state: BatchState = BatchState.RUNNING
    completed_at: float | None = None
    config_snapshot_json: str = "{}"


@dataclass
class BatchResult:
    """Result of processing a single decision in batch."""

    session_id: str
    file_id: str
    processed_at: float
    outcome: str  # 'accepted', 'review', 'rejected', 'error'
    crg_mbid: str | None = None
    rr_mbid: str | None = None
    confidence: float = 0.0
    rationale: str | None = None
    model_id: str | None = None
    error_message: str | None = None
    adjudication_id: str | None = None
    prompt_json: str | None = None
    response_json: str | None = None


class BatchProcessor:
    """Batch processor for LLM adjudication."""

    def __init__(
        self,
        decisions_db: DecisionsDB,
        adjudicator: LLMAdjudicator | None = None,
        config: Config | None = None,
        music_graph_db: MusicGraphDB | None = None,
        review_queue: ReviewQueue | None = None,
        auto_accept_threshold: float = 0.85,
        review_threshold: float = 0.60,
        rate_limit_per_min: int = 10,
    ):
        self.decisions_db = decisions_db
        self.adjudicator = adjudicator
        self.config = config
        self.music_graph_db = music_graph_db
        self.review_queue = review_queue
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold
        self.rate_limit_per_min = rate_limit_per_min
        self.db_path = decisions_db.db_path
        self._init_batch_schema()
        self._request_times: list[float] = []

    def _init_batch_schema(self) -> None:
        """Initialize batch adjudication schema."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS llm_batch_session (
                session_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                completed_at REAL,
                total_count INTEGER NOT NULL,
                processed_count INTEGER NOT NULL,
                accepted_count INTEGER NOT NULL,
                reviewed_count INTEGER NOT NULL,
                rejected_count INTEGER NOT NULL,
                error_count INTEGER NOT NULL,
                config_snapshot_json TEXT NOT NULL,
                state TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_batch_result (
                session_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                processed_at REAL NOT NULL,
                outcome TEXT NOT NULL,
                crg_mbid TEXT,
                rr_mbid TEXT,
                confidence REAL,
                rationale TEXT,
                model_id TEXT,
                error_message TEXT,
                adjudication_id TEXT,
                prompt_json TEXT,
                response_json TEXT,
                PRIMARY KEY (session_id, file_id)
            );

            CREATE INDEX IF NOT EXISTS idx_batch_result_session
                ON llm_batch_result(session_id);
            CREATE INDEX IF NOT EXISTS idx_batch_result_outcome
                ON llm_batch_result(outcome);
            """
        )
        conn.commit()
        conn.close()

    def _rate_limit(self) -> None:
        """Apply rate limiting (requests per minute)."""
        if self.rate_limit_per_min <= 0:
            return

        now = time.time()
        cutoff = now - 60.0  # 1 minute window

        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if t > cutoff]

        # If at limit, wait
        if len(self._request_times) >= self.rate_limit_per_min:
            oldest = self._request_times[0]
            wait_time = 60.0 - (now - oldest) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)

        # Record this request
        self._request_times.append(time.time())

    def create_session(
        self,
        total_count: int,
        config_snapshot: dict[str, Any] | None = None,
    ) -> str:
        """Create a new batch session."""
        import json

        session_id = f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        session = BatchSession(
            session_id=session_id,
            started_at=time.time(),
            total_count=total_count,
            config_snapshot_json=json.dumps(config_snapshot or {}),
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO llm_batch_session
                (session_id, started_at, completed_at, total_count, processed_count,
                 accepted_count, reviewed_count, rejected_count, error_count,
                 config_snapshot_json, state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.session_id,
                session.started_at,
                session.completed_at,
                session.total_count,
                session.processed_count,
                session.accepted_count,
                session.reviewed_count,
                session.rejected_count,
                session.error_count,
                session.config_snapshot_json,
                session.state.value,
            ),
        )
        conn.commit()
        conn.close()

        return session_id

    def get_session(self, session_id: str) -> BatchSession | None:
        """Get batch session by ID."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM llm_batch_session WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return BatchSession(
            session_id=row["session_id"],
            started_at=row["started_at"],
            total_count=row["total_count"],
            processed_count=row["processed_count"],
            accepted_count=row["accepted_count"],
            reviewed_count=row["reviewed_count"],
            rejected_count=row["rejected_count"],
            error_count=row["error_count"],
            state=BatchState(row["state"]),
            completed_at=row["completed_at"],
            config_snapshot_json=row["config_snapshot_json"],
        )

    def update_session_progress(
        self,
        session_id: str,
        processed: int,
        accepted: int,
        reviewed: int,
        rejected: int,
        errors: int,
    ) -> None:
        """Update session progress counters."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE llm_batch_session
            SET processed_count = ?,
                accepted_count = ?,
                reviewed_count = ?,
                rejected_count = ?,
                error_count = ?
            WHERE session_id = ?
            """,
            (processed, accepted, reviewed, rejected, errors, session_id),
        )
        conn.commit()
        conn.close()

    def complete_session(self, session_id: str, state: BatchState = BatchState.COMPLETED) -> None:
        """Mark session as completed."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            UPDATE llm_batch_session
            SET completed_at = ?,
                state = ?
            WHERE session_id = ?
            """,
            (time.time(), state.value, session_id),
        )
        conn.commit()
        conn.close()

    def store_result(self, result: BatchResult) -> None:
        """Store batch adjudication result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_batch_result
                (session_id, file_id, processed_at, outcome, crg_mbid, rr_mbid,
                 confidence, rationale, model_id, error_message, adjudication_id,
                 prompt_json, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.session_id,
                result.file_id,
                result.processed_at,
                result.outcome,
                result.crg_mbid,
                result.rr_mbid,
                result.confidence,
                result.rationale,
                result.model_id,
                result.error_message,
                result.adjudication_id,
                result.prompt_json,
                result.response_json,
            ),
        )
        conn.commit()
        conn.close()

    def get_processed_file_ids(self, session_id: str) -> set[str]:
        """Get set of file_ids already processed in this session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_id FROM llm_batch_result WHERE session_id = ?",
            (session_id,),
        )
        file_ids = {row[0] for row in cursor.fetchall()}
        conn.close()
        return file_ids

    def get_session_results(self, session_id: str) -> list[dict[str, Any]]:
        """Get all results for a session."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM llm_batch_result WHERE session_id = ? ORDER BY processed_at",
            (session_id,),
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def process_decision(
        self,
        session_id: str,
        decision: dict[str, Any],
    ) -> BatchResult:
        """Process a single decision with LLM adjudication."""
        from chart_binder.llm.adjudicator import AdjudicationOutcome

        file_id = decision["file_id"]
        work_key = decision.get("work_key", "")

        try:
            # Apply rate limiting
            self._rate_limit()

            # Build evidence bundle
            import json

            evidence_bundle: dict[str, Any] = {
                "artist": {"name": "Unknown"},
                "recording_candidates": [],
                "provenance": {"sources_used": ["decisions_db"]},
            }

            # Extract artist and title from work_key
            if " // " in work_key:
                parts = work_key.split(" // ", 1)
                evidence_bundle["artist"] = {"name": parts[0]}
                if len(parts) > 1:
                    evidence_bundle["recording_title"] = parts[1]
            else:
                evidence_bundle["work_key_raw"] = work_key

            # Include decision trace info if available
            trace_compact = decision.get("trace_compact", "")
            if trace_compact:
                evidence_bundle["decision_trace_compact"] = trace_compact

            # Include config snapshot from decision
            config_json = decision.get("config_snapshot_json", "{}")
            try:
                evidence_bundle["config_snapshot"] = json.loads(config_json)
            except json.JSONDecodeError:
                pass

            # Try to enrich evidence from music graph if available
            if self.music_graph_db:
                # If we have mb_recording_id, try to get recording info
                mb_recording_id = decision.get("mb_recording_id")
                if mb_recording_id:
                    recording = self.music_graph_db.get_recording(mb_recording_id)
                    if recording:
                        evidence_bundle["recording_candidates"] = [
                            {
                                "title": recording.get("title", ""),
                                "mb_recording_id": mb_recording_id,
                                "rg_candidates": [],
                            }
                        ]
                        # If we have artist info, get it
                        artist_mbid = recording.get("artist_mbid")
                        if artist_mbid:
                            artist = self.music_graph_db.get_artist(artist_mbid)
                            if artist:
                                evidence_bundle["artist"] = {
                                    "name": artist.get("name", "Unknown"),
                                    "mb_artist_id": artist_mbid,
                                    "begin_area_country": artist.get("begin_area_country"),
                                }

            # Include existing decision info for context
            evidence_bundle["existing_decision"] = {
                "mb_rg_id": decision.get("mb_rg_id"),
                "mb_release_id": decision.get("mb_release_id"),
                "state": decision.get("state"),
            }

            log.info(f"Adjudicating {work_key} (file_id: {file_id[:16]}...)")

            # Build decision trace dict from compact string
            decision_trace: dict[str, Any] | None = None
            if trace_compact:
                decision_trace = {"trace_compact": trace_compact}

            # Check adjudicator is available
            if not self.adjudicator:
                raise ValueError("Adjudicator not configured for batch processing")

            # Call LLM adjudicator
            adjudication_result = self.adjudicator.adjudicate(evidence_bundle, decision_trace)

            # Determine outcome based on confidence
            if adjudication_result.outcome == AdjudicationOutcome.ERROR:
                outcome = "error"
                log.warning(f"  ERROR: {adjudication_result.error_message}")
            elif adjudication_result.confidence >= self.auto_accept_threshold:
                outcome = "accepted"
                log.info(
                    f"  ACCEPTED: confidence={adjudication_result.confidence:.2f} "
                    f"CRG={adjudication_result.crg_mbid}"
                )
            elif adjudication_result.confidence >= self.review_threshold:
                outcome = "review"
                log.info(
                    f"  REVIEW: confidence={adjudication_result.confidence:.2f} "
                    f"CRG={adjudication_result.crg_mbid}"
                )
            else:
                outcome = "rejected"
                log.info(
                    f"  REJECTED: confidence={adjudication_result.confidence:.2f} "
                    f"(below threshold {self.review_threshold})"
                )
                if adjudication_result.rationale:
                    log.debug(f"  Rationale: {adjudication_result.rationale}")

            result = BatchResult(
                session_id=session_id,
                file_id=file_id,
                processed_at=time.time(),
                outcome=outcome,
                crg_mbid=adjudication_result.crg_mbid,
                rr_mbid=adjudication_result.rr_mbid,
                confidence=adjudication_result.confidence,
                rationale=adjudication_result.rationale,
                model_id=adjudication_result.model_id,
                error_message=adjudication_result.error_message,
                adjudication_id=adjudication_result.adjudication_id,
                prompt_json=adjudication_result.prompt_json,
                response_json=adjudication_result.response_json,
            )

        except Exception as e:
            log.error(f"Error processing decision {file_id}: {e}", exc_info=True)
            result = BatchResult(
                session_id=session_id,
                file_id=file_id,
                processed_at=time.time(),
                outcome="error",
                error_message=str(e),
            )

        return result
