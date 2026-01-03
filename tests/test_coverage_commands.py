"""Tests for coverage CLI commands."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from chart_binder.decisions_db import DecisionsDB, DecisionState
from chart_binder.drift import DriftDetector


@pytest.fixture
def decisions_db(tmp_path: Path) -> DecisionsDB:
    """Create a temporary decisions database."""
    db = DecisionsDB(tmp_path / "decisions.sqlite")
    return db


@pytest.fixture
def populated_decisions_db(decisions_db: DecisionsDB) -> DecisionsDB:
    """Create a decisions database with sample data."""
    db = decisions_db
    now = time.time()

    # Create file artifacts
    for i in range(5):
        db.upsert_file_artifact(
            file_id=f"file-{i}",
            library_root="/music",
            relative_path=f"song-{i}.mp3",
            created_at=now,
        )

    # Create decisions with various states
    decisions = [
        # Normal decided
        ("file-0", "queen_bohemian_rhapsody", DecisionState.DECIDED),
        # Indeterminate
        ("file-1", "beatles_hey_jude", DecisionState.INDETERMINATE),
        ("file-2", "eagles_hotel_california", DecisionState.INDETERMINATE),
        # Stale evidence
        ("file-3", "led_zeppelin_stairway", DecisionState.STALE_EVIDENCE),
        # Stale rules
        ("file-4", "pink_floyd_wish_you_were_here", DecisionState.STALE_RULES),
    ]

    for file_id, work_key, state in decisions:
        db.upsert_decision(
            file_id=file_id,
            work_key=work_key,
            mb_rg_id=f"rg-{file_id}",
            mb_release_id=f"rel-{file_id}",
            mb_recording_id=f"rec-{file_id}",
            ruleset_version="canon-1.0",
            config_snapshot={},
            evidence_hash=f"hash-{file_id}",
            trace_compact=f"TRACE-{work_key}",
            state=state,
        )

    return db


class TestDecisionsDBGetStale:
    """Tests for DecisionsDB.get_stale_decisions()."""

    def test_get_stale_returns_stale_and_indeterminate(self, populated_decisions_db: DecisionsDB):
        """Test that get_stale_decisions returns all stale/indeterminate decisions."""
        stale = populated_decisions_db.get_stale_decisions()

        # Should return 4 decisions (2 indeterminate, 1 stale_evidence, 1 stale_rules)
        assert len(stale) == 4

        # Check states
        states = {d["state"] for d in stale}
        assert DecisionState.INDETERMINATE in states
        assert DecisionState.STALE_EVIDENCE in states
        assert DecisionState.STALE_RULES in states
        assert DecisionState.DECIDED not in states

    def test_get_stale_empty_db(self, decisions_db: DecisionsDB):
        """Test get_stale_decisions on empty database."""
        stale = decisions_db.get_stale_decisions()
        assert stale == []


class TestDriftDetectorReviewDrift:
    """Tests for DriftDetector.review_drift()."""

    def test_review_drift_returns_summaries(self, populated_decisions_db: DecisionsDB):
        """Test that review_drift returns StaleDecisionSummary objects."""
        detector = DriftDetector(populated_decisions_db)
        summaries = detector.review_drift()

        assert len(summaries) == 4

        # Check all have required fields
        for summary in summaries:
            assert summary.file_id is not None
            assert summary.state in (
                DecisionState.INDETERMINATE,
                DecisionState.STALE_EVIDENCE,
                DecisionState.STALE_RULES,
                DecisionState.STALE_BOTH,
                DecisionState.STALE_NONDETERMINISTIC,
            )
            assert summary.work_key is not None
            assert summary.mb_rg_id is not None

    def test_review_drift_empty_db(self, decisions_db: DecisionsDB):
        """Test review_drift on empty database."""
        detector = DriftDetector(decisions_db)
        summaries = detector.review_drift()
        assert summaries == []


class TestFilterIndeterminate:
    """Tests for filtering indeterminate decisions."""

    def test_filter_by_work_key_prefix(self, populated_decisions_db: DecisionsDB):
        """Test filtering decisions by work_key prefix."""
        stale = populated_decisions_db.get_stale_decisions()

        # Filter to only INDETERMINATE
        indeterminate = [d for d in stale if d["state"] == DecisionState.INDETERMINATE]
        assert len(indeterminate) == 2

        # Filter by work_key prefix
        beatles = [d for d in indeterminate if d.get("work_key", "").startswith("beatles")]
        assert len(beatles) == 1
        assert beatles[0]["work_key"] == "beatles_hey_jude"


class TestGroupByState:
    """Tests for grouping stale decisions by state."""

    def test_group_by_state(self, populated_decisions_db: DecisionsDB):
        """Test grouping stale decisions by state."""
        detector = DriftDetector(populated_decisions_db)
        summaries = detector.review_drift()

        by_state: dict[str, list] = {}
        for summary in summaries:
            state_key = summary.state.value
            if state_key not in by_state:
                by_state[state_key] = []
            by_state[state_key].append(summary)

        assert "indeterminate" in by_state
        assert len(by_state["indeterminate"]) == 2

        assert "stale_evidence" in by_state
        assert len(by_state["stale_evidence"]) == 1

        assert "stale_rules" in by_state
        assert len(by_state["stale_rules"]) == 1


class TestDecisionStateEnum:
    """Tests for DecisionState enum."""

    def test_stale_states_values(self):
        """Test stale state enum values."""
        assert DecisionState.STALE_EVIDENCE.value == "stale_evidence"
        assert DecisionState.STALE_RULES.value == "stale_rules"
        assert DecisionState.STALE_BOTH.value == "stale_both"
        assert DecisionState.INDETERMINATE.value == "indeterminate"
