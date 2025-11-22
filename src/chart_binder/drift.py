"""
Drift detection for canonicalization decisions.

Compares stored decisions with recomputed decisions to detect:
- Evidence changes (upstream data drift)
- Rules/config changes
- Combinations of both

Per Epic 6 - Decision Store & Drift.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from chart_binder.decisions_db import DecisionsDB, DecisionState

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection for a decision."""

    file_id: str
    has_drift: bool
    old_state: DecisionState
    new_state: DecisionState
    old_evidence_hash: str
    new_evidence_hash: str
    old_ruleset_version: str
    new_ruleset_version: str
    old_crg: str
    new_crg: str
    old_rr: str
    new_rr: str
    drift_category: str | None = None  # STALE-EVIDENCE | STALE-RULES | STALE-BOTH


@dataclass
class StaleDecisionSummary:
    """Summary of a stale decision requiring review (no recomputation)."""

    file_id: str
    state: DecisionState
    evidence_hash: str
    ruleset_version: str
    mb_rg_id: str
    mb_release_id: str
    work_key: str
    updated_at: float


class DriftDetector:
    """
    Detect drift in canonicalization decisions.

    Compares stored decisions with recomputed decisions to determine
    if evidence or rules have changed.
    """

    def __init__(self, db: DecisionsDB):
        self.db = db

    @staticmethod
    def compute_evidence_hash(evidence_bundle: dict[str, Any]) -> str:
        """
        Compute deterministic hash of evidence bundle.

        Removes volatile fields (timestamps, HTTP headers) and sorts keys
        for canonical JSON representation.
        """
        # Deep copy to avoid modifying original
        bundle = copy.deepcopy(evidence_bundle)

        # Remove volatile fields
        def strip_volatile(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: strip_volatile(v)
                    for k, v in obj.items()
                    if k not in {"fetched_at", "fetched_at_utc", "created_at", "updated_at"}
                }
            elif isinstance(obj, list):
                return [strip_volatile(item) for item in obj]
            else:
                return obj

        clean_bundle = strip_volatile(bundle)

        # Canonical JSON (sorted keys, no whitespace)
        canonical_json = json.dumps(clean_bundle, sort_keys=True, separators=(",", ":"))

        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def detect_drift(
        self,
        file_id: str,
        new_evidence_bundle: dict[str, Any],
        new_decision: dict[str, Any],
        new_ruleset_version: str,
    ) -> DriftResult:
        """
        Detect drift for a file by comparing stored vs recomputed decision.

        Args:
            file_id: File artifact ID
            new_evidence_bundle: Newly computed evidence bundle
            new_decision: Newly computed decision dict with keys:
                - mb_rg_id, mb_release_id, work_key, trace_compact
            new_ruleset_version: Current ruleset version

        Returns:
            DriftResult indicating if/how decision has drifted
        """
        # Get stored decision
        stored = self.db.get_decision(file_id)
        if not stored:
            # No stored decision = UNDECIDED
            return DriftResult(
                file_id=file_id,
                has_drift=True,
                old_state=DecisionState.UNDECIDED,
                new_state=DecisionState.DECIDED,
                old_evidence_hash="",
                new_evidence_hash=self.compute_evidence_hash(new_evidence_bundle),
                old_ruleset_version="",
                new_ruleset_version=new_ruleset_version,
                old_crg="",
                new_crg=new_decision["mb_rg_id"],
                old_rr="",
                new_rr=new_decision["mb_release_id"],
                drift_category=None,
            )

        # Compute new evidence hash
        new_evidence_hash = self.compute_evidence_hash(new_evidence_bundle)

        old_evidence_hash = stored["evidence_hash"]
        old_ruleset_version = stored["ruleset_version"]
        old_crg = stored["mb_rg_id"]
        old_rr = stored["mb_release_id"]
        new_crg = new_decision["mb_rg_id"]
        new_rr = new_decision["mb_release_id"]

        # Check if decision changed
        decision_changed = old_crg != new_crg or old_rr != new_rr

        # Detect drift type
        evidence_changed = new_evidence_hash != old_evidence_hash
        rules_changed = new_ruleset_version != old_ruleset_version

        if not decision_changed:
            # No drift - decision is the same
            return DriftResult(
                file_id=file_id,
                has_drift=False,
                old_state=DecisionState.DECIDED,
                new_state=DecisionState.DECIDED,
                old_evidence_hash=old_evidence_hash,
                new_evidence_hash=new_evidence_hash,
                old_ruleset_version=old_ruleset_version,
                new_ruleset_version=new_ruleset_version,
                old_crg=old_crg,
                new_crg=new_crg,
                old_rr=old_rr,
                new_rr=new_rr,
                drift_category=None,
            )

        # Decision changed - determine drift category
        if evidence_changed and rules_changed:
            drift_category = "STALE-BOTH"
            new_state = DecisionState.STALE_BOTH
        elif evidence_changed:
            drift_category = "STALE-EVIDENCE"
            new_state = DecisionState.STALE_EVIDENCE
        elif rules_changed:
            drift_category = "STALE-RULES"
            new_state = DecisionState.STALE_RULES
        else:
            # Decision changed but neither evidence nor rules changed
            # This indicates non-determinism in the resolver logic
            drift_category = "STALE-NONDETERMINISTIC"
            new_state = DecisionState.STALE_NONDETERMINISTIC
            logger.warning(
                "Non-deterministic resolver behavior detected for file_id=%s: "
                "decision changed from (CRG=%s, RR=%s) to (CRG=%s, RR=%s) "
                "with identical evidence hash and ruleset version",
                file_id,
                old_crg,
                old_rr,
                new_crg,
                new_rr,
            )

        return DriftResult(
            file_id=file_id,
            has_drift=True,
            old_state=DecisionState(stored["state"]),
            new_state=new_state,
            old_evidence_hash=old_evidence_hash,
            new_evidence_hash=new_evidence_hash,
            old_ruleset_version=old_ruleset_version,
            new_ruleset_version=new_ruleset_version,
            old_crg=old_crg,
            new_crg=new_crg,
            old_rr=old_rr,
            new_rr=new_rr,
            drift_category=drift_category,
        )

    def review_drift(self) -> list[StaleDecisionSummary]:
        """
        Review all decisions for drift.

        Returns list of StaleDecisionSummary for decisions in STALE-* or
        INDETERMINATE states that require review.

        Note: This method does NOT recompute decisions. It only returns
        summaries of existing stale decisions. To detect drift with actual
        before/after comparison, use detect_drift() with recomputed evidence.
        """
        stale_decisions = self.db.get_stale_decisions()

        return [
            StaleDecisionSummary(
                file_id=decision["file_id"],
                state=DecisionState(decision["state"]),
                evidence_hash=decision["evidence_hash"],
                ruleset_version=decision["ruleset_version"],
                mb_rg_id=decision["mb_rg_id"],
                mb_release_id=decision["mb_release_id"],
                work_key=decision["work_key"],
                updated_at=decision["updated_at"],
            )
            for decision in stale_decisions
        ]


## Tests


def test_evidence_hash_deterministic():
    """Test evidence hash is deterministic."""
    evidence1 = {
        "artist": {"name": "Artist", "mbid": "123"},
        "recordings": [{"title": "Song", "mbid": "456"}],
        "timeline_facts": {"year": 2020},
    }

    evidence2 = {
        "timeline_facts": {"year": 2020},
        "artist": {"mbid": "123", "name": "Artist"},
        "recordings": [{"title": "Song", "mbid": "456"}],
    }

    hash1 = DriftDetector.compute_evidence_hash(evidence1)
    hash2 = DriftDetector.compute_evidence_hash(evidence2)

    assert hash1 == hash2


def test_evidence_hash_strips_volatile():
    """Test evidence hash strips volatile fields."""
    evidence1 = {
        "artist": {"name": "Artist", "fetched_at": 1234567890},
        "recordings": [{"title": "Song", "created_at": "2020-01-01"}],
    }

    evidence2 = {
        "artist": {"name": "Artist", "fetched_at": 9999999999},
        "recordings": [{"title": "Song", "created_at": "2099-12-31"}],
    }

    hash1 = DriftDetector.compute_evidence_hash(evidence1)
    hash2 = DriftDetector.compute_evidence_hash(evidence2)

    # Hashes should be same despite different volatile fields
    assert hash1 == hash2


def test_detect_drift_no_stored_decision(tmp_path):
    """Test drift detection when no stored decision exists."""
    from chart_binder.decisions_db import DecisionsDB

    db = DecisionsDB(tmp_path / "decisions.sqlite")
    detector = DriftDetector(db)

    evidence = {"artist": {"name": "Artist"}}
    decision = {"mb_rg_id": "rg123", "mb_release_id": "rel123"}

    result = detector.detect_drift(
        file_id="file123",
        new_evidence_bundle=evidence,
        new_decision=decision,
        new_ruleset_version="canon-1.0",
    )

    assert result.has_drift
    assert result.old_state == DecisionState.UNDECIDED
    assert result.new_state == DecisionState.DECIDED


def test_detect_drift_evidence_changed(tmp_path):
    """Test drift detection when evidence changes."""
    from chart_binder.decisions_db import DecisionsDB

    db = DecisionsDB(tmp_path / "decisions.sqlite")
    detector = DriftDetector(db)

    file_id = "file123"

    # Store initial decision
    db.upsert_file_artifact(file_id=file_id, library_root="/music", relative_path="song.mp3")

    old_evidence = {"artist": {"name": "Artist"}}
    old_hash = detector.compute_evidence_hash(old_evidence)

    db.upsert_decision(
        file_id=file_id,
        work_key="artist_song",
        mb_rg_id="rg123",
        mb_release_id="rel123",
        mb_recording_id="rec123",
        ruleset_version="canon-1.0",
        config_snapshot={},
        evidence_hash=old_hash,
        trace_compact="TRACE",
    )

    # Detect drift with new evidence (different CRG)
    new_evidence = {"artist": {"name": "Artist", "country": "US"}}  # Added field
    new_decision = {"mb_rg_id": "rg456", "mb_release_id": "rel456"}

    result = detector.detect_drift(
        file_id=file_id,
        new_evidence_bundle=new_evidence,
        new_decision=new_decision,
        new_ruleset_version="canon-1.0",  # Same ruleset
    )

    assert result.has_drift
    assert result.drift_category == "STALE-EVIDENCE"
    assert result.new_state == DecisionState.STALE_EVIDENCE
    assert result.old_crg == "rg123"
    assert result.new_crg == "rg456"


def test_detect_drift_rules_changed(tmp_path):
    """Test drift detection when rules change."""
    from chart_binder.decisions_db import DecisionsDB

    db = DecisionsDB(tmp_path / "decisions.sqlite")
    detector = DriftDetector(db)

    file_id = "file123"

    # Store initial decision
    db.upsert_file_artifact(file_id=file_id, library_root="/music", relative_path="song.mp3")

    evidence = {"artist": {"name": "Artist"}}
    evidence_hash = detector.compute_evidence_hash(evidence)

    db.upsert_decision(
        file_id=file_id,
        work_key="artist_song",
        mb_rg_id="rg123",
        mb_release_id="rel123",
        mb_recording_id="rec123",
        ruleset_version="canon-1.0",
        config_snapshot={},
        evidence_hash=evidence_hash,
        trace_compact="TRACE",
    )

    # Detect drift with new ruleset (different CRG)
    new_decision = {"mb_rg_id": "rg456", "mb_release_id": "rel456"}

    result = detector.detect_drift(
        file_id=file_id,
        new_evidence_bundle=evidence,  # Same evidence
        new_decision=new_decision,
        new_ruleset_version="canon-2.0",  # Different ruleset
    )

    assert result.has_drift
    assert result.drift_category == "STALE-RULES"
    assert result.new_state == DecisionState.STALE_RULES
