from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer


@dataclass
class Candidate:
    """A candidate (recording, release_group) pair with associated releases."""

    recording_mbid: str
    release_group_mbid: str
    artist_mbid: str | None = None
    title: str = ""
    artist_name: str = ""
    length_ms: int | None = None
    isrcs: list[str] = field(default_factory=list)
    discovery_method: str = "unknown"  # isrc, acoustid, title_artist_length


@dataclass
class CandidateSet:
    """Set of candidates for a file/recording."""

    file_path: Path | None = None
    candidates: list[Candidate] = field(default_factory=list)
    normalized_title: str = ""
    normalized_artist: str = ""
    length_ms: int | None = None


@dataclass
class EvidenceBundle:
    """Evidence bundle v1 for deterministic decision making."""

    artist: dict[str, Any] = field(default_factory=dict)
    recordings: list[dict[str, Any]] = field(default_factory=list)
    release_groups: list[dict[str, Any]] = field(default_factory=list)
    timeline_facts: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    evidence_hash: str = ""


class CandidateBuilder:
    """
    Candidate discovery and evidence bundle construction.

    Discovers candidates via:
    - ISRC lookups in musicgraph.sqlite
    - AcoustID lookups (stubbed for now)
    - Title + artist_core + length bucket fuzzy matching
    """

    def __init__(self, musicgraph_db: MusicGraphDB, normalizer: Normalizer):
        self.db = musicgraph_db
        self.normalizer = normalizer

    def discover_by_isrc(self, isrc: str) -> list[Candidate]:
        """Discover candidates by ISRC."""
        candidates = []

        recordings = self._find_recordings_by_isrc(isrc)

        for rec in recordings:
            release_groups = self._find_release_groups_for_recording(rec["mbid"])

            for rg in release_groups:
                candidates.append(
                    Candidate(
                        recording_mbid=rec["mbid"],
                        release_group_mbid=rg["mbid"],
                        artist_mbid=rec.get("artist_mbid"),
                        title=rec["title"],
                        artist_name=rg.get("artist_name", ""),
                        length_ms=rec.get("length_ms"),
                        isrcs=[isrc],
                        discovery_method="isrc",
                    )
                )

        return candidates

    def discover_by_title_artist_length(
        self, title: str, artist: str, length_ms: int | None = None
    ) -> list[Candidate]:
        """Discover candidates by normalized title+artist+length bucket."""
        # Normalize inputs
        title_result = self.normalizer.normalize_title(title)
        artist_result = self.normalizer.normalize_artist(artist)

        title_core = title_result.core
        artist_core = artist_result.core

        # Length bucket: ±10% tolerance
        length_min = int(length_ms * 0.9) if length_ms else None
        length_max = int(length_ms * 1.1) if length_ms else None

        # Find recordings by fuzzy match
        recordings = self._find_recordings_by_fuzzy_match(
            title_core, artist_core, length_min, length_max
        )

        candidates = []
        for rec in recordings:
            release_groups = self._find_release_groups_for_recording(rec["mbid"])

            for rg in release_groups:
                candidates.append(
                    Candidate(
                        recording_mbid=rec["mbid"],
                        release_group_mbid=rg["mbid"],
                        artist_mbid=rec.get("artist_mbid"),
                        title=rec["title"],
                        artist_name=rg.get("artist_name", ""),
                        length_ms=rec.get("length_ms"),
                        discovery_method="title_artist_length",
                    )
                )

        return candidates

    def build_evidence_bundle(self, candidate_set: CandidateSet) -> EvidenceBundle:
        """Construct evidence bundle v1 from candidate set."""
        bundle = EvidenceBundle()

        # For now, stub implementation
        # In full version, would gather all evidence for decision rules
        bundle.recordings = [
            {"mbid": c.recording_mbid, "title": c.title, "length_ms": c.length_ms}
            for c in candidate_set.candidates
        ]

        bundle.release_groups = [{"mbid": c.release_group_mbid} for c in candidate_set.candidates]

        bundle.provenance = {
            "sources_used": ["MB"],  # For now, only MusicBrainz
            "discovery_methods": list({c.discovery_method for c in candidate_set.candidates}),
        }

        # Hash the evidence bundle
        bundle.evidence_hash = self._hash_evidence(bundle)

        return bundle

    def _hash_evidence(self, bundle: EvidenceBundle) -> str:
        """
        Hash evidence bundle deterministically.

        Removes timestamps and cache ages, then computes SHA256.
        """
        # Create canonical representation
        canonical = {
            "artist": bundle.artist,
            "recordings": sorted(
                bundle.recordings, key=lambda r: r.get("mbid", "")
            ),  # Deterministic order
            "release_groups": sorted(bundle.release_groups, key=lambda rg: rg.get("mbid", "")),
            "timeline_facts": bundle.timeline_facts,
            "provenance": {
                k: v for k, v in bundle.provenance.items() if k != "cache_age"
            },  # Remove volatile fields
        }

        # Serialize to canonical JSON
        json_bytes = json.dumps(
            canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode("utf-8")

        return hashlib.sha256(json_bytes).hexdigest()

    def _find_recordings_by_isrc(self, isrc: str) -> list[dict[str, Any]]:
        """Find recordings by ISRC in musicgraph DB."""
        # Stub: In full implementation, would query musicgraph.sqlite
        # SELECT * FROM recording WHERE isrcs_json LIKE '%"isrc%'
        return []

    def _find_release_groups_for_recording(self, recording_mbid: str) -> list[dict[str, Any]]:
        """Find release groups that contain this recording."""
        # Stub: In full implementation, would join through recording_release and release
        # SELECT DISTINCT rg.* FROM release_group rg
        # JOIN release r ON r.release_group_mbid = rg.mbid
        # JOIN recording_release rr ON rr.release_mbid = r.mbid
        # WHERE rr.recording_mbid = ?
        return []

    def _find_recordings_by_fuzzy_match(
        self,
        title_core: str,
        artist_core: str,
        length_min: int | None,
        length_max: int | None,
    ) -> list[dict[str, Any]]:
        """Find recordings by fuzzy title+artist+length match."""
        # Stub: In full implementation, would normalize all recordings and match
        # This would require either:
        # 1. Pre-normalized title_core/artist_core columns
        # 2. Full-scan normalization (slow)
        # 3. Work key index
        return []


## Tests


def test_candidate_discovery_by_isrc(tmp_path):
    from chart_binder.musicgraph import MusicGraphDB

    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    # Stub test - would need fixtures
    candidates = builder.discover_by_isrc("USRC17607839")
    assert isinstance(candidates, list)


def test_evidence_bundle_hashing(tmp_path):
    from chart_binder.musicgraph import MusicGraphDB

    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    candidate_set = CandidateSet(
        file_path=Path("/tmp/test.mp3"),
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                title="Test Song",
                artist_name="Test Artist",
                discovery_method="isrc",
            )
        ],
    )

    bundle = builder.build_evidence_bundle(candidate_set)

    # Hash should be deterministic
    assert len(bundle.evidence_hash) == 64  # SHA256 hex length
    assert bundle.evidence_hash.isalnum()

    # Same input should produce same hash
    bundle2 = builder.build_evidence_bundle(candidate_set)
    assert bundle.evidence_hash == bundle2.evidence_hash


def test_evidence_hash_determinism(tmp_path):
    from chart_binder.musicgraph import MusicGraphDB

    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    # Different candidate order should produce same hash (sorted internally)
    candidate_set1 = CandidateSet(
        candidates=[
            Candidate(recording_mbid="rec-2", release_group_mbid="rg-2", discovery_method="isrc"),
            Candidate(recording_mbid="rec-1", release_group_mbid="rg-1", discovery_method="isrc"),
        ]
    )

    candidate_set2 = CandidateSet(
        candidates=[
            Candidate(recording_mbid="rec-1", release_group_mbid="rg-1", discovery_method="isrc"),
            Candidate(recording_mbid="rec-2", release_group_mbid="rg-2", discovery_method="isrc"),
        ]
    )

    bundle1 = builder.build_evidence_bundle(candidate_set1)
    bundle2 = builder.build_evidence_bundle(candidate_set2)

    # Hash should be the same despite different input order
    assert bundle1.evidence_hash == bundle2.evidence_hash
