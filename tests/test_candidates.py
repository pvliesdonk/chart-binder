"""Unit tests for candidate builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from chart_binder.candidates import Candidate, CandidateBuilder, CandidateSet, DiscoveryMethod
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer


@pytest.fixture
def candidate_builder(tmp_path):
    """Create a CandidateBuilder instance for testing."""
    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    return CandidateBuilder(db, norm)


def test_candidate_discovery_by_isrc(candidate_builder):
    """Test ISRC-based candidate discovery."""
    # Stub test - would need fixtures
    candidates = candidate_builder.discover_by_isrc("USRC17607839")
    assert isinstance(candidates, list)


def test_evidence_bundle_hashing(candidate_builder):
    """Test evidence bundle hash determinism."""
    candidate_set = CandidateSet(
        file_path=Path("/tmp/test.mp3"),
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                title="Test Song",
                artist_name="Test Artist",
                discovery_method=DiscoveryMethod.ISRC,
            )
        ],
    )

    bundle = candidate_builder.build_evidence_bundle(candidate_set)

    # Hash should be deterministic
    assert len(bundle.evidence_hash) == 64  # SHA256 hex length
    assert bundle.evidence_hash.isalnum()

    # Same input should produce same hash
    bundle2 = candidate_builder.build_evidence_bundle(candidate_set)
    assert bundle.evidence_hash == bundle2.evidence_hash


def test_evidence_hash_determinism(candidate_builder):
    """Test evidence hash is order-independent."""
    # Different candidate order should produce same hash (sorted internally)
    candidate_set1 = CandidateSet(
        candidates=[
            Candidate(
                recording_mbid="rec-2",
                release_group_mbid="rg-2",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                discovery_method=DiscoveryMethod.ISRC,
            ),
        ]
    )

    candidate_set2 = CandidateSet(
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-2",
                release_group_mbid="rg-2",
                discovery_method=DiscoveryMethod.ISRC,
            ),
        ]
    )

    bundle1 = candidate_builder.build_evidence_bundle(candidate_set1)
    bundle2 = candidate_builder.build_evidence_bundle(candidate_set2)

    # Hash should be the same despite different input order
    assert bundle1.evidence_hash == bundle2.evidence_hash


def test_discovery_method_enum():
    """Test DiscoveryMethod enum values."""
    assert DiscoveryMethod.ISRC == "isrc"
    assert DiscoveryMethod.ACOUSTID == "acoustid"
    assert DiscoveryMethod.TITLE_ARTIST_LENGTH == "title_artist_length"
    assert DiscoveryMethod.UNKNOWN == "unknown"


def test_discovery_methods_sorting():
    """Test discovery_methods list is sorted for deterministic hashing."""
    candidate_set = CandidateSet(
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
            ),
            Candidate(
                recording_mbid="rec-2",
                release_group_mbid="rg-2",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-3",
                release_group_mbid="rg-3",
                discovery_method=DiscoveryMethod.ACOUSTID,
            ),
        ]
    )

    builder = CandidateBuilder(MusicGraphDB(Path("/tmp/test.db")), Normalizer())
    bundle = builder.build_evidence_bundle(candidate_set)

    # Should be sorted alphabetically
    assert bundle.provenance["discovery_methods"] == [
        "acoustid",
        "isrc",
        "title_artist_length",
    ]


def test_duplicate_deduplication():
    """Test that duplicate recordings and release_groups are deduplicated."""
    candidate_set = CandidateSet(
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                title="Song",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-1",  # Same recording
                release_group_mbid="rg-2",  # Different RG
                title="Song",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-2",
                release_group_mbid="rg-1",  # Same RG as first
                title="Other Song",
                discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
            ),
        ]
    )

    builder = CandidateBuilder(MusicGraphDB(Path("/tmp/test.db")), Normalizer())
    bundle = builder.build_evidence_bundle(candidate_set)

    # Should have 2 unique recordings
    assert len(bundle.recordings) == 2
    recording_mbids = {r["mbid"] for r in bundle.recordings}
    assert recording_mbids == {"rec-1", "rec-2"}

    # Should have 2 unique release_groups
    assert len(bundle.release_groups) == 2
    rg_mbids = {rg["mbid"] for rg in bundle.release_groups}
    assert rg_mbids == {"rg-1", "rg-2"}
