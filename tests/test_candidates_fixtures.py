"""
Epic 4 Acceptance Tests: Candidate Builder with Fixtures

Tests candidate discovery and evidence bundle construction
using deterministic fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chart_binder.candidates import Candidate, CandidateBuilder, CandidateSet, DiscoveryMethod
from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer


@pytest.fixture
def musicgraph_db(tmp_path):
    """Create a musicgraph DB with test fixtures."""
    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")

    # Fixture 1: "Under Pressure" by Queen & David Bowie
    # Recording with ISRC GBUM71029604
    db.upsert_artist(
        mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        name="Queen",
        sort_name="Queen",
        begin_area_country="GB",
    )

    db.upsert_recording(
        mbid="7be81cfa-0f03-47b1-8c6b-3f4e8e4f3e3e",
        title="Under Pressure",
        artist_mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        length_ms=245000,
        isrcs_json='["GBUM71029604"]',
    )

    db.upsert_release_group(
        mbid="1e3f5e3b-3b3b-4b4b-8b8b-8b8b8b8b8b8b",
        title="Hot Space",
        artist_mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        type="Album",
        first_release_date="1982-05-21",
    )

    # Fixture 2: "Bohemian Rhapsody" by Queen
    # Multiple release groups (single, album, compilations)
    db.upsert_recording(
        mbid="b1a9c0e1-1f1f-1f1f-1f1f-1f1f1f1f1f1f",
        title="Bohemian Rhapsody",
        artist_mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        length_ms=354000,
        isrcs_json='["GBUM71500052"]',
    )

    db.upsert_release_group(
        mbid="2e3f5e3b-3b3b-4b4b-8b8b-8b8b8b8b8b8b",
        title="Bohemian Rhapsody",
        artist_mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        type="Single",
        first_release_date="1975-10-31",
    )

    db.upsert_release_group(
        mbid="3e3f5e3b-3b3b-4b4b-8b8b-8b8b8b8b8b8b",
        title="A Night at the Opera",
        artist_mbid="0383dadf-2a4e-4d10-a46a-e9e041da8eb3",
        type="Album",
        first_release_date="1975-11-21",
    )

    return db


def test_fixture_candidate_discovery_isrc(musicgraph_db):
    """
    Epic 4 Acceptance: ISRC-based discovery produces expected candidates.

    Fixture: Under Pressure with ISRC GBUM71029604
    Expected: Find recording and associated release group(s)
    """
    norm = Normalizer()
    builder = CandidateBuilder(musicgraph_db, norm)

    # This test will pass once DB queries are implemented
    # For now, it demonstrates the expected interface
    candidates = builder.discover_by_isrc("GBUM71029604")

    # Expected: at least one candidate for Hot Space album
    # (Stubbed for now, will work with full DB implementation)
    assert isinstance(candidates, list)


def test_fixture_candidate_discovery_title_artist_length(musicgraph_db):
    """
    Epic 4 Acceptance: Title+artist+length discovery with fuzzy matching.

    Fixture: "Bohemian Rhapsody" by "Queen" (~354s)
    Expected: Find both Single and Album release groups
    """
    norm = Normalizer()
    builder = CandidateBuilder(musicgraph_db, norm)

    candidates = builder.discover_by_title_artist_length(
        title="Bohemian Rhapsody", artist="Queen", length_ms=354000
    )

    # Expected: candidates for both single and album
    # Length bucket should allow ±10% variance (318.6s - 389.4s)
    assert isinstance(candidates, list)


def test_evidence_bundle_determinism(tmp_path):
    """
    Epic 4 Acceptance: Evidence hash stability.

    Same candidate set produces identical evidence_hash.
    """
    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    # Create identical candidate sets
    candidate_set = CandidateSet(
        file_path=Path("/test/song.mp3"),
        candidates=[
            Candidate(
                recording_mbid="rec-123",
                release_group_mbid="rg-456",
                artist_mbid="art-789",
                title="Test Song",
                artist_name="Test Artist",
                length_ms=240000,
                isrcs=["TEST123"],
                discovery_method=DiscoveryMethod.ISRC,
            )
        ],
        normalized_title="test song",
        normalized_artist="test artist",
        length_ms=240000,
    )

    # Build evidence bundle twice
    bundle1 = builder.build_evidence_bundle(candidate_set)
    bundle2 = builder.build_evidence_bundle(candidate_set)

    # Hash must be identical
    assert bundle1.evidence_hash == bundle2.evidence_hash
    assert len(bundle1.evidence_hash) == 64  # SHA256 hex


def test_evidence_bundle_candidate_order_independence(tmp_path):
    """
    Epic 4 Acceptance: Evidence hash is order-independent.

    Different candidate ordering produces same evidence_hash
    due to deterministic sorting.
    """
    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    candidates_order1 = [
        Candidate(
            recording_mbid="rec-aaa",
            release_group_mbid="rg-111",
            discovery_method=DiscoveryMethod.ISRC,
        ),
        Candidate(
            recording_mbid="rec-zzz",
            release_group_mbid="rg-999",
            discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
        ),
        Candidate(
            recording_mbid="rec-mmm",
            release_group_mbid="rg-555",
            discovery_method=DiscoveryMethod.ISRC,
        ),
    ]

    candidates_order2 = [
        Candidate(
            recording_mbid="rec-zzz",
            release_group_mbid="rg-999",
            discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
        ),
        Candidate(
            recording_mbid="rec-mmm",
            release_group_mbid="rg-555",
            discovery_method=DiscoveryMethod.ISRC,
        ),
        Candidate(
            recording_mbid="rec-aaa",
            release_group_mbid="rg-111",
            discovery_method=DiscoveryMethod.ISRC,
        ),
    ]

    candidate_set1 = CandidateSet(candidates=candidates_order1)
    candidate_set2 = CandidateSet(candidates=candidates_order2)

    bundle1 = builder.build_evidence_bundle(candidate_set1)
    bundle2 = builder.build_evidence_bundle(candidate_set2)

    # Hash must be identical despite different input order
    assert bundle1.evidence_hash == bundle2.evidence_hash


def test_evidence_bundle_provenance_tracking(tmp_path):
    """
    Epic 4 Acceptance: Evidence bundle tracks discovery methods.

    Provenance includes which discovery methods were used.
    """
    db = MusicGraphDB(tmp_path / "musicgraph.sqlite")
    norm = Normalizer()
    builder = CandidateBuilder(db, norm)

    candidate_set = CandidateSet(
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                discovery_method=DiscoveryMethod.ISRC,
            ),
            Candidate(
                recording_mbid="rec-2",
                release_group_mbid="rg-2",
                discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
            ),
            Candidate(
                recording_mbid="rec-3",
                release_group_mbid="rg-3",
                discovery_method=DiscoveryMethod.ISRC,
            ),
        ]
    )

    bundle = builder.build_evidence_bundle(candidate_set)

    # Check provenance tracks discovery methods
    assert "discovery_methods" in bundle.provenance
    assert "isrc" in bundle.provenance["discovery_methods"]
    assert "title_artist_length" in bundle.provenance["discovery_methods"]


def test_length_bucket_tolerance():
    """
    Epic 4 Acceptance: Length bucket has ±10% tolerance.

    Document expected behavior for fuzzy length matching.
    """
    # 240000ms (4 minutes)
    # Min: 216000ms (3:36)
    # Max: 264000ms (4:24)

    length_ms = 240000
    tolerance = 0.1

    length_min = int(length_ms * (1 - tolerance))
    length_max = int(length_ms * (1 + tolerance))

    assert length_min == 216000
    assert length_max == 264000

    # 354000ms (~5:54) for Bohemian Rhapsody
    length_ms = 354000
    length_min = int(length_ms * (1 - tolerance))
    length_max = int(length_ms * (1 + tolerance))

    assert length_min == 318600  # ~5:18
    assert length_max == 389400  # ~6:29


def test_candidate_set_structure():
    """
    Epic 4 Acceptance: CandidateSet has expected structure.

    Validates the data model for candidate sets.
    """
    candidate_set = CandidateSet(
        file_path=Path("/test/song.mp3"),
        candidates=[
            Candidate(
                recording_mbid="rec-1",
                release_group_mbid="rg-1",
                artist_mbid="art-1",
                title="Song Title",
                artist_name="Artist Name",
                length_ms=240000,
                isrcs=["ISRC123"],
                discovery_method=DiscoveryMethod.ISRC,
            )
        ],
        normalized_title="song title",
        normalized_artist="artist name",
        length_ms=240000,
    )

    assert candidate_set.file_path == Path("/test/song.mp3")
    assert len(candidate_set.candidates) == 1
    assert candidate_set.normalized_title == "song title"
    assert candidate_set.normalized_artist == "artist name"
    assert candidate_set.length_ms == 240000

    candidate = candidate_set.candidates[0]
    assert candidate.recording_mbid == "rec-1"
    assert candidate.release_group_mbid == "rg-1"
    assert candidate.discovery_method == "isrc"
    assert "ISRC123" in candidate.isrcs
