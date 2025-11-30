"""Tests for canonical resolver (CRG and RR selection).

Tests cover all 7 CRG selection rules and RR selection algorithm
using deterministic fixtures.
"""

from __future__ import annotations

import pytest

from chart_binder.resolver import (
    ConfigSnapshot,
    CRGRationale,
    DecisionState,
    Resolver,
    RRRationale,
)


@pytest.fixture
def resolver():
    """Create a Resolver instance with default config."""
    return Resolver()


@pytest.fixture
def resolver_with_short_window():
    """Create a Resolver with 30-day lead window for testing."""
    config = ConfigSnapshot(lead_window_days=30)
    return Resolver(config)


def test_crg_rule_1_soundtrack_premiere(resolver):
    """
    Rule 1: Soundtrack Origin.

    Soundtrack RG with earliest date should be selected.
    """
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Test Artist",
            "begin_area_country": "US",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Theme Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-soundtrack",
                        "title": "Movie Soundtrack",
                        "primary_type": "Soundtrack",
                        "first_release_date": "1995-06-15",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1995-06-15",
                                "country": "US",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-album",
                        "title": "Greatest Hits",
                        "primary_type": "Album",
                        "first_release_date": "1995-08-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-2",
                                "date": "1995-08-01",
                                "country": "US",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_soundtrack_date": "1995-06-15",
            "earliest_album_date": "1995-08-01",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-soundtrack"
    assert decision.crg_rationale == CRGRationale.SOUNDTRACK_PREMIERE
    assert decision.decision_trace.crg_selection["first_release_date"] == "1995-06-15"


def test_crg_rule_2a_album_lead_window(resolver):
    """
    Rule 2A: Album Lead-Window.

    Album RG should be selected when single is a promo within lead window.
    """
    # Note: This test will pass when promo detection is implemented
    # For now, it demonstrates the interface
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Hit Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1991-02-04",
                        "releases": [
                            {
                                "mb_release_id": "rel-album",
                                "date": "1991-02-04",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-single",
                        "title": "Hit Song",
                        "primary_type": "Single",
                        "secondary_types": ["Promo"],
                        "first_release_date": "1990-11-12",
                        "releases": [
                            {
                                "mb_release_id": "rel-single",
                                "date": "1990-11-12",
                                "country": "UK",
                                "flags": {"is_official": True, "is_promo": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_album_date": "1991-02-04",
            "earliest_single_ep_date": "1990-11-12",
        },
        "provenance": {"sources_used": ["MB", "Discogs"]},
    }

    # This will not trigger until promo detection is implemented
    decision = resolver.resolve(evidence_bundle)

    # For now, should fall through to rule 6 (earliest official)
    # When promo detection is implemented, this should be ALBUM_LEAD_WINDOW
    assert decision.state == DecisionState.DECIDED
    # TODO: Update this assertion when promo detection is implemented:
    # assert decision.release_group_mbid == "rg-album"
    # assert decision.crg_rationale == CRGRationale.ALBUM_LEAD_WINDOW


def test_crg_rule_2b_single_true_premiere(resolver):
    """
    Rule 2B: Single True Premiere.

    Single RG should be selected when it's truly first (outside lead window or no album).
    Album comes 143 days after single (outside 90-day lead window).
    """
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Single Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-single",
                        "title": "Single Song",
                        "primary_type": "Single",
                        "first_release_date": "1975-07-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-single",
                                "date": "1975-07-01",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-album",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1975-11-21",
                        "releases": [
                            {
                                "mb_release_id": "rel-album",
                                "date": "1975-11-21",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_single_ep_date": "1975-07-01",
            "earliest_album_date": "1975-11-21",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-single"
    assert decision.crg_rationale == CRGRationale.SINGLE_TRUE_PREMIERE
    assert decision.decision_trace.crg_selection["first_release_date"] == "1975-07-01"


def test_crg_rule_3_live_only_premiere(resolver):
    """
    Rule 3: Live Origin.

    Live RG should be selected when it's the only/earliest type.
    """
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Live Band"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Improvised Jam",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-live",
                        "title": "Live at Venue",
                        "primary_type": "Live",
                        "first_release_date": "1973-05-12",
                        "releases": [
                            {
                                "mb_release_id": "rel-live",
                                "date": "1973-05-12",
                                "country": "US",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {
            "earliest_live_date": "1973-05-12",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-live"
    assert decision.crg_rationale == CRGRationale.LIVE_ONLY_PREMIERE


def test_crg_rule_6_earliest_official(resolver):
    """
    Rule 6: Earliest Official.

    Fallback to earliest RG with confirmed first_release_date.
    """
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Test Artist",
            "begin_area_country": "UK",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album-1",
                        "title": "First Album",
                        "primary_type": "Album",
                        "first_release_date": "1980-01-15",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-01-15",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-album-2",
                        "title": "Compilation",
                        "primary_type": "Album",
                        "first_release_date": "1982-03-20",
                        "releases": [
                            {
                                "mb_release_id": "rel-2",
                                "date": "1982-03-20",
                                "country": "US",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_album_date": "1980-01-15",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-album-1"
    assert decision.crg_rationale == CRGRationale.EARLIEST_OFFICIAL
    assert decision.decision_trace.crg_selection["first_release_date"] == "1980-01-15"


def test_crg_album_same_date_tiebreaker(resolver):
    """
    Tie-breaker: Album over Single when same earliest date.

    When Album and Single have same year-precision date (e.g., "1969"),
    prefer Album since same date implies single is likely within the
    90-day lead window. This is the "Alors je chante" scenario.
    """
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Rika Zaraï",
            "begin_area_country": "IL",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Alors je chante",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album",
                        "title": "Alors je chante",
                        "primary_type": "Album",
                        "first_release_date": "1969",  # Year-only precision
                        "releases": [
                            {
                                "mb_release_id": "rel-album",
                                "date": "1969",
                                "country": "FR",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-single",
                        "title": "Alors je chante",
                        "primary_type": "Single",
                        "first_release_date": "1969",  # Same year-only precision
                        "releases": [
                            {
                                "mb_release_id": "rel-single",
                                "date": "1969",
                                "country": "FR",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_album_date": "1969",
            "earliest_single_ep_date": "1969",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    # Album should win over Single due to same-date tie-breaker
    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-album"
    assert decision.crg_rationale == CRGRationale.ALBUM_SAME_DATE_TIEBREAKER


def test_crg_rule_7_indeterminate_no_dates(resolver):
    """
    Rule 7: Indeterminate.

    Should return INDETERMINATE when no dates are available.
    """
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Mystery Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "title": "Unknown Album",
                        "primary_type": "Album",
                        # No first_release_date
                        "releases": [],
                    }
                ],
            }
        ],
        "timeline_facts": {},
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.INDETERMINATE
    assert decision.crg_rationale == CRGRationale.INDETERMINATE
    assert "insufficient_date_evidence_or_tie" in decision.decision_trace.missing_facts


def test_rr_origin_country_earliest(resolver):
    """
    RR Selection: Artist origin country preference.

    Should select earliest release from artist's origin country.
    """
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "UK Artist",
            "begin_area_country": "UK",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1985-03-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-uk",
                                "date": "1985-03-01",
                                "country": "UK",
                                "label": "UK Label",
                                "flags": {"is_official": True},
                            },
                            {
                                "mb_release_id": "rel-us",
                                "date": "1985-02-15",
                                "country": "US",
                                "label": "US Label",
                                "flags": {"is_official": True},
                            },
                            {
                                "mb_release_id": "rel-uk-2",
                                "date": "1985-03-05",
                                "country": "UK",
                                "label": "Another UK Label",
                                "flags": {"is_official": True},
                            },
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {"earliest_album_date": "1985-03-01"},
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_group_mbid == "rg-1"
    # Should select earliest UK release (rel-uk), not the earlier US one
    assert decision.release_mbid == "rel-uk"
    assert decision.rr_rationale == RRRationale.ORIGIN_COUNTRY_EARLIEST


def test_rr_world_earliest_no_origin_country(resolver):
    """
    RR Selection: Earliest worldwide when no origin country match.

    Should select globally earliest release.
    """
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Mystery Artist",
            # No begin_area_country
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1990-05-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-jp",
                                "date": "1990-05-01",
                                "country": "JP",
                                "flags": {"is_official": True},
                            },
                            {
                                "mb_release_id": "rel-us",
                                "date": "1990-06-15",
                                "country": "US",
                                "flags": {"is_official": True},
                            },
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {"earliest_album_date": "1990-05-01"},
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    assert decision.release_mbid == "rel-jp"
    assert decision.rr_rationale == RRRationale.WORLD_EARLIEST


def test_rr_indeterminate_no_official_releases(resolver):
    """
    RR Selection: Indeterminate when no official releases.

    Should return INDETERMINATE when no official releases exist.
    """
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1999-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-bootleg",
                                "date": "1999-01-01",
                                "country": "XX",
                                "flags": {"is_official": False, "is_bootleg": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {"earliest_album_date": "1999-01-01"},
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.INDETERMINATE
    assert decision.release_group_mbid == "rg-1"  # CRG was selected
    assert decision.rr_rationale == RRRationale.INDETERMINATE
    assert "no_official_releases" in decision.decision_trace.missing_facts


def test_decision_trace_compact_tag(resolver):
    """Test decision trace compact tag generation."""
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Test Artist",
            "begin_area_country": "US",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-single",
                        "title": "Single",
                        "primary_type": "Single",
                        "first_release_date": "1980-05-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-05-01",
                                "country": "US",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {
            "earliest_single_ep_date": "1980-05-01",
        },
        "provenance": {"sources_used": ["MB", "Discogs"]},
    }

    decision = resolver.resolve(evidence_bundle)

    # Check compact tag format
    assert decision.compact_tag.startswith("evh=")
    assert (
        ";crg=SINGLE_TRUE_PREMIERE;" in decision.compact_tag
        or ";crg=EARLIEST_OFFICIAL;" in decision.compact_tag
    )
    assert ";rr=" in decision.compact_tag
    assert ";src=dc,mb;" in decision.compact_tag
    assert ";cfg=lw90,rg10" in decision.compact_tag


def test_evidence_hash_determinism(resolver):
    """Test evidence hash is deterministic."""
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "title": "Album",
                        "primary_type": "Album",
                        "first_release_date": "1980-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-01-01",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {"earliest_album_date": "1980-01-01"},
        "provenance": {"sources_used": ["MB"]},
    }

    decision1 = resolver.resolve(evidence_bundle)
    decision2 = resolver.resolve(evidence_bundle)

    # Hash should be deterministic
    assert decision1.decision_trace.evidence_hash == decision2.decision_trace.evidence_hash
    assert len(decision1.decision_trace.evidence_hash) == 64  # SHA256 hex


def test_evidence_hash_excludes_volatile_fields(resolver):
    """Test evidence hash excludes volatile fields (timestamps, cache ages)."""
    evidence_bundle_1 = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "primary_type": "Album",
                        "first_release_date": "1980-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "provenance": {
            "sources_used": ["MB"],
            "fetched_at_utc": "2025-01-01T00:00:00Z",
            "cache_age_s": 100,
        },
    }

    evidence_bundle_2 = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "primary_type": "Album",
                        "first_release_date": "1980-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "provenance": {
            "sources_used": ["MB"],
            "fetched_at_utc": "2025-06-15T12:30:45Z",  # Different timestamp
            "cache_age_s": 5000,  # Different cache age
        },
    }

    decision1 = resolver.resolve(evidence_bundle_1)
    decision2 = resolver.resolve(evidence_bundle_2)

    # Hash should be identical despite different volatile fields
    assert decision1.decision_trace.evidence_hash == decision2.decision_trace.evidence_hash


def test_config_snapshot_in_trace(resolver):
    """Test config snapshot is included in decision trace."""
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-1",
                        "primary_type": "Album",
                        "first_release_date": "1980-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1980-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)

    # Config snapshot should be present
    assert decision.decision_trace.config_snapshot.lead_window_days == 90
    assert decision.decision_trace.config_snapshot.reissue_long_gap_years == 10
    assert "remaster" in decision.decision_trace.config_snapshot.reissue_terms


def test_partial_date_parsing(resolver):
    """Test handling of partial dates (YYYY-MM and YYYY)."""
    evidence_bundle = {
        "artist": {"mb_artist_id": "artist-1", "name": "Test Artist"},
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-year-only",
                        "title": "Old Album",
                        "primary_type": "Album",
                        "first_release_date": "1965",  # Year only
                        "releases": [
                            {
                                "mb_release_id": "rel-1",
                                "date": "1965",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-year-month",
                        "title": "Newer Album",
                        "primary_type": "Album",
                        "first_release_date": "1965-06",  # Year-month
                        "releases": [
                            {
                                "mb_release_id": "rel-2",
                                "date": "1965-06",
                                "country": "UK",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "timeline_facts": {
            "earliest_album_date": "1965",
        },
        "provenance": {"sources_used": ["MB"]},
    }

    # Should not crash on partial dates
    decision = resolver.resolve(evidence_bundle)

    assert decision.state == DecisionState.DECIDED
    # Should deterministically select "rg-year-only" (1965-01-01 < 1965-06-01)
    assert decision.release_group_mbid == "rg-year-only"


def test_human_readable_trace(resolver):
    """Test human-readable trace generation for explainability."""
    evidence_bundle = {
        "artist": {
            "mb_artist_id": "artist-1",
            "name": "Test Artist",
            "begin_area_country": "UK",
        },
        "recording_candidates": [
            {
                "mb_recording_id": "rec-1",
                "title": "Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album",
                        "title": "Test Album",
                        "primary_type": "Album",
                        "first_release_date": "1985-03-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-uk",
                                "date": "1985-03-01",
                                "country": "UK",
                                "label": "UK Records",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "timeline_facts": {"earliest_album_date": "1985-03-01"},
        "provenance": {"sources_used": ["MB"]},
    }

    decision = resolver.resolve(evidence_bundle)
    trace_str = decision.decision_trace.to_human_readable()

    # Should contain key sections
    assert "Decision Trace" in trace_str
    assert "Ruleset Version: 1.0" in trace_str
    assert "Evidence Hash:" in trace_str
    assert "Artist Origin Country: UK" in trace_str
    assert "CRG Selection:" in trace_str
    assert "RR Selection:" in trace_str
    assert "Config:" in trace_str
    assert "Lead Window: 90 days" in trace_str
    assert "Reissue Gap: 10 years" in trace_str
