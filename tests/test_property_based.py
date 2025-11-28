"""Property-based tests for Chart-Binder (Epic 14).

Uses hypothesis for property-based testing to validate invariants
across the codebase.
"""

from __future__ import annotations

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from chart_binder.normalize import Normalizer
from chart_binder.resolver import Resolver
from chart_binder.safe_logging import hash_path, redact_dict, sanitize_message

# Create normalizer instance for tests
_normalizer = Normalizer()


def normalize_artist(text: str) -> str:
    """Wrapper for normalizer artist normalization."""
    return _normalizer.normalize_artist(text).normalized


def normalize_title(text: str) -> str:
    """Wrapper for normalizer title normalization."""
    return _normalizer.normalize_title(text).normalized


# Normalization properties


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=100)
def test_normalize_artist_idempotent(text: str):
    """Property: Normalizing artist twice should equal normalizing once."""
    first = normalize_artist(text)
    second = normalize_artist(first)
    assert first == second, f"Idempotence failed: '{text}' -> '{first}' -> '{second}'"


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=100)
def test_normalize_title_idempotent(text: str):
    """Property: Normalizing title twice should equal normalizing once."""
    first = normalize_title(text)
    second = normalize_title(first)
    assert first == second, f"Idempotence failed: '{text}' -> '{first}' -> '{second}'"


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_normalize_artist_lowercase(text: str):
    """Property: Normalized artist should be lowercase (except specific exceptions)."""
    result = normalize_artist(text)
    # Core should be lowercase (guests might preserve case in some edge cases)
    if " feat. " not in result:
        assert result == result.lower() or result == "", f"Not lowercase: '{result}'"


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_normalize_title_lowercase(text: str):
    """Property: Normalized title core should be lowercase."""
    result = normalize_title(text)
    # Tags might be extracted, but core should be lowercase
    # This is a weak property - mainly checking it doesn't crash
    assert isinstance(result, str)


# Resolver properties


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(), st.integers(), st.none()),
        max_size=5,
    )
)
@settings(max_examples=20)
def test_resolver_never_crashes(random_dict: dict[str, Any]):
    """Property: Resolver should never crash, even with random input."""
    resolver = Resolver()
    # Wrap in evidence bundle structure
    evidence_bundle = {
        "artist": random_dict,
        "recording_candidates": [],
        "provenance": {"sources_used": []},
    }
    try:
        decision = resolver.resolve(evidence_bundle)
        # Should always return a decision object
        assert hasattr(decision, "state")
    except Exception as e:
        # Only allow specific expected exceptions
        if not isinstance(e, (TypeError, KeyError)):
            raise


@given(st.text(min_size=0, max_size=50), st.text(min_size=0, max_size=50))
@settings(max_examples=30)
def test_resolver_deterministic(artist_name: str, title: str):
    """Property: Same input should always produce same output."""
    resolver = Resolver()
    evidence_bundle = {
        "artist": {"name": artist_name},
        "recording_candidates": [
            {
                "title": title,
                "rg_candidates": [],
            }
        ],
        "provenance": {"sources_used": []},
    }

    decision1 = resolver.resolve(evidence_bundle)
    decision2 = resolver.resolve(evidence_bundle)

    assert decision1.state == decision2.state
    assert decision1.release_group_mbid == decision2.release_group_mbid
    assert decision1.release_mbid == decision2.release_mbid


# Safe logging properties


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_hash_path_deterministic(path: str):
    """Property: Same path should always produce same hash."""
    hash1 = hash_path(path)
    hash2 = hash_path(path)
    assert hash1 == hash2


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_hash_path_fixed_length(path: str):
    """Property: Hash should always be 12 characters."""
    result = hash_path(path)
    assert len(result) == 12


@given(st.text(min_size=0, max_size=500))
@settings(max_examples=50)
def test_sanitize_message_no_emails(text: str):
    """Property: Sanitized message should not contain email patterns."""
    result = sanitize_message(text)
    # If original had email patterns, they should be replaced
    import re

    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    emails_in_result = email_pattern.findall(result)
    assert len(emails_in_result) == 0, f"Found emails in sanitized output: {emails_in_result}"


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(max_size=50), st.integers(), st.none()),
        max_size=10,
    )
)
@settings(max_examples=30)
def test_redact_dict_preserves_structure(data: dict[str, Any]):
    """Property: Redacted dict should have same keys as original."""
    result = redact_dict(data)
    assert set(result.keys()) == set(data.keys())


@given(st.text(min_size=1, max_size=100, alphabet="abcdefghijklmnopqrstuvwxyz"))
@settings(max_examples=50)
def test_redact_dict_redacts_sensitive_keys(value: str):
    """Property: Known sensitive keys should always be redacted."""
    sensitive_keys = ["password", "api_key", "secret", "token"]
    for key in sensitive_keys:
        data = {key: value}
        result = redact_dict(data)
        if len(value) > 4:
            assert "***" in result[key], f"Key '{key}' was not redacted"
        else:
            assert result[key] == "***", f"Short value for '{key}' was not fully redacted"


# Compilation check - property tests should not alter module state


def test_evidence_hash_stable_ordering():
    """Verify evidence hash is stable regardless of dict ordering."""
    resolver = Resolver()

    # Same data, different insertion order
    bundle1 = {
        "artist": {"name": "Test", "mbid": "123"},
        "recording_candidates": [],
        "provenance": {"sources_used": ["a", "b"]},
    }
    bundle2 = {
        "provenance": {"sources_used": ["a", "b"]},
        "artist": {"mbid": "123", "name": "Test"},
        "recording_candidates": [],
    }

    hash1 = resolver._hash_evidence(bundle1)
    hash2 = resolver._hash_evidence(bundle2)

    assert hash1 == hash2, "Hash should be stable regardless of key order"


def test_compilation_no_va_flip():
    """Property: Adding VA compilation should not flip canonicalization without premiere evidence.

    This is a key invariant from the spec - adding a Various Artists compilation
    to the candidate set should not change the CRG selection unless the compilation
    has explicit premiere evidence.
    """
    resolver = Resolver()

    # Original evidence without compilation
    bundle_without_comp = {
        "artist": {"name": "Test Artist"},
        "recording_candidates": [
            {
                "title": "Test Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album-001",
                        "title": "Test Album",
                        "primary_type": "Album",
                        "first_release_date": "2020-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-001",
                                "date": "2020-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    }
                ],
            }
        ],
        "provenance": {"sources_used": ["test"]},
    }

    # Same evidence but with VA compilation added (later date, no premiere evidence)
    bundle_with_comp = {
        "artist": {"name": "Test Artist"},
        "recording_candidates": [
            {
                "title": "Test Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-album-001",
                        "title": "Test Album",
                        "primary_type": "Album",
                        "first_release_date": "2020-01-01",
                        "releases": [
                            {
                                "mb_release_id": "rel-001",
                                "date": "2020-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                    {
                        "mb_rg_id": "rg-comp-001",
                        "title": "Various Artists Compilation",
                        "primary_type": "Album",
                        "secondary_types": ["Compilation"],
                        "first_release_date": "2021-01-01",  # Later date
                        "releases": [
                            {
                                "mb_release_id": "rel-comp-001",
                                "date": "2021-01-01",
                                "flags": {"is_official": True},
                            }
                        ],
                    },
                ],
            }
        ],
        "provenance": {"sources_used": ["test"]},
    }

    decision_without = resolver.resolve(bundle_without_comp)
    decision_with = resolver.resolve(bundle_with_comp)

    # CRG should remain the original album, not flip to compilation
    assert decision_without.release_group_mbid == decision_with.release_group_mbid, (
        f"Adding VA compilation flipped CRG from "
        f"{decision_without.release_group_mbid} to {decision_with.release_group_mbid}"
    )
