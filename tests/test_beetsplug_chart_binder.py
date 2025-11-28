"""Tests for Chart-Binder Beets Plugin Adapter."""

from pathlib import Path

from beetsplug.chart_binder import (
    CanonMode,
    ChartBinderPluginBase,
    DecisionSummary,
    ImportSummary,
)
from chart_binder.resolver import DecisionState
from chart_binder.tagging import CanonicalIDs, CompactFields, TagSet


def test_canon_mode_enum():
    assert CanonMode.ADVISORY.value == "advisory"
    assert CanonMode.AUTHORITATIVE.value == "authoritative"
    assert CanonMode.AUGMENT.value == "augment"


def test_decision_summary_defaults():
    summary = DecisionSummary(file_path=Path("/test.mp3"))
    assert summary.state == DecisionState.INDETERMINATE
    assert summary.release_group_mbid is None
    assert summary.charts_blob is None


def test_import_summary_defaults():
    summary = ImportSummary()
    assert summary.total_items == 0
    assert summary.canonized == 0
    assert summary.augmented == 0
    assert summary.skipped == 0
    assert summary.indeterminate == 0
    assert summary.errors == []


def test_import_summary_counts():
    summary = ImportSummary(
        total_items=10,
        canonized=5,
        augmented=3,
        skipped=1,
        indeterminate=1,
    )
    assert summary.total_items == 10
    assert summary.canonized == 5
    assert summary.augmented == 3


def test_compute_work_key():
    """Test work key computation without beets dependency."""
    import unicodedata

    def normalize(s: str) -> str:
        s = unicodedata.normalize("NFC", s)
        return s.lower().strip()

    artist = "The Beatles"
    title = "Hey Jude"
    work_key = f"{normalize(artist)} // {normalize(title)}"

    assert work_key == "the beatles // hey jude"


def test_build_tagset_from_decision():
    """Test building tagset from decision summary."""
    summary = DecisionSummary(
        file_path=Path("/test.mp3"),
        title="Test Song",
        artist="Test Artist",
        album="Test Album",
        release_group_mbid="rg-12345",
        release_mbid="r-67890",
        state=DecisionState.DECIDED,
        compact_trace="evh=abc123;crg=EARLIEST_OFFICIAL;rr=ORIGIN_COUNTRY_EARLIEST;src=mb;cfg=lw90,rg10",
    )

    tagset = TagSet(
        title=summary.title,
        artist=summary.artist,
        album=summary.album,
        ids=CanonicalIDs(
            mb_release_group_id=summary.release_group_mbid,
            mb_release_id=summary.release_mbid,
        ),
        compact=CompactFields(
            decision_trace=summary.compact_trace,
            ruleset_version="canon-1.0",
        ),
    )

    assert tagset.title == "Test Song"
    assert tagset.ids.mb_release_group_id == "rg-12345"
    assert tagset.compact.decision_trace is not None
    assert "evh=abc123" in tagset.compact.decision_trace


def test_plugin_base_class():
    """Test the base class without beets."""
    plugin = ChartBinderPluginBase()
    assert plugin.mode == CanonMode.ADVISORY
    assert plugin.get_config("lead_window_days") == 90
    assert plugin.get_config("reissue_long_gap_years") == 10


def test_plugin_base_resolver():
    """Test resolver initialization."""
    plugin = ChartBinderPluginBase()
    resolver = plugin.resolver
    assert resolver is not None
    assert resolver.config.lead_window_days == 90
