"""Tests for tag assembly and writers (Epic 7)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from chart_binder.tagging import (
    CanonicalIDs,
    CompactFields,
    ID3TagWriter,
    MP4TagWriter,
    ReleaseType,
    TagSet,
    VorbisTagWriter,
    WriteReport,
    get_writer_for_file,
    verify,
    write_tags,
)


def create_minimal_mp3(path: Path) -> None:
    """Create a minimal valid MP3 file for testing."""
    # Minimal MP3: ID3v2.4 header + minimal frame
    id3_header = b"ID3\x04\x00\x00\x00\x00\x00\x00"  # ID3v2.4 header with 0 size
    # Add a minimal MPEG frame (sync word + header)
    # Frame sync (11 bits) + version (2 bits) + layer (2 bits) + protection (1 bit) = 0xFF 0xFB
    # Then bitrate/samplerate/padding/private
    mpeg_frame = b"\xff\xfb\x90\x00" + b"\x00" * 417  # ~417 bytes for a valid frame

    with open(path, "wb") as f:
        f.write(id3_header)
        f.write(mpeg_frame)


def create_minimal_flac(path: Path) -> None:
    """Create a minimal valid FLAC file for testing."""
    # FLAC signature
    flac_sig = b"fLaC"

    # Minimal STREAMINFO metadata block (last block, type 0, length 34)
    # Block header: 1 (last) << 7 | 0 (type) = 0x80, then 24-bit length = 34
    block_header = bytes([0x80, 0x00, 0x00, 0x22])

    # STREAMINFO (34 bytes):
    # - min/max block size (4 bytes): 4096/4096
    # - min/max frame size (6 bytes): 0/0 (unknown)
    # - sample rate (20 bits) + channels (3 bits) + bits/sample (5 bits) + total samples (36 bits) = 8 bytes
    # - MD5 (16 bytes)
    streaminfo = (
        struct.pack(">HH", 4096, 4096)  # block sizes
        + b"\x00\x00\x00\x00\x00\x00"  # frame sizes (unknown)
        + struct.pack(">I", (44100 << 12) | (0 << 9) | (15 << 4) | 0)[1:]  # 3 bytes
        + struct.pack(">I", 0)  # total samples low
        + b"\x00"  # total samples high
        + b"\x00" * 16  # MD5
    )

    with open(path, "wb") as f:
        f.write(flac_sig)
        f.write(block_header)
        f.write(streaminfo)


def create_minimal_m4a(path: Path) -> None:
    """Create a minimal valid M4A file for testing."""
    # Minimal ftyp box
    ftyp = b"\x00\x00\x00\x14ftypM4A \x00\x00\x00\x00M4A "
    # Minimal moov box with empty content
    moov = b"\x00\x00\x00\x08moov"
    # Add mdat box (media data)
    mdat = b"\x00\x00\x00\x08mdat"

    with open(path, "wb") as f:
        f.write(ftyp)
        f.write(moov)
        f.write(mdat)


# TagSet tests


def test_tagset_creation():
    """Test TagSet dataclass creation with default values."""
    tagset = TagSet()
    assert tagset.title is None
    assert tagset.artist is None
    assert tagset.album is None
    assert tagset.ids is not None
    assert tagset.compact is not None


def test_tagset_with_values():
    """Test TagSet with values populated."""
    tagset = TagSet(
        title="Bohemian Rhapsody",
        artist="Queen",
        album="A Night at the Opera",
        original_year="1975",
        track_number=11,
        track_total=12,
        release_type=ReleaseType.ALBUM,
        ids=CanonicalIDs(
            mb_recording_id="rec-123",
            mb_release_group_id="rg-456",
            mb_release_id="rel-789",
        ),
        compact=CompactFields(
            charts_blob='{"v":1,"c":[["t2000",1971,1,"y"]]}',
            decision_trace="evh=abc123;crg=ALBUM_LEAD_WINDOW;rr=ORIGIN_COUNTRY_EARLIEST;src=mb;cfg=lw90,rg10",
        ),
    )

    assert tagset.title == "Bohemian Rhapsody"
    assert tagset.artist == "Queen"
    assert tagset.track_number == 11
    assert tagset.release_type == ReleaseType.ALBUM
    assert tagset.ids.mb_recording_id == "rec-123"
    assert tagset.compact.charts_blob is not None


def test_canonical_ids():
    """Test CanonicalIDs dataclass."""
    ids = CanonicalIDs(
        mb_recording_id="rec-123",
        mb_release_group_id="rg-456",
        spotify_track_id="spotify:track:abc",
    )
    assert ids.mb_recording_id == "rec-123"
    assert ids.discogs_master_id is None


def test_compact_fields():
    """Test CompactFields dataclass."""
    compact = CompactFields(
        charts_blob='{"v":1,"c":[]}',
        decision_trace="evh=abc;crg=X;rr=Y;src=mb;cfg=lw90,rg10",
        ruleset_version="canon-1.0.norm-v1",
        evidence_hash="sha256:abc123",
    )
    assert compact.charts_blob is not None
    assert compact.ruleset_version == "canon-1.0.norm-v1"


def test_release_type_enum():
    """Test ReleaseType enum values."""
    assert ReleaseType.ALBUM.value == "album"
    assert ReleaseType.SINGLE.value == "single"
    assert ReleaseType.SOUNDTRACK.value == "soundtrack"


# Writer selection tests


def test_get_writer_for_mp3():
    """Test writer selection for MP3 files."""
    writer = get_writer_for_file(Path("test.mp3"))
    assert isinstance(writer, ID3TagWriter)


def test_get_writer_for_flac():
    """Test writer selection for FLAC files."""
    writer = get_writer_for_file(Path("test.flac"))
    assert isinstance(writer, VorbisTagWriter)


def test_get_writer_for_m4a():
    """Test writer selection for M4A files."""
    writer = get_writer_for_file(Path("test.m4a"))
    assert isinstance(writer, MP4TagWriter)


def test_get_writer_unsupported():
    """Test writer selection for unsupported format."""
    with pytest.raises(ValueError, match="Unsupported audio format"):
        get_writer_for_file(Path("test.wav"))


# ID3 writer tests (with actual file operations)


def test_id3_write_and_read_augment_mode(tmp_path: Path):
    """Test ID3 tag writing in augment-only mode."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    tagset = TagSet(
        title="Test Title",  # Should be skipped in augment mode
        ids=CanonicalIDs(
            mb_recording_id="rec-123",
            mb_release_group_id="rg-456",
        ),
        compact=CompactFields(
            charts_blob='{"v":1,"c":[["t2000",100,50,"y"]]}',
            decision_trace="evh=test123;crg=EARLIEST_OFFICIAL;rr=WORLD_EARLIEST;src=mb;cfg=lw90,rg10",
        ),
    )

    # Write in augment mode (authoritative=False)
    report = writer.write_tags(mp3_path, tagset, authoritative=False)

    assert "title" in report.fields_skipped  # Core fields skipped
    assert "mb_recording_id" in report.fields_written
    assert "charts_blob" in report.fields_written

    # Read back and verify
    read_tags = writer.read_tags(mp3_path)
    assert read_tags.get("mb_recording_id") == "rec-123"
    assert read_tags.get("mb_release_group_id") == "rg-456"
    assert read_tags.get("charts_blob") == '{"v":1,"c":[["t2000",100,50,"y"]]}'


def test_id3_write_authoritative_mode(tmp_path: Path):
    """Test ID3 tag writing in authoritative mode."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    tagset = TagSet(
        title="Canonical Title",
        artist="Canonical Artist",
        album="Canonical Album",
        original_year="1985",
        track_number=5,
        track_total=10,
        release_type=ReleaseType.ALBUM,
        ids=CanonicalIDs(mb_release_group_id="rg-test"),
    )

    # Write in authoritative mode
    report = writer.write_tags(mp3_path, tagset, authoritative=True)

    assert "title" in report.fields_written
    assert "artist" in report.fields_written
    assert "album" in report.fields_written
    assert "original_year" in report.fields_written

    # Read back and verify
    read_tags = writer.read_tags(mp3_path)
    assert read_tags.get("title") == "Canonical Title"
    assert read_tags.get("artist") == "Canonical Artist"
    assert read_tags.get("album") == "Canonical Album"
    assert read_tags.get("original_year") == "1985"
    assert read_tags.get("track_number") == 5
    assert read_tags.get("track_total") == 10


def test_id3_dry_run(tmp_path: Path):
    """Test ID3 dry run mode doesn't write."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    tagset = TagSet(
        title="Should Not Write",
        ids=CanonicalIDs(mb_recording_id="dry-run-id"),
    )

    # Dry run
    report = writer.write_tags(mp3_path, tagset, authoritative=True, dry_run=True)

    assert report.dry_run is True
    assert "title" in report.fields_written  # Reported as would-be-written

    # Verify nothing was actually written
    read_tags = writer.read_tags(mp3_path)
    assert read_tags.get("title") is None
    assert read_tags.get("mb_recording_id") is None


def test_id3_verify(tmp_path: Path):
    """Test ID3 round-trip verification."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    tagset = TagSet(
        title="Verify Test",
        artist="Verify Artist",
        ids=CanonicalIDs(mb_recording_id="verify-123"),
        compact=CompactFields(charts_blob='{"v":1,"c":[]}'),
    )

    writer.write_tags(mp3_path, tagset, authoritative=True)

    # Verify should pass
    assert writer.verify(mp3_path, tagset) is True

    # Modify tagset and verify should fail
    tagset.title = "Different Title"
    assert writer.verify(mp3_path, tagset) is False


def test_id3_original_stashing(tmp_path: Path):
    """Test ID3 original value stashing on first write."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    # First write with some original values
    original_tagset = TagSet(title="Original Title", artist="Original Artist")
    writer.write_tags(mp3_path, original_tagset, authoritative=True)

    # Second write should stash originals
    new_tagset = TagSet(
        title="New Title",
        artist="New Artist",
        ids=CanonicalIDs(mb_recording_id="new-id"),
    )
    report = writer.write_tags(mp3_path, new_tagset, authoritative=True)

    assert "title" in report.originals_stashed
    assert "artist" in report.originals_stashed

    # Verify originals were stashed
    read_tags = writer.read_tags(mp3_path)
    assert read_tags.get("orig_title") == "Original Title"
    assert read_tags.get("orig_artist") == "Original Artist"
    assert read_tags.get("title") == "New Title"


def test_id3_original_stashing_non_destructive(tmp_path: Path):
    """Test ORIG_ tags are never overwritten on consecutive runs.

    This is critical for recovery: even if we botch multiple tagging jobs,
    we can always find back the ORIGINAL tags from before chart-binder touched the file.
    """
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    writer = ID3TagWriter()

    # First write: user's original file with their tags
    original_tagset = TagSet(title="User Original", artist="User Artist", album="User Album")
    writer.write_tags(mp3_path, original_tagset, authoritative=True)

    # Second write: first chart-binder tagging (stashes originals)
    first_cb_tagset = TagSet(
        title="Chart-Binder V1",
        artist="CB Artist V1",
        album="CB Album V1",
        ids=CanonicalIDs(mb_recording_id="cb-v1"),
    )
    report1 = writer.write_tags(mp3_path, first_cb_tagset, authoritative=True)

    assert "title" in report1.originals_stashed
    read_tags1 = writer.read_tags(mp3_path)
    assert read_tags1.get("orig_title") == "User Original"
    assert read_tags1.get("title") == "Chart-Binder V1"

    # Third write: another chart-binder tagging (MUST NOT overwrite ORIG_)
    second_cb_tagset = TagSet(
        title="Chart-Binder V2",
        artist="CB Artist V2",
        album="CB Album V2",
        ids=CanonicalIDs(mb_recording_id="cb-v2"),
    )
    report2 = writer.write_tags(mp3_path, second_cb_tagset, authoritative=True)

    # ORIG_ tags should NOT be reported as stashed (already existed)
    assert "title" not in report2.originals_stashed
    assert "artist" not in report2.originals_stashed

    # Verify ORIG_ still contains USER's original, not "Chart-Binder V1"
    read_tags2 = writer.read_tags(mp3_path)
    assert read_tags2.get("orig_title") == "User Original"  # NOT "Chart-Binder V1"
    assert read_tags2.get("orig_artist") == "User Artist"  # NOT "CB Artist V1"
    assert read_tags2.get("orig_album") == "User Album"  # NOT "CB Album V1"
    assert read_tags2.get("title") == "Chart-Binder V2"  # Current title IS updated

    # Fourth write: yet another run (STILL must not overwrite ORIG_)
    third_cb_tagset = TagSet(
        title="Chart-Binder V3",
        artist="CB Artist V3",
    )
    writer.write_tags(mp3_path, third_cb_tagset, authoritative=True)

    read_tags3 = writer.read_tags(mp3_path)
    assert read_tags3.get("orig_title") == "User Original"  # Still the original!
    assert read_tags3.get("title") == "Chart-Binder V3"


# High-level API tests


def test_write_tags_function(tmp_path: Path):
    """Test high-level write_tags function."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    tagset = TagSet(
        ids=CanonicalIDs(mb_release_group_id="high-level-test"),
        compact=CompactFields(decision_trace="evh=test;crg=X;rr=Y;src=mb;cfg=lw90,rg10"),
    )

    report = write_tags(mp3_path, tagset)

    assert isinstance(report, WriteReport)
    assert report.file_path == mp3_path
    assert "mb_release_group_id" in report.fields_written


def test_verify_function(tmp_path: Path):
    """Test high-level verify function."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    # Write some tags first
    writer = ID3TagWriter()
    tagset = TagSet(
        title="Verify Func Test",
        ids=CanonicalIDs(mb_recording_id="verify-func-123"),
        compact=CompactFields(charts_blob='{"v":1,"c":[]}'),
        release_type=ReleaseType.SINGLE,
    )
    writer.write_tags(mp3_path, tagset, authoritative=True)

    # Use verify function
    read_tagset = verify(mp3_path)

    assert read_tagset.title == "Verify Func Test"
    assert read_tagset.ids.mb_recording_id == "verify-func-123"
    assert read_tagset.compact.charts_blob == '{"v":1,"c":[]}'
    assert read_tagset.release_type == ReleaseType.SINGLE


# CHARTS blob integration


def test_charts_blob_writing(tmp_path: Path):
    """Test CHARTS blob is written and read correctly."""
    mp3_path = tmp_path / "test.mp3"
    create_minimal_mp3(mp3_path)

    charts_blob = json.dumps(
        {
            "v": 1,
            "c": [
                ["t2000", 36250, 30, "y"],
                ["t40", 266, 4, "w"],
            ],
        },
        separators=(",", ":"),
    )

    tagset = TagSet(compact=CompactFields(charts_blob=charts_blob))

    write_tags(mp3_path, tagset)
    read_tagset = verify(mp3_path)

    assert read_tagset.compact.charts_blob == charts_blob

    # Parse and verify JSON structure
    parsed = json.loads(read_tagset.compact.charts_blob or "")
    assert parsed["v"] == 1
    assert len(parsed["c"]) == 2
    assert parsed["c"][0][0] == "t2000"


# WriteReport tests


def test_write_report_structure():
    """Test WriteReport dataclass structure."""
    report = WriteReport(
        file_path=Path("test.mp3"),
        fields_written=["title", "artist"],
        fields_skipped=["album"],
        originals_stashed=["title"],
        errors=[],
        dry_run=False,
    )

    assert report.file_path == Path("test.mp3")
    assert len(report.fields_written) == 2
    assert "title" in report.originals_stashed
    assert report.dry_run is False
