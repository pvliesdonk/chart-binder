"""Tag assembly and writers for audio files.

Implements the tag writer map v1 for ID3v2.4 (MP3), Vorbis/FLAC, and MP4.
See: docs/appendix/tag_writer_map_v1.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReleaseType(StrEnum):
    """Canonical release type values."""

    ALBUM = "album"
    SINGLE = "single"
    EP = "ep"
    SOUNDTRACK = "soundtrack"
    LIVE = "live"
    COMPILATION = "compilation"
    REMIX = "remix"
    OTHER = "other"


@dataclass
class CanonicalIDs:
    """Canonical IDs for external services."""

    mb_recording_id: str | None = None
    mb_release_group_id: str | None = None
    mb_release_id: str | None = None
    discogs_master_id: str | None = None
    discogs_release_id: str | None = None
    spotify_track_id: str | None = None
    spotify_album_id: str | None = None
    wikidata_qid: str | None = None


@dataclass
class CompactFields:
    """Compact JSON fields for decision trace and CHARTS."""

    charts_blob: str | None = None  # Minified JSON
    decision_trace: str | None = None  # evh=...;crg=...;rr=...;src=...;cfg=...
    ruleset_version: str | None = None
    evidence_hash: str | None = None


@dataclass
class TagSet:
    """
    Canonical tagset assembled from decision.

    Contains all fields to write to audio files.
    """

    # Core fields (overwrite only if authoritative)
    title: str | None = None
    artist: str | None = None  # No guests in core
    album: str | None = None  # RG title or RR title
    album_artist: str | None = None
    original_year: str | None = None  # YYYY or YYYY-MM-DD
    track_number: int | None = None
    track_total: int | None = None
    disc_number: int | None = None
    disc_total: int | None = None
    label: str | None = None
    country: str | None = None  # ISO-3166-1 alpha-2
    media_format: str | None = None
    release_type: ReleaseType | None = None

    # Canonical IDs (always safe to add)
    ids: CanonicalIDs = field(default_factory=CanonicalIDs)

    # Compact fields
    compact: CompactFields = field(default_factory=CompactFields)

    # Stashed originals (written once, never overwritten)
    orig_title: str | None = None
    orig_artist: str | None = None
    orig_album: str | None = None
    orig_date: str | None = None


@dataclass
class WriteReport:
    """Report of what was written/skipped."""

    file_path: Path
    fields_written: list[str] = field(default_factory=list)
    fields_skipped: list[str] = field(default_factory=list)
    originals_stashed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False


class TagWriter(ABC):
    """
    Abstract base class for format-specific tag writers.

    Subclasses implement format-specific read/write logic.
    """

    @abstractmethod
    def read_tags(self, file_path: Path) -> dict[str, Any]:
        """Read existing tags from file."""
        pass

    @abstractmethod
    def write_tags(
        self,
        file_path: Path,
        tagset: TagSet,
        authoritative: bool = False,
        dry_run: bool = False,
    ) -> WriteReport:
        """
        Write tags to file.

        Args:
            file_path: Path to audio file
            tagset: TagSet to write
            authoritative: If True, overwrite core fields; else augment only
            dry_run: If True, don't write, just report what would be written
        """
        pass

    @abstractmethod
    def verify(self, file_path: Path, tagset: TagSet) -> bool:
        """Verify written tags match tagset (round-trip verification)."""
        pass

    def _should_stash_original(self, existing_tags: dict[str, Any], field_key: str) -> bool:
        """Check if we should stash an original value (first write only)."""
        orig_key = f"orig_{field_key}"
        # Stash if original field exists and we haven't stashed before
        return field_key in existing_tags and orig_key not in existing_tags


class ID3TagWriter(TagWriter):
    """
    Tag writer for ID3v2.4 (MP3) files.

    Uses mutagen for low-level tag manipulation.
    """

    # ID3 frame mappings
    CORE_FRAMES = {
        "title": "TIT2",
        "artist": "TPE1",
        "album": "TALB",
        "album_artist": "TPE2",
        "original_year": "TDOR",
        "track_number": "TRCK",
        "disc_number": "TPOS",
        "label": "TPUB",
        "media_format": "TMED",
    }

    TXXX_FRAMES = {
        "country": "COUNTRY",
        "release_type": "CANONICAL_RELEASE_TYPE",
        "mb_recording_id": "MB_RECORDING_ID",
        "mb_release_group_id": "MB_RELEASE_GROUP_ID",
        "mb_release_id": "MB_RELEASE_ID",
        "discogs_master_id": "DISCOGS_MASTER_ID",
        "discogs_release_id": "DISCOGS_RELEASE_ID",
        "spotify_track_id": "SPOTIFY_TRACK_ID",
        "spotify_album_id": "SPOTIFY_ALBUM_ID",
        "wikidata_qid": "WIKIDATA_QID",
        "charts_blob": "CHARTS",
        "decision_trace": "TAG_DECISION_TRACE",
        "ruleset_version": "CANON_RULESET_VERSION",
        "evidence_hash": "CANON_EVIDENCE_HASH",
        "orig_title": "ORIG_TITLE",
        "orig_artist": "ORIG_ARTIST",
        "orig_album": "ORIG_ALBUM",
        "orig_date": "ORIG_DATE",
    }

    def read_tags(self, file_path: Path) -> dict[str, Any]:
        """Read ID3 tags from MP3 file."""
        from mutagen.id3 import ID3, ID3NoHeaderError
        from mutagen.id3._frames import TXXX

        try:
            tags = ID3(file_path)
        except ID3NoHeaderError:
            return {}

        result: dict[str, Any] = {}

        # Read core frames
        for field_name, frame_id in self.CORE_FRAMES.items():
            frame = tags.get(frame_id)
            if frame:
                if frame_id == "TRCK":
                    # Parse track/total
                    text = str(frame)
                    if "/" in text:
                        track, total = text.split("/", 1)
                        result["track_number"] = int(track) if track else None
                        result["track_total"] = int(total) if total else None
                    else:
                        result["track_number"] = int(text) if text else None
                elif frame_id == "TPOS":
                    # Parse disc/total
                    text = str(frame)
                    if "/" in text:
                        disc, total = text.split("/", 1)
                        result["disc_number"] = int(disc) if disc else None
                        result["disc_total"] = int(total) if total else None
                    else:
                        result["disc_number"] = int(text) if text else None
                else:
                    result[field_name] = str(frame)

        # Read TXXX frames
        for frame in tags.getall("TXXX"):
            if isinstance(frame, TXXX):
                desc = frame.desc  # pyright: ignore[reportAttributeAccessIssue]
                for field_name, txxx_name in self.TXXX_FRAMES.items():
                    if desc == txxx_name:
                        result[field_name] = str(frame)
                        break

        return result

    def write_tags(
        self,
        file_path: Path,
        tagset: TagSet,
        authoritative: bool = False,
        dry_run: bool = False,
    ) -> WriteReport:
        """Write ID3 tags to MP3 file."""
        from mutagen.id3 import ID3, ID3NoHeaderError
        from mutagen.id3._frames import TALB, TDOR, TIT2, TMED, TPE1, TPE2, TPOS, TPUB, TRCK, TXXX

        report = WriteReport(file_path=file_path, dry_run=dry_run)

        # Read existing tags
        try:
            tags = ID3(file_path)
            existing = self.read_tags(file_path)
        except ID3NoHeaderError:
            tags = ID3()
            existing = {}

        # Helper to write TXXX frame
        def write_txxx(name: str, value: str | None, field_name: str) -> None:
            if value is None:
                return
            if not dry_run:
                # Remove existing TXXX with same description
                txxx_frames = [f for f in tags.getall("TXXX") if f.desc == name]
                for _ in txxx_frames:
                    tags.delall(f"TXXX:{name}")
                tags.add(TXXX(encoding=3, desc=name, text=[value]))
            report.fields_written.append(field_name)

        # Stash originals if needed (first write only)
        if authoritative:
            if self._should_stash_original(existing, "title") and existing.get("title"):
                write_txxx("ORIG_TITLE", existing["title"], "orig_title")
                report.originals_stashed.append("title")
            if self._should_stash_original(existing, "artist") and existing.get("artist"):
                write_txxx("ORIG_ARTIST", existing["artist"], "orig_artist")
                report.originals_stashed.append("artist")
            if self._should_stash_original(existing, "album") and existing.get("album"):
                write_txxx("ORIG_ALBUM", existing["album"], "orig_album")
                report.originals_stashed.append("album")
            if self._should_stash_original(existing, "original_year") and existing.get(
                "original_year"
            ):
                write_txxx("ORIG_DATE", existing["original_year"], "orig_date")
                report.originals_stashed.append("original_year")

        # Write core fields (only if authoritative)
        if authoritative:
            if tagset.title is not None:
                if not dry_run:
                    tags.delall("TIT2")
                    tags.add(TIT2(encoding=3, text=[tagset.title]))
                report.fields_written.append("title")
            else:
                report.fields_skipped.append("title")

            if tagset.artist is not None:
                if not dry_run:
                    tags.delall("TPE1")
                    tags.add(TPE1(encoding=3, text=[tagset.artist]))
                report.fields_written.append("artist")
            else:
                report.fields_skipped.append("artist")

            if tagset.album is not None:
                if not dry_run:
                    tags.delall("TALB")
                    tags.add(TALB(encoding=3, text=[tagset.album]))
                report.fields_written.append("album")
            else:
                report.fields_skipped.append("album")

            if tagset.album_artist is not None:
                if not dry_run:
                    tags.delall("TPE2")
                    tags.add(TPE2(encoding=3, text=[tagset.album_artist]))
                report.fields_written.append("album_artist")
            else:
                report.fields_skipped.append("album_artist")

            if tagset.original_year is not None:
                if not dry_run:
                    tags.delall("TDOR")
                    tags.add(TDOR(encoding=3, text=[tagset.original_year]))
                    # Also mirror to TXXX for ID3v2.3 compatibility (no report entry)
                    txxx_frames = [f for f in tags.getall("TXXX") if f.desc == "ORIGINALYEAR"]
                    for _ in txxx_frames:
                        tags.delall("TXXX:ORIGINALYEAR")
                    tags.add(TXXX(encoding=3, desc="ORIGINALYEAR", text=[tagset.original_year]))
                report.fields_written.append("original_year")
            else:
                report.fields_skipped.append("original_year")

            if tagset.track_number is not None:
                track_str = str(tagset.track_number)
                if tagset.track_total:
                    track_str += f"/{tagset.track_total}"
                if not dry_run:
                    tags.delall("TRCK")
                    tags.add(TRCK(encoding=3, text=[track_str]))
                report.fields_written.append("track_number")
            else:
                report.fields_skipped.append("track_number")

            if tagset.disc_number is not None:
                disc_str = str(tagset.disc_number)
                if tagset.disc_total:
                    disc_str += f"/{tagset.disc_total}"
                if not dry_run:
                    tags.delall("TPOS")
                    tags.add(TPOS(encoding=3, text=[disc_str]))
                report.fields_written.append("disc_number")
            else:
                report.fields_skipped.append("disc_number")

            if tagset.label is not None:
                if not dry_run:
                    tags.delall("TPUB")
                    tags.add(TPUB(encoding=3, text=[tagset.label]))
                report.fields_written.append("label")
            else:
                report.fields_skipped.append("label")

            if tagset.media_format is not None:
                if not dry_run:
                    tags.delall("TMED")
                    tags.add(TMED(encoding=3, text=[tagset.media_format]))
                report.fields_written.append("media_format")
            else:
                report.fields_skipped.append("media_format")
        else:
            # In augment-only mode, skip all core fields
            report.fields_skipped.extend(
                [
                    "title",
                    "artist",
                    "album",
                    "album_artist",
                    "original_year",
                    "track_number",
                    "disc_number",
                    "label",
                    "media_format",
                ]
            )

        # Write TXXX fields (always safe to add)
        if tagset.country:
            write_txxx("COUNTRY", tagset.country, "country")
        if tagset.release_type:
            write_txxx("CANONICAL_RELEASE_TYPE", tagset.release_type.value, "release_type")

        # Canonical IDs
        if tagset.ids.mb_recording_id:
            write_txxx("MB_RECORDING_ID", tagset.ids.mb_recording_id, "mb_recording_id")
        if tagset.ids.mb_release_group_id:
            write_txxx("MB_RELEASE_GROUP_ID", tagset.ids.mb_release_group_id, "mb_release_group_id")
        if tagset.ids.mb_release_id:
            write_txxx("MB_RELEASE_ID", tagset.ids.mb_release_id, "mb_release_id")
        if tagset.ids.discogs_master_id:
            write_txxx("DISCOGS_MASTER_ID", tagset.ids.discogs_master_id, "discogs_master_id")
        if tagset.ids.discogs_release_id:
            write_txxx("DISCOGS_RELEASE_ID", tagset.ids.discogs_release_id, "discogs_release_id")
        if tagset.ids.spotify_track_id:
            write_txxx("SPOTIFY_TRACK_ID", tagset.ids.spotify_track_id, "spotify_track_id")
        if tagset.ids.spotify_album_id:
            write_txxx("SPOTIFY_ALBUM_ID", tagset.ids.spotify_album_id, "spotify_album_id")
        if tagset.ids.wikidata_qid:
            write_txxx("WIKIDATA_QID", tagset.ids.wikidata_qid, "wikidata_qid")

        # Compact fields
        if tagset.compact.charts_blob:
            write_txxx("CHARTS", tagset.compact.charts_blob, "charts_blob")
        if tagset.compact.decision_trace:
            write_txxx("TAG_DECISION_TRACE", tagset.compact.decision_trace, "decision_trace")
        if tagset.compact.ruleset_version:
            write_txxx("CANON_RULESET_VERSION", tagset.compact.ruleset_version, "ruleset_version")
        if tagset.compact.evidence_hash:
            write_txxx("CANON_EVIDENCE_HASH", tagset.compact.evidence_hash, "evidence_hash")

        # Save tags
        if not dry_run:
            tags.save(file_path, v2_version=4)

        return report

    def verify(self, file_path: Path, tagset: TagSet) -> bool:
        """Verify written tags match tagset."""
        read_tags = self.read_tags(file_path)

        # Check core fields
        if tagset.title and read_tags.get("title") != tagset.title:
            return False
        if tagset.artist and read_tags.get("artist") != tagset.artist:
            return False
        if tagset.album and read_tags.get("album") != tagset.album:
            return False

        # Check IDs
        if tagset.ids.mb_recording_id:
            if read_tags.get("mb_recording_id") != tagset.ids.mb_recording_id:
                return False
        if tagset.ids.mb_release_group_id:
            if read_tags.get("mb_release_group_id") != tagset.ids.mb_release_group_id:
                return False

        # Check compact fields
        if tagset.compact.charts_blob:
            if read_tags.get("charts_blob") != tagset.compact.charts_blob:
                return False
        if tagset.compact.decision_trace:
            if read_tags.get("decision_trace") != tagset.compact.decision_trace:
                return False

        return True


class VorbisTagWriter(TagWriter):
    """
    Tag writer for Vorbis comments (FLAC, OGG).

    Uses mutagen for low-level tag manipulation.
    """

    FIELD_MAPPINGS = {
        "title": "TITLE",
        "artist": "ARTIST",
        "album": "ALBUM",
        "album_artist": "ALBUMARTIST",
        "original_year": "ORIGINALYEAR",
        "track_number": "TRACKNUMBER",
        "track_total": "TRACKTOTAL",
        "disc_number": "DISCNUMBER",
        "disc_total": "DISCTOTAL",
        "label": "LABEL",
        "country": "COUNTRY",
        "media_format": "MEDIA",
        "release_type": "CANONICAL_RELEASE_TYPE",
        "mb_recording_id": "MB_RECORDING_ID",
        "mb_release_group_id": "MB_RELEASE_GROUP_ID",
        "mb_release_id": "MB_RELEASE_ID",
        "discogs_master_id": "DISCOGS_MASTER_ID",
        "discogs_release_id": "DISCOGS_RELEASE_ID",
        "spotify_track_id": "SPOTIFY_TRACK_ID",
        "spotify_album_id": "SPOTIFY_ALBUM_ID",
        "wikidata_qid": "WIKIDATA_QID",
        "charts_blob": "CHARTS",
        "decision_trace": "TAG_DECISION_TRACE",
        "ruleset_version": "CANON_RULESET_VERSION",
        "evidence_hash": "CANON_EVIDENCE_HASH",
        "orig_title": "ORIG_TITLE",
        "orig_artist": "ORIG_ARTIST",
        "orig_album": "ORIG_ALBUM",
        "orig_date": "ORIG_DATE",
    }

    def read_tags(self, file_path: Path) -> dict[str, Any]:
        """Read Vorbis comments from FLAC/OGG file."""
        from mutagen import File

        audio = File(file_path)
        if audio is None or audio.tags is None:
            return {}

        result: dict[str, Any] = {}
        for field_name, vorbis_key in self.FIELD_MAPPINGS.items():
            values = audio.tags.get(vorbis_key)  # pyright: ignore[reportAttributeAccessIssue]
            if values:
                # Vorbis comments can be multi-valued; take first for core fields
                result[field_name] = values[0] if len(values) == 1 else values
                # Convert numeric fields
                if field_name in ("track_number", "track_total", "disc_number", "disc_total"):
                    try:
                        result[field_name] = int(values[0])
                    except (ValueError, IndexError):
                        # If conversion fails, leave value as-is (string or list).
                        pass

        return result

    def write_tags(
        self,
        file_path: Path,
        tagset: TagSet,
        authoritative: bool = False,
        dry_run: bool = False,
    ) -> WriteReport:
        """Write Vorbis comments to FLAC/OGG file."""
        from mutagen import File

        report = WriteReport(file_path=file_path, dry_run=dry_run)

        audio = File(file_path)
        if audio is None:
            report.errors.append("Could not open file")
            return report

        existing = self.read_tags(file_path)

        def write_field(vorbis_key: str, value: str | None, field_name: str) -> None:
            if value is None:
                report.fields_skipped.append(field_name)
                return
            if not dry_run and audio.tags is not None:
                audio.tags[vorbis_key] = [value]  # pyright: ignore[reportIndexIssue]
            report.fields_written.append(field_name)

        # Stash originals if needed (first write only)
        if authoritative:
            if self._should_stash_original(existing, "title") and existing.get("title"):
                write_field("ORIG_TITLE", str(existing["title"]), "orig_title")
                report.originals_stashed.append("title")
            if self._should_stash_original(existing, "artist") and existing.get("artist"):
                write_field("ORIG_ARTIST", str(existing["artist"]), "orig_artist")
                report.originals_stashed.append("artist")
            if self._should_stash_original(existing, "album") and existing.get("album"):
                write_field("ORIG_ALBUM", str(existing["album"]), "orig_album")
                report.originals_stashed.append("album")
            if self._should_stash_original(existing, "original_year") and existing.get(
                "original_year"
            ):
                write_field("ORIG_DATE", str(existing["original_year"]), "orig_date")
                report.originals_stashed.append("original_year")

        # Write core fields (only if authoritative)
        if authoritative:
            write_field("TITLE", tagset.title, "title")
            write_field("ARTIST", tagset.artist, "artist")
            write_field("ALBUM", tagset.album, "album")
            write_field("ALBUMARTIST", tagset.album_artist, "album_artist")
            write_field("ORIGINALYEAR", tagset.original_year, "original_year")
            write_field(
                "TRACKNUMBER",
                str(tagset.track_number) if tagset.track_number else None,
                "track_number",
            )
            write_field(
                "TRACKTOTAL", str(tagset.track_total) if tagset.track_total else None, "track_total"
            )
            write_field(
                "DISCNUMBER", str(tagset.disc_number) if tagset.disc_number else None, "disc_number"
            )
            write_field(
                "DISCTOTAL", str(tagset.disc_total) if tagset.disc_total else None, "disc_total"
            )
            write_field("LABEL", tagset.label, "label")
            write_field("MEDIA", tagset.media_format, "media_format")
        else:
            report.fields_skipped.extend(
                [
                    "title",
                    "artist",
                    "album",
                    "album_artist",
                    "original_year",
                    "track_number",
                    "track_total",
                    "disc_number",
                    "disc_total",
                    "label",
                    "media_format",
                ]
            )

        # Always write TXXX-equivalent fields
        write_field("COUNTRY", tagset.country, "country")
        write_field(
            "CANONICAL_RELEASE_TYPE",
            tagset.release_type.value if tagset.release_type else None,
            "release_type",
        )

        # Canonical IDs
        write_field("MB_RECORDING_ID", tagset.ids.mb_recording_id, "mb_recording_id")
        write_field("MB_RELEASE_GROUP_ID", tagset.ids.mb_release_group_id, "mb_release_group_id")
        write_field("MB_RELEASE_ID", tagset.ids.mb_release_id, "mb_release_id")
        write_field("DISCOGS_MASTER_ID", tagset.ids.discogs_master_id, "discogs_master_id")
        write_field("DISCOGS_RELEASE_ID", tagset.ids.discogs_release_id, "discogs_release_id")
        write_field("SPOTIFY_TRACK_ID", tagset.ids.spotify_track_id, "spotify_track_id")
        write_field("SPOTIFY_ALBUM_ID", tagset.ids.spotify_album_id, "spotify_album_id")
        write_field("WIKIDATA_QID", tagset.ids.wikidata_qid, "wikidata_qid")

        # Compact fields
        write_field("CHARTS", tagset.compact.charts_blob, "charts_blob")
        write_field("TAG_DECISION_TRACE", tagset.compact.decision_trace, "decision_trace")
        write_field("CANON_RULESET_VERSION", tagset.compact.ruleset_version, "ruleset_version")
        write_field("CANON_EVIDENCE_HASH", tagset.compact.evidence_hash, "evidence_hash")

        # Save
        if not dry_run:
            audio.save()

        return report

    def verify(self, file_path: Path, tagset: TagSet) -> bool:
        """Verify written tags match tagset."""
        read_tags = self.read_tags(file_path)

        if tagset.title and read_tags.get("title") != tagset.title:
            return False
        if tagset.artist and read_tags.get("artist") != tagset.artist:
            return False
        if tagset.album and read_tags.get("album") != tagset.album:
            return False

        if tagset.ids.mb_recording_id:
            if read_tags.get("mb_recording_id") != tagset.ids.mb_recording_id:
                return False
        if tagset.ids.mb_release_group_id:
            if read_tags.get("mb_release_group_id") != tagset.ids.mb_release_group_id:
                return False

        if tagset.compact.charts_blob:
            if read_tags.get("charts_blob") != tagset.compact.charts_blob:
                return False

        return True


class MP4TagWriter(TagWriter):
    """
    Tag writer for MP4/M4A files.

    Uses mutagen for low-level tag manipulation.
    """

    # Standard MP4 atom mappings
    ATOM_MAPPINGS = {
        "title": "\xa9nam",
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "album_artist": "aART",
        "original_year": "\xa9day",
        "track_number": "trkn",  # Special: tuple (track, total)
        "disc_number": "disk",  # Special: tuple (disc, total)
    }

    # Custom iTunes atoms for canonical fields
    CUSTOM_ATOM_PREFIX = "----:com.apple.iTunes:"

    CUSTOM_ATOMS = {
        "label": "LABEL",
        "country": "COUNTRY",
        "media_format": "MEDIA",
        "release_type": "CANONICAL_RELEASE_TYPE",
        "mb_recording_id": "MB_RECORDING_ID",
        "mb_release_group_id": "MB_RELEASE_GROUP_ID",
        "mb_release_id": "MB_RELEASE_ID",
        "discogs_master_id": "DISCOGS_MASTER_ID",
        "discogs_release_id": "DISCOGS_RELEASE_ID",
        "spotify_track_id": "SPOTIFY_TRACK_ID",
        "spotify_album_id": "SPOTIFY_ALBUM_ID",
        "wikidata_qid": "WIKIDATA_QID",
        "charts_blob": "CHARTS",
        "decision_trace": "TAG_DECISION_TRACE",
        "ruleset_version": "CANON_RULESET_VERSION",
        "evidence_hash": "CANON_EVIDENCE_HASH",
        "orig_title": "ORIG_TITLE",
        "orig_artist": "ORIG_ARTIST",
        "orig_album": "ORIG_ALBUM",
        "orig_date": "ORIG_DATE",
    }

    def read_tags(self, file_path: Path) -> dict[str, Any]:
        """Read MP4 tags from file."""
        from mutagen.mp4 import MP4

        try:
            audio = MP4(file_path)
        except Exception:
            return {}

        result: dict[str, Any] = {}

        # Read standard atoms
        for field_name, atom in self.ATOM_MAPPINGS.items():
            if audio.tags is not None and atom in audio.tags:  # pyright: ignore[reportOperatorIssue]
                values = audio.tags[atom]  # pyright: ignore[reportOptionalSubscript]
                if atom == "trkn" and values:
                    # Track number is a tuple (track, total)
                    result["track_number"] = values[0][0]
                    result["track_total"] = values[0][1] if len(values[0]) > 1 else None
                elif atom == "disk" and values:
                    # Disc number is a tuple (disc, total)
                    result["disc_number"] = values[0][0]
                    result["disc_total"] = values[0][1] if len(values[0]) > 1 else None
                elif values:
                    result[field_name] = values[0]

        # Read custom atoms
        for custom_field, atom_name in self.CUSTOM_ATOMS.items():
            full_atom = f"{self.CUSTOM_ATOM_PREFIX}{atom_name}"
            if audio.tags is not None and full_atom in audio.tags:  # pyright: ignore[reportOperatorIssue]
                values = audio.tags[full_atom]  # pyright: ignore[reportOptionalSubscript]
                if values:
                    # MP4FreeForm returns bytes
                    result[custom_field] = (
                        values[0].decode("utf-8")
                        if isinstance(values[0], bytes)
                        else str(values[0])
                    )

        return result

    def write_tags(
        self,
        file_path: Path,
        tagset: TagSet,
        authoritative: bool = False,
        dry_run: bool = False,
    ) -> WriteReport:
        """Write MP4 tags to file."""
        from mutagen.mp4 import MP4, MP4FreeForm

        report = WriteReport(file_path=file_path, dry_run=dry_run)

        try:
            audio = MP4(file_path)
        except Exception as e:
            report.errors.append(f"Could not open file: {e}")
            return report

        existing = self.read_tags(file_path)

        def write_custom(atom_name: str, value: str | None, field_name: str) -> None:
            if value is None:
                report.fields_skipped.append(field_name)
                return
            full_atom = f"{self.CUSTOM_ATOM_PREFIX}{atom_name}"
            if not dry_run and audio.tags is not None:
                audio.tags[full_atom] = [MP4FreeForm(value.encode("utf-8"))]  # pyright: ignore[reportOptionalSubscript]
            report.fields_written.append(field_name)

        # Stash originals if needed (first write only)
        if authoritative:
            if self._should_stash_original(existing, "title") and existing.get("title"):
                write_custom("ORIG_TITLE", str(existing["title"]), "orig_title")
                report.originals_stashed.append("title")
            if self._should_stash_original(existing, "artist") and existing.get("artist"):
                write_custom("ORIG_ARTIST", str(existing["artist"]), "orig_artist")
                report.originals_stashed.append("artist")
            if self._should_stash_original(existing, "album") and existing.get("album"):
                write_custom("ORIG_ALBUM", str(existing["album"]), "orig_album")
                report.originals_stashed.append("album")
            if self._should_stash_original(existing, "original_year") and existing.get(
                "original_year"
            ):
                write_custom("ORIG_DATE", str(existing["original_year"]), "orig_date")
                report.originals_stashed.append("original_year")

        # Write core fields (only if authoritative)
        if authoritative:
            if tagset.title is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["\xa9nam"] = [tagset.title]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("title")
            else:
                report.fields_skipped.append("title")

            if tagset.artist is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["\xa9ART"] = [tagset.artist]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("artist")
            else:
                report.fields_skipped.append("artist")

            if tagset.album is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["\xa9alb"] = [tagset.album]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("album")
            else:
                report.fields_skipped.append("album")

            if tagset.album_artist is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["aART"] = [tagset.album_artist]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("album_artist")
            else:
                report.fields_skipped.append("album_artist")

            if tagset.original_year is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["\xa9day"] = [tagset.original_year]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("original_year")
            else:
                report.fields_skipped.append("original_year")

            if tagset.track_number is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["trkn"] = [(tagset.track_number, tagset.track_total or 0)]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("track_number")
            else:
                report.fields_skipped.append("track_number")

            if tagset.disc_number is not None:
                if not dry_run and audio.tags is not None:
                    audio.tags["disk"] = [(tagset.disc_number, tagset.disc_total or 0)]  # pyright: ignore[reportOptionalSubscript]
                report.fields_written.append("disc_number")
            else:
                report.fields_skipped.append("disc_number")

            write_custom("LABEL", tagset.label, "label")
            write_custom("MEDIA", tagset.media_format, "media_format")
        else:
            report.fields_skipped.extend(
                [
                    "title",
                    "artist",
                    "album",
                    "album_artist",
                    "original_year",
                    "track_number",
                    "disc_number",
                    "label",
                    "media_format",
                ]
            )

        # Always write custom fields
        write_custom("COUNTRY", tagset.country, "country")
        write_custom(
            "CANONICAL_RELEASE_TYPE",
            tagset.release_type.value if tagset.release_type else None,
            "release_type",
        )

        # Canonical IDs
        write_custom("MB_RECORDING_ID", tagset.ids.mb_recording_id, "mb_recording_id")
        write_custom("MB_RELEASE_GROUP_ID", tagset.ids.mb_release_group_id, "mb_release_group_id")
        write_custom("MB_RELEASE_ID", tagset.ids.mb_release_id, "mb_release_id")
        write_custom("DISCOGS_MASTER_ID", tagset.ids.discogs_master_id, "discogs_master_id")
        write_custom("DISCOGS_RELEASE_ID", tagset.ids.discogs_release_id, "discogs_release_id")
        write_custom("SPOTIFY_TRACK_ID", tagset.ids.spotify_track_id, "spotify_track_id")
        write_custom("SPOTIFY_ALBUM_ID", tagset.ids.spotify_album_id, "spotify_album_id")
        write_custom("WIKIDATA_QID", tagset.ids.wikidata_qid, "wikidata_qid")

        # Compact fields
        write_custom("CHARTS", tagset.compact.charts_blob, "charts_blob")
        write_custom("TAG_DECISION_TRACE", tagset.compact.decision_trace, "decision_trace")
        write_custom("CANON_RULESET_VERSION", tagset.compact.ruleset_version, "ruleset_version")
        write_custom("CANON_EVIDENCE_HASH", tagset.compact.evidence_hash, "evidence_hash")

        # Save
        if not dry_run:
            audio.save()

        return report

    def verify(self, file_path: Path, tagset: TagSet) -> bool:
        """Verify written tags match tagset."""
        read_tags = self.read_tags(file_path)

        if tagset.title and read_tags.get("title") != tagset.title:
            return False
        if tagset.artist and read_tags.get("artist") != tagset.artist:
            return False
        if tagset.album and read_tags.get("album") != tagset.album:
            return False

        if tagset.ids.mb_recording_id:
            if read_tags.get("mb_recording_id") != tagset.ids.mb_recording_id:
                return False
        if tagset.ids.mb_release_group_id:
            if read_tags.get("mb_release_group_id") != tagset.ids.mb_release_group_id:
                return False

        if tagset.compact.charts_blob:
            if read_tags.get("charts_blob") != tagset.compact.charts_blob:
                return False

        return True


def get_writer_for_file(file_path: Path) -> TagWriter:
    """Get appropriate tag writer for file based on extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".mp3":
        return ID3TagWriter()
    elif suffix in (".flac", ".ogg"):
        return VorbisTagWriter()
    elif suffix in (".mp4", ".m4a"):
        return MP4TagWriter()
    else:
        raise ValueError(f"Unsupported audio format: {suffix}")


def assemble_tags(
    decision: Any,  # CanonicalDecision
    charts_blob: str | None = None,
    external_ids: dict[str, str] | None = None,
) -> TagSet:
    """
    Assemble canonical tagset from decision.

    Args:
        decision: CanonicalDecision from resolver. Expected to have:
            - release_group_mbid: str | None
            - release_mbid: str | None
            - decision_trace: DecisionTrace | None (with evidence_hash, ruleset_version)
            - compact_tag: str
        charts_blob: Optional CHARTS JSON blob
        external_ids: Optional dict of external IDs to include

    Returns:
        TagSet ready for writing
    """
    tagset = TagSet()

    # Set IDs from decision (with safe attribute access)
    tagset.ids.mb_release_group_id = getattr(decision, "release_group_mbid", None)
    tagset.ids.mb_release_id = getattr(decision, "release_mbid", None)

    # Set compact fields from decision trace
    decision_trace = getattr(decision, "decision_trace", None)
    if decision_trace:
        tagset.compact.decision_trace = getattr(decision, "compact_tag", None)
        tagset.compact.evidence_hash = getattr(decision_trace, "evidence_hash", None)
        tagset.compact.ruleset_version = getattr(decision_trace, "ruleset_version", None)

    # Set CHARTS blob if provided
    if charts_blob:
        tagset.compact.charts_blob = charts_blob

    # Set external IDs if provided
    if external_ids:
        if "mb_recording_id" in external_ids:
            tagset.ids.mb_recording_id = external_ids["mb_recording_id"]
        if "discogs_master_id" in external_ids:
            tagset.ids.discogs_master_id = external_ids["discogs_master_id"]
        if "discogs_release_id" in external_ids:
            tagset.ids.discogs_release_id = external_ids["discogs_release_id"]
        if "spotify_track_id" in external_ids:
            tagset.ids.spotify_track_id = external_ids["spotify_track_id"]
        if "spotify_album_id" in external_ids:
            tagset.ids.spotify_album_id = external_ids["spotify_album_id"]
        if "wikidata_qid" in external_ids:
            tagset.ids.wikidata_qid = external_ids["wikidata_qid"]

    return tagset


def write_tags(
    file_path: Path,
    tagset: TagSet,
    authoritative: bool = False,
    dry_run: bool = False,
) -> WriteReport:
    """
    Write tags to audio file.

    Args:
        file_path: Path to audio file
        tagset: TagSet to write
        authoritative: If True, overwrite core fields; else augment only
        dry_run: If True, don't write, just report what would be written

    Returns:
        WriteReport with details of what was written/skipped
    """
    writer = get_writer_for_file(file_path)
    return writer.write_tags(file_path, tagset, authoritative=authoritative, dry_run=dry_run)


def verify(file_path: Path) -> TagSet:
    """
    Read and verify tags from audio file.

    Returns TagSet populated from file tags.
    """
    writer = get_writer_for_file(file_path)
    tags = writer.read_tags(file_path)

    tagset = TagSet()

    # Core fields
    tagset.title = tags.get("title")
    tagset.artist = tags.get("artist")
    tagset.album = tags.get("album")
    tagset.album_artist = tags.get("album_artist")
    tagset.original_year = tags.get("original_year")
    tagset.track_number = tags.get("track_number")
    tagset.track_total = tags.get("track_total")
    tagset.disc_number = tags.get("disc_number")
    tagset.disc_total = tags.get("disc_total")
    tagset.label = tags.get("label")
    tagset.country = tags.get("country")
    tagset.media_format = tags.get("media_format")

    # Release type
    if tags.get("release_type"):
        try:
            tagset.release_type = ReleaseType(tags["release_type"])
        except ValueError:
            # Ignore invalid release_type values; leave tagset.release_type unset.
            pass

    # Canonical IDs
    tagset.ids.mb_recording_id = tags.get("mb_recording_id")
    tagset.ids.mb_release_group_id = tags.get("mb_release_group_id")
    tagset.ids.mb_release_id = tags.get("mb_release_id")
    tagset.ids.discogs_master_id = tags.get("discogs_master_id")
    tagset.ids.discogs_release_id = tags.get("discogs_release_id")
    tagset.ids.spotify_track_id = tags.get("spotify_track_id")
    tagset.ids.spotify_album_id = tags.get("spotify_album_id")
    tagset.ids.wikidata_qid = tags.get("wikidata_qid")

    # Compact fields
    tagset.compact.charts_blob = tags.get("charts_blob")
    tagset.compact.decision_trace = tags.get("decision_trace")
    tagset.compact.ruleset_version = tags.get("ruleset_version")
    tagset.compact.evidence_hash = tags.get("evidence_hash")

    # Stashed originals
    tagset.orig_title = tags.get("orig_title")
    tagset.orig_artist = tags.get("orig_artist")
    tagset.orig_album = tags.get("orig_album")
    tagset.orig_date = tags.get("orig_date")

    return tagset
