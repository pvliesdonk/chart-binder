from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer


class DiscoveryMethod(StrEnum):
    """Discovery method for candidates."""

    ISRC = "isrc"
    ACOUSTID = "acoustid"
    TITLE_ARTIST_LENGTH = "title_artist_length"
    UNKNOWN = "unknown"


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
    discovery_method: DiscoveryMethod = DiscoveryMethod.UNKNOWN


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
        """
        Discover candidates by ISRC.

        Note: Currently returns empty list until DB query methods are implemented.
        See TODO markers in _find_recordings_by_isrc and _find_release_groups_for_recording.
        """
        candidates = []

        recordings = self._find_recordings_by_isrc(isrc)

        for rec in recordings:
            release_groups = self._find_release_groups_for_recording(rec["mbid"])

            # Get full ISRC list from record, ensure queried ISRC is included
            record_isrcs = rec.get("isrcs", [])
            if isrc not in record_isrcs:
                record_isrcs = [isrc, *record_isrcs]

            for rg in release_groups:
                candidates.append(
                    Candidate(
                        recording_mbid=rec["mbid"],
                        release_group_mbid=rg["mbid"],
                        artist_mbid=rec.get("artist_mbid"),
                        title=rec["title"],
                        artist_name=rg.get("artist_name", ""),
                        length_ms=rec.get("length_ms"),
                        isrcs=record_isrcs,
                        discovery_method=DiscoveryMethod.ISRC,
                    )
                )

        return candidates

    def discover_by_title_artist_length(
        self, title: str, artist: str, length_ms: int | None = None
    ) -> list[Candidate]:
        """
        Discover candidates by normalized title+artist+length bucket.

        Length bucket uses ±10% tolerance for fuzzy matching.

        Note: Currently returns empty list until DB query methods are implemented.
        See TODO markers in _find_recordings_by_fuzzy_match and _find_release_groups_for_recording.
        """
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
                        isrcs=rec.get("isrcs", []),
                        discovery_method=DiscoveryMethod.TITLE_ARTIST_LENGTH,
                    )
                )

        return candidates

    def build_evidence_bundle(self, candidate_set: CandidateSet) -> EvidenceBundle:
        """
        Construct evidence bundle v1 from candidate set.

        TODO: Gather full evidence for decision rules
        Currently includes minimal fields:
        - recordings: mbid, title, length_ms (deduplicated)
        - release_groups: mbid only (deduplicated)
        - provenance: sources_used, discovery_methods (sorted)

        Full implementation should include:
        - artist.begin_area_country, wikidata_country
        - recording.flags (is_live, is_remix, etc.)
        - release_group.primary_type, secondary_types, first_release_date, labels, countries
        - releases within each RG with flags (is_official, is_promo, etc.)
        - timeline_facts (earliest_soundtrack_date, earliest_album_date, etc.)
        """
        bundle = EvidenceBundle()

        # TODO: Expand to full evidence bundle fields per spec
        # Deduplicate recordings by mbid
        recordings_map: dict[str, dict[str, Any]] = {}
        for c in candidate_set.candidates:
            if c.recording_mbid not in recordings_map:
                recordings_map[c.recording_mbid] = {
                    "mbid": c.recording_mbid,
                    "title": c.title,
                    "length_ms": c.length_ms,
                }
        bundle.recordings = list(recordings_map.values())

        # Deduplicate release_groups by mbid
        release_groups_map: dict[str, dict[str, Any]] = {}
        for c in candidate_set.candidates:
            if c.release_group_mbid not in release_groups_map:
                release_groups_map[c.release_group_mbid] = {"mbid": c.release_group_mbid}
        bundle.release_groups = list(release_groups_map.values())

        bundle.provenance = {
            "sources_used": ["MB"],  # TODO: Track actual sources used (MB, Discogs, Spotify)
            # Sort discovery_methods for deterministic hashing
            "discovery_methods": sorted({c.discovery_method for c in candidate_set.candidates}),
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
        """
        Find recordings by ISRC in musicgraph DB.

        TODO: Implement ISRC lookup query
        Expected query:
            SELECT * FROM recording
            WHERE json_extract(isrcs_json, '$') LIKE '%' || ? || '%'

        Returns empty list until implemented.
        """
        # TODO: Implement DB query for ISRC lookup
        return []

    def _find_release_groups_for_recording(self, recording_mbid: str) -> list[dict[str, Any]]:
        """
        Find release groups that contain this recording.

        TODO: Implement release_group lookup via recording_release join
        Expected query:
            SELECT DISTINCT rg.* FROM release_group rg
            JOIN release r ON r.release_group_mbid = rg.mbid
            JOIN recording_release rr ON rr.release_mbid = r.mbid
            WHERE rr.recording_mbid = ?

        Returns empty list until implemented.
        """
        # TODO: Implement DB join query for release_group discovery
        return []

    def _find_recordings_by_fuzzy_match(
        self,
        title_core: str,
        artist_core: str,
        length_min: int | None,
        length_max: int | None,
    ) -> list[dict[str, Any]]:
        """
        Find recordings by fuzzy title+artist+length match.

        TODO: Implement fuzzy matching query
        Options:
        1. Pre-normalized title_core/artist_core columns (requires schema change)
        2. Full-scan normalization (slow but works with current schema)
        3. Work key index (optimal, requires schema change)

        For now, consider option 2 with LIMIT for initial implementation.

        Returns empty list until implemented.
        """
        # TODO: Implement fuzzy matching query (consider performance implications)
        return []
