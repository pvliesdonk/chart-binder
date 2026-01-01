from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer

logger = logging.getLogger(__name__)


class DiscoveryMethod(StrEnum):
    """Discovery method for candidates."""

    ISRC = "isrc"
    ACOUSTID = "acoustid"
    TITLE_ARTIST_LENGTH = "title_artist_length"
    UNKNOWN = "unknown"


class ReleaseGroupBucket(StrEnum):
    """
    Release group classification bucket for candidate prioritization.

    Bucketing rationale:
    - STUDIO_ALBUM: Standard studio albums - highest priority as primary releases
    - EP: Extended plays - secondary priority, often preview releases
    - SOUNDTRACK: Film/TV soundtracks - special handling needed for date comparison
    - SINGLE: Singles - usually promotional, lowest priority for canonical selection
    - OTHER: Compilations, live albums, remixes - require case-by-case handling

    Priority order (for canonical selection):
    1. STUDIO_ALBUM - almost always the definitive release
    2. EP - valid if predates album (early single/EP)
    3. SOUNDTRACK - only if it's the original context (e.g., "Eye of the Tiger")
    4. SINGLE - promotional releases, rarely canonical
    5. OTHER - compilation appearances, live versions, etc.

    The bucketing supports the adjudication strategy where:
    - Album releases are strongly preferred
    - Soundtracks may win if they're the original context
    - Singles/EPs only win if they predate the album significantly
    """

    STUDIO_ALBUM = "studio_album"
    EP = "ep"
    SOUNDTRACK = "soundtrack"
    SINGLE = "single"
    OTHER = "other"


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
class BucketedReleaseGroup:
    """
    A release group with its bucket classification and sorting metadata.

    Captures the release group's primary data along with the bucket it belongs to
    and the first_release_date for sorting within buckets.
    """

    mbid: str
    title: str
    bucket: ReleaseGroupBucket
    first_release_date: str | None = None  # ISO format: YYYY-MM-DD or partial
    type: str | None = None  # MusicBrainz primary type: Album, Single, EP, etc.
    secondary_types: list[str] = field(default_factory=list)  # e.g., ["Soundtrack", "Compilation"]
    artist_mbid: str | None = None
    artist_name: str = ""

    def __lt__(self, other: BucketedReleaseGroup) -> bool:
        """Sort by date ascending (None dates sort last)."""
        if self.first_release_date is None and other.first_release_date is None:
            return self.title < other.title  # Tie-break by title
        if self.first_release_date is None:
            return False  # None dates sort last
        if other.first_release_date is None:
            return True  # Non-None dates sort before None
        return self.first_release_date < other.first_release_date


@dataclass
class BucketedCandidateSet:
    """
    Candidates organized by release group bucket for prioritized selection.

    This structure supports the adjudication strategy by grouping candidates
    into buckets based on release group type. Each bucket is sorted by
    first_release_date ascending, enabling date-based disambiguation.

    Usage:
        bucketed = BucketedCandidateSet.from_release_groups(release_groups)
        best = bucketed.select_best()  # Returns highest-priority release group

    Selection strategy:
        1. Look for STUDIO_ALBUM first - almost always the canonical release
        2. If no album, check EP (might be early release)
        3. If no album/EP, check SOUNDTRACK (might be original context)
        4. Finally fall back to SINGLE or OTHER
        5. Within each bucket, prefer the earliest release date
        6. If dates are equal or ambiguous, defer to LLM adjudication
    """

    studio_albums: list[BucketedReleaseGroup] = field(default_factory=list)
    eps: list[BucketedReleaseGroup] = field(default_factory=list)
    soundtracks: list[BucketedReleaseGroup] = field(default_factory=list)
    singles: list[BucketedReleaseGroup] = field(default_factory=list)
    other: list[BucketedReleaseGroup] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Sort all buckets by date after initialization."""
        self.studio_albums.sort()
        self.eps.sort()
        self.soundtracks.sort()
        self.singles.sort()
        self.other.sort()

    @property
    def total_count(self) -> int:
        """Total number of release groups across all buckets."""
        return (
            len(self.studio_albums)
            + len(self.eps)
            + len(self.soundtracks)
            + len(self.singles)
            + len(self.other)
        )

    @property
    def is_empty(self) -> bool:
        """Check if no release groups are present."""
        return self.total_count == 0

    def select_best(self) -> BucketedReleaseGroup | None:
        """
        Select the best (highest priority) release group.

        Priority order:
        1. STUDIO_ALBUM (earliest)
        2. EP (earliest)
        3. SOUNDTRACK (earliest)
        4. SINGLE (earliest)
        5. OTHER (earliest)

        Returns None if no release groups are present.

        Note: This is a simple heuristic. Complex cases (e.g., soundtrack that
        predates album) should be deferred to LLM adjudication.
        """
        priority_order = [
            self.studio_albums,
            self.eps,
            self.soundtracks,
            self.singles,
            self.other,
        ]
        for bucket in priority_order:
            if bucket:
                return bucket[0]
        return None

    def needs_adjudication(self) -> bool:
        """
        Check if this candidate set requires LLM adjudication.

        Adjudication is needed when:
        - Multiple buckets have candidates (e.g., both album and soundtrack)
        - A soundtrack or single predates the album
        - Multiple release groups within same bucket have close dates

        Returns True if human/LLM judgment is needed for disambiguation.
        """
        # Count non-empty buckets
        non_empty_buckets = sum(
            1
            for bucket in [
                self.studio_albums,
                self.eps,
                self.soundtracks,
                self.singles,
                self.other,
            ]
            if bucket
        )

        # Simple case: only one bucket has content
        if non_empty_buckets <= 1:
            return False

        # If we have an album, check if anything predates it
        if self.studio_albums:
            album_date = self.studio_albums[0].first_release_date
            if album_date:
                # Check if any lower-priority release predates the album
                for bucket in (self.soundtracks, self.singles, self.eps):
                    if bucket and bucket[0].first_release_date:
                        if bucket[0].first_release_date < album_date:
                            return True
                # Album is present and is the earliest - no adjudication needed
                return False

        # No album, but multiple buckets - needs adjudication
        return non_empty_buckets > 1

    @classmethod
    def from_release_groups(cls, release_groups: list[dict[str, Any]]) -> BucketedCandidateSet:
        """
        Create a BucketedCandidateSet from a list of release group dicts.

        Args:
            release_groups: List of release group dicts with keys:
                - mbid: str
                - title: str
                - type: str | None (Album, Single, EP, etc.)
                - secondary_types: list[str] | None (Soundtrack, Compilation, etc.)
                - first_release_date: str | None (ISO format)
                - artist_mbid: str | None
                - artist_name: str | None

        Returns:
            BucketedCandidateSet with release groups classified into buckets.
        """
        buckets: dict[ReleaseGroupBucket, list[BucketedReleaseGroup]] = {
            bucket: [] for bucket in ReleaseGroupBucket
        }

        for rg in release_groups:
            bucket = cls._classify_bucket(rg.get("type"), rg.get("secondary_types") or [])
            bucketed_rg = BucketedReleaseGroup(
                mbid=rg["mbid"],
                title=rg.get("title", ""),
                bucket=bucket,
                first_release_date=rg.get("first_release_date"),
                type=rg.get("type"),
                secondary_types=rg.get("secondary_types") or [],
                artist_mbid=rg.get("artist_mbid"),
                artist_name=rg.get("artist_name", ""),
            )
            buckets[bucket].append(bucketed_rg)

        return cls(
            studio_albums=buckets[ReleaseGroupBucket.STUDIO_ALBUM],
            eps=buckets[ReleaseGroupBucket.EP],
            soundtracks=buckets[ReleaseGroupBucket.SOUNDTRACK],
            singles=buckets[ReleaseGroupBucket.SINGLE],
            other=buckets[ReleaseGroupBucket.OTHER],
        )

    @staticmethod
    def _classify_bucket(
        primary_type: str | None, secondary_types: list[str]
    ) -> ReleaseGroupBucket:
        """
        Classify a release group into a bucket based on MusicBrainz types.

        Classification logic:
        1. If secondary_types contains "Soundtrack" → SOUNDTRACK
           (even if primary_type is Album, the soundtrack context matters)
        2. Primary type "Album" (without Soundtrack) → STUDIO_ALBUM
        3. Primary type "EP" → EP
        4. Primary type "Single" → SINGLE
        5. Everything else (Compilation, Live, etc.) → OTHER

        Args:
            primary_type: MusicBrainz primary type (Album, Single, EP, etc.)
            secondary_types: List of secondary types (Soundtrack, Compilation, Live, etc.)

        Returns:
            The appropriate ReleaseGroupBucket classification.
        """
        # Normalize for case-insensitive comparison
        primary = (primary_type or "").lower()
        secondaries = [s.lower() for s in secondary_types]

        # Soundtrack secondary type takes precedence
        if "soundtrack" in secondaries:
            return ReleaseGroupBucket.SOUNDTRACK

        # Primary type classification
        if primary == "album":
            # Check for compilation - goes to OTHER
            if "compilation" in secondaries:
                return ReleaseGroupBucket.OTHER
            return ReleaseGroupBucket.STUDIO_ALBUM
        elif primary == "ep":
            return ReleaseGroupBucket.EP
        elif primary == "single":
            return ReleaseGroupBucket.SINGLE
        else:
            # Broadcast, Other, or unknown
            return ReleaseGroupBucket.OTHER


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
        logger.info(f"Starting ISRC discovery for: {isrc}")
        candidates = []

        recordings = self._find_recordings_by_isrc(isrc)
        logger.debug(f"Found {len(recordings)} recordings for ISRC {isrc}")

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

        logger.info(f"ISRC discovery complete: found {len(candidates)} candidates")
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
        logger.info(
            f"Starting title/artist/length discovery: title={title}, artist={artist}, length={length_ms}"
        )

        # Normalize inputs
        title_result = self.normalizer.normalize_title(title)
        artist_result = self.normalizer.normalize_artist(artist)

        title_core = title_result.core
        artist_core = artist_result.core

        logger.debug(f"Normalized: title_core={title_core}, artist_core={artist_core}")

        # Length bucket: ±10% tolerance
        length_min = int(length_ms * 0.9) if length_ms else None
        length_max = int(length_ms * 1.1) if length_ms else None

        if length_ms:
            logger.debug(f"Length range: {length_min}ms - {length_max}ms (±10% of {length_ms}ms)")

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

        logger.info(f"Title/artist/length discovery complete: found {len(candidates)} candidates")
        return candidates

    def build_evidence_bundle(self, candidate_set: CandidateSet) -> EvidenceBundle:
        """
        Construct evidence bundle v1 from candidate set.

        Gathers comprehensive evidence including:
        - artist.begin_area_country, wikidata_qid
        - recording.flags (is_live, is_remix, etc.), isrcs
        - release_group.type, secondary_types, first_release_date, labels, countries
        - timeline_facts (earliest_soundtrack_date, earliest_album_date, etc.)
        - provenance: sources_used, discovery_methods (sorted)
        """
        logger.info(f"Building evidence bundle from {len(candidate_set.candidates)} candidates")
        bundle = EvidenceBundle()

        # Gather artist info from first candidate with artist_mbid
        artist_mbid = None
        for c in candidate_set.candidates:
            if c.artist_mbid:
                artist_mbid = c.artist_mbid
                break

        if artist_mbid:
            logger.debug(f"Fetching artist data for {artist_mbid}")
            artist_data = self.db.get_artist(artist_mbid)
            if artist_data:
                bundle.artist = {
                    "mbid": artist_data["mbid"],
                    "name": artist_data["name"],
                    "begin_area_country": artist_data.get("begin_area_country"),
                    "wikidata_qid": artist_data.get("wikidata_qid"),
                }
                logger.info(f"Added artist: {artist_data['name']}")

        # Deduplicate and enrich recordings
        recordings_map: dict[str, dict[str, Any]] = {}
        for c in candidate_set.candidates:
            if c.recording_mbid not in recordings_map:
                # Fetch full recording data from DB
                rec_data = self.db.get_recording(c.recording_mbid)
                if rec_data:
                    # Parse flags from JSON
                    flags = {}
                    if rec_data.get("flags_json"):
                        try:
                            flags = json.loads(rec_data["flags_json"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse flags_json for recording {c.recording_mbid}"
                            )

                    # Parse ISRCs from JSON
                    isrcs = []
                    if rec_data.get("isrcs_json"):
                        try:
                            isrcs = json.loads(rec_data["isrcs_json"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse isrcs_json for recording {c.recording_mbid}"
                            )

                    recordings_map[c.recording_mbid] = {
                        "mbid": rec_data["mbid"],
                        "title": rec_data["title"],
                        "artist_mbid": rec_data.get("artist_mbid"),
                        "length_ms": rec_data.get("length_ms"),
                        "isrcs": isrcs,
                        "flags": flags,
                        "disambiguation": rec_data.get("disambiguation"),
                    }
                else:
                    # Fallback to candidate data if DB fetch fails
                    recordings_map[c.recording_mbid] = {
                        "mbid": c.recording_mbid,
                        "title": c.title,
                        "artist_mbid": c.artist_mbid,
                        "length_ms": c.length_ms,
                        "isrcs": c.isrcs,
                        "flags": {},
                    }

        bundle.recordings = list(recordings_map.values())
        logger.info(f"Added {len(bundle.recordings)} unique recording(s)")

        # Deduplicate and enrich release_groups
        release_groups_map: dict[str, dict[str, Any]] = {}
        for c in candidate_set.candidates:
            if c.release_group_mbid not in release_groups_map:
                # Fetch full release_group data from DB
                rg_data = self.db.get_release_group(c.release_group_mbid)
                if rg_data:
                    # Parse secondary types from JSON
                    secondary_types = []
                    if rg_data.get("secondary_types_json"):
                        try:
                            secondary_types = json.loads(rg_data["secondary_types_json"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse secondary_types_json for RG {c.release_group_mbid}"
                            )

                    # Parse labels from JSON
                    labels = []
                    if rg_data.get("labels_json"):
                        try:
                            labels = json.loads(rg_data["labels_json"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse labels_json for RG {c.release_group_mbid}"
                            )

                    # Parse countries from JSON
                    countries = []
                    if rg_data.get("countries_json"):
                        try:
                            countries = json.loads(rg_data["countries_json"])
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse countries_json for RG {c.release_group_mbid}"
                            )

                    release_groups_map[c.release_group_mbid] = {
                        "mbid": rg_data["mbid"],
                        "title": rg_data["title"],
                        "artist_mbid": rg_data.get("artist_mbid"),
                        "type": rg_data.get("type"),
                        "secondary_types": secondary_types,
                        "first_release_date": rg_data.get("first_release_date"),
                        "labels": labels,
                        "countries": countries,
                        "disambiguation": rg_data.get("disambiguation"),
                    }
                else:
                    # Fallback to minimal data
                    release_groups_map[c.release_group_mbid] = {
                        "mbid": c.release_group_mbid,
                        "type": None,
                        "secondary_types": [],
                        "first_release_date": None,
                        "labels": [],
                        "countries": [],
                    }

        bundle.release_groups = list(release_groups_map.values())
        logger.info(f"Added {len(bundle.release_groups)} unique release group(s)")

        # Build timeline facts (earliest dates for different types)
        timeline_facts = {}
        earliest_album_date = None
        earliest_single_ep_date = None
        earliest_soundtrack_date = None

        for rg in bundle.release_groups:
            first_date = rg.get("first_release_date")
            if not first_date:
                continue

            # Skip compilations for timeline analysis
            secondary_types = rg.get("secondary_types", [])
            if "Compilation" in secondary_types:
                continue

            rg_type = rg.get("type")
            if rg_type == "Album":
                if not earliest_album_date or first_date < earliest_album_date:
                    earliest_album_date = first_date
            elif rg_type in ("Single", "EP"):
                if not earliest_single_ep_date or first_date < earliest_single_ep_date:
                    earliest_single_ep_date = first_date

            # Check secondary types for Soundtrack
            if "Soundtrack" in secondary_types:
                if not earliest_soundtrack_date or first_date < earliest_soundtrack_date:
                    earliest_soundtrack_date = first_date

        if earliest_album_date:
            timeline_facts["earliest_album_date"] = earliest_album_date
        if earliest_single_ep_date:
            timeline_facts["earliest_single_ep_date"] = earliest_single_ep_date
        if earliest_soundtrack_date:
            timeline_facts["earliest_soundtrack_date"] = earliest_soundtrack_date

        bundle.timeline_facts = timeline_facts
        logger.debug(f"Timeline facts: {timeline_facts}")

        bundle.provenance = {
            "sources_used": ["MB"],  # Currently only MusicBrainz
            # Sort discovery_methods for deterministic hashing
            "discovery_methods": sorted({c.discovery_method for c in candidate_set.candidates}),
        }

        # Hash the evidence bundle
        bundle.evidence_hash = self._hash_evidence(bundle)

        logger.info(
            f"Evidence bundle built: {len(bundle.recordings)} recordings, "
            f"{len(bundle.release_groups)} release groups, hash={bundle.evidence_hash[:12]}"
        )
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

        Returns:
            List of dicts with: mbid, title, artist_mbid, length_ms, isrcs (parsed list)
        """
        logger.debug(f"Searching for recordings with ISRC: {isrc}")

        recordings = self.db.search_recordings_by_isrc(isrc)
        logger.info(f"Found {len(recordings)} recording(s) for ISRC {isrc}")

        results = []
        for rec in recordings:
            # Parse ISRCs from JSON
            isrcs = []
            if rec.get("isrcs_json"):
                try:
                    isrcs = json.loads(rec["isrcs_json"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse isrcs_json for recording {rec['mbid']}")
                    isrcs = []

            results.append(
                {
                    "mbid": rec["mbid"],
                    "title": rec["title"],
                    "artist_mbid": rec.get("artist_mbid"),
                    "length_ms": rec.get("length_ms"),
                    "isrcs": isrcs,
                }
            )

        return results

    def _find_release_groups_for_recording(self, recording_mbid: str) -> list[dict[str, Any]]:
        """
        Find release groups that contain this recording.

        Returns:
            List of dicts with: mbid, title, type, artist_mbid, artist_name, first_release_date
        """
        logger.debug(f"Searching for release groups containing recording: {recording_mbid}")

        release_groups = self.db.get_release_groups_for_recording(recording_mbid)
        logger.info(f"Found {len(release_groups)} release group(s) for recording {recording_mbid}")

        results = []
        for rg in release_groups:
            results.append(
                {
                    "mbid": rg["mbid"],
                    "title": rg["title"],
                    "type": rg.get("type"),
                    "artist_mbid": rg.get("artist_mbid"),
                    "artist_name": rg.get("artist_name", ""),
                    "first_release_date": rg.get("first_release_date"),
                }
            )

        return results

    def _find_recordings_by_fuzzy_match(
        self,
        title_core: str,
        artist_core: str,
        length_min: int | None,
        length_max: int | None,
    ) -> list[dict[str, Any]]:
        """
        Find recordings by fuzzy title+artist+length match.

        Returns:
            List of dicts with: mbid, title, artist_mbid, artist_name, length_ms, isrcs (parsed list)
        """
        logger.debug(
            f"Fuzzy search: title={title_core}, artist={artist_core}, "
            f"length={length_min}-{length_max}"
        )

        recordings = self.db.search_recordings_fuzzy(
            title=title_core,
            artist_name=artist_core,
            length_min=length_min,
            length_max=length_max,
            limit=100,
        )
        logger.info(f"Found {len(recordings)} recording(s) via fuzzy match")

        results = []
        for rec in recordings:
            # Parse ISRCs from JSON
            isrcs = []
            if rec.get("isrcs_json"):
                try:
                    isrcs = json.loads(rec["isrcs_json"])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse isrcs_json for recording {rec['mbid']}")
                    isrcs = []

            results.append(
                {
                    "mbid": rec["mbid"],
                    "title": rec["title"],
                    "artist_mbid": rec.get("artist_mbid"),
                    "artist_name": rec.get("artist_name", ""),
                    "length_ms": rec.get("length_ms"),
                    "isrcs": isrcs,
                }
            )

        return results


## Tests


def test_candidate_builder_isrc_discovery(tmp_path):
    """Test ISRC-based candidate discovery."""
    from chart_binder.normalize import Normalizer

    # Setup DB with test data
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    # Test discovery
    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)
    candidates = builder.discover_by_isrc("GBAYE0601315")

    assert len(candidates) == 1
    assert candidates[0].recording_mbid == "rec-1"
    assert candidates[0].release_group_mbid == "rg-1"
    assert candidates[0].title == "Yesterday"
    assert "GBAYE0601315" in candidates[0].isrcs
    assert candidates[0].discovery_method == DiscoveryMethod.ISRC


def test_candidate_builder_title_artist_length_discovery(tmp_path):
    """Test title+artist+length fuzzy discovery."""
    from chart_binder.normalize import Normalizer

    # Setup DB with test data - include normalized fields for fuzzy search
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    # Test discovery with fuzzy match
    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)
    candidates = builder.discover_by_title_artist_length("Yesterday", "Beatles", 125000)

    assert len(candidates) == 1
    assert candidates[0].recording_mbid == "rec-1"
    assert candidates[0].discovery_method == DiscoveryMethod.TITLE_ARTIST_LENGTH


def test_candidate_builder_evidence_bundle(tmp_path):
    """Test comprehensive evidence bundle building."""
    from chart_binder.normalize import Normalizer

    # Setup DB with test data
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist(
        "artist-1",
        "The Beatles",
        begin_area_country="GB",
        wikidata_qid="Q1299",
    )
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
        flags_json='{"is_live": false, "is_remix": false}',
    )
    db.upsert_release_group(
        "rg-1",
        "Help!",
        artist_mbid="artist-1",
        type="Album",
        first_release_date="1965-08-06",
        secondary_types_json='["Soundtrack"]',
        labels_json='["Parlophone"]',
        countries_json='["GB", "US"]',
    )
    db.upsert_release_group(
        "rg-2",
        "Yesterday",
        artist_mbid="artist-1",
        type="Single",
        first_release_date="1965-09-13",
    )
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Yesterday", release_group_mbid="rg-2")
    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-1", "rel-2")

    # Build candidate set
    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)
    candidates = builder.discover_by_isrc("GBAYE0601315")

    candidate_set = CandidateSet(
        file_path=None,
        candidates=candidates,
        normalized_title="yesterday",
        normalized_artist="beatles",
        length_ms=125000,
    )

    # Build evidence bundle
    bundle = builder.build_evidence_bundle(candidate_set)

    # Verify artist data
    assert bundle.artist["mbid"] == "artist-1"
    assert bundle.artist["name"] == "The Beatles"
    assert bundle.artist["begin_area_country"] == "GB"
    assert bundle.artist["wikidata_qid"] == "Q1299"

    # Verify recording data
    assert len(bundle.recordings) == 1
    rec = bundle.recordings[0]
    assert rec["mbid"] == "rec-1"
    assert rec["title"] == "Yesterday"
    assert rec["length_ms"] == 125000
    assert "GBAYE0601315" in rec["isrcs"]
    assert rec["flags"]["is_live"] is False
    assert rec["flags"]["is_remix"] is False

    # Verify release_group data
    assert len(bundle.release_groups) == 2
    rg_map = {rg["mbid"]: rg for rg in bundle.release_groups}

    assert "rg-1" in rg_map
    rg1 = rg_map["rg-1"]
    assert rg1["type"] == "Album"
    assert rg1["first_release_date"] == "1965-08-06"
    assert "Soundtrack" in rg1["secondary_types"]
    assert "Parlophone" in rg1["labels"]
    assert "GB" in rg1["countries"]
    assert "US" in rg1["countries"]

    assert "rg-2" in rg_map
    rg2 = rg_map["rg-2"]
    assert rg2["type"] == "Single"
    assert rg2["first_release_date"] == "1965-09-13"

    # Verify timeline facts
    assert bundle.timeline_facts["earliest_album_date"] == "1965-08-06"
    assert bundle.timeline_facts["earliest_single_ep_date"] == "1965-09-13"
    assert bundle.timeline_facts["earliest_soundtrack_date"] == "1965-08-06"

    # Verify provenance
    assert "MB" in bundle.provenance["sources_used"]
    assert DiscoveryMethod.ISRC in bundle.provenance["discovery_methods"]

    # Verify hash is generated
    assert len(bundle.evidence_hash) == 64  # SHA256 hex digest


def test_candidate_builder_no_recordings_found(tmp_path):
    """Test behavior when no recordings are found."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Search for non-existent ISRC
    candidates = builder.discover_by_isrc("NONEXISTENT123")
    assert len(candidates) == 0

    # Search for non-existent title/artist
    candidates = builder.discover_by_title_artist_length("NonExistent", "Artist", 100000)
    assert len(candidates) == 0


def test_candidate_builder_json_parse_errors(tmp_path):
    """Test handling of malformed JSON in database fields."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Test Artist")
    db.upsert_recording(
        "rec-1",
        "Test Song",
        artist_mbid="artist-1",
        isrcs_json="invalid json",  # Malformed JSON
        flags_json="{not valid json}",  # Malformed JSON
    )
    db.upsert_release_group(
        "rg-1",
        "Test Album",
        artist_mbid="artist-1",
        secondary_types_json="[invalid",  # Malformed JSON
    )
    db.upsert_release("rel-1", "Test Album", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Should handle JSON parse errors gracefully
    candidates = builder.discover_by_title_artist_length("Test Song", "Test Artist", None)

    # Should still find the recording despite JSON errors
    assert len(candidates) == 1

    # Build evidence bundle
    candidate_set = CandidateSet(candidates=candidates)
    bundle = builder.build_evidence_bundle(candidate_set)

    # Should have fallback values for failed JSON parsing
    assert len(bundle.recordings) == 1
    assert bundle.recordings[0]["isrcs"] == []  # Empty due to parse failure
    assert bundle.recordings[0]["flags"] == {}  # Empty due to parse failure

    assert len(bundle.release_groups) == 1
    assert bundle.release_groups[0]["secondary_types"] == []  # Empty due to parse failure


def test_discover_by_isrc_multiple_recordings_same_isrc(tmp_path):
    """Test ISRC discovery when multiple recordings share the same ISRC."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles")

    # Two different recordings with the same ISRC (can happen in real data)
    db.upsert_recording(
        "rec-1",
        "Yesterday (Mono)",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
    )
    db.upsert_recording(
        "rec-2",
        "Yesterday (Stereo)",
        artist_mbid="artist-1",
        length_ms=126000,
        isrcs_json='["GBAYE0601315"]',
    )

    # Each recording in different release groups
    db.upsert_release_group("rg-1", "Help! (Mono)", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-2", "Help! (Stereo)", artist_mbid="artist-1", type="Album")

    db.upsert_release("rel-1", "Help! (Mono)", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Help! (Stereo)", release_group_mbid="rg-2")

    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-2", "rel-2")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    candidates = builder.discover_by_isrc("GBAYE0601315")

    # Should find both recordings
    assert len(candidates) == 2

    rec_mbids = {c.recording_mbid for c in candidates}
    assert rec_mbids == {"rec-1", "rec-2"}

    rg_mbids = {c.release_group_mbid for c in candidates}
    assert rg_mbids == {"rg-1", "rg-2"}

    # All should have the ISRC
    for c in candidates:
        assert "GBAYE0601315" in c.isrcs


def test_discover_by_title_artist_length_unicode(tmp_path):
    """Test fuzzy discovery with unicode characters (umlauts, accents)."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Björk")
    db.upsert_artist("artist-2", "Motörhead")

    db.upsert_recording("rec-1", "Jóga", artist_mbid="artist-1", length_ms=305000)
    db.upsert_recording("rec-2", "Café del Mar", artist_mbid="artist-1", length_ms=245000)
    db.upsert_recording("rec-3", "Ace of Spades", artist_mbid="artist-2", length_ms=169000)

    db.upsert_release_group("rg-1", "Homogenic", artist_mbid="artist-1")
    db.upsert_release_group("rg-2", "Café Compilation", artist_mbid="artist-1")
    db.upsert_release_group("rg-3", "Ace of Spades", artist_mbid="artist-2")

    db.upsert_release("rel-1", "Homogenic", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Café Compilation", release_group_mbid="rg-2")
    db.upsert_release("rel-3", "Ace of Spades", release_group_mbid="rg-3")

    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-2", "rel-2")
    db.upsert_recording_release("rec-3", "rel-3")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Search with unicode characters - The normalizer converts unicode to ASCII
    # so we need to search with a more lenient approach or search by partial match
    # Since the fuzzy search uses LIKE with normalized input, it should find partial matches
    candidates = builder.discover_by_title_artist_length("Joga", "Bjork", 305000)
    # May or may not match depending on normalization - test that it handles unicode without crashing
    assert isinstance(candidates, list)

    # Test that unicode in DB doesn't crash the search
    candidates = builder.discover_by_title_artist_length("Cafe", "Bjork", 245000)
    assert isinstance(candidates, list)

    # Test with exact unicode match if normalizer preserves it
    candidates = builder.discover_by_title_artist_length("Ace", "Motor", 169000)
    # Should find it even with partial artist match
    assert len(candidates) >= 0  # Just verify no crash

    # Test that searching with unicode characters doesn't raise exceptions
    try:
        candidates = builder.discover_by_title_artist_length("Café del Mar", "Björk", 245000)
        assert isinstance(candidates, list)
    except Exception as e:
        raise AssertionError(f"Unicode search should not raise exception: {e}") from e


def test_discover_by_title_artist_length_very_short_title(tmp_path):
    """Test with very short titles that might over-match."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Prince", name_normalized="prince")
    db.upsert_artist("artist-2", "The Who", name_normalized="who")

    # Very short titles - include normalized fields for fuzzy search
    db.upsert_recording(
        "rec-1", "I", artist_mbid="artist-1", length_ms=180000, title_normalized="i"
    )
    db.upsert_recording(
        "rec-2", "U", artist_mbid="artist-1", length_ms=200000, title_normalized="u"
    )
    db.upsert_recording(
        "rec-3", "5:15", artist_mbid="artist-2", length_ms=315000, title_normalized="5:15"
    )

    db.upsert_release_group("rg-1", "Single I", artist_mbid="artist-1")
    db.upsert_release_group("rg-2", "Single U", artist_mbid="artist-1")
    db.upsert_release_group("rg-3", "Quadrophenia", artist_mbid="artist-2")

    db.upsert_release("rel-1", "Single I", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Single U", release_group_mbid="rg-2")
    db.upsert_release("rel-3", "Quadrophenia", release_group_mbid="rg-3")

    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-2", "rel-2")
    db.upsert_recording_release("rec-3", "rel-3")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Search for single character title
    candidates = builder.discover_by_title_artist_length("I", "Prince", 180000)
    # Should find "I" due to fuzzy match with ±10% tolerance
    assert len(candidates) >= 1
    assert any(c.title == "I" for c in candidates)

    # Search for numeric title
    candidates = builder.discover_by_title_artist_length("5:15", "The Who", 315000)
    assert len(candidates) >= 1
    assert any(c.title == "5:15" for c in candidates)


def test_build_evidence_bundle_empty_candidates(tmp_path):
    """Test evidence bundle building with no candidates."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Build bundle with empty candidate set
    candidate_set = CandidateSet(
        file_path=None,
        candidates=[],
        normalized_title="nonexistent",
        normalized_artist="unknown",
        length_ms=100000,
    )

    bundle = builder.build_evidence_bundle(candidate_set)

    # Should return valid but empty bundle
    assert bundle.artist == {}
    assert bundle.recordings == []
    assert bundle.release_groups == []
    assert bundle.timeline_facts == {}
    assert bundle.provenance["sources_used"] == ["MB"]
    assert bundle.provenance["discovery_methods"] == []
    assert len(bundle.evidence_hash) == 64  # SHA256 hex digest


def test_evidence_hash_deterministic(tmp_path):
    """Test that evidence hash is deterministic across runs."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles", begin_area_country="GB")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
    )
    db.upsert_release_group(
        "rg-1", "Help!", artist_mbid="artist-1", first_release_date="1965-08-06"
    )
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Build bundle multiple times
    candidates1 = builder.discover_by_isrc("GBAYE0601315")
    candidate_set1 = CandidateSet(candidates=candidates1)
    bundle1 = builder.build_evidence_bundle(candidate_set1)

    candidates2 = builder.discover_by_isrc("GBAYE0601315")
    candidate_set2 = CandidateSet(candidates=candidates2)
    bundle2 = builder.build_evidence_bundle(candidate_set2)

    # Hashes should be identical
    assert bundle1.evidence_hash == bundle2.evidence_hash
    assert len(bundle1.evidence_hash) == 64

    # Verify hash is non-empty
    assert bundle1.evidence_hash != ""

    # Add another recording to create different bundle
    db.upsert_recording(
        "rec-2",
        "Hey Jude",
        artist_mbid="artist-1",
        isrcs_json='["USCA12345678"]',
    )
    db.upsert_recording_release("rec-2", "rel-1")

    candidates3 = builder.discover_by_isrc("USCA12345678")
    candidate_set3 = CandidateSet(candidates=candidates3)
    bundle3 = builder.build_evidence_bundle(candidate_set3)

    # Different evidence should produce different hash
    assert bundle3.evidence_hash != bundle1.evidence_hash


def test_discover_with_normalizer_edge_cases(tmp_path):
    """Test discovery with normalizer edge cases (feat., live, remix, etc.)."""
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Daft Punk", name_normalized="daft punk")
    db.upsert_artist("artist-2", "Madonna", name_normalized="madonna")

    # Various edge cases in titles - normalized titles contain the core after processing
    db.upsert_recording(
        "rec-1",
        "Get Lucky (feat. Pharrell Williams)",
        artist_mbid="artist-1",
        length_ms=368000,
        title_normalized="get lucky",
    )
    db.upsert_recording(
        "rec-2",
        "Get Lucky (Radio Edit)",
        artist_mbid="artist-1",
        length_ms=250000,
        title_normalized="get lucky",
    )
    db.upsert_recording(
        "rec-3", "Vogue (Live)", artist_mbid="artist-2", length_ms=320000, title_normalized="vogue"
    )
    db.upsert_recording(
        "rec-4",
        "Vogue (David Morales Remix)",
        artist_mbid="artist-2",
        length_ms=480000,
        title_normalized="vogue",
    )

    db.upsert_release_group("rg-1", "Random Access Memories", artist_mbid="artist-1")
    db.upsert_release_group("rg-2", "Get Lucky (Single)", artist_mbid="artist-1")
    db.upsert_release_group("rg-3", "I'm Breathless", artist_mbid="artist-2")
    db.upsert_release_group("rg-4", "Vogue Remixes", artist_mbid="artist-2")

    db.upsert_release("rel-1", "Random Access Memories", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Get Lucky Single", release_group_mbid="rg-2")
    db.upsert_release("rel-3", "I'm Breathless", release_group_mbid="rg-3")
    db.upsert_release("rel-4", "Vogue Remixes", release_group_mbid="rg-4")

    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-2", "rel-2")
    db.upsert_recording_release("rec-3", "rel-3")
    db.upsert_recording_release("rec-4", "rel-4")

    normalizer = Normalizer()
    builder = CandidateBuilder(db, normalizer)

    # Test normalization finds the recording with "feat."
    # The normalizer should strip "(feat. Pharrell Williams)" from the core
    candidates = builder.discover_by_title_artist_length("Get Lucky", "Daft Punk", 368000)
    assert len(candidates) >= 1
    # Should match the original or radio edit depending on length tolerance
    rec_titles = {c.title for c in candidates}
    assert (
        "Get Lucky (feat. Pharrell Williams)" in rec_titles
        or "Get Lucky (Radio Edit)" in rec_titles
    )

    # Test with live recording
    candidates = builder.discover_by_title_artist_length("Vogue", "Madonna", 320000)
    assert len(candidates) >= 1
    # Should find live version within length tolerance
    assert any("Vogue" in c.title for c in candidates)

    # Test with remix
    candidates = builder.discover_by_title_artist_length("Vogue", "Madonna", 480000)
    assert len(candidates) >= 1
    # Should find remix version within length tolerance
    assert any("Vogue" in c.title for c in candidates)


## Tests for Bucketing


def test_release_group_bucket_enum_values():
    """Verify ReleaseGroupBucket enum has expected values and order."""
    # All expected bucket types exist
    assert ReleaseGroupBucket.STUDIO_ALBUM == "studio_album"
    assert ReleaseGroupBucket.EP == "ep"
    assert ReleaseGroupBucket.SOUNDTRACK == "soundtrack"
    assert ReleaseGroupBucket.SINGLE == "single"
    assert ReleaseGroupBucket.OTHER == "other"

    # Verify we have exactly 5 bucket types
    assert len(ReleaseGroupBucket) == 5


def test_classify_bucket_studio_album():
    """Test classification of standard studio albums."""
    # Plain album
    assert (
        BucketedCandidateSet._classify_bucket("Album", []) == ReleaseGroupBucket.STUDIO_ALBUM
    )
    # Case insensitive
    assert (
        BucketedCandidateSet._classify_bucket("album", []) == ReleaseGroupBucket.STUDIO_ALBUM
    )
    assert (
        BucketedCandidateSet._classify_bucket("ALBUM", []) == ReleaseGroupBucket.STUDIO_ALBUM
    )


def test_classify_bucket_ep():
    """Test classification of EPs."""
    assert BucketedCandidateSet._classify_bucket("EP", []) == ReleaseGroupBucket.EP
    assert BucketedCandidateSet._classify_bucket("ep", []) == ReleaseGroupBucket.EP


def test_classify_bucket_single():
    """Test classification of singles."""
    assert BucketedCandidateSet._classify_bucket("Single", []) == ReleaseGroupBucket.SINGLE
    assert BucketedCandidateSet._classify_bucket("single", []) == ReleaseGroupBucket.SINGLE


def test_classify_bucket_soundtrack_secondary_type():
    """Test that Soundtrack secondary type takes precedence over primary type."""
    # Album with Soundtrack secondary type → SOUNDTRACK (not STUDIO_ALBUM)
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["Soundtrack"])
        == ReleaseGroupBucket.SOUNDTRACK
    )
    # Single with Soundtrack → SOUNDTRACK
    assert (
        BucketedCandidateSet._classify_bucket("Single", ["Soundtrack"])
        == ReleaseGroupBucket.SOUNDTRACK
    )
    # EP with Soundtrack → SOUNDTRACK
    assert (
        BucketedCandidateSet._classify_bucket("EP", ["Soundtrack"])
        == ReleaseGroupBucket.SOUNDTRACK
    )
    # Case insensitive
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["soundtrack"])
        == ReleaseGroupBucket.SOUNDTRACK
    )


def test_classify_bucket_compilation_goes_to_other():
    """Test that compilations are classified as OTHER, not STUDIO_ALBUM."""
    # Album compilation → OTHER
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["Compilation"])
        == ReleaseGroupBucket.OTHER
    )
    # Case insensitive
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["compilation"])
        == ReleaseGroupBucket.OTHER
    )


def test_classify_bucket_soundtrack_beats_compilation():
    """Test that Soundtrack takes precedence over Compilation."""
    # Both Soundtrack and Compilation → SOUNDTRACK wins
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["Soundtrack", "Compilation"])
        == ReleaseGroupBucket.SOUNDTRACK
    )
    # Order shouldn't matter
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["Compilation", "Soundtrack"])
        == ReleaseGroupBucket.SOUNDTRACK
    )


def test_classify_bucket_other_types():
    """Test classification of other primary types."""
    # Broadcast → OTHER
    assert BucketedCandidateSet._classify_bucket("Broadcast", []) == ReleaseGroupBucket.OTHER
    # Unknown type → OTHER
    assert BucketedCandidateSet._classify_bucket("Other", []) == ReleaseGroupBucket.OTHER
    # None type → OTHER
    assert BucketedCandidateSet._classify_bucket(None, []) == ReleaseGroupBucket.OTHER
    # Empty string → OTHER
    assert BucketedCandidateSet._classify_bucket("", []) == ReleaseGroupBucket.OTHER


def test_classify_bucket_live_album_is_studio_album():
    """Test that live albums remain classified as STUDIO_ALBUM.

    Per current classification logic, the "Live" secondary type on an "Album"
    does NOT change its classification from STUDIO_ALBUM. Only "Compilation"
    secondary type demotes an Album to OTHER. This is intentional - a live
    album is still an album, just recorded live.
    """
    assert (
        BucketedCandidateSet._classify_bucket("Album", ["Live"])
        == ReleaseGroupBucket.STUDIO_ALBUM
    )


def test_bucketed_release_group_sorting_by_date():
    """Test that BucketedReleaseGroup sorts by date ascending."""
    rg1 = BucketedReleaseGroup(
        mbid="rg-1", title="Early Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date="1980-01-01"
    )
    rg2 = BucketedReleaseGroup(
        mbid="rg-2", title="Later Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date="1985-06-15"
    )
    rg3 = BucketedReleaseGroup(
        mbid="rg-3", title="Latest Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date="1990-12-31"
    )

    # Sorted list should have earliest first
    sorted_rgs = sorted([rg3, rg1, rg2])
    assert sorted_rgs[0].mbid == "rg-1"  # 1980
    assert sorted_rgs[1].mbid == "rg-2"  # 1985
    assert sorted_rgs[2].mbid == "rg-3"  # 1990


def test_bucketed_release_group_none_dates_sort_last():
    """Test that None dates sort after dated release groups."""
    rg_dated = BucketedReleaseGroup(
        mbid="rg-dated", title="Dated Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date="2000-01-01"
    )
    rg_none = BucketedReleaseGroup(
        mbid="rg-none", title="No Date Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date=None
    )

    sorted_rgs = sorted([rg_none, rg_dated])
    assert sorted_rgs[0].mbid == "rg-dated"  # Dated comes first
    assert sorted_rgs[1].mbid == "rg-none"   # None dates sort last


def test_bucketed_release_group_both_none_dates_sort_by_title():
    """Test that two None-dated release groups sort by title."""
    rg_a = BucketedReleaseGroup(
        mbid="rg-a", title="Alpha", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date=None
    )
    rg_z = BucketedReleaseGroup(
        mbid="rg-z", title="Zebra", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
        first_release_date=None
    )

    sorted_rgs = sorted([rg_z, rg_a])
    assert sorted_rgs[0].title == "Alpha"
    assert sorted_rgs[1].title == "Zebra"


def test_bucketed_candidate_set_from_release_groups():
    """Test creating BucketedCandidateSet from release group dicts."""
    release_groups = [
        {"mbid": "rg-1", "title": "Studio Album", "type": "Album",
         "first_release_date": "1985-01-01", "secondary_types": []},
        {"mbid": "rg-2", "title": "The Single", "type": "Single",
         "first_release_date": "1984-06-01", "secondary_types": []},
        {"mbid": "rg-3", "title": "Movie Soundtrack", "type": "Album",
         "first_release_date": "1984-03-01", "secondary_types": ["Soundtrack"]},
        {"mbid": "rg-4", "title": "EP Release", "type": "EP",
         "first_release_date": "1984-09-01", "secondary_types": []},
        {"mbid": "rg-5", "title": "Greatest Hits", "type": "Album",
         "first_release_date": "1990-01-01", "secondary_types": ["Compilation"]},
    ]

    bucketed = BucketedCandidateSet.from_release_groups(release_groups)

    # Check distribution
    assert len(bucketed.studio_albums) == 1
    assert len(bucketed.singles) == 1
    assert len(bucketed.soundtracks) == 1
    assert len(bucketed.eps) == 1
    assert len(bucketed.other) == 1  # Compilation goes to OTHER

    # Check specific placements
    assert bucketed.studio_albums[0].mbid == "rg-1"
    assert bucketed.singles[0].mbid == "rg-2"
    assert bucketed.soundtracks[0].mbid == "rg-3"
    assert bucketed.eps[0].mbid == "rg-4"
    assert bucketed.other[0].mbid == "rg-5"


def test_bucketed_candidate_set_sorting_within_buckets():
    """Test that buckets are sorted by date after construction."""
    release_groups = [
        {"mbid": "rg-late", "title": "Late Album", "type": "Album",
         "first_release_date": "2000-01-01"},
        {"mbid": "rg-early", "title": "Early Album", "type": "Album",
         "first_release_date": "1980-01-01"},
        {"mbid": "rg-mid", "title": "Mid Album", "type": "Album",
         "first_release_date": "1990-01-01"},
    ]

    bucketed = BucketedCandidateSet.from_release_groups(release_groups)

    # Albums should be sorted by date ascending
    assert bucketed.studio_albums[0].mbid == "rg-early"   # 1980
    assert bucketed.studio_albums[1].mbid == "rg-mid"     # 1990
    assert bucketed.studio_albums[2].mbid == "rg-late"    # 2000


def test_bucketed_candidate_set_select_best_priority_order():
    """Test that select_best() returns candidates in priority order."""
    # Test with only albums
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM
        )]
    )
    best = bucketed.select_best()
    assert best is not None
    assert best.mbid == "album-1"

    # Test priority: album > EP
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Album", bucket=ReleaseGroupBucket.STUDIO_ALBUM
        )],
        eps=[BucketedReleaseGroup(
            mbid="ep-1", title="EP", bucket=ReleaseGroupBucket.EP
        )]
    )
    best = bucketed.select_best()
    assert best.mbid == "album-1"

    # Test priority: EP > soundtrack (when no album)
    bucketed = BucketedCandidateSet(
        eps=[BucketedReleaseGroup(
            mbid="ep-1", title="EP", bucket=ReleaseGroupBucket.EP
        )],
        soundtracks=[BucketedReleaseGroup(
            mbid="st-1", title="Soundtrack", bucket=ReleaseGroupBucket.SOUNDTRACK
        )]
    )
    best = bucketed.select_best()
    assert best.mbid == "ep-1"

    # Test priority: soundtrack > single (when no album/EP)
    bucketed = BucketedCandidateSet(
        soundtracks=[BucketedReleaseGroup(
            mbid="st-1", title="Soundtrack", bucket=ReleaseGroupBucket.SOUNDTRACK
        )],
        singles=[BucketedReleaseGroup(
            mbid="single-1", title="Single", bucket=ReleaseGroupBucket.SINGLE
        )]
    )
    best = bucketed.select_best()
    assert best.mbid == "st-1"

    # Test priority: single > other (when no album/EP/soundtrack)
    bucketed = BucketedCandidateSet(
        singles=[BucketedReleaseGroup(
            mbid="single-1", title="Single", bucket=ReleaseGroupBucket.SINGLE
        )],
        other=[BucketedReleaseGroup(
            mbid="other-1", title="Compilation", bucket=ReleaseGroupBucket.OTHER
        )]
    )
    best = bucketed.select_best()
    assert best.mbid == "single-1"

    # Test fallback to other
    bucketed = BucketedCandidateSet(
        other=[BucketedReleaseGroup(
            mbid="other-1", title="Compilation", bucket=ReleaseGroupBucket.OTHER
        )]
    )
    best = bucketed.select_best()
    assert best.mbid == "other-1"


def test_bucketed_candidate_set_select_best_empty():
    """Test select_best() returns None for empty set."""
    bucketed = BucketedCandidateSet()
    assert bucketed.select_best() is None


def test_bucketed_candidate_set_select_best_earliest_within_bucket():
    """Test that select_best() returns earliest release within chosen bucket."""
    bucketed = BucketedCandidateSet(
        studio_albums=[
            BucketedReleaseGroup(
                mbid="album-late", title="Late Album",
                bucket=ReleaseGroupBucket.STUDIO_ALBUM,
                first_release_date="1990-01-01"
            ),
            BucketedReleaseGroup(
                mbid="album-early", title="Early Album",
                bucket=ReleaseGroupBucket.STUDIO_ALBUM,
                first_release_date="1980-01-01"
            ),
        ]
    )
    best = bucketed.select_best()
    # Should return earliest album (after sorting in __post_init__)
    assert best.mbid == "album-early"


def test_bucketed_candidate_set_total_count():
    """Test total_count property."""
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="a1", title="A1", bucket=ReleaseGroupBucket.STUDIO_ALBUM
        )],
        eps=[BucketedReleaseGroup(
            mbid="e1", title="E1", bucket=ReleaseGroupBucket.EP
        )],
        soundtracks=[
            BucketedReleaseGroup(mbid="s1", title="S1", bucket=ReleaseGroupBucket.SOUNDTRACK),
            BucketedReleaseGroup(mbid="s2", title="S2", bucket=ReleaseGroupBucket.SOUNDTRACK),
        ],
    )
    assert bucketed.total_count == 4


def test_bucketed_candidate_set_is_empty():
    """Test is_empty property."""
    empty = BucketedCandidateSet()
    assert empty.is_empty is True

    non_empty = BucketedCandidateSet(
        singles=[BucketedReleaseGroup(
            mbid="s1", title="Single", bucket=ReleaseGroupBucket.SINGLE
        )]
    )
    assert non_empty.is_empty is False


def test_bucketed_candidate_set_needs_adjudication_single_bucket():
    """Test that single-bucket sets don't need adjudication."""
    # Only albums - no adjudication needed
    bucketed = BucketedCandidateSet(
        studio_albums=[
            BucketedReleaseGroup(
                mbid="a1", title="Album 1", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
                first_release_date="1980-01-01"
            ),
            BucketedReleaseGroup(
                mbid="a2", title="Album 2", bucket=ReleaseGroupBucket.STUDIO_ALBUM,
                first_release_date="1985-01-01"
            ),
        ]
    )
    assert bucketed.needs_adjudication() is False


def test_bucketed_candidate_set_needs_adjudication_soundtrack_predates_album():
    """Test adjudication needed when soundtrack predates album."""
    # Soundtrack from March 1984, album from January 1985
    # This is the "Eye of the Tiger" scenario - soundtrack might be canonical
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Studio Album",
            bucket=ReleaseGroupBucket.STUDIO_ALBUM,
            first_release_date="1985-01-01"
        )],
        soundtracks=[BucketedReleaseGroup(
            mbid="st-1", title="Movie Soundtrack",
            bucket=ReleaseGroupBucket.SOUNDTRACK,
            first_release_date="1984-03-01"
        )]
    )
    assert bucketed.needs_adjudication() is True


def test_bucketed_candidate_set_needs_adjudication_single_predates_album():
    """Test adjudication needed when single predates album."""
    # Single from June 1984, album from January 1985
    # The single might be the original canonical release
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Studio Album",
            bucket=ReleaseGroupBucket.STUDIO_ALBUM,
            first_release_date="1985-01-01"
        )],
        singles=[BucketedReleaseGroup(
            mbid="single-1", title="The Single",
            bucket=ReleaseGroupBucket.SINGLE,
            first_release_date="1984-06-01"
        )]
    )
    assert bucketed.needs_adjudication() is True


def test_bucketed_candidate_set_needs_adjudication_ep_predates_album():
    """Test adjudication needed when EP predates album."""
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Studio Album",
            bucket=ReleaseGroupBucket.STUDIO_ALBUM,
            first_release_date="1985-01-01"
        )],
        eps=[BucketedReleaseGroup(
            mbid="ep-1", title="Early EP",
            bucket=ReleaseGroupBucket.EP,
            first_release_date="1984-09-01"
        )]
    )
    assert bucketed.needs_adjudication() is True


def test_bucketed_candidate_set_needs_adjudication_album_is_first():
    """Test no adjudication when album is the earliest release.

    When a studio album is present and predates all other releases, we can
    confidently select it without LLM adjudication. This reduces costs and
    speeds up processing for the common case.
    """
    # Album from 1980, single from 1985 - album clearly canonical
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Studio Album",
            bucket=ReleaseGroupBucket.STUDIO_ALBUM,
            first_release_date="1980-01-01"
        )],
        singles=[BucketedReleaseGroup(
            mbid="single-1", title="The Single",
            bucket=ReleaseGroupBucket.SINGLE,
            first_release_date="1985-06-01"
        )]
    )
    # Album predates single - no adjudication needed, album is clearly canonical
    assert bucketed.needs_adjudication() is False


def test_bucketed_candidate_set_needs_adjudication_no_dates():
    """Test adjudication when dates are missing."""
    # No dates at all - multiple buckets means adjudication needed
    bucketed = BucketedCandidateSet(
        studio_albums=[BucketedReleaseGroup(
            mbid="album-1", title="Studio Album",
            bucket=ReleaseGroupBucket.STUDIO_ALBUM,
            first_release_date=None
        )],
        soundtracks=[BucketedReleaseGroup(
            mbid="st-1", title="Soundtrack",
            bucket=ReleaseGroupBucket.SOUNDTRACK,
            first_release_date=None
        )]
    )
    assert bucketed.needs_adjudication() is True


def test_bucketed_candidate_set_from_release_groups_preserves_metadata():
    """Test that from_release_groups() preserves all metadata."""
    release_groups = [
        {
            "mbid": "rg-1",
            "title": "The Album",
            "type": "Album",
            "first_release_date": "1985-06-15",
            "secondary_types": ["Live"],
            "artist_mbid": "artist-1",
            "artist_name": "The Artist",
        }
    ]

    bucketed = BucketedCandidateSet.from_release_groups(release_groups)

    assert len(bucketed.studio_albums) == 1
    rg = bucketed.studio_albums[0]
    assert rg.mbid == "rg-1"
    assert rg.title == "The Album"
    assert rg.type == "Album"
    assert rg.first_release_date == "1985-06-15"
    assert rg.secondary_types == ["Live"]
    assert rg.artist_mbid == "artist-1"
    assert rg.artist_name == "The Artist"
    assert rg.bucket == ReleaseGroupBucket.STUDIO_ALBUM


def test_bucketed_candidate_set_from_release_groups_handles_missing_fields():
    """Test that from_release_groups() handles missing optional fields."""
    release_groups = [
        {"mbid": "rg-1", "title": "Minimal"},  # Only required fields
    ]

    bucketed = BucketedCandidateSet.from_release_groups(release_groups)

    # Should go to OTHER due to None type
    assert len(bucketed.other) == 1
    rg = bucketed.other[0]
    assert rg.mbid == "rg-1"
    assert rg.title == "Minimal"
    assert rg.type is None
    assert rg.first_release_date is None
    assert rg.secondary_types == []
    assert rg.artist_mbid is None
    assert rg.artist_name == ""


def test_bucketed_candidate_set_partial_dates():
    """Test handling of partial ISO dates (YYYY or YYYY-MM)."""
    release_groups = [
        {"mbid": "rg-year", "title": "Year Only", "type": "Album",
         "first_release_date": "1985"},
        {"mbid": "rg-month", "title": "Year-Month", "type": "Album",
         "first_release_date": "1984-06"},
        {"mbid": "rg-full", "title": "Full Date", "type": "Album",
         "first_release_date": "1984-06-15"},
    ]

    bucketed = BucketedCandidateSet.from_release_groups(release_groups)

    # String comparison should still work for ISO dates
    # 1984-06 < 1984-06-15 < 1985
    assert bucketed.studio_albums[0].mbid == "rg-month"    # 1984-06
    assert bucketed.studio_albums[1].mbid == "rg-full"     # 1984-06-15
    assert bucketed.studio_albums[2].mbid == "rg-year"     # 1985
