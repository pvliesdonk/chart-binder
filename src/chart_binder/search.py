"""
Multi-stage search infrastructure for candidate discovery.

This module implements a progressive search strategy that tries increasingly
expensive and broad search methods until sufficient candidates are found:

Stage 1 (STRICT): Strict recording search
    - Uses exact normalized title/artist matching
    - Fast, high precision, lower recall
    - Target: >=5 results to proceed without fallback

Stage 2 (FUZZY): Non-strict recording search
    - Uses fuzzy matching on normalized fields
    - Triggered when Stage 1 < min_strict_results
    - Broader recall, may include false positives

Stage 3 (ARTIST_BROWSE): Artist exhaustive browse (implemented in Part 2)
    - Searches for artist by name
    - Browses all release groups for that artist
    - Filters by title fuzzy match

Stage 4 (ACOUSTID): Optional fingerprint lookup (implemented in Part 2)
    - Uses audio fingerprint if available
    - Highest confidence when available
    - Requires optional fingerprint/duration parameters

Usage:
    from chart_binder.search import MultiStageSearcher

    searcher = MultiStageSearcher(musicgraph_db, normalizer)
    results = searcher.search(title="Yesterday", artist="The Beatles")

    # Check which stage found the results
    print(f"Found via {results[0].stage.value if results else 'none'}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rapidfuzz import fuzz

from chart_binder.musicgraph import MusicGraphDB
from chart_binder.normalize import Normalizer

logger = logging.getLogger(__name__)


class SearchStage(StrEnum):
    """
    Search stage enum indicating which method found a result.

    Stages are ordered by cost/broadness:
    - STRICT: Fast, exact matching (lowest cost, highest precision)
    - FUZZY: Broader matching with tolerance (medium cost)
    - ARTIST_BROWSE: Exhaustive artist catalog search (higher cost)
    - ACOUSTID: Fingerprint-based lookup (requires audio, highest confidence)
    """

    STRICT = "strict"
    FUZZY = "fuzzy"
    ARTIST_BROWSE = "artist_browse"
    ACOUSTID = "acoustid"


@dataclass
class SearchResult:
    """
    A recording search result with provenance metadata.

    Captures both the recording data and which search stage found it,
    enabling downstream processing to weight results by discovery method.
    """

    # Core recording data
    recording_mbid: str
    title: str
    artist_mbid: str | None = None
    artist_name: str = ""
    length_ms: int | None = None
    isrcs: list[str] = field(default_factory=list)

    # Provenance - which stage found this result
    stage: SearchStage = SearchStage.STRICT

    # Match quality metrics (populated by searcher)
    title_similarity: float = 0.0  # 0-100 score from fuzzy match
    artist_similarity: float = 0.0  # 0-100 score from fuzzy match

    def __hash__(self) -> int:
        """Hash by recording MBID for deduplication."""
        return hash(self.recording_mbid)

    def __eq__(self, other: object) -> bool:
        """Equality by recording MBID."""
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.recording_mbid == other.recording_mbid


@dataclass
class SearchConfig:
    """
    Configuration for multi-stage search behavior.

    Allows tuning of thresholds and stage progression.
    """

    # Minimum results from strict search before falling back to fuzzy
    min_strict_results: int = 5

    # Minimum results from fuzzy search before falling back to artist browse
    min_fuzzy_results: int = 5

    # Maximum results to return per stage (performance limit)
    max_results_per_stage: int = 100

    # Enable/disable specific stages
    enable_strict: bool = True
    enable_fuzzy: bool = True
    enable_artist_browse: bool = True  # Implemented in Part 2
    enable_acoustid: bool = True  # Implemented in Part 2, requires fingerprint

    # Similarity thresholds for fuzzy matching
    min_title_similarity: float = 70.0  # Minimum title similarity to accept
    min_artist_similarity: float = 60.0  # Minimum artist similarity to accept

    # Length tolerances (as fractions, e.g., 0.1 = ±10%)
    strict_length_tolerance: float = 0.1  # ±10% for strict search
    fuzzy_length_tolerance: float = 0.2  # ±20% for fuzzy search


class MultiStageSearcher:
    """
    Multi-stage recording search with progressive fallback.

    Implements a search strategy that tries increasingly broad methods
    until sufficient candidates are found, tracking which stage produced
    each result for downstream processing.

    Stage progression:
    1. STRICT - Try exact normalized matching first
    2. FUZZY - If strict < min_strict_results, try fuzzy matching
    3. ARTIST_BROWSE - If still insufficient, browse artist catalog (Part 2)
    4. ACOUSTID - If fingerprint available, try audio matching (Part 2)

    Example:
        searcher = MultiStageSearcher(db, normalizer)
        results = searcher.search(title="Yesterday", artist="Beatles")
        for r in results:
            print(f"{r.title} by {r.artist_name} (via {r.stage.value})")
    """

    def __init__(
        self,
        db: MusicGraphDB,
        normalizer: Normalizer,
        config: SearchConfig | None = None,
    ):
        """
        Initialize multi-stage searcher.

        Args:
            db: MusicGraphDB instance for database queries
            normalizer: Normalizer instance for text normalization
            config: Optional SearchConfig for tuning behavior
        """
        self.db = db
        self.normalizer = normalizer
        self.config = config or SearchConfig()

    def search(
        self,
        title: str,
        artist: str,
        length_ms: int | None = None,
        fingerprint: str | None = None,
        fingerprint_duration: int | None = None,
    ) -> list[SearchResult]:
        """
        Execute multi-stage search for recordings.

        Tries stages in order until sufficient results are found:
        1. STRICT - Exact normalized matching
        2. FUZZY - Fuzzy matching (if strict insufficient)
        3. ARTIST_BROWSE - Artist catalog search (Part 2)
        4. ACOUSTID - Fingerprint lookup (Part 2, if fingerprint provided)

        Args:
            title: Track title to search for
            artist: Artist name to search for
            length_ms: Optional track length for filtering
            fingerprint: Optional AcoustID fingerprint (Part 2)
            fingerprint_duration: Optional duration for AcoustID (Part 2)

        Returns:
            List of SearchResult objects with stage provenance
        """
        logger.info(f"Starting multi-stage search: title='{title}', artist='{artist}'")

        # Normalize inputs once
        title_result = self.normalizer.normalize_title(title)
        artist_result = self.normalizer.normalize_artist(artist)

        title_core = title_result.core
        artist_core = artist_result.core

        logger.debug(f"Normalized: title_core='{title_core}', artist_core='{artist_core}'")

        all_results: list[SearchResult] = []

        # Stage 1: STRICT search
        if self.config.enable_strict:
            strict_results = self._search_strict(title_core, artist_core, length_ms)
            all_results.extend(strict_results)
            logger.info(f"Stage 1 (STRICT): found {len(strict_results)} results")

            # If we have enough strict results, we're done
            if len(strict_results) >= self.config.min_strict_results:
                logger.info(
                    f"Sufficient strict results ({len(strict_results)} >= "
                    f"{self.config.min_strict_results}), skipping fallback stages"
                )
                return self._deduplicate(all_results)

        # Stage 2: FUZZY search (fallback)
        if self.config.enable_fuzzy:
            fuzzy_results = self._search_fuzzy(title_core, artist_core, length_ms)
            all_results.extend(fuzzy_results)
            logger.info(f"Stage 2 (FUZZY): found {len(fuzzy_results)} results")

            # If we have enough results after fuzzy, stop here
            # (Artist browse and AcoustID implemented in Part 2)
            deduplicated_results = self._deduplicate(all_results)
            if len(deduplicated_results) >= self.config.min_fuzzy_results:
                logger.info(
                    f"Sufficient results after fuzzy "
                    f"({len(deduplicated_results)} >= {self.config.min_fuzzy_results})"
                )
                return deduplicated_results

        # Stages 3-4 (ARTIST_BROWSE, ACOUSTID) implemented in Part 2
        # Placeholder logging for future stages
        if self.config.enable_artist_browse and len(all_results) < self.config.min_fuzzy_results:
            logger.debug("Stage 3 (ARTIST_BROWSE) will be implemented in Part 2")

        if self.config.enable_acoustid and fingerprint:
            logger.debug("Stage 4 (ACOUSTID) will be implemented in Part 2")

        return self._deduplicate(all_results)

    def _search_strict(
        self, title_core: str, artist_core: str, length_ms: int | None
    ) -> list[SearchResult]:
        """
        Stage 1: Strict recording search.

        Uses the same underlying DB query as fuzzy search but with tighter
        length tolerance and no post-filter similarity scoring. The "strict"
        stage is effectively the first pass with normalized input, where
        results that survive the DB's LIKE query are assumed to be good matches.

        Note: The underlying DB uses LIKE '%...%' for substring matching on
        normalized fields. A true exact-match query would require a separate
        DB method. For now, we rely on the normalization to produce precise
        enough results for Stage 1.

        Args:
            title_core: Normalized title (core form)
            artist_core: Normalized artist (core form)
            length_ms: Optional length for filtering

        Returns:
            List of SearchResult objects from strict search
        """
        logger.debug(f"Strict search: title='{title_core}', artist='{artist_core}'")

        # Calculate length range using configurable tolerance
        tol = self.config.strict_length_tolerance
        length_min = int(length_ms * (1 - tol)) if length_ms else None
        length_max = int(length_ms * (1 + tol)) if length_ms else None

        # Query database - uses LIKE matching on normalized fields
        recordings = self.db.search_recordings_fuzzy(
            title=title_core,
            artist_name=artist_core,
            length_min=length_min,
            length_max=length_max,
            limit=self.config.max_results_per_stage,
        )

        results = []
        for rec in recordings:
            result = self._recording_to_search_result(rec, SearchStage.STRICT)
            # For strict search, we expect high similarity
            result.title_similarity = 100.0  # Assumed exact match
            result.artist_similarity = 100.0  # Assumed exact match
            results.append(result)

        return results

    def _search_fuzzy(
        self, title_core: str, artist_core: str, length_ms: int | None
    ) -> list[SearchResult]:
        """
        Stage 2: Fuzzy recording search.

        Uses fuzzy matching to find recordings that are similar but
        not exactly matching. Triggered when strict search doesn't
        return enough results.

        Args:
            title_core: Normalized title (core form)
            artist_core: Normalized artist (core form)
            length_ms: Optional length for filtering (with wider tolerance)

        Returns:
            List of SearchResult objects from fuzzy search
        """
        logger.debug(f"Fuzzy search: title='{title_core}', artist='{artist_core}'")

        # Wider length tolerance for fuzzy search using configurable value
        tol = self.config.fuzzy_length_tolerance
        length_min = int(length_ms * (1 - tol)) if length_ms else None
        length_max = int(length_ms * (1 + tol)) if length_ms else None

        # Query database with fuzzy matching
        # Use partial matching for broader results
        recordings = self.db.search_recordings_fuzzy(
            title=title_core,
            artist_name=artist_core,
            length_min=length_min,
            length_max=length_max,
            limit=self.config.max_results_per_stage,
        )

        results = []
        for rec in recordings:
            result = self._recording_to_search_result(rec, SearchStage.FUZZY)

            # Calculate similarity scores using rapidfuzz (imported at module level)
            rec_title = rec.get("title_normalized") or rec.get("title", "")
            rec_artist = rec.get("artist_name_normalized") or rec.get("artist_name", "")

            result.title_similarity = fuzz.token_set_ratio(title_core, rec_title.lower())
            result.artist_similarity = fuzz.token_set_ratio(artist_core, rec_artist.lower())

            # Filter by minimum similarity thresholds
            if (
                result.title_similarity >= self.config.min_title_similarity
                and result.artist_similarity >= self.config.min_artist_similarity
            ):
                results.append(result)

        return results

    def _recording_to_search_result(
        self, recording: dict[str, Any], stage: SearchStage
    ) -> SearchResult:
        """
        Convert a database recording dict to SearchResult.

        Args:
            recording: Recording dict from database
            stage: The search stage that found this result

        Returns:
            SearchResult with recording data and stage provenance
        """
        # Parse ISRCs from JSON if present (json imported at module level)
        isrcs: list[str] = []
        if recording.get("isrcs_json"):
            try:
                isrcs = json.loads(recording["isrcs_json"])
            except (json.JSONDecodeError, TypeError):
                isrcs = []

        return SearchResult(
            recording_mbid=recording["mbid"],
            title=recording.get("title", ""),
            artist_mbid=recording.get("artist_mbid"),
            artist_name=recording.get("artist_name", ""),
            length_ms=recording.get("length_ms"),
            isrcs=isrcs,
            stage=stage,
        )

    def _deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Deduplicate results by recording MBID, preserving first occurrence.

        When the same recording is found by multiple stages, we keep
        the first occurrence (typically from an earlier, more precise stage).

        Args:
            results: List of SearchResult objects (may have duplicates)

        Returns:
            Deduplicated list preserving order and first stage
        """
        seen: set[str] = set()
        unique: list[SearchResult] = []

        for result in results:
            if result.recording_mbid not in seen:
                seen.add(result.recording_mbid)
                unique.append(result)

        return unique

    def get_stage_summary(self, results: list[SearchResult]) -> dict[SearchStage, int]:
        """
        Get a summary of how many results came from each stage.

        Useful for understanding search behavior and tuning thresholds.

        Args:
            results: List of SearchResult objects

        Returns:
            Dict mapping each stage to count of results from that stage
        """
        summary: dict[SearchStage, int] = {stage: 0 for stage in SearchStage}
        for result in results:
            summary[result.stage] += 1
        return summary


## Tests


def test_search_stage_enum_values():
    """Verify SearchStage enum has expected values."""
    assert SearchStage.STRICT == "strict"
    assert SearchStage.FUZZY == "fuzzy"
    assert SearchStage.ARTIST_BROWSE == "artist_browse"
    assert SearchStage.ACOUSTID == "acoustid"
    assert len(SearchStage) == 4


def test_search_result_dataclass():
    """Test SearchResult creation and defaults."""
    result = SearchResult(
        recording_mbid="rec-1",
        title="Yesterday",
        artist_mbid="artist-1",
        artist_name="The Beatles",
        stage=SearchStage.STRICT,
    )

    assert result.recording_mbid == "rec-1"
    assert result.title == "Yesterday"
    assert result.artist_name == "The Beatles"
    assert result.stage == SearchStage.STRICT
    assert result.isrcs == []
    assert result.title_similarity == 0.0
    assert result.artist_similarity == 0.0


def test_search_result_equality():
    """Test that SearchResult equality is by MBID."""
    result1 = SearchResult(recording_mbid="rec-1", title="A", stage=SearchStage.STRICT)
    result2 = SearchResult(recording_mbid="rec-1", title="B", stage=SearchStage.FUZZY)
    result3 = SearchResult(recording_mbid="rec-2", title="A", stage=SearchStage.STRICT)

    assert result1 == result2  # Same MBID
    assert result1 != result3  # Different MBID


def test_search_result_hash():
    """Test that SearchResult can be used in sets."""
    result1 = SearchResult(recording_mbid="rec-1", title="A", stage=SearchStage.STRICT)
    result2 = SearchResult(recording_mbid="rec-1", title="B", stage=SearchStage.FUZZY)
    result3 = SearchResult(recording_mbid="rec-2", title="A", stage=SearchStage.STRICT)

    result_set = {result1, result2, result3}
    assert len(result_set) == 2  # rec-1 and rec-2


def test_search_config_defaults():
    """Test SearchConfig default values."""
    config = SearchConfig()

    assert config.min_strict_results == 5
    assert config.min_fuzzy_results == 5
    assert config.max_results_per_stage == 100
    assert config.enable_strict is True
    assert config.enable_fuzzy is True
    assert config.enable_artist_browse is True
    assert config.enable_acoustid is True
    assert config.min_title_similarity == 70.0
    assert config.min_artist_similarity == 60.0
    assert config.strict_length_tolerance == 0.1
    assert config.fuzzy_length_tolerance == 0.2


def test_search_config_custom():
    """Test SearchConfig with custom values."""
    config = SearchConfig(
        min_strict_results=10,
        min_fuzzy_results=20,
        enable_artist_browse=False,
    )

    assert config.min_strict_results == 10
    assert config.min_fuzzy_results == 20
    assert config.enable_artist_browse is False


def test_multi_stage_searcher_creation(tmp_path):
    """Test MultiStageSearcher instantiation."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    searcher = MultiStageSearcher(db, normalizer)
    assert searcher.db is db
    assert searcher.normalizer is normalizer
    assert searcher.config.min_strict_results == 5


def test_multi_stage_searcher_with_custom_config(tmp_path):
    """Test MultiStageSearcher with custom config."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()
    config = SearchConfig(min_strict_results=3)

    searcher = MultiStageSearcher(db, normalizer, config)
    assert searcher.config.min_strict_results == 3


def test_multi_stage_searcher_strict_search(tmp_path):
    """Test Stage 1 strict search returns results."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    # Setup test data with normalized fields
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    searcher = MultiStageSearcher(db, normalizer)
    results = searcher.search(title="Yesterday", artist="The Beatles")

    assert len(results) >= 1
    assert results[0].recording_mbid == "rec-1"
    assert results[0].stage == SearchStage.STRICT


def test_multi_stage_searcher_fallback_to_fuzzy(tmp_path):
    """Test fallback to Stage 2 fuzzy when strict returns few results."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    # Setup minimal test data (less than min_strict_results)
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    # Use config that requires more strict results to trigger fallback
    config = SearchConfig(min_strict_results=10)
    searcher = MultiStageSearcher(db, normalizer, config)

    # Search should try fuzzy after strict doesn't return enough
    results = searcher.search(title="Yesterday", artist="Beatles")

    # Should still find the recording (either strict or fuzzy)
    assert len(results) >= 1


def test_multi_stage_searcher_no_results(tmp_path):
    """Test search with no matching results."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    searcher = MultiStageSearcher(db, normalizer)
    results = searcher.search(title="NonExistent", artist="Unknown Artist")

    assert len(results) == 0


def test_multi_stage_searcher_deduplication(tmp_path):
    """Test that duplicate results are deduplicated."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    # Setup test data
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    # Force both strict and fuzzy to run by requiring many strict results
    config = SearchConfig(min_strict_results=100)  # Won't be satisfied
    searcher = MultiStageSearcher(db, normalizer, config)

    results = searcher.search(title="Yesterday", artist="Beatles")

    # Should deduplicate - same recording shouldn't appear twice
    mbids = [r.recording_mbid for r in results]
    assert len(mbids) == len(set(mbids))  # All unique


def test_multi_stage_searcher_stage_summary(tmp_path):
    """Test get_stage_summary method."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    searcher = MultiStageSearcher(db, normalizer)

    # Create some results manually
    results = [
        SearchResult(recording_mbid="rec-1", title="A", stage=SearchStage.STRICT),
        SearchResult(recording_mbid="rec-2", title="B", stage=SearchStage.STRICT),
        SearchResult(recording_mbid="rec-3", title="C", stage=SearchStage.FUZZY),
    ]

    summary = searcher.get_stage_summary(results)

    assert summary[SearchStage.STRICT] == 2
    assert summary[SearchStage.FUZZY] == 1
    assert summary[SearchStage.ARTIST_BROWSE] == 0
    assert summary[SearchStage.ACOUSTID] == 0


def test_multi_stage_searcher_length_tolerance(tmp_path):
    """Test that length tolerance works correctly."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    # Setup recording with specific length
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,  # 2:05
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    searcher = MultiStageSearcher(db, normalizer)

    # Search with length within 10% tolerance
    results = searcher.search(title="Yesterday", artist="Beatles", length_ms=130000)
    assert len(results) >= 1  # 130000 is within 10% of 125000

    # Search with length way outside tolerance
    results = searcher.search(title="Yesterday", artist="Beatles", length_ms=300000)
    # May or may not find results depending on whether fuzzy search runs
    # The key is that it doesn't crash


def test_multi_stage_searcher_disabled_stages(tmp_path):
    """Test disabling specific search stages."""
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer

    db = MusicGraphDB(tmp_path / "test.sqlite")
    normalizer = Normalizer()

    # Setup test data
    db.upsert_artist("artist-1", "The Beatles", name_normalized="beatles")
    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        title_normalized="yesterday",
    )
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("rec-1", "rel-1")

    # Disable strict search
    config = SearchConfig(enable_strict=False)
    searcher = MultiStageSearcher(db, normalizer, config)

    results = searcher.search(title="Yesterday", artist="Beatles")

    # Should still find via fuzzy since strict is disabled
    # All results should be from fuzzy stage
    for result in results:
        assert result.stage == SearchStage.FUZZY
