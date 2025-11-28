"""Search tool for LLM adjudication in Chart-Binder.

Provides a search interface that the LLM can use to gather additional
context about recordings, artists, and releases from the music graph
database and external sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chart_binder.musicgraph import MusicGraphDB

log = logging.getLogger(__name__)


class SearchResultType(StrEnum):
    """Type of search result."""

    ARTIST = "artist"
    RECORDING = "recording"
    RELEASE_GROUP = "release_group"
    RELEASE = "release"
    CHART_ENTRY = "chart_entry"


@dataclass
class SearchResult:
    """Single search result from the tool."""

    result_type: SearchResultType
    id: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 1.0

    def to_context_string(self) -> str:
        """Format result as context string for LLM."""
        lines = [f"[{self.result_type.value}] {self.title}"]
        lines.append(f"  ID: {self.id}")

        for key, value in self.metadata.items():
            if value:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


@dataclass
class SearchResponse:
    """Response from a search operation."""

    query: str
    results: list[SearchResult]
    total_count: int
    truncated: bool = False

    def to_context_string(self) -> str:
        """Format all results as context for LLM prompt."""
        if not self.results:
            return f"No results found for query: {self.query}"

        lines = [f"Search results for '{self.query}' ({self.total_count} total):"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"\n{i}. {result.to_context_string()}")

        if self.truncated:
            lines.append(f"\n... and {self.total_count - len(self.results)} more results")

        return "\n".join(lines)


class SearchTool:
    """Search tool for LLM context gathering.

    Provides methods to search the music graph database for:
    - Artists by name or MBID
    - Recordings by title, ISRC, or MBID
    - Release groups by title or MBID
    - Releases by catalog number, barcode, or MBID
    - Chart entries by title/artist

    Results are formatted for LLM consumption.
    """

    def __init__(self, music_graph_db: MusicGraphDB | None = None):
        # Use Any type for dynamic attribute access
        self._db: Any = music_graph_db
        self._max_results = 10

    @property
    def db(self) -> Any:
        """Get the music graph database instance."""
        return self._db

    def set_database(self, db: MusicGraphDB) -> None:
        """Set the music graph database instance."""
        self._db = db

    def search_artist(
        self,
        query: str,
        *,
        by_mbid: bool = False,
    ) -> SearchResponse:
        """Search for artists by name or MBID.

        Args:
            query: Artist name or MBID to search for
            by_mbid: If True, search by exact MBID match

        Returns:
            SearchResponse with matching artists
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=query, results=[], total_count=0)

        if by_mbid:
            artist = self._db.get_artist(query)
            if artist:
                results.append(
                    SearchResult(
                        result_type=SearchResultType.ARTIST,
                        id=artist["mbid"],
                        title=artist["name"],
                        metadata={
                            "sort_name": artist.get("sort_name"),
                            "country": artist.get("begin_area_country"),
                            "disambiguation": artist.get("disambiguation"),
                        },
                    )
                )
        else:
            # Use search_artists if available, otherwise search by name directly
            if hasattr(self._db, "search_artists"):
                artists = self._db.search_artists(query, limit=self._max_results)
                for artist in artists:
                    results.append(
                        SearchResult(
                            result_type=SearchResultType.ARTIST,
                            id=artist["mbid"],
                            title=artist["name"],
                            metadata={
                                "sort_name": artist.get("sort_name"),
                                "country": artist.get("begin_area_country"),
                            },
                        )
                    )

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            truncated=len(results) >= self._max_results,
        )

    def search_recording(
        self,
        query: str,
        *,
        by_mbid: bool = False,
        by_isrc: bool = False,
        artist: str | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> SearchResponse:
        """Search for recordings by title, MBID, or ISRC.

        Args:
            query: Recording title, MBID, or ISRC
            by_mbid: Search by exact MBID
            by_isrc: Search by ISRC code
            artist: Optional artist name filter

        Returns:
            SearchResponse with matching recordings
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=query, results=[], total_count=0)

        if by_mbid:
            recording = self._db.get_recording(query)
            if recording:
                results.append(self._recording_to_result(recording))
        elif by_isrc:
            # Use search_recordings_by_isrc if available
            if hasattr(self._db, "search_recordings_by_isrc"):
                recordings = self._db.search_recordings_by_isrc(query)
                results.extend(
                    self._recording_to_result(r) for r in recordings[: self._max_results]
                )
        else:
            # Use search_recordings if available
            if hasattr(self._db, "search_recordings"):
                recordings = self._db.search_recordings(
                    query, artist=artist, limit=self._max_results
                )
                results.extend(self._recording_to_result(r) for r in recordings)

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            truncated=len(results) >= self._max_results,
        )

    def search_release_group(
        self,
        query: str,
        *,
        by_mbid: bool = False,
        artist: str | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> SearchResponse:
        """Search for release groups by title or MBID.

        Args:
            query: Release group title or MBID
            by_mbid: Search by exact MBID
            artist: Optional artist name filter

        Returns:
            SearchResponse with matching release groups
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=query, results=[], total_count=0)

        if by_mbid:
            # Use get_release_group if available
            if hasattr(self._db, "get_release_group"):
                rg = self._db.get_release_group(query)
                if rg:
                    results.append(self._release_group_to_result(rg))
        else:
            # Use search_release_groups if available
            if hasattr(self._db, "search_release_groups"):
                rgs = self._db.search_release_groups(query, artist=artist, limit=self._max_results)
                results.extend(self._release_group_to_result(rg) for rg in rgs)

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            truncated=len(results) >= self._max_results,
        )

    def search_release(
        self,
        query: str,
        *,
        by_mbid: bool = False,
        by_barcode: bool = False,
        by_catno: bool = False,
        release_group_mbid: str | None = None,  # pyright: ignore[reportUnusedParameter]
    ) -> SearchResponse:
        """Search for releases by title, MBID, barcode, or catalog number.

        Args:
            query: Release title, MBID, barcode, or catalog number
            by_mbid: Search by exact MBID
            by_barcode: Search by barcode
            by_catno: Search by catalog number
            release_group_mbid: Filter by release group

        Returns:
            SearchResponse with matching releases
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=query, results=[], total_count=0)

        if by_mbid:
            # Use get_release if available
            if hasattr(self._db, "get_release"):
                release = self._db.get_release(query)
                if release:
                    results.append(self._release_to_result(release))
        elif by_barcode:
            if hasattr(self._db, "search_releases_by_barcode"):
                releases = self._db.search_releases_by_barcode(query)
                results.extend(self._release_to_result(r) for r in releases[: self._max_results])
        elif by_catno:
            if hasattr(self._db, "search_releases_by_catno"):
                releases = self._db.search_releases_by_catno(query)
                results.extend(self._release_to_result(r) for r in releases[: self._max_results])
        else:
            if hasattr(self._db, "search_releases"):
                releases = self._db.search_releases(
                    query,
                    release_group_mbid=release_group_mbid,
                    limit=self._max_results,
                )
                results.extend(self._release_to_result(r) for r in releases)

        return SearchResponse(
            query=query,
            results=results,
            total_count=len(results),
            truncated=len(results) >= self._max_results,
        )

    def get_release_group_releases(self, rg_mbid: str) -> SearchResponse:
        """Get all releases in a release group.

        Args:
            rg_mbid: Release group MBID

        Returns:
            SearchResponse with releases in the group
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=rg_mbid, results=[], total_count=0)

        if hasattr(self._db, "get_releases_in_group"):
            releases = self._db.get_releases_in_group(rg_mbid)
            results.extend(self._release_to_result(r) for r in releases[: self._max_results])
            return SearchResponse(
                query=f"releases in RG {rg_mbid}",
                results=results,
                total_count=len(releases),
                truncated=len(releases) > self._max_results,
            )

        return SearchResponse(
            query=f"releases in RG {rg_mbid}",
            results=results,
            total_count=0,
        )

    def get_recording_release_groups(self, recording_mbid: str) -> SearchResponse:
        """Get all release groups containing a recording.

        Args:
            recording_mbid: Recording MBID

        Returns:
            SearchResponse with release groups containing the recording
        """
        results: list[SearchResult] = []

        if self._db is None:
            return SearchResponse(query=recording_mbid, results=[], total_count=0)

        if hasattr(self._db, "get_release_groups_for_recording"):
            rgs = self._db.get_release_groups_for_recording(recording_mbid)
            results.extend(self._release_group_to_result(rg) for rg in rgs[: self._max_results])
            return SearchResponse(
                query=f"RGs for recording {recording_mbid}",
                results=results,
                total_count=len(rgs),
                truncated=len(rgs) > self._max_results,
            )

        return SearchResponse(
            query=f"RGs for recording {recording_mbid}",
            results=results,
            total_count=0,
        )

    def _recording_to_result(self, recording: dict[str, Any]) -> SearchResult:
        """Convert recording dict to SearchResult."""
        return SearchResult(
            result_type=SearchResultType.RECORDING,
            id=recording.get("mbid") or recording.get("mb_recording_id", ""),
            title=recording.get("title", ""),
            metadata={
                "artist": recording.get("artist_credit") or recording.get("artist_mbid"),
                "duration_ms": recording.get("duration_ms") or recording.get("length_ms"),
                "isrc": recording.get("isrc") or recording.get("isrcs_json"),
                "disambiguation": recording.get("disambiguation"),
            },
        )

    def _release_group_to_result(self, rg: dict[str, Any]) -> SearchResult:
        """Convert release group dict to SearchResult."""
        return SearchResult(
            result_type=SearchResultType.RELEASE_GROUP,
            id=rg.get("mbid") or rg.get("mb_rg_id", ""),
            title=rg.get("title", ""),
            metadata={
                "artist": rg.get("artist_credit") or rg.get("artist_mbid"),
                "primary_type": rg.get("primary_type") or rg.get("type"),
                "secondary_types": rg.get("secondary_types") or rg.get("secondary_types_json"),
                "first_release_date": rg.get("first_release_date"),
            },
        )

    def _release_to_result(self, release: dict[str, Any]) -> SearchResult:
        """Convert release dict to SearchResult."""
        return SearchResult(
            result_type=SearchResultType.RELEASE,
            id=release.get("mbid") or release.get("mb_release_id", ""),
            title=release.get("title", ""),
            metadata={
                "date": release.get("date"),
                "country": release.get("country"),
                "label": release.get("label"),
                "catalog_number": release.get("catalog_number") or release.get("catno"),
                "barcode": release.get("barcode"),
                "status": release.get("status"),
                "format": release.get("format"),
            },
        )

    def format_for_llm(
        self,
        searches: list[tuple[str, SearchResponse]],
    ) -> str:
        """Format multiple search responses for LLM context.

        Args:
            searches: List of (description, response) pairs

        Returns:
            Formatted string for LLM prompt
        """
        sections = []
        for description, response in searches:
            sections.append(f"## {description}\n{response.to_context_string()}")
        return "\n\n".join(sections)


## Tests


def test_search_result_to_context():
    """Test SearchResult context string formatting."""
    result = SearchResult(
        result_type=SearchResultType.ARTIST,
        id="abc123",
        title="Test Artist",
        metadata={"country": "US", "type": "Person"},
    )
    context = result.to_context_string()
    assert "[artist] Test Artist" in context
    assert "ID: abc123" in context
    assert "country: US" in context


def test_search_response_empty():
    """Test empty SearchResponse formatting."""
    response = SearchResponse(query="nothing", results=[], total_count=0)
    context = response.to_context_string()
    assert "No results found" in context


def test_search_response_with_results():
    """Test SearchResponse with results."""
    results = [
        SearchResult(
            result_type=SearchResultType.ARTIST,
            id="id1",
            title="Artist 1",
        ),
        SearchResult(
            result_type=SearchResultType.ARTIST,
            id="id2",
            title="Artist 2",
        ),
    ]
    response = SearchResponse(query="test", results=results, total_count=2)
    context = response.to_context_string()
    assert "Artist 1" in context
    assert "Artist 2" in context
    assert "2 total" in context


def test_search_tool_no_db():
    """Test SearchTool without database."""
    tool = SearchTool()
    response = tool.search_artist("Test")
    assert response.total_count == 0
    assert response.results == []


def test_format_for_llm():
    """Test formatting multiple searches for LLM."""
    tool = SearchTool()
    searches = [
        ("Artist search", SearchResponse("artist", [], 0)),
        ("Recording search", SearchResponse("recording", [], 0)),
    ]
    formatted = tool.format_for_llm(searches)
    assert "## Artist search" in formatted
    assert "## Recording search" in formatted
