"""Search tool for LLM adjudication in Chart-Binder.

Provides a search interface that the LLM can use to gather additional
context about recordings, artists, and releases. Hybrid approach:
- ID lookups: Local DB first, fetch from API if not found
- Free text searches: Pass directly to external APIs (MusicBrainz, Discogs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chart_binder.fetcher import UnifiedFetcher as Fetcher
    from chart_binder.musicbrainz import MusicBrainzClient
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

    Hybrid approach:
    - ID lookups (MBID): Check local DB first, fetch from API if not found
    - Free text searches: Pass directly to MusicBrainz API

    Provides methods to search for:
    - Artists by name or MBID
    - Recordings by title, ISRC, or MBID
    - Release groups by title or MBID
    - Releases by MBID, barcode, or catalog number

    Results are formatted for LLM consumption.
    """

    def __init__(
        self,
        music_graph_db: MusicGraphDB | None = None,
        fetcher: Fetcher | None = None,
        mb_client: MusicBrainzClient | None = None,
    ):
        # Use Any type for dynamic attribute access
        self._db: Any = music_graph_db
        self._fetcher: Any = fetcher
        self._mb_client: Any = mb_client
        self._max_results = 10

    @property
    def db(self) -> Any:
        """Get the music graph database instance."""
        return self._db

    def set_database(self, db: MusicGraphDB) -> None:
        """Set the music graph database instance."""
        self._db = db

    def set_fetcher(self, fetcher: Fetcher) -> None:
        """Set the fetcher for API calls."""
        self._fetcher = fetcher

    def set_mb_client(self, mb_client: MusicBrainzClient) -> None:
        """Set the MusicBrainz client for direct API calls."""
        self._mb_client = mb_client

    def search_artist(
        self,
        query: str,
        *,
        by_mbid: bool = False,
    ) -> SearchResponse:
        """Search for artists by name or MBID.

        Hybrid approach:
        - by_mbid=True: Check local DB, fetch from API if not found
        - by_mbid=False: Search MusicBrainz API directly

        Args:
            query: Artist name or MBID to search for
            by_mbid: If True, lookup by exact MBID

        Returns:
            SearchResponse with matching artists
        """
        results: list[SearchResult] = []

        if by_mbid:
            # ID lookup: local first, then fetch
            artist = self._db.get_artist(query) if self._db else None

            if not artist and self._fetcher:
                # Fetch from API and hydrate
                try:
                    self._fetcher.fetch_artist(query)
                    artist = self._db.get_artist(query) if self._db else None
                except Exception as e:
                    log.debug(f"Failed to fetch artist {query}: {e}")

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
            # Free text search: direct to MusicBrainz API
            if self._mb_client:
                try:
                    api_results = self._mb_client.search_artists(query, limit=self._max_results)
                    for artist in api_results:
                        results.append(
                            SearchResult(
                                result_type=SearchResultType.ARTIST,
                                id=artist.get("id", ""),
                                title=artist.get("name", ""),
                                metadata={
                                    "sort_name": artist.get("sort-name"),
                                    "country": artist.get("country"),
                                    "disambiguation": artist.get("disambiguation"),
                                },
                            )
                        )
                except Exception as e:
                    log.debug(f"MusicBrainz artist search failed: {e}")
            elif self._db and hasattr(self._db, "search_artists"):
                # Fallback to local DB search
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
        artist: str | None = None,
    ) -> SearchResponse:
        """Search for recordings by title, MBID, or ISRC.

        Hybrid approach:
        - by_mbid=True: Check local DB, fetch from API if not found
        - by_isrc=True: Search MusicBrainz API by ISRC
        - Otherwise: Search MusicBrainz API by title/artist

        Args:
            query: Recording title, MBID, or ISRC
            by_mbid: Lookup by exact MBID
            by_isrc: Search by ISRC code
            artist: Optional artist name filter (used with text search)

        Returns:
            SearchResponse with matching recordings
        """
        results: list[SearchResult] = []

        if by_mbid:
            # ID lookup: local first, then fetch
            recording = self._db.get_recording(query) if self._db else None

            if not recording and self._fetcher:
                try:
                    self._fetcher.fetch_recording(query)
                    recording = self._db.get_recording(query) if self._db else None
                except Exception as e:
                    log.debug(f"Failed to fetch recording {query}: {e}")

            if recording:
                results.append(self._recording_to_result(recording))

        elif by_isrc:
            # ISRC search: MusicBrainz API
            if self._mb_client:
                try:
                    api_results = self._mb_client.search_recordings(
                        isrc=query, limit=self._max_results
                    )
                    for rec in api_results:
                        results.append(
                            SearchResult(
                                result_type=SearchResultType.RECORDING,
                                id=rec.mbid,
                                title=rec.title,
                                metadata={
                                    "artist": rec.artist_name,
                                    "artist_mbid": rec.artist_mbid,
                                    "duration_ms": rec.length_ms,
                                    "isrcs": rec.isrcs,
                                },
                            )
                        )
                except Exception as e:
                    log.debug(f"MusicBrainz ISRC search failed: {e}")

        else:
            # Free text search: MusicBrainz API
            if self._mb_client:
                try:
                    api_results = self._mb_client.search_recordings(
                        title=query, artist=artist, limit=self._max_results
                    )
                    for rec in api_results:
                        results.append(
                            SearchResult(
                                result_type=SearchResultType.RECORDING,
                                id=rec.mbid,
                                title=rec.title,
                                metadata={
                                    "artist": rec.artist_name,
                                    "artist_mbid": rec.artist_mbid,
                                    "duration_ms": rec.length_ms,
                                },
                            )
                        )
                except Exception as e:
                    log.debug(f"MusicBrainz recording search failed: {e}")

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
        artist: str | None = None,
    ) -> SearchResponse:
        """Search for release groups by title or MBID.

        Hybrid approach:
        - by_mbid=True: Check local DB, fetch from API if not found
        - Otherwise: Search MusicBrainz API by title/artist

        Args:
            query: Release group title or MBID
            by_mbid: Lookup by exact MBID
            artist: Optional artist name filter (used with text search)

        Returns:
            SearchResponse with matching release groups
        """
        results: list[SearchResult] = []

        if by_mbid:
            # ID lookup: local first, then fetch
            rg = None
            if self._db and hasattr(self._db, "get_release_group"):
                rg = self._db.get_release_group(query)

            if not rg and self._fetcher:
                try:
                    self._fetcher.fetch_release_group(query)
                    if self._db:
                        rg = self._db.get_release_group(query)
                except Exception as e:
                    log.debug(f"Failed to fetch release group {query}: {e}")

            if rg:
                results.append(self._release_group_to_result(rg))
        else:
            # Free text search: MusicBrainz API
            if self._mb_client and hasattr(self._mb_client, "search_release_groups"):
                try:
                    api_results = self._mb_client.search_release_groups(
                        title=query, artist=artist, limit=self._max_results
                    )
                    for rg in api_results:
                        results.append(
                            SearchResult(
                                result_type=SearchResultType.RELEASE_GROUP,
                                id=rg.get("id", ""),
                                title=rg.get("title", ""),
                                metadata={
                                    "primary_type": rg.get("primary-type"),
                                    "secondary_types": rg.get("secondary-types", []),
                                    "first_release_date": rg.get("first-release-date"),
                                },
                            )
                        )
                except Exception as e:
                    log.debug(f"MusicBrainz release group search failed: {e}")

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
        release_group_mbid: str | None = None,
    ) -> SearchResponse:
        """Search for releases by MBID, barcode, or catalog number.

        Hybrid approach:
        - by_mbid=True: Check local DB, fetch from API if not found
        - by_barcode/by_catno: Search local DB (fetcher handles barcode lookups)

        Args:
            query: Release MBID, barcode, or catalog number
            by_mbid: Lookup by exact MBID
            by_barcode: Search by barcode
            by_catno: Search by catalog number
            release_group_mbid: Filter by release group

        Returns:
            SearchResponse with matching releases
        """
        results: list[SearchResult] = []

        if by_mbid:
            # ID lookup: local first, then fetch
            release = None
            if self._db and hasattr(self._db, "get_release"):
                release = self._db.get_release(query)

            if not release and self._fetcher:
                try:
                    self._fetcher.fetch_release(query)
                    if self._db:
                        release = self._db.get_release(query)
                except Exception as e:
                    log.debug(f"Failed to fetch release {query}: {e}")

            if release:
                results.append(self._release_to_result(release))

        elif by_barcode:
            # Barcode lookup via fetcher (searches multiple sources)
            if self._fetcher:
                try:
                    search_results = self._fetcher.search_recordings(barcode=query)
                    for r in search_results[: self._max_results]:
                        if r.get("discogs_release_id"):
                            results.append(
                                SearchResult(
                                    result_type=SearchResultType.RELEASE,
                                    id=f"discogs:{r['discogs_release_id']}",
                                    title=r.get("title", ""),
                                    metadata={
                                        "source": "discogs",
                                        "artist": r.get("artist_name"),
                                    },
                                )
                            )
                except Exception as e:
                    log.debug(f"Barcode search failed: {e}")

        elif by_catno:
            # Catalog number: local DB search
            if self._db and hasattr(self._db, "search_releases_by_catno"):
                releases = self._db.search_releases_by_catno(query)
                results.extend(self._release_to_result(r) for r in releases[: self._max_results])

        else:
            # General release search: local DB
            if self._db and hasattr(self._db, "search_releases"):
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


def test_search_tool_with_real_db_methods(tmp_path):
    """Test SearchTool uses actual DB methods when available."""
    from chart_binder.musicgraph import MusicGraphDB

    # Create a populated database
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles", sort_name="Beatles, The", begin_area_country="GB")
    db.upsert_artist("artist-2", "The Rolling Stones")

    db.upsert_recording(
        "rec-1",
        "Yesterday",
        artist_mbid="artist-1",
        length_ms=125000,
        isrcs_json='["GBAYE0601315"]',
    )
    db.upsert_recording(
        "rec-2",
        "Hey Jude",
        artist_mbid="artist-1",
        length_ms=431000,
    )

    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-2", "Abbey Road", artist_mbid="artist-1", type="Album")

    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")
    db.upsert_release("rel-2", "Abbey Road", release_group_mbid="rg-2")

    db.upsert_recording_release("rec-1", "rel-1")
    db.upsert_recording_release("rec-2", "rel-2")

    # Create search tool with database only (no MB client for text searches)
    tool = SearchTool(music_graph_db=db)

    # Test artist search by MBID (uses local DB)
    response = tool.search_artist("artist-1", by_mbid=True)
    assert response.total_count == 1
    assert response.results[0].title == "The Beatles"
    assert response.results[0].metadata["sort_name"] == "Beatles, The"

    # Test recording search by MBID (uses local DB)
    response = tool.search_recording("rec-2", by_mbid=True)
    assert response.total_count == 1
    assert response.results[0].title == "Hey Jude"

    # Test release group search by MBID (uses local DB)
    response = tool.search_release_group("rg-1", by_mbid=True)
    assert response.total_count == 1
    assert response.results[0].title == "Help!"

    # Test get releases in group (uses local DB)
    response = tool.get_release_group_releases("rg-1")
    assert response.total_count >= 1
    assert any(r.title == "Help!" for r in response.results)

    # Test get release groups for recording (uses local DB)
    response = tool.get_recording_release_groups("rec-1")
    assert response.total_count >= 1
    assert any(r.title == "Help!" for r in response.results)

    # Note: Text searches (search_artist, search_recording without by_mbid,
    # search_release_group without by_mbid, by_isrc) now go to MusicBrainz API
    # and are not tested here without a mock MB client


def test_search_tool_format_for_llm_multiple_searches(tmp_path):
    """Test format_for_llm with multiple search types using MBID lookups."""
    from chart_binder.musicgraph import MusicGraphDB

    # Create a populated database
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_recording("rec-1", "Yesterday", artist_mbid="artist-1", length_ms=125000)
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release("rel-1", "Help!", release_group_mbid="rg-1")

    tool = SearchTool(music_graph_db=db)

    # Use MBID lookups (text searches require MB client)
    artist_search = tool.search_artist("artist-1", by_mbid=True)
    recording_search = tool.search_recording("rec-1", by_mbid=True)
    rg_search = tool.search_release_group("rg-1", by_mbid=True)

    # Format all searches together
    searches = [
        ("Artist Results", artist_search),
        ("Recording Results", recording_search),
        ("Release Group Results", rg_search),
    ]
    formatted = tool.format_for_llm(searches)

    # Verify all sections are present
    assert "## Artist Results" in formatted
    assert "## Recording Results" in formatted
    assert "## Release Group Results" in formatted

    # Verify actual results appear
    assert "The Beatles" in formatted
    assert "Yesterday" in formatted
    assert "Help!" in formatted

    # Verify structure
    assert "[artist]" in formatted
    assert "[recording]" in formatted
    assert "[release_group]" in formatted
