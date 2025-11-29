"""SearxNG client for external web searches in Chart-Binder.

SearxNG is a self-hosted metasearch engine that aggregates results
from multiple search engines while respecting privacy.

This module provides:
- SearxNGClient: Low-level HTTP client for SearxNG API
- SearxNGSearchTool: High-level tool for music-specific searches
- Integration with existing SearchResponse/SearchResult format
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from chart_binder.llm.search_tool import SearchResponse, SearchResult, SearchResultType

log = logging.getLogger(__name__)


@dataclass
class SearxNGResult:
    """Single search result from SearxNG."""

    title: str
    url: str
    content: str  # snippet/description
    engine: str
    score: float = 1.0


@dataclass
class SearxNGResponse:
    """Response from a SearxNG search operation."""

    query: str
    results: list[SearxNGResult]
    total_count: int
    error: str | None = None

    def to_context_string(self) -> str:
        """Format results for LLM consumption."""
        if self.error:
            return f"Search error for '{self.query}': {self.error}"

        if not self.results:
            return f"No web results found for query: {self.query}"

        lines = [f"Web search results for '{self.query}' ({self.total_count} results):"]
        for i, result in enumerate(self.results, 1):
            lines.append(f"\n{i}. {result.title}")
            lines.append(f"   URL: {result.url}")
            if result.content:
                # Truncate long snippets
                snippet = result.content[:200] + "..." if len(result.content) > 200 else result.content
                lines.append(f"   {snippet}")
            lines.append(f"   Source: {result.engine}")

        return "\n".join(lines)


class SearxNGClient:
    """HTTP client for SearxNG metasearch engine.

    Provides low-level access to SearxNG's JSON API for performing
    web searches across multiple search engines.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        """Initialize SearxNG client.

        Args:
            base_url: Base URL of the SearxNG instance (e.g., http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def search(
        self,
        query: str,
        categories: list[str] | None = None,
        engines: list[str] | None = None,
        max_results: int = 10,
    ) -> SearxNGResponse:
        """Perform a search query via SearxNG.

        Args:
            query: Search query string
            categories: Optional list of categories to search (default: ["general"])
            engines: Optional list of specific engines to use
            max_results: Maximum number of results to return

        Returns:
            SearxNGResponse with search results or error information
        """
        if not query or not query.strip():
            return SearxNGResponse(
                query=query,
                results=[],
                total_count=0,
                error="Empty query provided",
            )

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
        }

        if categories:
            params["categories"] = ",".join(categories)
        else:
            params["categories"] = "general"

        if engines:
            params["engines"] = ",".join(engines)

        try:
            response = self._client.get(
                f"{self.base_url}/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            # Parse results from SearxNG response
            raw_results = data.get("results", [])
            results: list[SearxNGResult] = []

            for raw_result in raw_results[:max_results]:
                results.append(
                    SearxNGResult(
                        title=raw_result.get("title", ""),
                        url=raw_result.get("url", ""),
                        content=raw_result.get("content", ""),
                        engine=raw_result.get("engine", "unknown"),
                        score=raw_result.get("score", 1.0),
                    )
                )

            return SearxNGResponse(
                query=query,
                results=results,
                total_count=len(results),
                error=None,
            )

        except httpx.TimeoutException as e:
            log.warning(f"SearxNG search timeout for query '{query}': {e}")
            return SearxNGResponse(
                query=query,
                results=[],
                total_count=0,
                error=f"Search timeout: {e}",
            )
        except httpx.HTTPStatusError as e:
            log.warning(f"SearxNG HTTP error for query '{query}': {e}")
            return SearxNGResponse(
                query=query,
                results=[],
                total_count=0,
                error=f"HTTP error {e.response.status_code}: {e}",
            )
        except httpx.HTTPError as e:
            log.warning(f"SearxNG connection error for query '{query}': {e}")
            return SearxNGResponse(
                query=query,
                results=[],
                total_count=0,
                error=f"Connection error: {e}",
            )
        except Exception as e:
            log.error(f"Unexpected error in SearxNG search for query '{query}': {e}")
            return SearxNGResponse(
                query=query,
                results=[],
                total_count=0,
                error=f"Unexpected error: {e}",
            )

    def is_available(self) -> bool:
        """Check if the SearxNG instance is reachable.

        Returns:
            True if SearxNG is available, False otherwise
        """
        try:
            response = self._client.get(f"{self.base_url}/", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()

    def __enter__(self) -> SearxNGClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


class SearxNGSearchTool:
    """High-level search tool for music-specific web searches.

    Wraps SearxNGClient and provides methods tailored for music metadata
    verification and discovery. Converts results to the standard SearchResponse
    format used by other search tools in Chart-Binder.
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 10.0):
        """Initialize the search tool.

        Args:
            base_url: SearxNG instance URL (default from config)
            timeout: Request timeout in seconds
        """
        self.client = SearxNGClient(base_url=base_url, timeout=timeout)

    def search_web(self, query: str, max_results: int = 10) -> SearchResponse:
        """General web search.

        Args:
            query: Search query string
            max_results: Maximum number of results

        Returns:
            SearchResponse with web search results
        """
        searxng_response = self.client.search(query, max_results=max_results)
        return self._convert_to_search_response(searxng_response)

    def search_music_info(
        self,
        artist: str,
        title: str,
        max_results: int = 10,
    ) -> SearchResponse:
        """Search for music information about a specific recording.

        Combines artist and title into an optimized search query.

        Args:
            artist: Artist name
            title: Recording/song title
            max_results: Maximum number of results

        Returns:
            SearchResponse with music-related web results
        """
        query = f'"{artist}" "{title}" music'
        searxng_response = self.client.search(query, max_results=max_results)
        return self._convert_to_search_response(searxng_response)

    def verify_release(
        self,
        artist: str,
        title: str,
        year: str | None = None,
        max_results: int = 10,
    ) -> SearchResponse:
        """Search for release verification information.

        Optimized for finding authoritative sources about music releases,
        such as official discographies, music databases, and reviews.

        Args:
            artist: Artist name
            title: Release title
            year: Optional release year
            max_results: Maximum number of results

        Returns:
            SearchResponse with release verification results
        """
        # Build query with year if provided
        if year:
            query = f'"{artist}" "{title}" {year} release discography'
        else:
            query = f'"{artist}" "{title}" release discography'

        searxng_response = self.client.search(query, max_results=max_results)
        return self._convert_to_search_response(searxng_response)

    def is_available(self) -> bool:
        """Check if SearxNG is available.

        Returns:
            True if SearxNG instance is reachable
        """
        return self.client.is_available()

    def _convert_to_search_response(self, searxng_response: SearxNGResponse) -> SearchResponse:
        """Convert SearxNG response to standard SearchResponse format.

        Args:
            searxng_response: Response from SearxNG client

        Returns:
            SearchResponse compatible with other search tools
        """
        results: list[SearchResult] = []

        for searxng_result in searxng_response.results:
            # Map SearxNG results to SearchResult format
            # Use a generic "external" type since web results don't fit MusicBrainz types
            result = SearchResult(
                result_type=SearchResultType.CHART_ENTRY,  # Generic type for external content
                id=searxng_result.url,  # Use URL as unique ID
                title=searxng_result.title,
                metadata={
                    "url": searxng_result.url,
                    "snippet": searxng_result.content,
                    "source_engine": searxng_result.engine,
                },
                score=searxng_result.score,
            )
            results.append(result)

        return SearchResponse(
            query=searxng_response.query,
            results=results,
            total_count=searxng_response.total_count,
            truncated=False,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self.client.close()

    def __enter__(self) -> SearxNGSearchTool:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


## Tests


def test_searxng_result():
    """Test SearxNGResult dataclass."""
    result = SearxNGResult(
        title="Test Result",
        url="https://example.com",
        content="This is a test snippet",
        engine="google",
        score=0.95,
    )
    assert result.title == "Test Result"
    assert result.url == "https://example.com"
    assert result.content == "This is a test snippet"
    assert result.engine == "google"
    assert result.score == 0.95


def test_searxng_response_empty():
    """Test SearxNGResponse with no results."""
    response = SearxNGResponse(query="test", results=[], total_count=0)
    context = response.to_context_string()
    assert "No web results found" in context
    assert "test" in context


def test_searxng_response_with_error():
    """Test SearxNGResponse with error."""
    response = SearxNGResponse(
        query="test",
        results=[],
        total_count=0,
        error="Connection failed",
    )
    context = response.to_context_string()
    assert "Search error" in context
    assert "Connection failed" in context


def test_searxng_response_with_results():
    """Test SearxNGResponse with results."""
    results = [
        SearxNGResult(
            title="Result 1",
            url="https://example.com/1",
            content="First result",
            engine="google",
        ),
        SearxNGResult(
            title="Result 2",
            url="https://example.com/2",
            content="Second result",
            engine="duckduckgo",
        ),
    ]
    response = SearxNGResponse(query="test query", results=results, total_count=2)
    context = response.to_context_string()
    assert "test query" in context
    assert "Result 1" in context
    assert "Result 2" in context
    assert "https://example.com/1" in context
    assert "google" in context


def test_searxng_response_long_snippet():
    """Test SearxNGResponse truncates long snippets."""
    long_content = "x" * 300
    results = [
        SearxNGResult(
            title="Long Result",
            url="https://example.com",
            content=long_content,
            engine="test",
        )
    ]
    response = SearxNGResponse(query="test", results=results, total_count=1)
    context = response.to_context_string()
    assert "..." in context
    assert len(context) < len(long_content) + 200


def test_searxng_client_empty_query():
    """Test SearxNG client with empty query."""
    client = SearxNGClient(base_url="http://localhost:8080")
    response = client.search("")
    assert response.error is not None
    assert "Empty query" in response.error
    assert response.total_count == 0


def test_searxng_client_unavailable():
    """Test SearxNG client availability check when not running."""
    client = SearxNGClient(base_url="http://localhost:99999", timeout=1.0)
    assert client.is_available() is False


def test_searxng_tool_search_web():
    """Test SearxNGSearchTool web search."""
    tool = SearxNGSearchTool(base_url="http://localhost:99999", timeout=1.0)
    response = tool.search_web("test query")
    # Should return empty results due to unavailable server
    assert isinstance(response, SearchResponse)
    assert response.query == "test query"


def test_searxng_tool_search_music_info():
    """Test SearxNGSearchTool music info search."""
    tool = SearxNGSearchTool(base_url="http://localhost:99999", timeout=1.0)
    response = tool.search_music_info(artist="The Beatles", title="Hey Jude")
    assert isinstance(response, SearchResponse)
    assert "The Beatles" in response.query
    assert "Hey Jude" in response.query


def test_searxng_tool_verify_release():
    """Test SearxNGSearchTool release verification."""
    tool = SearxNGSearchTool(base_url="http://localhost:99999", timeout=1.0)
    response = tool.verify_release(
        artist="Pink Floyd",
        title="The Dark Side of the Moon",
        year="1973",
    )
    assert isinstance(response, SearchResponse)
    assert "Pink Floyd" in response.query
    assert "1973" in response.query


def test_searxng_tool_verify_release_no_year():
    """Test SearxNGSearchTool release verification without year."""
    tool = SearxNGSearchTool(base_url="http://localhost:99999", timeout=1.0)
    response = tool.verify_release(
        artist="Led Zeppelin",
        title="Led Zeppelin IV",
    )
    assert isinstance(response, SearchResponse)
    assert "Led Zeppelin" in response.query
    assert "Led Zeppelin IV" in response.query


def test_searxng_tool_context_manager():
    """Test SearxNGSearchTool context manager."""
    with SearxNGSearchTool(base_url="http://localhost:8080") as tool:
        assert tool.client is not None
    # After exit, client should be closed (no exception raised)


def test_searxng_client_context_manager():
    """Test SearxNGClient context manager."""
    with SearxNGClient(base_url="http://localhost:8080") as client:
        assert client._client is not None
    # After exit, client should be closed (no exception raised)


def test_convert_to_search_response():
    """Test conversion from SearxNG to SearchResponse format."""
    tool = SearxNGSearchTool(base_url="http://localhost:8080")
    searxng_response = SearxNGResponse(
        query="test",
        results=[
            SearxNGResult(
                title="Test Result",
                url="https://example.com",
                content="Test content",
                engine="google",
                score=0.9,
            )
        ],
        total_count=1,
    )

    search_response = tool._convert_to_search_response(searxng_response)

    assert search_response.query == "test"
    assert len(search_response.results) == 1
    assert search_response.results[0].title == "Test Result"
    assert search_response.results[0].id == "https://example.com"
    assert search_response.results[0].metadata["url"] == "https://example.com"
    assert search_response.results[0].metadata["snippet"] == "Test content"
    assert search_response.results[0].metadata["source_engine"] == "google"
    assert search_response.results[0].score == 0.9
