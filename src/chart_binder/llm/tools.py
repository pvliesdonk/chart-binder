"""LangChain tool wrappers for MusicBrainz search operations.

This module converts SearchTool methods to LangChain's @tool format
for use with create_agent(). Each wrapper delegates to SearchTool and
returns standardized JSON responses.

All tools return JSON strings following a consistent format:
- result: status (success, no_results, error)
- content: formatted result data for LLM consumption
- action: guidance on what to do next

Usage:
    from chart_binder.llm.tools import create_music_tools
    from langchain.agents import create_agent

    tools = create_music_tools(search_tool)
    agent = create_agent(model, tools=tools)
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool, tool

if TYPE_CHECKING:
    from chart_binder.llm.search_tool import SearchTool
    from chart_binder.llm.searxng import SearxNGSearchTool

log = logging.getLogger(__name__)

# Module-level tool instances (set via create_music_tools)
_search_tool: SearchTool | None = None
_web_search_tool: SearxNGSearchTool | None = None


def _json_response(
    result: str,
    content: str | None = None,
    action: str | None = None,
    **extra: Any,
) -> str:
    """Create standardized JSON response.

    Args:
        result: Status code (success, no_results, error)
        content: Formatted content for LLM consumption
        action: Guidance on next steps
        **extra: Additional fields to include

    Returns:
        JSON string with response
    """
    response = {"result": result}
    if content is not None:
        response["content"] = content
    if action is not None:
        response["action"] = action
    response.update(extra)
    return json.dumps(response, indent=2)


@tool
def search_artist(query: str) -> str:
    """Search MusicBrainz for artists by name.

    Use this to find artist MBIDs when you only have the artist name.
    Returns matching artists with their MBIDs, countries, and disambiguation info.

    Args:
        query: Artist name to search for (e.g., "The Beatles", "Queen")

    Returns:
        JSON with search results including MBIDs and metadata
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_artist(query)

        if not response.results:
            return _json_response(
                "no_results",
                query=query,
                action="Try a different spelling or use web_search for more context",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            count=response.total_count,
            action="Use get_artist with an MBID to get detailed info",
        )

    except Exception as e:
        log.error("Artist search error: %s", e)
        return _json_response("error", error=str(e), action="Try again or use web_search")


@tool
def get_artist(mbid: str) -> str:
    """Get detailed artist information by MusicBrainz ID.

    Use this to get full details about an artist when you have their MBID.
    Checks local database first, fetches from MusicBrainz API if not found.

    Args:
        mbid: MusicBrainz artist ID (UUID format)

    Returns:
        JSON with artist details including name, country, disambiguation
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_artist(mbid, by_mbid=True)

        if not response.results:
            return _json_response(
                "no_results",
                mbid=mbid,
                action="Verify the MBID is correct or search by artist name",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            action="Use this artist info in your analysis",
        )

    except Exception as e:
        log.error("Artist lookup error: %s", e)
        return _json_response("error", error=str(e), mbid=mbid)


@tool
def search_recording(title: str, artist: str | None = None) -> str:
    """Search MusicBrainz for recordings by title and optionally artist.

    Use this to find recording MBIDs and identify which release groups contain a track.

    Args:
        title: Recording/track title to search for
        artist: Optional artist name to filter results

    Returns:
        JSON with matching recordings including MBIDs and artist info
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_recording(title, artist=artist)

        if not response.results:
            return _json_response(
                "no_results",
                title=title,
                artist=artist,
                action="Try searching with different spelling or without artist filter",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            count=response.total_count,
            action="Use get_recording with an MBID for details, or search_release_group to find albums",
        )

    except Exception as e:
        log.error("Recording search error: %s", e)
        return _json_response("error", error=str(e))


@tool
def get_recording(mbid: str) -> str:
    """Get detailed recording information by MusicBrainz ID.

    Use this to get full details about a specific recording/track.
    Checks local database first, fetches from MusicBrainz API if not found.

    Args:
        mbid: MusicBrainz recording ID (UUID format)

    Returns:
        JSON with recording details including title, artist, duration
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_recording(mbid, by_mbid=True)

        if not response.results:
            return _json_response(
                "no_results",
                mbid=mbid,
                action="Verify the MBID or search by recording title",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            action="Use this recording info in your analysis",
        )

    except Exception as e:
        log.error("Recording lookup error: %s", e)
        return _json_response("error", error=str(e), mbid=mbid)


@tool
def search_release_group(title: str, artist: str | None = None) -> str:
    """Search MusicBrainz for release groups (albums, singles, EPs) by title.

    A release group represents all versions of an album/single (different editions,
    remasters, etc.). Use this to find the canonical release group for a track.

    Args:
        title: Release group title to search for (album/single/EP name)
        artist: Optional artist name to filter results

    Returns:
        JSON with matching release groups including MBIDs, types, and first release dates
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_release_group(title, artist=artist)

        if not response.results:
            return _json_response(
                "no_results",
                title=title,
                artist=artist,
                action="Try different spelling or use web_search for more context",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            count=response.total_count,
            action="Use get_release_group for details or get_releases_in_group to see all releases",
        )

    except Exception as e:
        log.error("Release group search error: %s", e)
        return _json_response("error", error=str(e))


@tool
def get_release_group(mbid: str) -> str:
    """Get detailed release group information by MusicBrainz ID.

    Use this to get full details about a release group (album/single/EP).
    Checks local database first, fetches from MusicBrainz API if not found.

    Args:
        mbid: MusicBrainz release group ID (UUID format)

    Returns:
        JSON with release group details including type, first release date
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.search_release_group(mbid, by_mbid=True)

        if not response.results:
            return _json_response(
                "no_results",
                mbid=mbid,
                action="Verify the MBID or search by title",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            action="Use get_releases_in_group to see individual releases in this group",
        )

    except Exception as e:
        log.error("Release group lookup error: %s", e)
        return _json_response("error", error=str(e), mbid=mbid)


@tool
def get_releases_in_group(rg_mbid: str) -> str:
    """Get all releases within a release group.

    A release group can contain multiple releases (editions, countries, formats).
    Use this to find the representative release - typically the earliest release
    in the artist's origin country.

    Args:
        rg_mbid: MusicBrainz release group ID

    Returns:
        JSON with list of releases including dates, countries, and MBIDs
    """
    if _search_tool is None:
        return _json_response(
            "error",
            error="Search tool not initialized",
            action="Report this error - search tool should be configured",
        )

    try:
        response = _search_tool.get_release_group_releases(rg_mbid)

        if not response.results:
            return _json_response(
                "no_results",
                rg_mbid=rg_mbid,
                action="Verify the release group MBID is correct",
            )

        return _json_response(
            "success",
            content=response.to_context_string(),
            count=response.total_count,
            action="Select the representative release - prefer earliest in artist's origin country",
        )

    except Exception as e:
        log.error("Releases in group lookup error: %s", e)
        return _json_response("error", error=str(e), rg_mbid=rg_mbid)


@tool
def web_search(query: str) -> str:
    """Search the web for information about music releases.

    Use this when MusicBrainz data is insufficient. Good for:
    - Release date verification
    - Disambiguation between similar releases
    - Historical context about releases

    Args:
        query: Search query (e.g., "Queen Bohemian Rhapsody original release date 1975")

    Returns:
        JSON with search results including titles, URLs, and snippets
    """
    if _web_search_tool is None:
        return _json_response(
            "error",
            error="Web search not configured (SearxNG not available)",
            action="Use MusicBrainz tools instead",
        )

    try:
        response = _web_search_tool.search_web(query, max_results=5)

        if not response.results:
            return _json_response(
                "no_results",
                query=query,
                action="Try different search terms",
            )

        # Format results
        lines = [f"Web search results for '{query}':\n"]
        for i, result in enumerate(response.results[:5], 1):
            lines.append(f"{i}. {result.title}")
            if snippet := result.metadata.get("snippet"):
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"   {snippet}")
            if url := result.metadata.get("url"):
                lines.append(f"   URL: {url}")
            lines.append("")

        return _json_response(
            "success",
            content="\n".join(lines),
            count=len(response.results),
            action="Use web_fetch to get more details from a specific URL",
        )

    except Exception as e:
        log.error("Web search error: %s", e)
        return _json_response("error", error=str(e))


@tool
def web_fetch(url: str) -> str:
    """Fetch and extract content from a URL.

    Use this to get more details from a web page found via web_search.
    Extracts main content as text, stripping navigation and ads.

    Args:
        url: URL to fetch (must start with http:// or https://)

    Returns:
        JSON with extracted page content
    """
    if not url.startswith(("http://", "https://")):
        return _json_response(
            "error",
            error=f"Invalid URL (must start with http:// or https://): {url}",
            action="Provide a valid URL",
        )

    try:
        import httpx

        client = httpx.Client(timeout=10.0, follow_redirects=True)
        response = client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            return _json_response(
                "error",
                error=f"URL returned non-text content: {content_type}",
                action="Try a different URL",
            )

        content = response.text

        # Simple HTML tag stripping
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()

        # Truncate to reasonable length
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"

        return _json_response(
            "success",
            content=f"Content from {url}:\n\n{content}",
            action="Extract relevant information from the content",
        )

    except Exception as e:
        log.error("Web fetch error for %s: %s", url, e)
        return _json_response("error", error=str(e), url=url)


def create_music_tools(
    search_tool: SearchTool,
    web_search: SearxNGSearchTool | None = None,
) -> list[BaseTool]:
    """Create all music search tools configured with the given SearchTool.

    Args:
        search_tool: Configured SearchTool instance with DB and API access
        web_search: Optional SearxNG search tool for web searches

    Returns:
        List of LangChain tools ready for use with create_agent()
    """
    global _search_tool, _web_search_tool
    _search_tool = search_tool
    _web_search_tool = web_search

    tools: list[BaseTool] = [
        search_artist,
        get_artist,
        search_recording,
        get_recording,
        search_release_group,
        get_release_group,
        get_releases_in_group,
    ]

    # Add web tools if SearxNG is configured
    if web_search is not None:
        tools.extend([web_search_tool, web_fetch])
    else:
        # Add web tools anyway - they'll return helpful errors if not configured
        tools.extend([web_search_tool, web_fetch])

    return tools


# Alias for the tool function to avoid name collision
web_search_tool = web_search


## Tests


def test_json_response_success():
    """Test JSON response formatting."""
    response = _json_response("success", content="test content", action="do something")
    data = json.loads(response)
    assert data["result"] == "success"
    assert data["content"] == "test content"
    assert data["action"] == "do something"


def test_json_response_error():
    """Test JSON error response."""
    response = _json_response("error", error="something went wrong")
    data = json.loads(response)
    assert data["result"] == "error"
    assert data["error"] == "something went wrong"


def test_create_music_tools_without_web_search():
    """Test tool creation without web search."""
    from chart_binder.llm.search_tool import SearchTool

    search = SearchTool()
    tools = create_music_tools(search, web_search=None)

    # Should have 9 tools (7 music + 2 web that return errors)
    assert len(tools) == 9

    tool_names = [t.name for t in tools]
    assert "search_artist" in tool_names
    assert "get_artist" in tool_names
    assert "search_recording" in tool_names
    assert "get_recording" in tool_names
    assert "search_release_group" in tool_names
    assert "get_release_group" in tool_names
    assert "get_releases_in_group" in tool_names
    assert "web_search" in tool_names
    assert "web_fetch" in tool_names


def test_search_artist_no_tool():
    """Test search_artist when tool not initialized."""
    global _search_tool
    _search_tool = None

    result = search_artist.invoke({"query": "test"})
    data = json.loads(result)
    assert data["result"] == "error"
    assert "not initialized" in data["error"]


def test_web_fetch_invalid_url():
    """Test web_fetch with invalid URL."""
    result = web_fetch.invoke({"url": "not-a-url"})
    data = json.loads(result)
    assert data["result"] == "error"
    assert "Invalid URL" in data["error"]
