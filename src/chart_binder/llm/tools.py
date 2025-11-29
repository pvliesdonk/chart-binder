"""LLM tools for Chart-Binder adjudication.

Provides tools that LLMs can call to gather additional information:
- web_search: Search the web for music release information
- web_fetch: Fetch and extract content from specific URLs
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
from langchain_core.tools import tool

if TYPE_CHECKING:
    from chart_binder.llm.searxng import SearxNGSearchTool

log = logging.getLogger(__name__)

# Global search tool instance (set by adjudicator)
_searxng_tool: SearxNGSearchTool | None = None


def set_searxng_tool(searxng: SearxNGSearchTool | None) -> None:
    """Set the global SearxNG tool instance."""
    global _searxng_tool
    _searxng_tool = searxng


@tool
def web_search(query: str) -> str:
    """Search the web for information about music releases, artists, and recordings.

    Use this to find additional context about release dates, chart positions,
    authoritative sources, and other metadata not available in MusicBrainz.

    Args:
        query: Search query (e.g., "Queen Killer Queen single release date 1974")

    Returns:
        Search results as formatted text with titles, URLs, and snippets
    """
    if _searxng_tool is None:
        return "Error: Web search is not available (SearxNG not configured)"

    try:
        log.debug(f"LLM tool: web_search('{query}')")
        response = _searxng_tool.search_web(query, max_results=5)

        if not response.results:
            return f"No web results found for: {query}"

        # Format results for LLM
        lines = [f"Web search results for '{query}':\n"]
        for i, result in enumerate(response.results[:5], 1):
            lines.append(f"{i}. {result.title}")
            if result.metadata.get("snippet"):
                snippet = result.metadata["snippet"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                lines.append(f"   {snippet}")
            lines.append(f"   URL: {result.metadata.get('url', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        log.error(f"Web search tool error: {e}")
        return f"Error performing web search: {e}"


@tool
def web_fetch(url: str) -> str:
    """Fetch and extract text content from a specific URL.

    Use this to get detailed information from authoritative sources like
    official discographies, music databases, or reliable music journalism sites.

    Args:
        url: URL to fetch (must be http:// or https://)

    Returns:
        Extracted text content from the page (truncated to 2000 chars)
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: Invalid URL (must start with http:// or https://): {url}"

    try:
        log.debug(f"LLM tool: web_fetch('{url}')")

        # Fetch the URL with timeout
        client = httpx.Client(timeout=10.0, follow_redirects=True)
        response = client.get(url)
        response.raise_for_status()

        # Get content type
        content_type = response.headers.get("content-type", "")

        # Only process HTML/text content
        if "html" not in content_type and "text" not in content_type:
            return f"Error: URL returned non-text content: {content_type}"

        # Extract text (simple approach - just get the text)
        # For better extraction, could use beautifulsoup4 or html2text
        content = response.text

        # Simple HTML tag stripping (basic approach)
        import re

        # Remove script and style tags and their content
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content).strip()

        # Truncate to reasonable length
        if len(content) > 2000:
            content = content[:2000] + "...[truncated]"

        return f"Content from {url}:\n\n{content}"

    except httpx.HTTPStatusError as e:
        log.error(f"Web fetch HTTP error for {url}: {e}")
        return f"Error: HTTP {e.response.status_code} when fetching {url}"
    except httpx.TimeoutException:
        log.error(f"Web fetch timeout for {url}")
        return f"Error: Timeout when fetching {url}"
    except Exception as e:
        log.error(f"Web fetch error for {url}: {e}")
        return f"Error fetching URL: {e}"
