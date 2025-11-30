"""ReAct pattern LLM adjudicator using prompt engineering instead of tool binding.

This module implements the ReAct (Reasoning + Acting) pattern via prompts rather than
relying on native LLM tool calling APIs. This approach is more transparent, debuggable,
and works across different LLM providers without API-specific fragility.

Tools available:
- Web search/fetch (via SearxNG)
- MusicBrainz API queries (search artists, recordings, release groups)
- Local database lookups with API fallback for MBIDs
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from chart_binder.llm.adjudicator import (
    SYSTEM_PROMPT_V1,
    AdjudicationOutcome,
    AdjudicationResult,
)
from chart_binder.llm.search_tool import SearchTool

if TYPE_CHECKING:
    from chart_binder.config import LLMConfig
    from chart_binder.fetcher import UnifiedFetcher as Fetcher
    from chart_binder.llm.searxng import SearxNGSearchTool
    from chart_binder.musicbrainz import MusicBrainzClient
    from chart_binder.musicgraph import MusicGraphDB

log = logging.getLogger(__name__)

# ReAct pattern system prompt template
REACT_SYSTEM_PROMPT = """You are a music canonicalization expert helping identify the correct release group and representative release for chart entries.

You have access to these tools:

**Tool: search_artist**
Description: Search for artists by name. Returns artist info including MBID, country, disambiguation.
Arguments:
  - query (string): Artist name to search
Returns: List of matching artists with MBIDs

**Tool: get_artist**
Description: Get artist details by MBID. Fetches from API if not in local database.
Arguments:
  - mbid (string): MusicBrainz artist ID
Returns: Artist details including name, country, disambiguation

**Tool: search_recording**
Description: Search for recordings by title and artist. Searches MusicBrainz directly.
Arguments:
  - title (string): Recording title
  - artist (string, optional): Artist name to filter results
Returns: List of matching recordings with MBIDs

**Tool: get_recording**
Description: Get recording details by MBID. Fetches from API if not in local database.
Arguments:
  - mbid (string): MusicBrainz recording ID
Returns: Recording details including title, artist, duration

**Tool: search_release_group**
Description: Search for release groups by title. Searches MusicBrainz directly.
Arguments:
  - title (string): Release group title
  - artist (string, optional): Artist name to filter results
Returns: List of matching release groups with MBIDs, types, and first release dates

**Tool: get_release_group**
Description: Get release group details by MBID. Fetches from API if not in local database.
Arguments:
  - mbid (string): MusicBrainz release group ID
Returns: Release group details including title, type, first release date

**Tool: get_releases_in_group**
Description: Get all releases within a release group.
Arguments:
  - rg_mbid (string): Release group MBID
Returns: List of releases with dates, countries, and MBIDs

**Tool: web_search**
Description: Search the web for general information about music releases.
Arguments:
  - query (string): Search query (e.g., "Queen Bohemian Rhapsody original release date")
Returns: Search results with titles, URLs, and snippets

## How to Use Tools (ReAct Pattern)

When you need information, use this format:

Thought: [Explain what you're thinking and why you need to use a tool]
Action: tool_name
Action Input: {"arg1": "value1", "arg2": "value2"}
Observation: [This will be filled in with the tool result]

After receiving an Observation, you can:
- Use another tool if you need more information (repeat Thought/Action/Action Input)
- Provide your final answer if you have enough information

## Final Answer Format

When you have enough information, provide your answer in this format:

Thought: [Explain your final reasoning]
Final Answer:
```json
{
  "crg_mbid": "selected release group MBID",
  "rr_mbid": "selected release MBID within CRG",
  "confidence": 0.0-1.0,
  "rationale": "concise one-line explanation"
}
```

## Important Guidelines

- Always start with a Thought before taking an Action
- You can use multiple tools in sequence if needed
- Be concise but thorough in your reasoning
- You CAN propose MBIDs not in the original evidence if you find better candidates
- If the evidence seems wrong or incomplete, use tools to search for correct data
- Confidence should reflect how certain you are (0.0 = not confident, 1.0 = very confident)
"""


class ReActAdjudicator:
    """ReAct pattern LLM adjudicator using prompt engineering.

    This implementation uses the ReAct (Reasoning + Acting) pattern via prompts
    instead of relying on LLM provider's native tool calling APIs.

    Benefits:
    - More transparent and debuggable
    - Works consistently across providers
    - Less fragile than API-based tool calling
    - Easier to customize and extend

    Tools available:
    - MusicBrainz search/lookup (via SearchTool)
    - Web search (via SearxNG)
    - Local database with API fallback
    """

    MAX_ITERATIONS = 5  # Maximum reasoning/action loops

    def __init__(
        self,
        config: LLMConfig,
        search_tool: SearxNGSearchTool | None = None,
        music_search_tool: SearchTool | None = None,
        db: MusicGraphDB | None = None,
        fetcher: Fetcher | None = None,
        mb_client: MusicBrainzClient | None = None,
    ):
        self.config = config
        self.web_search_tool = search_tool  # Renamed for clarity

        # Initialize music search tool with dependencies
        if music_search_tool:
            self.music_search_tool = music_search_tool
        else:
            self.music_search_tool = SearchTool(
                music_graph_db=db,
                fetcher=fetcher,
                mb_client=mb_client,
            )

        # Initialize LLM based on provider (no tool binding)
        if config.provider == "openai":
            self.llm = ChatOpenAI(
                model=config.model_id,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_s,
            )
        elif config.provider == "ollama":
            self.llm = ChatOllama(
                model=config.model_id,
                temperature=config.temperature,
                num_predict=config.max_tokens,
                num_ctx=32768,
                base_url=config.ollama_base_url,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def adjudicate(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> AdjudicationResult:
        """Adjudicate using ReAct pattern.

        Args:
            evidence_bundle: Evidence bundle from resolver
            decision_trace: Decision trace with missing facts

        Returns:
            AdjudicationResult with CRG/RR selection
        """
        if not self.config.enabled:
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message="LLM adjudication is disabled",
            )

        # Build the initial prompt
        user_prompt = self._build_prompt(evidence_bundle, decision_trace)

        # System prompt includes ReAct instructions
        system_prompt = REACT_SYSTEM_PROMPT + "\n\n" + SYSTEM_PROMPT_V1

        log.debug("ReAct LLM Adjudication:")
        log.debug("=" * 70)
        log.debug("SYSTEM:")
        log.debug(system_prompt)
        log.debug("=" * 70)
        log.debug("USER:")
        log.debug(user_prompt)
        log.debug("=" * 70)

        try:
            # Run the ReAct loop
            result = self._react_loop(system_prompt, user_prompt, evidence_bundle)
            return result

        except Exception as e:
            log.error(f"LLM adjudication error: {e}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=str(e),
            )

    def _react_loop(
        self,
        system_prompt: str,
        user_prompt: str,
        evidence_bundle: dict[str, Any],
    ) -> AdjudicationResult:
        """Run the ReAct reasoning loop.

        The loop continues until:
        - The LLM provides a Final Answer
        - Maximum iterations reached
        - An error occurs
        """
        conversation = user_prompt
        full_interaction = []

        for iteration in range(self.MAX_ITERATIONS):
            log.debug(f"ReAct iteration {iteration + 1}/{self.MAX_ITERATIONS}")

            # Invoke LLM with current conversation
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=conversation),
            ]

            response = self.llm.invoke(messages)
            response_text = str(response.content)

            log.debug("LLM Response:")
            log.debug("=" * 70)
            log.debug(response_text)
            log.debug("=" * 70)

            full_interaction.append(f"Assistant: {response_text}")

            # Check if this is a final answer
            if "Final Answer:" in response_text:
                log.info(f"ReAct loop completed in {iteration + 1} iterations")
                return self._parse_final_answer(response_text, evidence_bundle, full_interaction)

            # Parse for tool calls
            tool_call = self._parse_tool_call(response_text)

            if tool_call is None:
                # No tool call and no final answer - might be malformed response
                log.warning("LLM response has no tool call and no final answer")
                # Try to parse as final answer anyway
                return self._parse_final_answer(response_text, evidence_bundle, full_interaction)

            # Execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            log.debug(f"Executing tool: {tool_name}({tool_args})")

            observation = self._execute_tool(tool_name, tool_args)

            log.debug(f"Tool result: {observation[:200]}...")

            # Append observation to conversation
            conversation += f"\n\n{response_text}\nObservation: {observation}"
            full_interaction.append(f"Observation: {observation}")

        # Max iterations reached without final answer
        log.warning(f"ReAct loop reached max iterations ({self.MAX_ITERATIONS})")
        return AdjudicationResult(
            outcome=AdjudicationOutcome.ERROR,
            error_message=f"Reached maximum iterations ({self.MAX_ITERATIONS}) without final answer",
            prompt_json=json.dumps({"system": system_prompt, "user": user_prompt}, indent=2),
            response_json="\n\n".join(full_interaction),
        )

    def _parse_tool_call(self, text: str) -> dict[str, Any] | None:
        """Parse a tool call from LLM response text.

        Expected format:
            Action: tool_name
            Action Input: {"arg": "value"}

        Returns:
            Dict with 'name' and 'args', or None if no tool call found
        """
        # Look for Action: and Action Input: patterns
        action_match = re.search(r"Action:\s*(\w+)", text, re.IGNORECASE)
        if not action_match:
            return None

        tool_name = action_match.group(1).strip()

        # Look for Action Input (JSON or plain text)
        input_match = re.search(
            r"Action Input:\s*({.*?}|\{[\s\S]*?\}|[^\n]+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if not input_match:
            log.warning(f"Found Action but no Action Input for tool: {tool_name}")
            return None

        input_str = input_match.group(1).strip()

        # Try to parse as JSON
        try:
            tool_args = json.loads(input_str)
        except json.JSONDecodeError:
            # If not JSON, treat as a single string argument
            # Common pattern: Action Input: "search query here"
            tool_args = {"query" if "search" in tool_name.lower() else "url": input_str.strip('"')}

        return {"name": tool_name, "args": tool_args}

    def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool and return the observation."""
        # Web tools
        if tool_name == "web_search":
            return self._tool_web_search(tool_args.get("query", ""))
        elif tool_name == "web_fetch":
            return self._tool_web_fetch(tool_args.get("url", ""))

        # Music search tools
        elif tool_name == "search_artist":
            return self._tool_search_artist(tool_args.get("query", ""))
        elif tool_name == "get_artist":
            return self._tool_get_artist(tool_args.get("mbid", ""))
        elif tool_name == "search_recording":
            return self._tool_search_recording(tool_args.get("title", ""), tool_args.get("artist"))
        elif tool_name == "get_recording":
            return self._tool_get_recording(tool_args.get("mbid", ""))
        elif tool_name == "search_release_group":
            return self._tool_search_release_group(
                tool_args.get("title", ""), tool_args.get("artist")
            )
        elif tool_name == "get_release_group":
            return self._tool_get_release_group(tool_args.get("mbid", ""))
        elif tool_name == "get_releases_in_group":
            return self._tool_get_releases_in_group(tool_args.get("rg_mbid", ""))
        else:
            return f"Error: Unknown tool '{tool_name}'"

    def _tool_web_search(self, query: str) -> str:
        """Execute web_search tool."""
        if not query:
            return "Error: Missing 'query' argument for web_search"

        if self.web_search_tool is None:
            return "Error: Web search is not available (SearxNG not configured)"

        try:
            log.debug(f"Tool execution: web_search('{query}')")
            response = self.web_search_tool.search_web(query, max_results=5)

            if not response.results:
                return f"No web results found for: {query}"

            # Format results
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

    def _tool_search_artist(self, query: str) -> str:
        """Execute search_artist tool."""
        if not query:
            return "Error: Missing 'query' argument for search_artist"

        try:
            log.debug(f"Tool execution: search_artist('{query}')")
            response = self.music_search_tool.search_artist(query)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Artist search error: {e}")
            return f"Error searching artists: {e}"

    def _tool_get_artist(self, mbid: str) -> str:
        """Execute get_artist tool."""
        if not mbid:
            return "Error: Missing 'mbid' argument for get_artist"

        try:
            log.debug(f"Tool execution: get_artist('{mbid}')")
            response = self.music_search_tool.search_artist(mbid, by_mbid=True)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Artist lookup error: {e}")
            return f"Error getting artist: {e}"

    def _tool_search_recording(self, title: str, artist: str | None) -> str:
        """Execute search_recording tool."""
        if not title:
            return "Error: Missing 'title' argument for search_recording"

        try:
            log.debug(f"Tool execution: search_recording('{title}', artist='{artist}')")
            response = self.music_search_tool.search_recording(title, artist=artist)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Recording search error: {e}")
            return f"Error searching recordings: {e}"

    def _tool_get_recording(self, mbid: str) -> str:
        """Execute get_recording tool."""
        if not mbid:
            return "Error: Missing 'mbid' argument for get_recording"

        try:
            log.debug(f"Tool execution: get_recording('{mbid}')")
            response = self.music_search_tool.search_recording(mbid, by_mbid=True)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Recording lookup error: {e}")
            return f"Error getting recording: {e}"

    def _tool_search_release_group(self, title: str, artist: str | None) -> str:
        """Execute search_release_group tool."""
        if not title:
            return "Error: Missing 'title' argument for search_release_group"

        try:
            log.debug(f"Tool execution: search_release_group('{title}', artist='{artist}')")
            response = self.music_search_tool.search_release_group(title, artist=artist)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Release group search error: {e}")
            return f"Error searching release groups: {e}"

    def _tool_get_release_group(self, mbid: str) -> str:
        """Execute get_release_group tool."""
        if not mbid:
            return "Error: Missing 'mbid' argument for get_release_group"

        try:
            log.debug(f"Tool execution: get_release_group('{mbid}')")
            response = self.music_search_tool.search_release_group(mbid, by_mbid=True)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Release group lookup error: {e}")
            return f"Error getting release group: {e}"

    def _tool_get_releases_in_group(self, rg_mbid: str) -> str:
        """Execute get_releases_in_group tool."""
        if not rg_mbid:
            return "Error: Missing 'rg_mbid' argument for get_releases_in_group"

        try:
            log.debug(f"Tool execution: get_releases_in_group('{rg_mbid}')")
            response = self.music_search_tool.get_release_group_releases(rg_mbid)
            return response.to_context_string()
        except Exception as e:
            log.error(f"Releases in group lookup error: {e}")
            return f"Error getting releases: {e}"

    def _tool_web_fetch(self, url: str) -> str:
        """Execute web_fetch tool."""
        if not url:
            return "Error: Missing 'url' argument for web_fetch"

        if not url.startswith(("http://", "https://")):
            return f"Error: Invalid URL (must start with http:// or https://): {url}"

        try:
            import httpx

            log.debug(f"Tool execution: web_fetch('{url}')")

            # Fetch the URL with timeout
            client = httpx.Client(timeout=10.0, follow_redirects=True)
            response = client.get(url)
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("content-type", "")

            # Only process HTML/text content
            if "html" not in content_type and "text" not in content_type:
                return f"Error: URL returned non-text content: {content_type}"

            # Extract text (simple approach)
            content = response.text

            # Simple HTML tag stripping
            content = re.sub(
                r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(
                r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE
            )
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()

            # Truncate to reasonable length
            if len(content) > 2000:
                content = content[:2000] + "...[truncated]"

            return f"Content from {url}:\n\n{content}"

        except Exception as e:
            log.error(f"Web fetch error for {url}: {e}")
            return f"Error fetching URL: {e}"

    def _parse_final_answer(
        self,
        response_text: str,
        evidence_bundle: dict[str, Any],
        full_interaction: list[str],
    ) -> AdjudicationResult:
        """Parse the final answer from LLM response."""
        try:
            # Extract JSON from response
            content = response_text

            # Look for JSON in code blocks
            if "```json" in content:
                json_start = content.index("```json") + 7
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.index("```") + 3
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()
            else:
                # Try to extract JSON object by finding { and }
                if "{" in content and "}" in content:
                    json_start = content.index("{")
                    brace_count = 0
                    json_end = json_start
                    for i in range(json_start, len(content)):
                        if content[i] == "{":
                            brace_count += 1
                        elif content[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break
                    content = content[json_start:json_end].strip()

            # Strip JSON comments
            lines = content.split("\n")
            cleaned_lines = []
            for line in lines:
                if "//" in line:
                    line = line[: line.index("//")].rstrip()
                if line.strip():
                    cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)

            # Parse JSON
            data = json.loads(content)

            crg_mbid = data.get("crg_mbid")
            rr_mbid = data.get("rr_mbid")
            confidence = float(data.get("confidence", 0.0))
            rationale = data.get("rationale", "")

            # Determine outcome based on confidence
            if confidence >= self.config.auto_accept_threshold:
                outcome = AdjudicationOutcome.ACCEPTED
            elif confidence >= self.config.review_threshold:
                outcome = AdjudicationOutcome.REVIEW
            else:
                outcome = AdjudicationOutcome.REJECTED

            return AdjudicationResult(
                outcome=outcome,
                crg_mbid=crg_mbid,
                rr_mbid=rr_mbid,
                confidence=confidence,
                rationale=rationale,
                model_id=self.config.model_id,
                prompt_json=json.dumps({"interaction": full_interaction}, indent=2),
                response_json=response_text,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"Failed to parse final answer: {e}")
            log.debug(f"Response content: {response_text}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=f"Failed to parse final answer: {e}",
                confidence=0.0,
                rationale=response_text[:500],
                response_json="\n\n".join(full_interaction),
            )

    def _build_prompt(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> str:
        """Build the user prompt from evidence bundle.

        This is similar to the agent_adjudicator version but adapted for ReAct.
        """
        lines = ["# Recording Evidence\n"]

        # Artist info
        artist = evidence_bundle.get("artist", {})
        if artist:
            lines.append("## Artist")
            lines.append(f"- Name: {artist.get('name', 'Unknown')}")
            if artist.get("mb_artist_id"):
                lines.append(f"- MBID: {artist['mb_artist_id']}")
            if artist.get("begin_area_country"):
                lines.append(f"- Origin Country: {artist['begin_area_country']}")
            lines.append("")

        # Recording title
        recording_title = evidence_bundle.get("recording_title", "")
        if recording_title:
            lines.append(f"## Recording Title: {recording_title}\n")

        # Release group candidates
        candidates = evidence_bundle.get("recording_candidates", [])
        if candidates:
            lines.append("## Release Group Candidates\n")

            rg_map: dict[str, dict[str, Any]] = {}
            for rec in candidates:
                for rg in rec.get("rg_candidates", []):
                    rg_mbid = rg.get("mb_rg_id", "")
                    if rg_mbid and rg_mbid not in rg_map:
                        rg_map[rg_mbid] = rg

            origin_country = artist.get("country", "")
            for i, (rg_mbid, rg) in enumerate(rg_map.items(), 1):
                title = rg.get("title", "Unknown")
                is_compilation = "Compilation" in rg.get("secondary_types", [])

                if is_compilation:
                    lines.append(f"{i}. **{title}** ⚠️ COMPILATION")
                else:
                    lines.append(f"{i}. **{title}**")

                lines.append(f"   - RG MBID: {rg_mbid}")
                lines.append(f"   - Type: {rg.get('primary_type', 'Unknown')}")
                if rg.get("secondary_types"):
                    lines.append(f"   - Secondary: {', '.join(rg['secondary_types'])}")
                if rg.get("first_release_date"):
                    lines.append(f"   - First Release: {rg['first_release_date']}")

                releases = rg.get("releases", [])
                if releases:
                    lines.append("   - Sample Releases:")
                    for r in releases[:3]:
                        parts = []
                        if r.get("date"):
                            parts.append(r["date"])
                        if r.get("country"):
                            country = r["country"]
                            if country == origin_country:
                                parts.append(f"[{country}] ✓ ORIGIN")
                            else:
                                parts.append(f"[{country}]")
                        if r.get("mb_release_id"):
                            parts.append(f"(MBID: {r['mb_release_id']})")
                        lines.append(f"     - {' '.join(parts)}")
                lines.append("")

        # Timeline facts
        timeline = evidence_bundle.get("timeline_facts", {})
        if timeline:
            lines.append("## Timeline Analysis")

            earliest_album = timeline.get("earliest_album_date")
            earliest_single = timeline.get("earliest_single_ep_date")

            if earliest_single:
                lines.append(f"- Earliest Single/EP: {earliest_single}")
            if earliest_album:
                lines.append(f"- Earliest Album: {earliest_album}")

            if earliest_single and earliest_album:
                try:
                    from datetime import datetime

                    def parse_date(date_str: str) -> datetime:
                        if len(date_str) == 7:
                            return datetime.strptime(date_str, "%Y-%m")
                        else:
                            return datetime.strptime(date_str, "%Y-%m-%d")

                    single_date = parse_date(earliest_single)
                    album_date = parse_date(earliest_album)
                    gap_days = (album_date - single_date).days

                    lines.append(f"\n**GAP: {gap_days} days**")
                    if gap_days <= 90 and gap_days > 0:
                        lines.append("→ Single is within 90-day lead window → **PREFER ALBUM**")
                    elif gap_days > 90:
                        lines.append("→ Single is >90 days before album → Prefer single")
                    elif gap_days < 0:
                        lines.append("→ Album came first")
                except Exception:
                    pass

            lines.append("")

        # Missing facts
        if decision_trace:
            missing_facts = decision_trace.get("missing_facts", [])
            if missing_facts:
                lines.append("## Why INDETERMINATE")
                for fact in missing_facts:
                    lines.append(f"- {fact}")
                lines.append("")

        # Instructions
        lines.append("## Your Task\n")
        lines.append("Determine the Canonical Release Group (CRG) and Representative Release (RR).")
        lines.append("")
        lines.append(
            "Use the Thought/Action/Observation pattern if you need additional information from the web."
        )
        lines.append("")
        lines.append("When ready, provide your Final Answer as JSON.")

        return "\n".join(lines)
