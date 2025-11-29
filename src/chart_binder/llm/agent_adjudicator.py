"""Tool-calling LLM adjudicator using LangChain.

This module provides an LLM adjudicator that can use tools (web search, web fetch)
to gather additional information when making canonicalization decisions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from chart_binder.llm.adjudicator import (
    SYSTEM_PROMPT_V1,
    AdjudicationOutcome,
    AdjudicationResult,
)
from chart_binder.llm.tools import set_searxng_tool, web_fetch, web_search

if TYPE_CHECKING:
    from chart_binder.config import LLMConfig
    from chart_binder.llm.searxng import SearxNGSearchTool

log = logging.getLogger(__name__)


class AgentAdjudicator:
    """Tool-calling LLM adjudicator with web search and fetch capabilities.

    Uses LangChain to give the LLM access to tools for gathering additional
    information about recordings and releases.
    """

    def __init__(
        self,
        config: LLMConfig,
        search_tool: SearxNGSearchTool | None = None,
    ):
        self.config = config
        self.search_tool = search_tool

        # Set global search tool for tool functions
        set_searxng_tool(search_tool)

        # Initialize LLM based on provider
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
                base_url=config.ollama_base_url,
                # NOTE: Don't use format="json" with tools - it conflicts with tool calling
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        # Bind tools to the LLM
        tools = []
        if search_tool is not None:
            tools.append(web_search)
            tools.append(web_fetch)
            log.info(f"LLM tools enabled: {[t.name for t in tools]}")

        if tools:
            # OpenAI requires strict mode for structured outputs
            if config.provider == "openai":
                self.llm_with_tools = self.llm.bind_tools(tools, strict=True)
            else:
                self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            self.llm_with_tools = self.llm

    def adjudicate(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> AdjudicationResult:
        """Adjudicate using tool-calling LLM.

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

        # Build the prompt
        prompt = self._build_prompt(evidence_bundle, decision_trace)

        # Create messages (typed as list of BaseMessage for tool use)
        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT_V1),
            HumanMessage(content=prompt),
        ]

        # Log the prompt at DEBUG level
        log.debug("LLM Tool-Calling Adjudication:")
        log.debug("=" * 70)
        log.debug("SYSTEM:")
        log.debug(SYSTEM_PROMPT_V1)
        log.debug("=" * 70)
        log.debug("USER:")
        log.debug(prompt)
        log.debug("=" * 70)

        try:
            # Invoke the LLM (it will call tools if needed)
            response = self.llm_with_tools.invoke(messages)

            # Log the response
            log.debug("LLM Response:")
            log.debug("=" * 70)
            log.debug(f"Content: {response.content}")
            log.debug(f"Type: {type(response)}")
            if hasattr(response, "tool_calls"):
                log.debug(f"tool_calls: {response.tool_calls}")
            if hasattr(response, "invalid_tool_calls"):
                log.debug(f"invalid_tool_calls: {response.invalid_tool_calls}")
            if hasattr(response, "additional_kwargs"):
                log.debug(f"additional_kwargs: {response.additional_kwargs}")
            if hasattr(response, "response_metadata"):
                log.debug(f"response_metadata: {json.dumps(response.response_metadata, indent=2)}")
            log.debug("=" * 70)

            # If there are tool calls, execute them and get final response
            if hasattr(response, "tool_calls") and response.tool_calls:
                log.info(f"LLM requested {len(response.tool_calls)} tool calls")

                # Add the assistant's response to messages
                messages.append(response)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    log.debug(f"Executing tool: {tool_name}({tool_args})")

                    # Execute the tool
                    if tool_name == "web_search":
                        tool_result = web_search.invoke(tool_args)
                    elif tool_name == "web_fetch":
                        tool_result = web_fetch.invoke(tool_args)
                    else:
                        tool_result = f"Unknown tool: {tool_name}"

                    log.debug(f"Tool result: {tool_result[:200]}...")

                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage

                    messages.append(
                        ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"],
                        )
                    )

                # Get final response after tool use
                final_response = self.llm.invoke(messages)
                response_content = final_response.content

                log.debug("Final LLM Response after tool use:")
                log.debug("=" * 70)
                log.debug(response_content)
                log.debug("=" * 70)
            else:
                response_content = response.content

            # Convert response_content to string if it's a list
            if isinstance(response_content, list):
                response_content_str = json.dumps(response_content)
            else:
                response_content_str = str(response_content)

            # Parse the response
            result = self._parse_response(response_content_str, evidence_bundle)
            result.prompt_json = json.dumps({"system": SYSTEM_PROMPT_V1, "user": prompt}, indent=2)
            result.response_json = response_content_str
            result.model_id = self.config.model_id

            return result

        except Exception as e:
            log.error(f"LLM adjudication error: {e}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=str(e),
            )

    def _build_prompt(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> str:
        """Build the user prompt from evidence bundle."""
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

        # Release group candidates (simplified view)
        candidates = evidence_bundle.get("recording_candidates", [])
        if candidates:
            lines.append("## Release Group Candidates\n")

            # Collect all unique release groups
            rg_map: dict[str, dict[str, Any]] = {}
            for rec in candidates:
                for rg in rec.get("rg_candidates", []):
                    rg_mbid = rg.get("mb_rg_id", "")
                    if rg_mbid and rg_mbid not in rg_map:
                        rg_map[rg_mbid] = rg

            # Show release groups
            for i, (rg_mbid, rg) in enumerate(rg_map.items(), 1):
                lines.append(f"{i}. **{rg.get('title', 'Unknown')}**")
                lines.append(f"   - RG MBID: {rg_mbid}")
                lines.append(f"   - Type: {rg.get('primary_type', 'Unknown')}")
                if rg.get("secondary_types"):
                    lines.append(f"   - Secondary: {', '.join(rg['secondary_types'])}")
                if rg.get("first_release_date"):
                    lines.append(f"   - First Release: {rg['first_release_date']}")

                # Show a few releases
                releases = rg.get("releases", [])
                if releases:
                    lines.append("   - Sample Releases:")
                    for r in releases[:3]:
                        parts = []
                        if r.get("date"):
                            parts.append(r["date"])
                        if r.get("country"):
                            parts.append(f"[{r['country']}]")
                        if r.get("mb_release_id"):
                            parts.append(f"({r['mb_release_id'][:8]}...)")
                        lines.append(f"     - {' '.join(parts)}")
                lines.append("")

        # Timeline facts
        timeline = evidence_bundle.get("timeline_facts", {})
        if timeline:
            lines.append("## Timeline")
            if timeline.get("earliest_album_date"):
                lines.append(f"- Earliest Album: {timeline['earliest_album_date']}")
            if timeline.get("earliest_single_date"):
                lines.append(f"- Earliest Single/EP: {timeline['earliest_single_date']}")
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
        lines.append("You have access to these tools:")
        if self.search_tool:
            lines.append("- `web_search(query)`: Search the web for additional information")
            lines.append("- `web_fetch(url)`: Fetch content from a specific URL")
            lines.append("")
            lines.append(
                "Use these tools if you need more information to make a confident decision."
            )
            lines.append("")

        lines.append("**IMPORTANT**: Return your final answer as a JSON object (you may use tools first if needed):")
        lines.append("```json")
        lines.append("{")
        lines.append('  "crg_mbid": "selected release group MBID",')
        lines.append('  "rr_mbid": "selected release MBID within CRG",')
        lines.append('  "confidence": 0.0-1.0,')
        lines.append('  "rationale": "brief explanation"')
        lines.append("}")
        lines.append("```")

        return "\n".join(lines)

    def _parse_response(
        self,
        response_content: str,
        evidence_bundle: dict[str, Any],
    ) -> AdjudicationResult:
        """Parse LLM JSON response."""
        try:
            # Handle markdown code blocks
            content = response_content.strip()
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
                    # Find the matching closing brace
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

            # Strip JSON comments (// style) - they're not valid JSON
            lines = content.split("\n")
            cleaned_lines = []
            for line in lines:
                # Remove inline comments
                if "//" in line:
                    line = line[: line.index("//")].rstrip()
                if line.strip():  # Only add non-empty lines
                    cleaned_lines.append(line)
            content = "\n".join(cleaned_lines)

            # Parse JSON
            data = json.loads(content)

            crg_mbid = data.get("crg_mbid")
            rr_mbid = data.get("rr_mbid")
            confidence = float(data.get("confidence", 0.0))
            rationale = data.get("rationale", "")

            # Determine outcome
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
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning(f"Failed to parse LLM response: {e}")
            log.debug(f"Response content: {response_content}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=f"Failed to parse LLM response: {e}",
                confidence=0.0,
                rationale=response_content[:500],  # Include start of response
            )
