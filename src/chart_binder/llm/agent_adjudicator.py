"""LangChain agent-based adjudicator for chart-binder.

Replaces the ReAct prompt-based approach with native LangChain tool calling
and structured output. Uses create_agent() for agent creation and
with_structured_output() for final answer validation.

Key improvements over ReAct:
- Native tool calling instead of prompt-based parsing
- Pydantic schema validation for structured output
- Validation/repair loop for robust error handling
- Better observability via LangChain callbacks
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from chart_binder.llm.adjudicator import (
    SYSTEM_PROMPT_V1,
    AdjudicationOutcome,
    AdjudicationResult,
)
from chart_binder.llm.langchain_provider import create_chat_model
from chart_binder.llm.search_tool import SearchTool
from chart_binder.llm.structured_output import (
    StructuredOutputStrategy,
    build_retry_feedback,
    format_validation_errors,
    strip_null_values,
    with_structured_output,
)
from chart_binder.llm.tools import create_music_tools

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool

    from chart_binder.config import LLMConfig
    from chart_binder.llm.searxng import SearxNGSearchTool
    from chart_binder.musicbrainz import MusicBrainzClient

log = logging.getLogger(__name__)


class SyncMusicBrainzClient:
    """Sync wrapper for MusicBrainzClient's async methods.

    SearchTool expects sync methods, but MusicBrainzClient has async methods.
    This wrapper runs async methods in an event loop for sync access.
    """

    def __init__(self, mb_client: MusicBrainzClient):
        self._client = mb_client

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            # No event loop exists
            return asyncio.run(coro)

    def search_recordings(
        self,
        query: str | None = None,
        isrc: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        limit: int = 25,
    ) -> list[Any]:
        """Sync wrapper for search_recordings."""
        return self._run_async(
            self._client.search_recordings(
                query=query, isrc=isrc, artist=artist, title=title, limit=limit
            )
        )

    def search_artists(self, query: str, limit: int = 25) -> list[Any]:
        """Sync wrapper for search_artists."""
        # MusicBrainzClient doesn't have search_artists, but SearchTool expects it
        # We'll implement a search using the generic search endpoint
        return self._run_async(self._search_artists_async(query, limit))

    async def _search_artists_async(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search for artists by name."""
        params = {"query": f'artist:"{query}"', "limit": str(limit), "fmt": "json"}
        await self._client._rate_limit()

        url = f"{self._client.BASE_URL}/artist"

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            cached = self._client.cache.get(cache_key)
            if cached:
                return cached.json().get("artists", [])

        response = await self._client._client.get(url, params=params)
        response.raise_for_status()

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            self._client.cache.put(cache_key, response)

        return response.json().get("artists", [])

    def search_release_groups(
        self, title: str, artist: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Sync wrapper for search_release_groups."""
        return self._run_async(self._search_release_groups_async(title, artist, limit))

    async def _search_release_groups_async(
        self, title: str, artist: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Search for release groups by title and optional artist."""
        query_parts = [f'releasegroup:"{title}"']
        if artist:
            query_parts.append(f'artist:"{artist}"')
        query = " AND ".join(query_parts)

        params = {"query": query, "limit": str(limit), "fmt": "json"}
        await self._client._rate_limit()

        url = f"{self._client.BASE_URL}/release-group"

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            cached = self._client.cache.get(cache_key)
            if cached:
                return cached.json().get("release-groups", [])

        response = await self._client._client.get(url, params=params)
        response.raise_for_status()

        if self._client.cache:
            cache_key = f"{url}?{self._client._make_cache_key(params)}"
            self._client.cache.put(cache_key, response)

        return response.json().get("release-groups", [])


class AdjudicationResponse(BaseModel):
    """Structured response from the adjudication agent.

    This schema is used with with_structured_output() to ensure
    the LLM returns a valid, parseable response.
    """

    crg_mbid: str | None = Field(
        default=None,
        description="Canonical Release Group MBID (UUID format)",
    )
    rr_mbid: str | None = Field(
        default=None,
        description="Representative Release MBID within the CRG",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 (no confidence) to 1.0 (very confident)",
    )
    rationale: str = Field(
        default="",
        description="Concise one-line explanation of the decision",
    )


# Agent system prompt - combines domain rules with tool usage instructions
AGENT_SYSTEM_PROMPT = """You are a music metadata expert specialized in determining the canonical release for recordings.

Your task is to analyze evidence about a recording and determine:
1. The Canonical Release Group (CRG): The authoritative release group where this recording first appeared
2. The Representative Release (RR): The specific release within the CRG to use for metadata

CRITICAL DECISION RULES (apply in order):

1. **Compilations are never canonical** - Exclude any release group with "Secondary: Compilation"

2. **Lead Single Window Rule (90 days)**:
   - If the earliest single/EP is within 90 days BEFORE an album, choose the ALBUM as CRG
   - Only if the single is >90 days before the album should you choose the single
   - Example: Single Oct 1974, Album Nov 1 1974 = ~30 days = Choose ALBUM

3. **Soundtrack Exception**: If the recording was created specifically for a soundtrack, the soundtrack is CRG

4. **Live vs Studio**: Live recordings are only CRG if no studio version exists

5. **Remixes**: Link to the original single/EP release, not remix compilations

6. **Representative Release Selection**:
   - Within the chosen CRG, select the specific release (RR)
   - PREFER releases from the artist's origin country
   - If origin country unavailable, prefer earliest release in the CRG

IMPORTANT - DECISION WORKFLOW:
1. FIRST: Analyze the evidence bundle provided - it usually contains all information needed
2. ONLY use tools if the evidence is truly unclear or missing critical dates
3. LIMIT tool calls to 2-3 maximum - don't get stuck searching
4. If web_fetch fails, proceed with available information
5. Make a decision even with incomplete data - assign lower confidence if uncertain

REASONING PROCESS:
1. Eliminate all compilations from candidates
2. Identify earliest single/EP and earliest album dates from the evidence
3. Calculate days between them - if ≤90 days, prefer album
4. Within chosen CRG, find releases matching origin country
5. State your confidence (0.0-1.0)

STOP CONDITION: Once you can identify a CRG and RR from the evidence (even with some uncertainty), provide your final answer. Do not keep searching indefinitely."""


class AgentAdjudicator:
    """LangChain agent-based adjudicator.

    Uses native LangChain tool calling and structured output for
    more reliable adjudication than the ReAct prompt approach.
    """

    MAX_AGENT_ITERATIONS = 15
    MAX_VALIDATION_RETRIES = 3

    def __init__(
        self,
        config: LLMConfig,
        search_tool: SearchTool | None = None,
        web_search_tool: SearxNGSearchTool | None = None,
        db_path: str | None = None,
    ):
        self.config = config
        self._web_search_tool = web_search_tool
        self._model: BaseChatModel | None = None
        self._tools: list[BaseTool] | None = None

        # Initialize SearchTool with database and MusicBrainz client if provided
        if search_tool is not None:
            self._search_tool = search_tool
        elif db_path:
            from pathlib import Path

            from chart_binder.http_cache import HttpCache
            from chart_binder.musicbrainz import MusicBrainzClient
            from chart_binder.musicgraph import MusicGraphDB

            db = MusicGraphDB(Path(db_path))

            # Create MusicBrainz client with cache for API searches
            cache_dir = Path(db_path).parent / "cache" / "musicbrainz"
            cache_dir.mkdir(parents=True, exist_ok=True)
            mb_cache = HttpCache(cache_dir, ttl_seconds=3600)
            async_mb_client = MusicBrainzClient(cache=mb_cache, rate_limit_per_sec=1.0)
            sync_mb_client = SyncMusicBrainzClient(async_mb_client)

            self._search_tool = SearchTool(music_graph_db=db, mb_client=sync_mb_client)
            log.debug(
                "SearchTool initialized with database and MusicBrainz client: %s", db_path
            )
        else:
            self._search_tool = SearchTool()
            log.warning("SearchTool initialized without database - searches will be limited")

    @property
    def model(self) -> BaseChatModel:
        """Get or create the chat model."""
        if self._model is None:
            self._model = create_chat_model(
                provider_name=str(self.config.provider),
                model=self.config.model_id,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_s,
                host=self.config.ollama_base_url,
            )
        return self._model

    @property
    def tools(self) -> list[BaseTool]:
        """Get or create the tool list."""
        if self._tools is None:
            self._tools = create_music_tools(
                self._search_tool,
                web_search=self._web_search_tool,
            )
        return self._tools

    def adjudicate(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> AdjudicationResult:
        """Adjudicate using the LangChain agent.

        Synchronous wrapper around async implementation.

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

        # Run async adjudication
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._adjudicate_async(evidence_bundle, decision_trace))

    async def _adjudicate_async(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> AdjudicationResult:
        """Async adjudication implementation.

        Uses a two-phase approach:
        1. Agent phase: Use tools to gather information
        2. Structured output phase: Generate validated final answer
        """
        adjudication_id = str(uuid.uuid4())
        start_time = time.time()

        # Build the evidence prompt
        user_prompt = self._build_prompt(evidence_bundle, decision_trace)

        log.debug("Agent Adjudication:")
        log.debug("=" * 70)
        log.debug("SYSTEM:")
        log.debug(AGENT_SYSTEM_PROMPT)
        log.debug("=" * 70)
        log.debug("USER:")
        log.debug(user_prompt)
        log.debug("=" * 70)

        try:
            # Phase 1: Run agent with tools to gather information
            agent_result = await self._run_agent_phase(user_prompt)

            # Phase 2: Get structured output with validation
            response, tokens_used = await self._get_structured_response(
                user_prompt,
                agent_result.get("context", ""),
                evidence_bundle,
            )

            # Determine outcome based on confidence
            if response.confidence >= self.config.auto_accept_threshold:
                outcome = AdjudicationOutcome.ACCEPTED
            elif response.confidence >= self.config.review_threshold:
                outcome = AdjudicationOutcome.REVIEW
            else:
                outcome = AdjudicationOutcome.REJECTED

            return AdjudicationResult(
                outcome=outcome,
                crg_mbid=response.crg_mbid,
                rr_mbid=response.rr_mbid,
                confidence=response.confidence,
                rationale=response.rationale,
                model_id=self.config.model_id,
                prompt_template_version="agent-v1",
                adjudication_id=adjudication_id,
                created_at=start_time,
                prompt_json=json.dumps({"system": AGENT_SYSTEM_PROMPT, "user": user_prompt}),
                response_json=response.model_dump_json(),
            )

        except Exception as e:
            log.error("Agent adjudication failed: %s", e)
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=str(e),
                adjudication_id=adjudication_id,
                created_at=start_time,
            )

    async def _run_agent_phase(self, user_prompt: str) -> dict[str, Any]:
        """Run the agent phase with tool calling.

        Uses LangChain's create_agent for native tool calling.

        Args:
            user_prompt: The evidence-based user prompt

        Returns:
            Dict with gathered context and agent messages
        """
        try:
            from langchain.agents import create_agent
        except ImportError:
            # Fall back to direct tool binding if create_agent not available
            log.warning("create_agent not available, using direct tool binding")
            return await self._run_simple_agent(user_prompt)

        # Create agent with tools
        agent = create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        # Run the agent
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": self.MAX_AGENT_ITERATIONS},
        )

        # Extract context from agent run
        messages = result.get("messages", [])
        context_parts = []

        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    context_parts.append(str(msg.content))

        return {
            "context": "\n\n".join(context_parts),
            "messages": messages,
        }

    async def _run_simple_agent(self, user_prompt: str) -> dict[str, Any]:
        """Simple agent fallback using tool binding.

        Used when create_agent is not available. Runs a simpler
        loop that binds tools to the model.

        Args:
            user_prompt: The evidence-based user prompt

        Returns:
            Dict with gathered context
        """
        # Bind tools to model
        model_with_tools = self.model.bind_tools(self.tools)

        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        context_parts = []

        for _ in range(self.MAX_AGENT_ITERATIONS):
            response = await model_with_tools.ainvoke(messages)
            messages.append(response)

            # Check for tool calls
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})

                    # Find and execute the tool
                    result = await self._execute_tool(tool_name, tool_args)
                    context_parts.append(f"[{tool_name}]: {result}")

                    # Add tool result to conversation
                    from langchain_core.messages import ToolMessage

                    messages.append(
                        ToolMessage(
                            content=result,
                            tool_call_id=tool_call.get("id", ""),
                        )
                    )
            else:
                # No more tool calls - agent is done
                if response.content:
                    context_parts.append(str(response.content))
                break

        return {"context": "\n\n".join(context_parts)}

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            Tool result as string
        """
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    # Tools are synchronous, but wrapped for async invocation
                    result = tool.invoke(tool_args)
                    return str(result)
                except Exception as e:
                    log.error("Tool %s failed: %s", tool_name, e)
                    return json.dumps({"result": "error", "error": str(e)})

        return json.dumps({"result": "error", "error": f"Unknown tool: {tool_name}"})

    async def _get_structured_response(
        self,
        user_prompt: str,
        agent_context: str,
        evidence_bundle: dict[str, Any],
    ) -> tuple[AdjudicationResponse, int]:
        """Get structured response with validation/repair loop.

        Args:
            user_prompt: Original evidence prompt
            agent_context: Context gathered by agent
            evidence_bundle: Original evidence for validation

        Returns:
            Tuple of (validated response, tokens used)
        """
        # Create model with structured output
        structured_model = with_structured_output(
            self.model,
            AdjudicationResponse,
            strategy=StructuredOutputStrategy.JSON_MODE,
            provider_name=str(self.config.provider),
        )

        # Build messages with agent context
        system_content = SYSTEM_PROMPT_V1
        user_content = user_prompt
        if agent_context:
            user_content += f"\n\n## Additional Context from Search\n{agent_context}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]

        total_tokens = 0
        last_errors: list[str] = []

        for attempt in range(1, self.MAX_VALIDATION_RETRIES + 1):
            log.debug("Structured output attempt %d/%d", attempt, self.MAX_VALIDATION_RETRIES)

            result = await structured_model.ainvoke(messages)

            # Extract tokens from result
            total_tokens += self._extract_tokens(result)

            # Handle Pydantic model result
            if isinstance(result, AdjudicationResponse):
                # Validate against evidence bundle
                validation_errors = self._validate_response(result, evidence_bundle)
                if not validation_errors:
                    return result, total_tokens

                # Add validation errors for retry
                last_errors = validation_errors
                if attempt < self.MAX_VALIDATION_RETRIES:
                    feedback = build_retry_feedback(validation_errors)
                    messages.append(HumanMessage(content=feedback))
                continue

            # Handle dict result (needs validation)
            if isinstance(result, dict):
                try:
                    cleaned = strip_null_values(result)
                    response = AdjudicationResponse.model_validate(cleaned)

                    # Validate against evidence bundle
                    validation_errors = self._validate_response(response, evidence_bundle)
                    if not validation_errors:
                        return response, total_tokens

                    last_errors = validation_errors
                    if attempt < self.MAX_VALIDATION_RETRIES:
                        feedback = build_retry_feedback(validation_errors)
                        messages.append(HumanMessage(content=feedback))
                    continue

                except ValidationError as e:
                    last_errors = [format_validation_errors(e.errors())]
                    if attempt < self.MAX_VALIDATION_RETRIES:
                        feedback = build_retry_feedback(last_errors)
                        messages.append(HumanMessage(content=feedback))
                    continue

            # Unexpected result type
            last_errors = [f"Unexpected result type: {type(result).__name__}"]
            if attempt < self.MAX_VALIDATION_RETRIES:
                messages.append(
                    HumanMessage(
                        content="Please provide a valid JSON response with crg_mbid, rr_mbid, confidence, and rationale fields."
                    )
                )

        # All retries exhausted - return best effort
        log.warning(
            "Structured output failed after %d attempts: %s",
            self.MAX_VALIDATION_RETRIES,
            last_errors,
        )
        return AdjudicationResponse(
            crg_mbid=None,
            rr_mbid=None,
            confidence=0.0,
            rationale=f"Failed to get valid response: {'; '.join(last_errors)}",
        ), total_tokens

    def _validate_response(
        self,
        response: AdjudicationResponse,
        evidence_bundle: dict[str, Any],
    ) -> list[str]:
        """Validate response against evidence bundle.

        Checks that selected MBIDs exist in the evidence.

        Args:
            response: Adjudication response to validate
            evidence_bundle: Evidence bundle with valid MBIDs

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Collect valid MBIDs from evidence
        valid_crg_mbids = set()
        valid_rr_mbids: dict[str, set[str]] = {}

        for rec in evidence_bundle.get("recording_candidates", []):
            for rg in rec.get("rg_candidates", []):
                rg_id = rg.get("mb_rg_id")
                if rg_id:
                    valid_crg_mbids.add(rg_id)
                    valid_rr_mbids[rg_id] = set()
                    for release in rg.get("releases", []):
                        if release.get("mb_release_id"):
                            valid_rr_mbids[rg_id].add(release["mb_release_id"])

        # Validate CRG MBID
        if response.crg_mbid and response.crg_mbid not in valid_crg_mbids:
            errors.append(
                f"Selected CRG MBID '{response.crg_mbid}' not in evidence bundle. "
                f"Valid options: {', '.join(list(valid_crg_mbids)[:5])}"
            )

        # Validate RR MBID
        if response.crg_mbid and response.rr_mbid:
            valid_releases = valid_rr_mbids.get(response.crg_mbid, set())
            if response.rr_mbid not in valid_releases:
                errors.append(
                    f"Selected RR MBID '{response.rr_mbid}' not in CRG '{response.crg_mbid}'. "
                    f"Valid options: {', '.join(list(valid_releases)[:5])}"
                )

        return errors

    def _extract_tokens(self, result: Any) -> int:
        """Extract token count from result.

        Args:
            result: Response from model invocation

        Returns:
            Total tokens used, or 0 if not available
        """
        # Check usage_metadata (Ollama, newer providers)
        if hasattr(result, "usage_metadata"):
            usage = getattr(result, "usage_metadata", None)
            if usage:
                tokens = usage.get("total_tokens")
                if tokens is not None:
                    return int(tokens)

        # Check response_metadata (OpenAI)
        if hasattr(result, "response_metadata"):
            metadata = getattr(result, "response_metadata", None) or {}
            if "token_usage" in metadata:
                tokens = metadata["token_usage"].get("total_tokens")
                if tokens is not None:
                    return int(tokens)

        return 0

    def _build_prompt(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> str:
        """Build the user prompt from evidence bundle.

        Same format as the original adjudicator for consistency.
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

            origin_country = artist.get("begin_area_country", "")
            for i, (rg_mbid, rg) in enumerate(rg_map.items(), 1):
                title = rg.get("title", "Unknown")
                is_compilation = "Compilation" in rg.get("secondary_types", [])

                if is_compilation:
                    lines.append(f"{i}. **{title}** [COMPILATION - NOT CANONICAL]")
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
                    lines.append("   - Releases:")
                    for r in releases[:5]:
                        parts = []
                        if r.get("date"):
                            parts.append(r["date"])
                        if r.get("country"):
                            country = r["country"]
                            if country == origin_country:
                                parts.append(f"[{country}] ORIGIN")
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
                            return datetime.strptime(date_str[:10], "%Y-%m-%d")

                    single_date = parse_date(earliest_single)
                    album_date = parse_date(earliest_album)
                    gap_days = (album_date - single_date).days

                    lines.append(f"\n**GAP: {gap_days} days**")
                    if 0 < gap_days <= 90:
                        lines.append("-> Single within 90-day lead window -> **PREFER ALBUM**")
                    elif gap_days > 90:
                        lines.append("-> Single >90 days before album -> Prefer single")
                    elif gap_days < 0:
                        lines.append("-> Album released first")
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
        lines.append("## Task")
        lines.append("Determine the Canonical Release Group (CRG) and Representative Release (RR).")
        lines.append("Use the available tools if you need additional information.")
        lines.append("Then provide your final answer with confidence score and rationale.")

        return "\n".join(lines)


## Tests


def test_adjudication_response_schema():
    """Test AdjudicationResponse schema validation."""
    response = AdjudicationResponse(
        crg_mbid="test-id",
        rr_mbid="test-release",
        confidence=0.85,
        rationale="Test rationale",
    )
    assert response.crg_mbid == "test-id"
    assert response.confidence == 0.85


def test_adjudication_response_defaults():
    """Test AdjudicationResponse default values."""
    response = AdjudicationResponse()
    assert response.crg_mbid is None
    assert response.rr_mbid is None
    assert response.confidence == 0.0
    assert response.rationale == ""


def test_adjudication_response_confidence_bounds():
    """Test confidence bounds validation."""
    import pytest

    with pytest.raises(ValidationError):
        AdjudicationResponse(confidence=1.5)

    with pytest.raises(ValidationError):
        AdjudicationResponse(confidence=-0.1)


def test_agent_adjudicator_disabled():
    """Test adjudicator when disabled."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(enabled=False)
    adjudicator = AgentAdjudicator(config)

    result = adjudicator.adjudicate({})
    assert result.outcome == AdjudicationOutcome.ERROR
    assert "disabled" in (result.error_message or "").lower()


def test_validate_response_valid():
    """Test response validation with valid MBIDs."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(enabled=True)
    adjudicator = AgentAdjudicator(config)

    evidence = {
        "recording_candidates": [
            {
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-123",
                        "releases": [{"mb_release_id": "rel-123"}],
                    }
                ]
            }
        ]
    }

    response = AdjudicationResponse(
        crg_mbid="rg-123",
        rr_mbid="rel-123",
        confidence=0.9,
    )

    errors = adjudicator._validate_response(response, evidence)
    assert errors == []


def test_validate_response_invalid_crg():
    """Test response validation with invalid CRG MBID."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(enabled=True)
    adjudicator = AgentAdjudicator(config)

    evidence = {
        "recording_candidates": [
            {
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-123",
                        "releases": [{"mb_release_id": "rel-123"}],
                    }
                ]
            }
        ]
    }

    response = AdjudicationResponse(
        crg_mbid="invalid-rg",
        confidence=0.9,
    )

    errors = adjudicator._validate_response(response, evidence)
    assert len(errors) == 1
    assert "invalid-rg" in errors[0]


def test_build_prompt():
    """Test prompt building."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(enabled=True)
    adjudicator = AgentAdjudicator(config)

    evidence = {
        "artist": {"name": "Test Artist", "begin_area_country": "US"},
        "recording_title": "Test Song",
        "recording_candidates": [
            {
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-123",
                        "title": "Test Album",
                        "primary_type": "Album",
                        "first_release_date": "2020-01-01",
                        "releases": [
                            {"mb_release_id": "rel-123", "date": "2020-01-01", "country": "US"}
                        ],
                    }
                ]
            }
        ],
    }

    prompt = adjudicator._build_prompt(evidence)
    assert "Test Artist" in prompt
    assert "Test Song" in prompt
    assert "Test Album" in prompt
    assert "rg-123" in prompt
    assert "US" in prompt
