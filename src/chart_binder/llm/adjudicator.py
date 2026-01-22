"""LLM adjudicator for Chart-Binder.

Provides LLM-based adjudication for INDETERMINATE canonicalization decisions.
Uses structured prompts to ask the LLM to select the canonical release group
and representative release, with confidence scoring.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from chart_binder.llm.providers import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    ProviderRegistry,
)
from chart_binder.llm.search_tool import SearchTool

if TYPE_CHECKING:
    from chart_binder.config import LLMConfig

log = logging.getLogger(__name__)


class AdjudicationOutcome(StrEnum):
    """Outcome of LLM adjudication."""

    ACCEPTED = "accepted"  # High confidence, auto-accepted
    REVIEW = "review"  # Medium confidence, needs human review
    REJECTED = "rejected"  # Low confidence, keep INDETERMINATE
    ERROR = "error"  # LLM error or timeout


@dataclass
class AdjudicationResult:
    """Result of LLM adjudication for a decision."""

    outcome: AdjudicationOutcome
    crg_mbid: str | None = None
    rr_mbid: str | None = None
    confidence: float = 0.0
    rationale: str = ""
    model_id: str = ""
    prompt_template_version: str = "v1"
    error_message: str | None = None
    adjudication_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # Audit trail
    prompt_json: str = ""
    response_json: str = ""


SYSTEM_PROMPT_V1 = """\
You are a music metadata expert. An automated system could not determine the \
canonical release for a recording and is asking for your judgment.

Given the evidence about a recording, determine:
- Canonical Release Group (CRG): The release group where this recording first \
appeared as an official release
- Representative Release (RR): A specific release within that group (prefer \
artist's origin country if known, otherwise earliest)

Use your music knowledge to make the best determination from the evidence provided.

Respond in valid JSON:
{
  "crg_mbid": "the release group ID you selected",
  "rr_mbid": "the release ID within the CRG",
  "confidence": 0.0 to 1.0,
  "rationale": "concise one-line explanation"
}

If you cannot determine the answer with reasonable confidence, set confidence \
below 0.60 and explain why."""


class LLMAdjudicator:
    """LLM-based adjudicator for INDETERMINATE decisions.

    Uses an LLM to analyze evidence bundles and make canonical
    release group selections when deterministic rules fail.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        provider: LLMProvider | None = None,
        search_tool: SearchTool | None = None,
    ):
        if config is None:
            from chart_binder.config import LLMConfig

            config = LLMConfig()
        self.config = config
        self._provider = provider
        self._search_tool = search_tool
        self._registry = ProviderRegistry()

    @property
    def provider(self) -> LLMProvider:
        """Get or create the LLM provider."""
        if self._provider is None:
            self._provider = self._registry.create_from_config(
                {
                    "provider": str(self.config.provider),
                    "model_id": self.config.model_id,
                    "ollama_base_url": self.config.ollama_base_url,
                    "api_key_env": self.config.api_key_env,
                }
            )
        return self._provider

    @property
    def search_tool(self) -> SearchTool:
        """Get or create the search tool."""
        if self._search_tool is None:
            self._search_tool = SearchTool()
        return self._search_tool

    def adjudicate(
        self,
        evidence_bundle: dict[str, Any],
        decision_trace: dict[str, Any] | None = None,
    ) -> AdjudicationResult:
        """Adjudicate an INDETERMINATE decision using LLM.

        Args:
            evidence_bundle: Evidence bundle from the resolver
            decision_trace: Optional decision trace with missing facts

        Returns:
            AdjudicationResult with CRG/RR selection and confidence
        """
        if not self.config.enabled:
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message="LLM adjudication is disabled",
            )

        # Build the prompt
        prompt = self._build_prompt(evidence_bundle, decision_trace)
        prompt_json = json.dumps(
            {"system": SYSTEM_PROMPT_V1, "user": prompt},
            indent=2,
        )

        # Log the full prompt at DEBUG level
        log.debug("LLM Adjudication Prompt:")
        log.debug("=" * 70)
        log.debug("SYSTEM PROMPT:")
        log.debug(SYSTEM_PROMPT_V1)
        log.debug("=" * 70)
        log.debug("USER PROMPT:")
        log.debug(prompt)
        log.debug("=" * 70)

        # Call the LLM
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT_V1),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = self.provider.chat(
                messages,
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                timeout=self.config.timeout_s,
            )
        except LLMProviderError as e:
            log.error(f"LLM adjudication failed: {e}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=str(e),
                prompt_json=prompt_json,
            )

        # Log the raw response at DEBUG level
        log.debug("Raw LLM Response:")
        log.debug("=" * 70)
        log.debug(json.dumps(response.raw_response, indent=2))
        log.debug("=" * 70)

        # Parse the response
        result = self._parse_response(response, evidence_bundle)
        result.prompt_json = prompt_json
        result.response_json = json.dumps(response.raw_response, indent=2)
        result.model_id = response.model
        result.prompt_template_version = self.config.prompt_template_version

        return result

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

        # Recording candidates
        candidates = evidence_bundle.get("recording_candidates", [])
        if candidates:
            lines.append("## Recording Candidates\n")
            for i, rec in enumerate(candidates, 1):
                lines.append(f"### Recording {i}")
                lines.append(f"- Title: {rec.get('title', 'Unknown')}")
                if rec.get("mb_recording_id"):
                    lines.append(f"- MBID: {rec['mb_recording_id']}")
                if rec.get("isrc"):
                    lines.append(f"- ISRC: {rec['isrc']}")

                rg_candidates = rec.get("rg_candidates", [])
                if rg_candidates:
                    lines.append("\n#### Release Group Candidates:")
                    for j, rg in enumerate(rg_candidates, 1):
                        lines.append(f"\n{j}. **{rg.get('title', 'Unknown')}**")
                        lines.append(f"   - RG MBID: {rg.get('mb_rg_id', 'Unknown')}")
                        lines.append(f"   - Type: {rg.get('primary_type', 'Unknown')}")
                        if rg.get("secondary_types"):
                            lines.append(
                                f"   - Secondary Types: {', '.join(rg['secondary_types'])}"
                            )
                        if rg.get("first_release_date"):
                            lines.append(f"   - First Release Date: {rg['first_release_date']}")

                        releases = rg.get("releases", [])
                        if releases:
                            lines.append("   - Releases:")
                            for r in releases[:5]:  # Limit to first 5
                                release_line = f"     - {r.get('mb_release_id', '?')[:8]}..."
                                if r.get("date"):
                                    release_line += f" ({r['date']})"
                                if r.get("country"):
                                    release_line += f" [{r['country']}]"
                                if r.get("label"):
                                    release_line += f" - {r['label']}"
                                lines.append(release_line)
                lines.append("")

        # Timeline facts
        timeline = evidence_bundle.get("timeline_facts", {})
        if timeline:
            lines.append("## Timeline Facts")
            if timeline.get("earliest_album_date"):
                lines.append(f"- Earliest Album: {timeline['earliest_album_date']}")
            if timeline.get("earliest_single_ep_date"):
                lines.append(f"- Earliest Single/EP: {timeline['earliest_single_ep_date']}")
            lines.append("")

        # Decision trace / missing facts
        if decision_trace:
            missing_facts = decision_trace.get("missing_facts", [])
            if missing_facts:
                lines.append("## Missing Facts (reasons for INDETERMINATE)")
                for fact in missing_facts:
                    lines.append(f"- {fact}")
                lines.append("")

        # Web search results (if SearxNG is enabled)
        if self._search_tool is not None:
            artist_name = artist.get("name", "Unknown")
            recording_title = evidence_bundle.get("recording_title", "")

            # Only search if we have both artist and title
            if artist_name != "Unknown" and recording_title:
                try:
                    from chart_binder.llm.searxng import SearxNGSearchTool

                    if isinstance(self._search_tool, SearxNGSearchTool):
                        log.debug(f"Performing web search for: {artist_name} - {recording_title}")
                        search_response = self._search_tool.search_music_info(
                            artist=artist_name,
                            title=recording_title,
                            max_results=5,
                        )
                        if search_response.results:
                            lines.append("## Web Search Results")
                            lines.append(
                                f"Additional context from web search for '{artist_name} - {recording_title}':\n"
                            )
                            for i, result in enumerate(search_response.results[:5], 1):
                                lines.append(f"{i}. {result.title}")
                                if result.metadata.get("snippet"):
                                    snippet = result.metadata["snippet"]
                                    # Truncate long snippets
                                    if len(snippet) > 150:
                                        snippet = snippet[:150] + "..."
                                    lines.append(f"   {snippet}")
                                lines.append(f"   Source: {result.metadata.get('url', 'N/A')}")
                                lines.append("")
                except Exception as e:
                    log.warning(f"Web search failed: {e}")

        lines.append("## Question")
        lines.append(
            "Based on the evidence above, which release group is the canonical premiere "
            "for this recording, and which release within it should be the representative release?"
        )

        return "\n".join(lines)

    def _parse_response(
        self,
        response: LLMResponse,
        evidence_bundle: dict[str, Any],
    ) -> AdjudicationResult:
        """Parse LLM response and validate the selection."""
        content = response.content.strip()

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in content:
                json_start = content.index("```json") + 7
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.index("```") + 3
                json_end = content.index("```", json_start)
                content = content[json_start:json_end].strip()

            data = json.loads(content)
        except (json.JSONDecodeError, ValueError) as e:
            log.warning(f"Failed to parse LLM response as JSON: {e}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=f"Failed to parse response: {e}",
            )

        crg_mbid = data.get("crg_mbid")
        rr_mbid = data.get("rr_mbid")
        confidence = float(data.get("confidence", 0.0))
        rationale = data.get("rationale", "")

        # Validate that the selected MBIDs exist in the evidence bundle
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

        if crg_mbid and crg_mbid not in valid_crg_mbids:
            log.warning(f"LLM selected invalid CRG MBID: {crg_mbid}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=f"Invalid CRG MBID: {crg_mbid} not in evidence bundle",
                confidence=confidence,
                rationale=rationale,
            )

        if crg_mbid and rr_mbid and rr_mbid not in valid_rr_mbids.get(crg_mbid, set()):
            log.warning(f"LLM selected invalid RR MBID: {rr_mbid}")
            return AdjudicationResult(
                outcome=AdjudicationOutcome.ERROR,
                error_message=f"Invalid RR MBID: {rr_mbid} not in CRG {crg_mbid}",
                confidence=confidence,
                rationale=rationale,
                crg_mbid=crg_mbid,
            )

        # Determine outcome based on confidence thresholds
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
        )


## Tests


def test_llm_config_defaults():
    """Test LLMConfig defaults."""
    from chart_binder.config import LLMConfig

    config = LLMConfig()
    assert config.enabled is False
    assert config.provider == "ollama"
    assert config.auto_accept_threshold == 0.85
    assert config.review_threshold == 0.60


def test_adjudication_result_defaults():
    """Test AdjudicationResult defaults."""
    result = AdjudicationResult(outcome=AdjudicationOutcome.ERROR)
    assert result.crg_mbid is None
    assert result.confidence == 0.0
    assert result.adjudication_id is not None


def test_adjudicator_disabled():
    """Test adjudicator returns error when disabled."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(enabled=False)
    adjudicator = LLMAdjudicator(config=config)
    result = adjudicator.adjudicate({})
    assert result.outcome == AdjudicationOutcome.ERROR
    assert "disabled" in (result.error_message or "")


def test_build_prompt():
    """Test prompt building from evidence bundle."""
    adjudicator = LLMAdjudicator()
    evidence = {
        "artist": {"name": "Test Artist", "begin_area_country": "US"},
        "recording_candidates": [
            {
                "title": "Test Song",
                "rg_candidates": [
                    {
                        "mb_rg_id": "rg-123",
                        "title": "Test Album",
                        "primary_type": "Album",
                        "first_release_date": "2020-01-01",
                        "releases": [{"mb_release_id": "rel-123", "date": "2020-01-01"}],
                    }
                ],
            }
        ],
    }
    prompt = adjudicator._build_prompt(evidence)
    assert "Test Artist" in prompt
    assert "Test Song" in prompt
    assert "Test Album" in prompt
    assert "rg-123" in prompt


def test_parse_response_valid():
    """Test parsing valid LLM response."""
    adjudicator = LLMAdjudicator()
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

    from chart_binder.llm.providers import LLMResponse

    response = LLMResponse(
        content='{"crg_mbid": "rg-123", "rr_mbid": "rel-123", "confidence": 0.9, "rationale": "test"}',
        model="test",
    )
    result = adjudicator._parse_response(response, evidence)
    assert result.outcome == AdjudicationOutcome.ACCEPTED
    assert result.crg_mbid == "rg-123"
    assert result.rr_mbid == "rel-123"
    assert result.confidence == 0.9


def test_parse_response_low_confidence():
    """Test parsing response with low confidence."""
    from chart_binder.config import LLMConfig

    config = LLMConfig(auto_accept_threshold=0.85, review_threshold=0.60)
    adjudicator = LLMAdjudicator(config=config)
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

    from chart_binder.llm.providers import LLMResponse

    # Medium confidence -> REVIEW
    response = LLMResponse(
        content='{"crg_mbid": "rg-123", "rr_mbid": "rel-123", "confidence": 0.7, "rationale": "test"}',
        model="test",
    )
    result = adjudicator._parse_response(response, evidence)
    assert result.outcome == AdjudicationOutcome.REVIEW

    # Low confidence -> REJECTED
    response = LLMResponse(
        content='{"crg_mbid": "rg-123", "rr_mbid": "rel-123", "confidence": 0.3, "rationale": "test"}',
        model="test",
    )
    result = adjudicator._parse_response(response, evidence)
    assert result.outcome == AdjudicationOutcome.REJECTED


def test_parse_response_invalid_mbid():
    """Test parsing response with invalid MBID."""
    adjudicator = LLMAdjudicator()
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

    from chart_binder.llm.providers import LLMResponse

    response = LLMResponse(
        content='{"crg_mbid": "invalid-id", "confidence": 0.9, "rationale": "test"}',
        model="test",
    )
    result = adjudicator._parse_response(response, evidence)
    assert result.outcome == AdjudicationOutcome.ERROR
    assert "Invalid CRG MBID" in (result.error_message or "")


def test_parse_response_json_in_markdown():
    """Test parsing JSON wrapped in markdown code block."""
    adjudicator = LLMAdjudicator()
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

    from chart_binder.llm.providers import LLMResponse

    response = LLMResponse(
        content='```json\n{"crg_mbid": "rg-123", "rr_mbid": "rel-123", "confidence": 0.9, "rationale": "test"}\n```',
        model="test",
    )
    result = adjudicator._parse_response(response, evidence)
    assert result.outcome == AdjudicationOutcome.ACCEPTED
    assert result.crg_mbid == "rg-123"
