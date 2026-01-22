"""LLM adjudication module for Chart-Binder (Epic 13).

This module provides:
- LangChain-based agent with native tool calling
- Structured output with Pydantic validation
- Provider-agnostic model factory (Ollama, OpenAI, Anthropic)
- Search tools for MusicBrainz context gathering
- Human review queue for low-confidence decisions
"""

from __future__ import annotations

from chart_binder.llm.adjudicator import (
    AdjudicationOutcome,
    AdjudicationResult,
    LLMAdjudicator,
)
from chart_binder.llm.agent_adjudicator import (
    AdjudicationResponse,
    AgentAdjudicator,
)
from chart_binder.llm.langchain_provider import (
    LangChainProviderError,
    create_chat_model,
    get_default_model,
)
from chart_binder.llm.providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderRegistry,
)
from chart_binder.llm.review_queue import ReviewAction, ReviewItem, ReviewQueue, ReviewSource
from chart_binder.llm.search_tool import SearchResult, SearchTool
from chart_binder.llm.searxng import (
    SearxNGClient,
    SearxNGResponse,
    SearxNGResult,
    SearxNGSearchTool,
)
from chart_binder.llm.structured_output import (
    StructuredOutputStrategy,
    with_structured_output,
)
from chart_binder.llm.tools import create_music_tools

__all__ = [
    # LangChain provider factory
    "create_chat_model",
    "get_default_model",
    "LangChainProviderError",
    # Legacy providers
    "LLMProvider",
    "ProviderRegistry",
    "OllamaProvider",
    "OpenAIProvider",
    # Agent adjudicator
    "AgentAdjudicator",
    "AdjudicationResponse",
    # Legacy adjudicator interface
    "LLMAdjudicator",
    "AdjudicationResult",
    "AdjudicationOutcome",
    # Structured output
    "StructuredOutputStrategy",
    "with_structured_output",
    # Tools
    "create_music_tools",
    # Search tool
    "SearchTool",
    "SearchResult",
    # SearxNG
    "SearxNGClient",
    "SearxNGResponse",
    "SearxNGResult",
    "SearxNGSearchTool",
    # Review queue
    "ReviewQueue",
    "ReviewItem",
    "ReviewAction",
    "ReviewSource",
]
