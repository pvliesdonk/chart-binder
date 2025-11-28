"""LLM adjudication module for Chart-Binder (Epic 13).

This module provides:
- Plugin system for multiple LLM providers (Ollama, OpenAI)
- Structured prompts for music metadata adjudication
- Confidence-based decision making
- Search tool for LLM context gathering
- Human review queue for low-confidence decisions
"""

from __future__ import annotations

from chart_binder.llm.adjudicator import (
    AdjudicationConfig,
    AdjudicationOutcome,
    AdjudicationResult,
    LLMAdjudicator,
)
from chart_binder.llm.providers import (
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderRegistry,
)
from chart_binder.llm.review_queue import ReviewAction, ReviewItem, ReviewQueue, ReviewSource
from chart_binder.llm.search_tool import SearchResult, SearchTool

__all__ = [
    # Providers
    "LLMProvider",
    "ProviderRegistry",
    "OllamaProvider",
    "OpenAIProvider",
    # Adjudication
    "LLMAdjudicator",
    "AdjudicationConfig",
    "AdjudicationResult",
    "AdjudicationOutcome",
    # Search tool
    "SearchTool",
    "SearchResult",
    # Review queue
    "ReviewQueue",
    "ReviewItem",
    "ReviewAction",
    "ReviewSource",
]
