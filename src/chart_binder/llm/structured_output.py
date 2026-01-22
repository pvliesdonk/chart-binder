"""Structured output utilities for LangChain models.

Provides strategy selection and wrapper functions for enforcing structured
output from LLMs using Pydantic schemas. Follows patterns from questfoundry.

Two strategies available:
- JSON_MODE: Uses provider's native JSON schema support (recommended)
- TOOL: Uses function/tool calling to enforce schema

JSON_MODE is preferred because:
- More reliable across providers (Ollama TOOL strategy can return None)
- Better error messages for schema violations
- Direct schema enforcement without tool call overhead
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel

log = logging.getLogger(__name__)

T = TypeVar("T", bound="BaseModel")


class StructuredOutputStrategy(str, Enum):
    """Strategy for enforcing structured output."""

    TOOL = "tool"  # Use function/tool calling
    JSON_MODE = "json_mode"  # Use native JSON schema support
    AUTO = "auto"  # Auto-select based on provider


# Provider-specific default strategies
# JSON_MODE is preferred for reliability
PROVIDER_STRATEGY_DEFAULTS: dict[str, StructuredOutputStrategy] = {
    "ollama": StructuredOutputStrategy.JSON_MODE,
    "openai": StructuredOutputStrategy.JSON_MODE,
    "anthropic": StructuredOutputStrategy.JSON_MODE,
}


def get_default_strategy(provider_name: str | None) -> StructuredOutputStrategy:
    """Get default structured output strategy for a provider.

    Args:
        provider_name: Provider identifier (ollama, openai, anthropic)

    Returns:
        Default strategy for the provider (JSON_MODE for all)
    """
    if provider_name is None:
        return StructuredOutputStrategy.JSON_MODE

    return PROVIDER_STRATEGY_DEFAULTS.get(
        provider_name.lower(),
        StructuredOutputStrategy.JSON_MODE,
    )


def with_structured_output(
    model: BaseChatModel,
    schema: type[T],
    strategy: StructuredOutputStrategy | None = None,
    provider_name: str | None = None,
) -> Runnable[Any, Any]:
    """Wrap a chat model with structured output enforcement.

    Creates a runnable that enforces the given Pydantic schema on model output.
    The strategy determines how the schema is enforced.

    Args:
        model: LangChain chat model to wrap
        schema: Pydantic model class defining the output structure
        strategy: Output strategy (auto-selected if None)
        provider_name: Provider name for strategy auto-detection

    Returns:
        Runnable that produces validated schema instances

    Example:
        ```python
        from pydantic import BaseModel

        class AdjudicationResponse(BaseModel):
            crg_mbid: str | None
            confidence: float
            rationale: str

        structured = with_structured_output(model, AdjudicationResponse)
        result = await structured.ainvoke(messages)
        # result is AdjudicationResponse instance
        ```
    """
    # Auto-select strategy if not specified
    if strategy is None or strategy == StructuredOutputStrategy.AUTO:
        strategy = get_default_strategy(provider_name)

    # Map strategy to LangChain method parameter
    method: str
    if strategy == StructuredOutputStrategy.TOOL:
        method = "function_calling"
    else:
        method = "json_schema"

    log.debug(
        "Configuring structured output: schema=%s strategy=%s method=%s",
        schema.__name__,
        strategy.value,
        method,
    )

    return model.with_structured_output(schema, method=method)


def strip_null_values(data: dict[str, Any]) -> dict[str, Any]:
    """Strip null values from a dictionary.

    LLMs often return null for optional fields. This removes them
    so Pydantic can use default values instead.

    Args:
        data: Dictionary potentially containing null values

    Returns:
        Dictionary with null values removed (recursively)
    """
    result = {}
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            result[key] = strip_null_values(value)
        elif isinstance(value, list):
            result[key] = [
                strip_null_values(v) if isinstance(v, dict) else v for v in value if v is not None
            ]
        else:
            result[key] = value
    return result


def format_validation_errors(errors: list[dict[str, Any]]) -> str:
    """Format Pydantic validation errors for LLM feedback.

    Creates a human-readable error message that helps the LLM
    understand what went wrong and how to fix it.

    Args:
        errors: List of error dicts from ValidationError.errors()

    Returns:
        Formatted error message
    """
    lines = []
    for error in errors:
        loc = ".".join(str(part) for part in error.get("loc", []))
        msg = error.get("msg", "Unknown error")
        if loc:
            lines.append(f"  - {loc}: {msg}")
        else:
            lines.append(f"  - {msg}")

    return "Validation errors:\n" + "\n".join(lines)


def build_retry_feedback(errors: list[str]) -> str:
    """Build error feedback message for retry attempt.

    Args:
        errors: List of error messages from previous attempt

    Returns:
        Formatted feedback message to append to conversation
    """
    error_list = "\n".join(f"  - {e}" for e in errors)
    return (
        "The previous output had validation errors:\n"
        f"{error_list}\n\n"
        "Please fix these issues and try again. "
        "Ensure all required fields are present and have valid values."
    )


## Tests


def test_get_default_strategy():
    """Test default strategy selection."""
    assert get_default_strategy("ollama") == StructuredOutputStrategy.JSON_MODE
    assert get_default_strategy("openai") == StructuredOutputStrategy.JSON_MODE
    assert get_default_strategy("anthropic") == StructuredOutputStrategy.JSON_MODE
    assert get_default_strategy("unknown") == StructuredOutputStrategy.JSON_MODE
    assert get_default_strategy(None) == StructuredOutputStrategy.JSON_MODE


def test_strip_null_values_simple():
    """Test stripping null values."""
    data = {"a": 1, "b": None, "c": "test"}
    result = strip_null_values(data)
    assert result == {"a": 1, "c": "test"}


def test_strip_null_values_nested():
    """Test stripping null values from nested dicts."""
    data = {
        "a": 1,
        "nested": {"x": None, "y": 2},
        "empty": None,
    }
    result = strip_null_values(data)
    assert result == {"a": 1, "nested": {"y": 2}}


def test_strip_null_values_list():
    """Test stripping null values from lists."""
    data = {
        "items": [{"a": 1}, None, {"b": None, "c": 2}],
    }
    result = strip_null_values(data)
    assert result == {"items": [{"a": 1}, {"c": 2}]}


def test_format_validation_errors():
    """Test validation error formatting."""
    errors = [
        {"loc": ("crg_mbid",), "msg": "field required"},
        {"loc": ("confidence",), "msg": "value is not a valid float"},
    ]
    formatted = format_validation_errors(errors)
    assert "crg_mbid: field required" in formatted
    assert "confidence: value is not a valid float" in formatted


def test_build_retry_feedback():
    """Test retry feedback message."""
    errors = ["Missing required field: crg_mbid", "Confidence must be 0.0-1.0"]
    feedback = build_retry_feedback(errors)
    assert "validation errors" in feedback
    assert "crg_mbid" in feedback
    assert "Confidence" in feedback
    assert "try again" in feedback.lower()
