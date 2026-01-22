"""LangChain callback handlers for observability.

Provides callback handlers that integrate with chart-binder's logging system,
including JSONL logging for LLM calls.
"""

# ruff: noqa: ARG002 - Callback interface methods require unused parameters

from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

    from chart_binder.llm.llm_logger import LLMLogger

log = logging.getLogger(__name__)


class LLMLoggingCallback(BaseCallbackHandler):
    """Callback handler that logs LLM calls to JSONL.

    Captures request/response pairs and writes them to the LLMLogger,
    preserving full prompts and responses for debugging.
    """

    def __init__(self, llm_logger: LLMLogger, stage: str = "adjudication") -> None:
        """Initialize the callback handler.

        Args:
            llm_logger: LLMLogger instance for writing entries.
            stage: Stage name to use in log entries.
        """
        super().__init__()
        self._llm_logger = llm_logger
        self._stage = stage
        self._pending_calls: dict[UUID, dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts processing."""
        # Extract model info
        model_kwargs = serialized.get("kwargs", {})
        model_name = (
            model_kwargs.get("model")
            or model_kwargs.get("model_name")
            or serialized.get("id", ["unknown"])[-1]
        )

        temperature = model_kwargs.get("temperature")

        # Flatten messages for storage
        flat_messages = []
        for message_batch in messages:
            for msg in message_batch:
                msg_content = msg.content
                if isinstance(msg_content, list):
                    # Handle multimodal content
                    msg_content = str(msg_content)
                flat_messages.append(
                    {
                        "role": msg.type,
                        "content": msg_content
                        if isinstance(msg_content, str)
                        else str(msg_content),
                    }
                )

        # Store pending call with start time
        self._pending_calls[run_id] = {
            "model": model_name,
            "messages": flat_messages,
            "start_time": perf_counter(),
            "temperature": temperature,
        }

        log.debug(
            "LLM call start: run_id=%s, model=%s, messages=%d",
            run_id,
            model_name,
            len(flat_messages),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM completes."""
        call_info = self._pending_calls.pop(run_id, {})

        # Calculate duration
        duration_seconds = 0.0
        if "start_time" in call_info:
            duration_seconds = perf_counter() - call_info["start_time"]

        # Extract response content
        content = ""
        tool_calls: list[dict[str, Any]] = []

        if (
            response.generations
            and len(response.generations) > 0
            and len(response.generations[0]) > 0
        ):
            gen = response.generations[0][0]
            content = gen.text if hasattr(gen, "text") else str(gen)

            # Check for tool calls
            if hasattr(gen, "message"):
                msg = gen.message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.get("id", ""),
                            "name": tc.get("name", ""),
                            "arguments": tc.get("args", {}),
                        }
                        for tc in msg.tool_calls
                    ]

        # Extract token usage
        total_tokens = 0
        llm_output = response.llm_output or {}

        if "token_usage" in llm_output:
            usage = llm_output["token_usage"]
            total_tokens = usage.get("total_tokens", 0)
        elif "usage_metadata" in llm_output:
            usage = llm_output["usage_metadata"]
            total_tokens = usage.get("total_tokens", 0)

        # Build entry
        temperature = call_info.get("temperature")
        entry_kwargs: dict[str, Any] = {
            "stage": self._stage,
            "model": call_info.get("model", "unknown"),
            "messages": call_info.get("messages", []),
            "content": content,
            "tokens_used": total_tokens,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "duration_seconds": duration_seconds,
            "tool_calls": tool_calls if tool_calls else None,
        }
        if temperature is not None:
            entry_kwargs["temperature"] = temperature

        # Write to logger
        entry = self._llm_logger.create_entry(**entry_kwargs)
        self._llm_logger.log(entry)

        log.debug(
            "LLM call end: run_id=%s, tokens=%d, tool_calls=%d",
            run_id,
            total_tokens,
            len(tool_calls),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        call_info = self._pending_calls.pop(run_id, {})

        log.warning(
            "LLM call error: run_id=%s, model=%s, error=%s",
            run_id,
            call_info.get("model", "unknown"),
            error,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts."""
        tool_name = serialized.get("name", "unknown")
        log.debug("Tool start: %s (run_id=%s)", tool_name, run_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes."""
        if isinstance(output, ToolMessage):
            content = output.content
            output_len = len(content) if isinstance(content, str) else None
        elif isinstance(output, str):
            output_len = len(output)
        else:
            output_len = None

        log.debug("Tool end: run_id=%s, output_length=%s", run_id, output_len)

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        log.debug("Agent action: tool=%s (run_id=%s)", action.tool, run_id)

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        log.debug("Agent finish: run_id=%s", run_id)


def create_logging_callbacks(
    llm_logger: LLMLogger, stage: str = "adjudication"
) -> list[BaseCallbackHandler]:
    """Create logging callback handlers.

    Args:
        llm_logger: LLMLogger instance.
        stage: Stage name for log entries.

    Returns:
        List of callback handlers for use with LangChain.
    """
    return [LLMLoggingCallback(llm_logger, stage)]


## Tests


def test_callback_flattens_messages() -> None:
    """Test that messages are properly flattened."""
    from unittest.mock import MagicMock
    from uuid import uuid4

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    logger = MagicMock()
    callback = LLMLoggingCallback(logger, stage="test")

    run_id = uuid4()
    messages = [
        [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]
    ]

    callback.on_chat_model_start(
        serialized={"kwargs": {"model": "test-model"}},
        messages=messages,
        run_id=run_id,
    )

    assert run_id in callback._pending_calls
    flat = callback._pending_calls[run_id]["messages"]
    assert len(flat) == 3
    assert flat[0]["role"] == "system"
    assert flat[1]["role"] == "human"
    assert flat[2]["role"] == "ai"
