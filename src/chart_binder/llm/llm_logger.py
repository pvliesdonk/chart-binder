"""JSONL logger for LLM calls.

Writes structured log entries for each LLM call to logs/llm_calls.jsonl.
Content is never truncated - full prompts and responses are preserved.

Only active when --log flag is passed to CLI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class LLMLogEntry:
    """Entry for LLM call logging."""

    timestamp: str
    stage: str
    model: str

    # Request
    messages: list[dict[str, str]]
    temperature: float
    max_tokens: int

    # Response
    content: str
    tokens_used: int
    finish_reason: str
    duration_seconds: float

    # Optional fields
    error: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMLogger:
    """Logger for LLM calls in JSONL format.

    Writes one JSON object per line to logs/llm_calls.jsonl.
    Content is never truncated - full prompts and responses are preserved.

    Attributes:
        log_path: Path to the JSONL log file.
        enabled: Whether logging is enabled.
    """

    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        """Initialize LLM logger.

        Args:
            log_dir: Directory for log files.
            enabled: Whether to actually write logs.
        """
        self.enabled = enabled
        self.log_path = log_dir / "llm_calls.jsonl"
        if enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: LLMLogEntry) -> None:
        """Append an entry to the JSONL log.

        Args:
            entry: Log entry to write.
        """
        if not self.enabled:
            return

        with self.log_path.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    @staticmethod
    def create_entry(
        stage: str,
        model: str,
        messages: list[dict[str, str]],
        content: str,
        tokens_used: int,
        finish_reason: str,
        duration_seconds: float,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        error: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        **metadata: Any,
    ) -> LLMLogEntry:
        """Create a log entry with current timestamp.

        Args:
            stage: Pipeline stage name.
            model: Model identifier used.
            messages: List of conversation messages.
            content: Response content.
            tokens_used: Total tokens used.
            finish_reason: Why generation stopped.
            duration_seconds: Time taken for call.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens allowed.
            error: Error message if call failed.
            tool_calls: List of tool calls from response (if any).
            **metadata: Additional metadata.

        Returns:
            LLMLogEntry ready for logging.
        """
        return LLMLogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            stage=stage,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            content=content,
            tokens_used=tokens_used,
            finish_reason=finish_reason,
            duration_seconds=duration_seconds,
            error=error,
            tool_calls=tool_calls,
            metadata=dict(metadata),
        )

    def read_entries(self) -> list[LLMLogEntry]:
        """Read all entries from the log file.

        Returns:
            List of log entries.
        """
        if not self.log_path.exists():
            return []

        entries = []
        with self.log_path.open() as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(LLMLogEntry(**data))
        return entries


## Tests


def test_llm_logger_disabled(tmp_path: Path) -> None:
    """Test that disabled logger doesn't write."""
    logger = LLMLogger(tmp_path, enabled=False)
    entry = LLMLogger.create_entry(
        stage="test",
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        content="Hi there",
        tokens_used=10,
        finish_reason="stop",
        duration_seconds=0.5,
    )
    logger.log(entry)
    assert not logger.log_path.exists()


def test_llm_logger_writes_jsonl(tmp_path: Path) -> None:
    """Test that logger writes valid JSONL."""
    logger = LLMLogger(tmp_path, enabled=True)
    entry = LLMLogger.create_entry(
        stage="adjudication",
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        content="Hi there!",
        tokens_used=25,
        finish_reason="stop",
        duration_seconds=1.2,
        temperature=0.7,
        max_tokens=1000,
    )
    logger.log(entry)

    # Verify file was created
    assert logger.log_path.exists()

    # Verify content is valid JSON
    with logger.log_path.open() as f:
        data = json.loads(f.readline())

    assert data["stage"] == "adjudication"
    assert data["model"] == "gpt-4"
    assert len(data["messages"]) == 2
    assert data["content"] == "Hi there!"
    assert data["tokens_used"] == 25


def test_llm_logger_read_entries(tmp_path: Path) -> None:
    """Test reading entries back from log file."""
    logger = LLMLogger(tmp_path, enabled=True)

    # Write two entries
    for i in range(2):
        entry = LLMLogger.create_entry(
            stage=f"stage-{i}",
            model="test-model",
            messages=[{"role": "user", "content": f"Message {i}"}],
            content=f"Response {i}",
            tokens_used=10 + i,
            finish_reason="stop",
            duration_seconds=0.5,
        )
        logger.log(entry)

    # Read them back
    entries = logger.read_entries()
    assert len(entries) == 2
    assert entries[0].stage == "stage-0"
    assert entries[1].stage == "stage-1"


def test_llm_logger_with_tool_calls(tmp_path: Path) -> None:
    """Test logging with tool calls."""
    logger = LLMLogger(tmp_path, enabled=True)
    entry = LLMLogger.create_entry(
        stage="agent",
        model="gpt-4",
        messages=[{"role": "user", "content": "Search for Beatles"}],
        content="",
        tokens_used=50,
        finish_reason="tool_calls",
        duration_seconds=2.0,
        tool_calls=[{"id": "call_1", "name": "search_artist", "arguments": {"query": "Beatles"}}],
    )
    logger.log(entry)

    entries = logger.read_entries()
    assert len(entries) == 1
    assert entries[0].tool_calls is not None
    assert len(entries[0].tool_calls) == 1
    assert entries[0].tool_calls[0]["name"] == "search_artist"
