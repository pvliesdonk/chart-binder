"""PII-safe logging utilities for Chart-Binder (Epic 14).

Provides utilities to ensure logs do not contain personally identifiable information:
- File path hashing/relativization
- Sensitive field redaction
- Safe log message formatting
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

# Fields that should be redacted in logs
REDACT_FIELDS = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "access_token",
        "refresh_token",
        "client_secret",
        "private_key",
    }
)

# Regex patterns for sensitive data
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "ip_v4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "uuid": re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I),
}


def hash_path(file_path: Path | str, length: int = 12) -> str:
    """Hash a file path for logging.

    Creates a deterministic, non-reversible hash of the full path.

    Args:
        file_path: Path to hash
        length: Length of hash to return

    Returns:
        Truncated SHA256 hash of the path
    """
    path_str = str(file_path)
    return hashlib.sha256(path_str.encode()).hexdigest()[:length]


def relativize_path(file_path: Path | str, library_root: Path | str | None = None) -> str:
    """Convert path to relative form for safe logging.

    If library_root is provided, returns path relative to it.
    Otherwise, returns just the filename with parent directory.

    Args:
        file_path: Path to convert
        library_root: Optional library root directory

    Returns:
        Relative or minimal path string
    """
    path = Path(file_path)

    if library_root:
        root = Path(library_root)
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass

    # Return minimal identifying info: parent/filename
    if path.parent.name:
        return f"{path.parent.name}/{path.name}"
    return path.name


def safe_path(
    file_path: Path | str,
    library_root: Path | str | None = None,
    use_hash: bool = False,
) -> str:
    """Get a safe representation of a path for logging.

    Args:
        file_path: Path to represent
        library_root: Optional library root for relativization
        use_hash: If True, return hash instead of relative path

    Returns:
        Safe path representation
    """
    if use_hash:
        return f"file:{hash_path(file_path)}"
    return relativize_path(file_path, library_root)


def redact_value(value: str, visible_chars: int = 4) -> str:
    """Redact a sensitive value, showing only first few characters.

    Args:
        value: Value to redact
        visible_chars: Number of characters to show

    Returns:
        Redacted string (e.g., "sk-a***")
    """
    if len(value) <= visible_chars:
        return "***"
    return f"{value[:visible_chars]}***"


def redact_dict(
    data: dict[str, Any],
    redact_fields: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Recursively redact sensitive fields in a dictionary.

    Args:
        data: Dictionary to redact
        redact_fields: Set of field names to redact (case-insensitive)

    Returns:
        New dictionary with sensitive fields redacted
    """
    if redact_fields is None:
        redact_fields = REDACT_FIELDS

    result: dict[str, Any] = {}
    for key, value in data.items():
        key_lower = key.lower()

        # Efficient check: exact match first, then substring as fallback
        should_redact = key_lower in redact_fields or any(
            field in key_lower for field in redact_fields
        )

        if should_redact and isinstance(value, str):
            result[key] = redact_value(value)
        elif isinstance(value, dict):
            result[key] = redact_dict(value, redact_fields)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item, redact_fields) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def sanitize_message(message: str) -> str:
    """Remove potential PII patterns from a log message.

    Args:
        message: Log message to sanitize

    Returns:
        Sanitized message with PII patterns redacted
    """
    result = message

    # Redact email addresses
    result = PATTERNS["email"].sub("[EMAIL]", result)

    # Note: We don't redact UUIDs as they are needed for debugging
    # and are not personally identifiable

    return result


@lru_cache(maxsize=1)
def _get_library_root() -> Path | None:
    """Get library root from environment."""
    root = os.environ.get("CHART_BINDER_LIBRARY_ROOT")
    return Path(root) if root else None


class SafeLogFormatter(logging.Formatter):
    """Log formatter that sanitizes messages for PII.

    Automatically sanitizes log messages and redacts sensitive data
    in the record's extra fields.
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        sanitize_messages: bool = True,
        hash_paths: bool = False,
    ):
        super().__init__(fmt, datefmt)
        self.sanitize_messages = sanitize_messages
        self.hash_paths = hash_paths
        self._library_root = _get_library_root()

    def format(self, record: logging.LogRecord) -> str:
        # Create a copy of the record to avoid mutating the original,
        # which could cause issues with multiple handlers
        record = logging.makeLogRecord(record.__dict__)

        # Sanitize the message
        if self.sanitize_messages:
            record.msg = sanitize_message(str(record.msg))

        # Handle file paths in args
        if record.args:
            record.args = self._sanitize_args(record.args)

        return super().format(record)

    def _sanitize_args(self, args: tuple[Any, ...] | Mapping[str, Any]) -> tuple[Any, ...]:
        """Sanitize formatting arguments."""
        if isinstance(args, Mapping):
            return tuple(self._sanitize_value(v) for v in args.values())
        return tuple(self._sanitize_value(arg) for arg in args)

    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize a single value."""
        if isinstance(value, Path):
            return safe_path(value, self._library_root, use_hash=self.hash_paths)
        if isinstance(value, str) and "/" in value and not value.startswith("http"):
            # Check for path-like pattern before expensive Path operations
            # Paths typically have file extensions or multiple slashes
            try:
                path = Path(value)
                # Check suffix first (no I/O), only check exists() if has extension
                if path.suffix:
                    return safe_path(path, self._library_root, use_hash=self.hash_paths)
            except (OSError, ValueError):
                # String cannot be parsed as a valid path; treat as non-path value
                pass
        return value


def configure_safe_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    hash_paths: bool = False,
) -> None:
    """Configure logging with PII-safe formatting.

    Args:
        level: Logging level
        format_string: Optional custom format string
        hash_paths: Whether to hash file paths
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = SafeLogFormatter(
        fmt=format_string,
        sanitize_messages=True,
        hash_paths=hash_paths,
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


## Tests


def test_hash_path():
    """Test path hashing."""
    path1 = Path("/home/user/music/song.mp3")
    path2 = Path("/home/user/music/song.mp3")
    path3 = Path("/home/user/music/other.mp3")

    hash1 = hash_path(path1)
    hash2 = hash_path(path2)
    hash3 = hash_path(path3)

    assert len(hash1) == 12
    assert hash1 == hash2  # Same path = same hash
    assert hash1 != hash3  # Different path = different hash


def test_relativize_path():
    """Test path relativization."""
    path = Path("/home/user/music/artist/album/song.mp3")

    # With library root
    rel = relativize_path(path, "/home/user/music")
    assert rel == "artist/album/song.mp3"

    # Without library root
    rel = relativize_path(path)
    assert rel == "album/song.mp3"


def test_safe_path():
    """Test safe path representation."""
    path = Path("/home/user/music/song.mp3")

    # Relative mode
    safe = safe_path(path)
    assert "user" not in safe
    assert "song.mp3" in safe

    # Hash mode
    safe = safe_path(path, use_hash=True)
    assert safe.startswith("file:")
    assert "song.mp3" not in safe


def test_redact_value():
    """Test value redaction."""
    assert redact_value("sk-secret-key-12345") == "sk-s***"
    assert redact_value("abc") == "***"
    assert redact_value("abcdef", 3) == "abc***"


def test_redact_dict():
    """Test dictionary redaction."""
    data = {
        "api_key": "sk-secret-12345",
        "user": "john",
        "nested": {
            "password": "hunter2",
            "name": "test",
        },
        "tokens": [
            {"access_token": "abc123", "type": "bearer"},
        ],
    }

    redacted = redact_dict(data)

    assert redacted["api_key"] == "sk-s***"
    assert redacted["user"] == "john"  # Not redacted
    assert redacted["nested"]["password"] == "hunt***"
    assert redacted["nested"]["name"] == "test"  # Not redacted
    assert redacted["tokens"][0]["access_token"] == "abc1***"
    assert redacted["tokens"][0]["type"] == "bearer"


def test_sanitize_message():
    """Test message sanitization."""
    msg = "Error for user@example.com at 192.168.1.1"
    sanitized = sanitize_message(msg)

    assert "[EMAIL]" in sanitized
    assert "user@example.com" not in sanitized
    # IP addresses are kept for debugging
    assert "192.168.1.1" in sanitized


def test_sanitize_message_preserves_uuids():
    """Test that UUIDs are preserved for debugging."""
    msg = "Error for MB ID 12345678-1234-1234-1234-123456789abc"
    sanitized = sanitize_message(msg)

    assert "12345678-1234-1234-1234-123456789abc" in sanitized


def test_safe_log_formatter():
    """Test SafeLogFormatter."""
    formatter = SafeLogFormatter(
        fmt="%(message)s",
        sanitize_messages=True,
        hash_paths=False,
    )

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Email: user@example.com",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert "[EMAIL]" in formatted
    assert "user@example.com" not in formatted
