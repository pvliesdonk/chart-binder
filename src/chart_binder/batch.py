"""Batch processing utilities for Chart-Binder (Epic 14).

Provides efficient batch processing for file operations with:
- Configurable batch sizes
- Progress tracking
- Error handling with continue-on-error mode
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchConfig:
    """Configuration for batch processing."""

    identify_batch_size: int = 50
    decide_batch_size: int = 50
    write_batch_size: int = 20
    verify_batch_size: int = 20
    continue_on_error: bool = True
    progress_callback: Callable[[int, int], None] | None = None


@dataclass
class BatchResult:
    """Result of a batch operation."""

    total: int = 0
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    errors: list[tuple[Any, Exception]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.processed == 0:
            return 100.0
        return (self.succeeded / self.processed) * 100

    def add_success(self) -> None:
        """Record a successful operation."""
        self.processed += 1
        self.succeeded += 1

    def add_failure(self, item: Any, error: Exception) -> None:
        """Record a failed operation."""
        self.processed += 1
        self.failed += 1
        self.errors.append((item, error))


def batch_iter(items: list[T], batch_size: int) -> Iterator[list[T]]:
    """Iterate over items in batches.

    Args:
        items: List of items to process
        batch_size: Maximum size of each batch

    Yields:
        Lists of items, each up to batch_size in length
    """
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def process_batch(
    items: list[T],
    processor: Callable[[T], R],
    config: BatchConfig | None = None,
) -> tuple[list[R], BatchResult]:
    """Process items in batches with error handling.

    Args:
        items: List of items to process
        processor: Function to apply to each item
        config: Batch processing configuration

    Returns:
        Tuple of (results list, batch result summary)
    """
    if config is None:
        config = BatchConfig()

    result = BatchResult(total=len(items))
    results: list[R] = []

    for i, item in enumerate(items):
        try:
            processed = processor(item)
            results.append(processed)
            result.add_success()

            if config.progress_callback:
                config.progress_callback(i + 1, len(items))

        except Exception as e:
            result.add_failure(item, e)
            if config.continue_on_error:
                log.warning(f"Batch processing error (continuing): {e}")
            else:
                log.error(f"Batch processing error (stopping): {e}")
                raise

    return results, result


def collect_audio_files(
    paths: list[Path],
    extensions: tuple[str, ...] = (".mp3", ".flac", ".ogg", ".m4a", ".opus", ".wav"),
    recursive: bool = True,
) -> list[Path]:
    """Collect audio files from paths.

    Args:
        paths: List of files or directories
        extensions: Audio file extensions to include
        recursive: Whether to search directories recursively

    Returns:
        List of audio file paths
    """
    audio_files: list[Path] = []

    for path in paths:
        if path.is_file():
            if path.suffix.lower() in extensions:
                audio_files.append(path)
        elif path.is_dir():
            if recursive:
                for ext in extensions:
                    audio_files.extend(path.rglob(f"*{ext}"))
            else:
                for ext in extensions:
                    audio_files.extend(path.glob(f"*{ext}"))

    # Sort for deterministic ordering
    return sorted(set(audio_files))


class BatchProcessor:
    """High-level batch processor for chart-binder operations.

    Provides batched identify, decide, and write operations with
    configurable batch sizes and progress tracking.
    """

    def __init__(self, config: BatchConfig | None = None):
        self.config = config or BatchConfig()

    def identify_batch(
        self,
        files: list[Path],
        fetcher: Any | None = None,
    ) -> tuple[list[dict[str, Any]], BatchResult]:
        """Identify recordings for a batch of files.

        Args:
            files: List of audio file paths
            fetcher: UnifiedFetcher instance (optional)

        Returns:
            Tuple of (evidence bundles, batch result)
        """
        from chart_binder.tagging import verify

        def identify_one(file_path: Path) -> dict[str, Any]:
            tagset = verify(file_path)
            bundle: dict[str, Any] = {
                "artifact": {"file_path": str(file_path)},
                "artist": {"name": tagset.artist or "Unknown"},
                "recording_candidates": [],
                "provenance": {"sources_used": ["local_tags"]},
            }

            if tagset.ids.mb_release_group_id:
                bundle["recording_candidates"] = [
                    {
                        "mb_recording_id": tagset.ids.mb_recording_id,
                        "title": tagset.title,
                        "rg_candidates": [
                            {
                                "mb_rg_id": tagset.ids.mb_release_group_id,
                                "title": tagset.album,
                                "first_release_date": tagset.original_year,
                                "releases": [],
                            }
                        ],
                    }
                ]

            return bundle

        return process_batch(files, identify_one, self._config_with_size("identify"))

    def decide_batch(
        self,
        evidence_bundles: list[dict[str, Any]],
    ) -> tuple[list[Any], BatchResult]:
        """Make decisions for a batch of evidence bundles.

        Args:
            evidence_bundles: List of evidence bundles

        Returns:
            Tuple of (decisions, batch result)
        """
        from chart_binder.resolver import ConfigSnapshot, Resolver

        resolver = Resolver(ConfigSnapshot())

        def decide_one(bundle: dict[str, Any]) -> Any:
            return resolver.resolve(bundle)

        return process_batch(evidence_bundles, decide_one, self._config_with_size("decide"))

    def write_batch(
        self,
        file_tagset_pairs: list[tuple[Path, Any]],
        authoritative: bool = False,
        dry_run: bool = False,
    ) -> tuple[list[Any], BatchResult]:
        """Write tags for a batch of files.

        Args:
            file_tagset_pairs: List of (file_path, tagset) tuples
            authoritative: Whether to overwrite existing tags
            dry_run: Whether to simulate writes

        Returns:
            Tuple of (write reports, batch result)
        """
        from chart_binder.tagging import write_tags

        def write_one(pair: tuple[Path, Any]) -> Any:
            file_path, tagset = pair
            return write_tags(file_path, tagset, authoritative=authoritative, dry_run=dry_run)

        return process_batch(file_tagset_pairs, write_one, self._config_with_size("write"))

    def _config_with_size(self, operation: str) -> BatchConfig:
        """Get config with appropriate batch size for operation.

        This creates a new BatchConfig that inherits error handling and
        progress callback from the parent config while preserving all
        operation-specific batch sizes.
        """
        return BatchConfig(
            identify_batch_size=self.config.identify_batch_size,
            decide_batch_size=self.config.decide_batch_size,
            write_batch_size=self.config.write_batch_size,
            verify_batch_size=self.config.verify_batch_size,
            continue_on_error=self.config.continue_on_error,
            progress_callback=self.config.progress_callback,
        )


## Tests


def test_batch_iter():
    """Test batch iteration."""
    items = list(range(10))
    batches = list(batch_iter(items, 3))
    assert len(batches) == 4
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]
    assert batches[2] == [6, 7, 8]
    assert batches[3] == [9]


def test_batch_iter_exact():
    """Test batch iteration with exact multiple."""
    items = list(range(6))
    batches = list(batch_iter(items, 3))
    assert len(batches) == 2
    assert batches[0] == [0, 1, 2]
    assert batches[1] == [3, 4, 5]


def test_batch_iter_empty():
    """Test batch iteration with empty list."""
    batches = list(batch_iter([], 5))
    assert batches == []


def test_process_batch_success():
    """Test successful batch processing."""
    items = [1, 2, 3, 4, 5]
    results, batch_result = process_batch(items, lambda x: x * 2)

    assert results == [2, 4, 6, 8, 10]
    assert batch_result.total == 5
    assert batch_result.processed == 5
    assert batch_result.succeeded == 5
    assert batch_result.failed == 0
    assert batch_result.success_rate == 100.0


def test_process_batch_with_errors():
    """Test batch processing with errors and continue-on-error."""

    def processor(x: int) -> int:
        if x == 3:
            raise ValueError("Error on 3")
        return x * 2

    config = BatchConfig(continue_on_error=True)
    results, batch_result = process_batch([1, 2, 3, 4, 5], processor, config)

    assert results == [2, 4, 8, 10]  # 3 failed
    assert batch_result.succeeded == 4
    assert batch_result.failed == 1
    assert len(batch_result.errors) == 1
    assert batch_result.errors[0][0] == 3


def test_process_batch_stop_on_error():
    """Test batch processing stops on error when configured."""

    def processor(x: int) -> int:
        if x == 3:
            raise ValueError("Error on 3")
        return x * 2

    config = BatchConfig(continue_on_error=False)

    try:
        process_batch([1, 2, 3, 4, 5], processor, config)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


def test_batch_result_success_rate():
    """Test success rate calculation."""
    result = BatchResult()
    assert result.success_rate == 100.0  # Empty = 100%

    result.add_success()
    result.add_success()
    result.add_failure("item", ValueError("test"))
    result.add_success()

    assert result.success_rate == 75.0


def test_collect_audio_files(tmp_path):
    """Test audio file collection."""
    # Create test files
    (tmp_path / "song1.mp3").touch()
    (tmp_path / "song2.flac").touch()
    (tmp_path / "document.pdf").touch()
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "song3.m4a").touch()

    # Test recursive
    files = collect_audio_files([tmp_path])
    assert len(files) == 3
    assert any("song1.mp3" in str(f) for f in files)
    assert any("song3.m4a" in str(f) for f in files)

    # Test non-recursive
    files = collect_audio_files([tmp_path], recursive=False)
    assert len(files) == 2


def test_batch_config_defaults():
    """Test BatchConfig default values."""
    config = BatchConfig()
    assert config.identify_batch_size == 50
    assert config.decide_batch_size == 50
    assert config.write_batch_size == 20
    assert config.continue_on_error is True


def test_batch_processor_creation():
    """Test BatchProcessor creation."""
    processor = BatchProcessor()
    assert processor.config.identify_batch_size == 50

    custom_config = BatchConfig(identify_batch_size=100)
    processor = BatchProcessor(custom_config)
    assert processor.config.identify_batch_size == 100
