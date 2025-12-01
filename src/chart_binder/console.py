"""Shared Rich console and progress utilities for chart-binder.

Provides a global Rich console instance and helpers for consistent
output formatting across all CLI commands.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.status import Status

# Global console instance (initialized in CLI)
_console: Console | None = None


def get_console() -> Console:
    """Get the global Rich console instance.

    Returns:
        The global Console instance

    Raises:
        RuntimeError: If console not initialized (should only happen in tests)
    """
    if _console is None:
        raise RuntimeError("Console not initialized. Call set_console() first.")
    return _console


def set_console(console: Console) -> None:
    """Set the global Rich console instance.

    Args:
        console: The Console instance to use globally
    """
    global _console
    _console = console


@contextmanager
def make_progress(
    transient: bool = False,
    show_percentage: bool = True,
) -> Iterator[Progress]:
    """Create a Rich Progress context for tracking operations.

    Args:
        transient: If True, progress bar disappears when complete
        show_percentage: Whether to show percentage complete

    Yields:
        Progress instance for tracking tasks

    Example:
        with make_progress() as progress:
            task = progress.add_task("Processing...", total=100)
            for i in range(100):
                progress.update(task, advance=1)
    """
    columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ]

    if show_percentage:
        from rich.progress import TaskProgressColumn

        columns.append(TaskProgressColumn())

    columns.append(TimeRemainingColumn())

    with Progress(*columns, transient=transient, console=get_console()) as progress:
        yield progress


@contextmanager
def status(
    message: str,
    spinner: str = "dots",
) -> Iterator[Status]:
    """Create a Rich Status context for showing ongoing operations.

    Args:
        message: Status message to display
        spinner: Spinner style name (default: "dots")

    Yields:
        Status instance for updating message

    Example:
        with status("Searching MusicBrainz...") as st:
            # Do work
            st.update("Hydrating results...")
    """
    with get_console().status(message, spinner=spinner) as st:
        yield st


def print(*args: Any, **kwargs: Any) -> None:
    """Print to the global console.

    Wrapper around console.print() that uses the global console instance.

    Args:
        *args: Positional arguments passed to console.print()
        **kwargs: Keyword arguments passed to console.print()
    """
    get_console().print(*args, **kwargs)


def print_error(message: str) -> None:
    """Print an error message in red.

    Args:
        message: Error message to display
    """
    get_console().print(f"[red]Error: {message}[/red]")


def print_warning(message: str) -> None:
    """Print a warning message in yellow.

    Args:
        message: Warning message to display
    """
    get_console().print(f"[yellow]Warning: {message}[/yellow]")


def print_success(message: str) -> None:
    """Print a success message in green.

    Args:
        message: Success message to display
    """
    get_console().print(f"[green]{message}[/green]")
