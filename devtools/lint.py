import subprocess

from funlog import log_calls
from rich import get_console, reconfigure
from rich import print as rprint

# Update as needed.
SRC_PATHS = ["src", "tests", "devtools"]
DOC_PATHS = ["README.md"]


reconfigure(emoji=not get_console().options.legacy_windows)  # No emojis on legacy windows.


def main():
    rprint()

    errcount = 0
    errcount += run(["codespell", "--write-changes", *SRC_PATHS, *DOC_PATHS])
    errcount += run(["ruff", "check", "--fix", *SRC_PATHS])
    errcount += run(["ruff", "format", *SRC_PATHS])
    errcount += run_pyright(["basedpyright", "--level", "error", "--stats", *SRC_PATHS])

    rprint()

    if errcount != 0:
        rprint(f"[bold red]:x: Lint failed with {errcount} errors.[/bold red]")
    else:
        rprint("[bold green]:white_check_mark: Lint passed![/bold green]")
    rprint()

    return errcount


@log_calls(level="warning", show_timing_only=True)
def run_pyright(cmd: list[str]) -> int:
    """Run basedpyright and check output for actual errors."""
    rprint()
    rprint(f"[bold green]>> {' '.join(cmd)}[/bold green]")
    errcount = 0
    try:
        result = subprocess.run(cmd, text=True, capture_output=True)
        rprint(result.stdout)
        if result.stderr:
            rprint(result.stderr)
        if "0 errors" not in result.stdout:
            errcount = 1
    except KeyboardInterrupt:
        rprint("[yellow]Keyboard interrupt - Cancelled[/yellow]")
        errcount = 1
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error: {e}[/bold red]")
        errcount = 1

    return errcount


@log_calls(level="warning", show_timing_only=True)
def run(cmd: list[str]) -> int:
    rprint()
    rprint(f"[bold green]>> {' '.join(cmd)}[/bold green]")
    errcount = 0
    try:
        subprocess.run(cmd, text=True, check=True)
    except KeyboardInterrupt:
        rprint("[yellow]Keyboard interrupt - Cancelled[/yellow]")
        errcount = 1
    except subprocess.CalledProcessError as e:
        rprint(f"[bold red]Error: {e}[/bold red]")
        errcount = 1

    return errcount


if __name__ == "__main__":
    exit(main())
