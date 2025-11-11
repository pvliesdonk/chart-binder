from __future__ import annotations

from pathlib import Path

import click

from chart_binder.config import Config


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration TOML file",
)
@click.option("--offline", is_flag=True, help="Run in offline mode")
@click.pass_context
def canon(ctx: click.Context, config: Path | None, offline: bool) -> None:
    """
    Chart-Binder: Charts-aware audio tagger.

    Pick the most canonical release, link MB/Discogs/Spotify IDs,
    and embed compact chart history.
    """
    cfg = Config.load(config)
    if offline:
        cfg.offline_mode = True
    ctx.obj = cfg


@canon.command()
@click.pass_obj
def scan(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Scan audio files and discover metadata."""
    click.echo("Scan subcommand (not yet implemented)")


@canon.command()
@click.option("--explain", is_flag=True, help="Show detailed decision rationale")
@click.pass_obj
def decide(config: Config, explain: bool) -> None:  # pyright: ignore[reportUnusedParameter]
    """Make canonicalization decisions."""
    click.echo("Decide subcommand (not yet implemented)")
    if explain:
        click.echo("(Would show detailed rationale)")


@canon.command()
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.pass_obj
def write(config: Config, dry_run: bool) -> None:  # pyright: ignore[reportUnusedParameter]
    """Write tags to audio files."""
    click.echo("Write subcommand (not yet implemented)")
    if dry_run:
        click.echo("(Dry run mode)")


@canon.group()
def cache() -> None:
    """Manage HTTP and entity caches."""
    pass


@cache.command()
@click.pass_obj
def status(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Show cache status and statistics."""
    click.echo("Cache status (not yet implemented)")


@cache.command()
@click.pass_obj
def purge(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Clear all caches."""
    click.echo("Cache purge (not yet implemented)")


@canon.command()
@click.pass_obj
def coverage(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Generate coverage reports."""
    click.echo("Coverage report (not yet implemented)")


@canon.group()
def charts() -> None:
    """Manage chart data."""
    pass


@charts.command()
@click.pass_obj
def ingest(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Ingest chart data from sources."""
    click.echo("Charts ingest (not yet implemented)")


@canon.group()
def drift() -> None:
    """Manage decision drift."""
    pass


@drift.command()
@click.pass_obj
def review(config: Config) -> None:  # pyright: ignore[reportUnusedParameter]
    """Review decision drift."""
    click.echo("Drift review (not yet implemented)")


def main() -> None:
    """Entry point for the canon CLI."""
    canon()


if __name__ == "__main__":
    main()
