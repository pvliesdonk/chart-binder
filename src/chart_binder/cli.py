from __future__ import annotations

import json
import logging
import sqlite3
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import click

from chart_binder.config import Config
from chart_binder.safe_logging import configure_safe_logging

# Supported audio file extensions
AUDIO_EXTENSIONS = ("*.mp3", "*.flac", "*.ogg", "*.m4a")


class OutputFormat(StrEnum):
    """Output format for CLI commands."""

    TEXT = "text"
    JSON = "json"


class ExitCode:
    """Standard exit codes for CLI commands."""

    SUCCESS = 0
    ERROR = 1
    NO_RESULTS = 2


def _collect_audio_files(paths: tuple[Path, ...]) -> list[Path]:
    """
    Collect audio files from paths (files or directories).

    Recursively searches directories for supported audio formats.
    """
    audio_files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for ext in AUDIO_EXTENSIONS:
                audio_files.extend(path.rglob(ext))
        else:
            audio_files.append(path)
    return audio_files


def _get_rationale_value(rationale: Any) -> str | None:
    """Safely extract rationale value from enum or string."""
    if rationale is None:
        return None
    return rationale.value if hasattr(rationale, "value") else str(rationale)


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration TOML file",
)
@click.option("--offline", is_flag=True, help="Run in offline mode (no network requests)")
@click.option("--frozen", is_flag=True, help="Use only cached data, fail if cache miss")
@click.option("--refresh", is_flag=True, help="Force refresh of cached data")
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.option("-v", "--verbose", count=True, help="Increase verbosity (-v for INFO, -vv for DEBUG)")
# Database options
@click.option("--db-music-graph", type=click.Path(path_type=Path), help="Music graph database path")
@click.option("--db-charts", type=click.Path(path_type=Path), help="Charts database path")
@click.option("--db-decisions", type=click.Path(path_type=Path), help="Decisions database path")
# Cache options
@click.option("--cache-dir", type=click.Path(path_type=Path), help="HTTP cache directory")
@click.option("--cache-ttl", type=int, help="Cache TTL in seconds")
@click.option("--no-cache", is_flag=True, help="Disable HTTP caching")
# LLM options
@click.option("--llm-provider", type=click.Choice(["ollama", "openai"]), help="LLM provider")
@click.option("--llm-model", help="LLM model ID")
@click.option("--llm-enabled/--llm-disabled", default=None, help="Enable/disable LLM adjudication")
@click.option("--llm-temperature", type=float, help="LLM temperature (0.0-2.0)")
# SearxNG options
@click.option("--searxng-url", help="SearxNG instance URL")
@click.option("--searxng-enabled/--searxng-disabled", default=None, help="Enable/disable SearxNG")
@click.pass_context
def canon(
    ctx: click.Context,
    config: Path | None,
    offline: bool,
    frozen: bool,
    refresh: bool,
    output: str,
    verbose: int,
    db_music_graph: Path | None,
    db_charts: Path | None,
    db_decisions: Path | None,
    cache_dir: Path | None,
    cache_ttl: int | None,
    no_cache: bool,
    llm_provider: str | None,
    llm_model: str | None,
    llm_enabled: bool | None,
    llm_temperature: float | None,
    searxng_url: str | None,
    searxng_enabled: bool | None,
) -> None:
    """
    Chart-Binder: Charts-aware audio tagger.

    Pick the most canonical release, link MB/Discogs/Spotify IDs,
    and embed compact chart history.
    """
    logger = logging.getLogger(__name__)

    # Load config (TOML + env vars)
    cfg = Config.load(config)
    if config:
        logger.info(f"Loaded config from {config}")

    # Apply CLI overrides (highest precedence: CLI > Env > Config File > Defaults)
    if offline or frozen:
        cfg.offline_mode = True

    # Database overrides
    if db_music_graph:
        cfg.database.music_graph_path = db_music_graph
    if db_charts:
        cfg.database.charts_path = db_charts
    if db_decisions:
        cfg.database.decisions_path = db_decisions

    # Cache overrides
    if cache_dir:
        cfg.http_cache.directory = cache_dir
    if cache_ttl is not None:
        cfg.http_cache.ttl_seconds = cache_ttl
    if no_cache:
        cfg.http_cache.enabled = False

    # LLM overrides
    if llm_enabled is not None:
        cfg.llm.enabled = llm_enabled
    if llm_provider:
        cfg.llm.provider = llm_provider  # type: ignore[assignment]
    if llm_model:
        cfg.llm.model_id = llm_model
    if llm_temperature is not None:
        cfg.llm.temperature = llm_temperature

    # SearxNG overrides
    if searxng_url:
        cfg.llm.searxng.url = searxng_url
    if searxng_enabled is not None:
        cfg.llm.searxng.enabled = searxng_enabled

    # Configure logging with CLI > Config precedence
    # If CLI verbose flag is provided, use it; otherwise use config setting
    if verbose > 0:
        # CLI flag takes precedence
        if verbose >= 2:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
    else:
        # Use config file setting
        level_str = cfg.logging.level.upper()
        log_level = getattr(logging, level_str, logging.WARNING)

    configure_safe_logging(
        level=log_level,
        format_string=cfg.logging.format,
        hash_paths=cfg.logging.hash_paths,
    )

    logger.debug(
        f"Logging configured: level={logging.getLevelName(log_level)}, hash_paths={cfg.logging.hash_paths}"
    )

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["offline"] = offline
    ctx.obj["frozen"] = frozen
    ctx.obj["refresh"] = refresh
    ctx.obj["output"] = OutputFormat(output)


@canon.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.pass_context
def scan(ctx: click.Context, paths: tuple[Path, ...]) -> None:
    """
    Scan audio files and discover metadata.

    Reads existing tags and prints facts about each file.
    Supports MP3, FLAC, OGG, and MP4/M4A files.
    """
    from chart_binder.tagging import verify

    logger = logging.getLogger(__name__)
    logger.info(f"Running scan command on {len(paths)} path(s)")

    output_format = ctx.obj["output"]
    results = []
    audio_files = _collect_audio_files(paths)
    logger.debug(f"Collected {len(audio_files)} audio files")

    for audio_file in audio_files:
        try:
            tagset = verify(audio_file)
            result = {
                "file": str(audio_file),
                "title": tagset.title,
                "artist": tagset.artist,
                "album": tagset.album,
                "original_year": tagset.original_year,
                "track_number": tagset.track_number,
                "disc_number": tagset.disc_number,
                "mb_recording_id": tagset.ids.mb_recording_id,
                "mb_release_group_id": tagset.ids.mb_release_group_id,
                "mb_release_id": tagset.ids.mb_release_id,
                "charts_blob": tagset.compact.charts_blob,
                "decision_trace": tagset.compact.decision_trace,
            }
            results.append(result)

            if output_format == OutputFormat.TEXT:
                click.echo(f"\n✔︎ {audio_file}")
                click.echo(f"  Title: {tagset.title or '(none)'}")
                click.echo(f"  Artist: {tagset.artist or '(none)'}")
                click.echo(f"  Album: {tagset.album or '(none)'}")
                click.echo(f"  Year: {tagset.original_year or '(none)'}")
                if tagset.ids.mb_release_group_id:
                    click.echo(f"  MB RG: {tagset.ids.mb_release_group_id}")
                if tagset.compact.decision_trace:
                    click.echo(f"  Trace: {tagset.compact.decision_trace}")

        except Exception as e:
            if output_format == OutputFormat.TEXT:
                click.echo(f"\n✘ {audio_file}: {e}", err=True)
            results.append({"file": str(audio_file), "error": str(e)})

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(results, indent=2))

    sys.exit(ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@canon.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--explain", is_flag=True, help="Show detailed decision rationale")
@click.pass_context
def decide(ctx: click.Context, paths: tuple[Path, ...], explain: bool) -> None:
    """
    Make canonicalization decisions for audio files.

    Resolves the Canonical Release Group (CRG) and Representative Release (RR)
    for each file based on available metadata.
    """
    from chart_binder.resolver import ConfigSnapshot, Resolver
    from chart_binder.tagging import verify

    logger = logging.getLogger(__name__)
    logger.info(f"Running decide command on {len(paths)} path(s), explain={explain}")

    output_format = ctx.obj["output"]
    results: list[dict[str, Any]] = []

    # Create resolver with config
    resolver_config = ConfigSnapshot(
        lead_window_days=90,
        reissue_long_gap_years=10,
    )
    resolver = Resolver(resolver_config)
    audio_files = _collect_audio_files(paths)
    logger.debug(f"Collected {len(audio_files)} audio files")

    for audio_file in audio_files:
        try:
            tagset = verify(audio_file)

            # Build minimal evidence bundle from existing tags
            evidence_bundle: dict[str, Any] = {
                "artifact": {
                    "file_path": str(audio_file),
                },
                "artist": {
                    "name": tagset.artist or "Unknown",
                    "mb_artist_id": None,
                },
                "recording_candidates": [],
                "timeline_facts": {},
                "provenance": {
                    "sources_used": ["local_tags"],
                },
            }

            # If MB IDs exist, build candidate structure
            if tagset.ids.mb_release_group_id:
                evidence_bundle["recording_candidates"] = [
                    {
                        "mb_recording_id": tagset.ids.mb_recording_id,
                        "title": tagset.title,
                        "rg_candidates": [
                            {
                                "mb_rg_id": tagset.ids.mb_release_group_id,
                                "title": tagset.album,
                                "primary_type": "Album",
                                "first_release_date": tagset.original_year,
                                "releases": [
                                    {
                                        "mb_release_id": tagset.ids.mb_release_id or "unknown",
                                        "date": tagset.original_year,
                                        "country": tagset.country,
                                        "label": tagset.label,
                                        "title": tagset.album,
                                        "flags": {"is_official": True},
                                    }
                                ],
                            }
                        ],
                    }
                ]

            decision = resolver.resolve(evidence_bundle)

            result = {
                "file": str(audio_file),
                "state": decision.state.value,
                "crg_mbid": decision.release_group_mbid,
                "rr_mbid": decision.release_mbid,
                "crg_rationale": _get_rationale_value(decision.crg_rationale),
                "rr_rationale": _get_rationale_value(decision.rr_rationale),
                "compact_tag": decision.compact_tag,
            }

            if explain:
                trace_dict: dict[str, Any] = {
                    "evidence_hash": decision.decision_trace.evidence_hash,
                    "considered_candidates": decision.decision_trace.considered_candidates,
                    "crg_selection": decision.decision_trace.crg_selection,
                    "rr_selection": decision.decision_trace.rr_selection,
                    "missing_facts": decision.decision_trace.missing_facts,
                }
                result["trace"] = json.dumps(trace_dict)

            results.append(result)

            if output_format == OutputFormat.TEXT:
                state_icon = "✔︎" if decision.state.value == "decided" else "∆"
                click.echo(f"\n{state_icon} {audio_file}")
                click.echo(f"  State: {decision.state.value}")
                if decision.release_group_mbid:
                    click.echo(f"  CRG: {decision.release_group_mbid}")
                    click.echo(f"       ({decision.crg_rationale})")
                if decision.release_mbid:
                    click.echo(f"  RR:  {decision.release_mbid}")
                    click.echo(f"       ({decision.rr_rationale})")
                click.echo(f"  Trace: {decision.compact_tag}")
                if explain:
                    click.echo("\n" + decision.decision_trace.to_human_readable())

        except Exception as e:
            if output_format == OutputFormat.TEXT:
                click.echo(f"\n✘ {audio_file}: {e}", err=True)
            results.append({"file": str(audio_file), "error": str(e)})

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(results, indent=2))

    sys.exit(ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@canon.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.option("--apply", is_flag=True, help="Apply changes (required for actual writes)")
@click.pass_context
def write(ctx: click.Context, paths: tuple[Path, ...], dry_run: bool, apply: bool) -> None:
    """
    Write canonical tags to audio files.

    Writes decision trace, canonical IDs, and optionally CHARTS blob to files.
    Use --dry-run to preview changes without writing.
    Use --apply to confirm actual writes (safety feature).
    """
    from chart_binder.tagging import (
        CanonicalIDs,
        CompactFields,
        TagSet,
        write_tags,
    )

    output_format = ctx.obj["output"]
    results = []

    if not dry_run and not apply:
        click.echo("Error: Use --dry-run to preview or --apply to write changes.", err=True)
        sys.exit(ExitCode.ERROR)

    audio_files = _collect_audio_files(paths)

    for audio_file in audio_files:
        try:
            # For now, create a minimal tagset for demonstration
            # In full implementation, this would come from decide()
            tagset = TagSet(
                ids=CanonicalIDs(),
                compact=CompactFields(
                    ruleset_version="canon-1.0",
                ),
            )

            report = write_tags(
                audio_file,
                tagset,
                authoritative=False,  # Augment-only mode
                dry_run=dry_run,
            )

            result = {
                "file": str(audio_file),
                "dry_run": report.dry_run,
                "fields_written": report.fields_written,
                "fields_skipped": report.fields_skipped,
                "originals_stashed": report.originals_stashed,
                "errors": report.errors,
            }
            results.append(result)

            if output_format == OutputFormat.TEXT:
                mode = "(dry run)" if dry_run else ""
                if report.errors:
                    click.echo(f"\n✘ {audio_file} {mode}")
                    for error in report.errors:
                        click.echo(f"  Error: {error}")
                else:
                    click.echo(f"\n✔︎ {audio_file} {mode}")
                    if report.fields_written:
                        click.echo(f"  Written: {', '.join(report.fields_written)}")
                    if report.originals_stashed:
                        click.echo(f"  Stashed: {', '.join(report.originals_stashed)}")

        except Exception as e:
            if output_format == OutputFormat.TEXT:
                click.echo(f"\n✘ {audio_file}: {e}", err=True)
            results.append({"file": str(audio_file), "error": str(e)})

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(results, indent=2))

    # Summary
    if output_format == OutputFormat.TEXT:
        total = len(results)
        errors = sum(1 for r in results if r.get("errors") or r.get("error"))
        click.echo(f"\nProcessed {total} files, {errors} errors")

    sys.exit(ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@canon.group()
def cache() -> None:
    """Manage HTTP and entity caches."""
    pass


@cache.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show cache status and statistics."""
    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    cache_dir = config.http_cache.directory
    cache_db = cache_dir / "cache_index.sqlite"

    result = {
        "cache_directory": str(cache_dir),
        "cache_enabled": config.http_cache.enabled,
        "ttl_seconds": config.http_cache.ttl_seconds,
        "entries": 0,
        "total_size_bytes": 0,
        "expired_entries": 0,
    }

    if cache_db.exists():
        import time

        with sqlite3.connect(cache_db) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM cache_entries")
            result["entries"] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at <= ?", (time.time(),)
            )
            result["expired_entries"] = cursor.fetchone()[0]

        # Calculate total size of cached files
        total_size = sum(f.stat().st_size for f in cache_dir.glob("*.cache") if f.is_file())
        result["total_size_bytes"] = total_size

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("Cache Status")
        click.echo("=" * 40)
        click.echo(f"  Directory: {result['cache_directory']}")
        click.echo(f"  Enabled: {result['cache_enabled']}")
        click.echo(f"  TTL: {result['ttl_seconds']} seconds")
        click.echo(f"  Entries: {result['entries']}")
        click.echo(f"  Expired: {result['expired_entries']}")
        total_bytes = result["total_size_bytes"]
        size_mb = int(total_bytes) / (1024 * 1024)
        click.echo(f"  Size: {size_mb:.2f} MB")

    sys.exit(ExitCode.SUCCESS)


@cache.command()
@click.option("--expired-only", is_flag=True, help="Only purge expired entries")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def purge(ctx: click.Context, expired_only: bool, force: bool) -> None:
    """Clear caches."""
    from chart_binder.http_cache import HttpCache

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    cache_dir = config.http_cache.directory

    if not cache_dir.exists():
        click.echo("Cache directory does not exist.")
        sys.exit(ExitCode.SUCCESS)

    cache = HttpCache(cache_dir, ttl_seconds=config.http_cache.ttl_seconds)

    removed = 0
    if expired_only:
        removed = cache.purge_expired()
        result = {"action": "purge_expired", "removed_entries": removed}
    else:
        if not force:
            click.confirm("Are you sure you want to clear all caches?", abort=True)
        cache.clear()
        result = {"action": "purge_all", "removed_entries": "all"}

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        if expired_only:
            click.echo(f"✔︎ Purged {removed} expired entries")
        else:
            click.echo("✔︎ All caches cleared")

    sys.exit(ExitCode.SUCCESS)


@canon.group()
def coverage() -> None:
    """Generate coverage reports."""
    pass


@coverage.command("chart")
@click.argument("chart_id")
@click.argument("period")
@click.pass_context
def coverage_chart(ctx: click.Context, chart_id: str, period: str) -> None:
    """Show coverage report for a chart run."""
    from chart_binder.charts_db import ChartsDB

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = ChartsDB(config.database.charts_path)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        click.echo(f"No chart run found for {chart_id} {period}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    report = db.get_coverage_report(run["run_id"])

    result = {
        "chart_id": chart_id,
        "period": period,
        "run_id": run["run_id"],
        "total_entries": report.total_entries,
        "linked_entries": report.linked_entries,
        "unlinked_entries": report.unlinked_entries,
        "coverage_pct": round(report.coverage_pct, 2),
        "by_method": report.by_method,
        "by_confidence": report.by_confidence,
    }

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Coverage Report: {chart_id} {period}")
        click.echo("=" * 40)
        click.echo(f"  Total entries: {report.total_entries}")
        click.echo(f"  Linked: {report.linked_entries}")
        click.echo(f"  Unlinked: {report.unlinked_entries}")
        click.echo(f"  Coverage: {report.coverage_pct:.1f}%")
        if report.by_method:
            click.echo("  By method:")
            for method, count in report.by_method.items():
                click.echo(f"    {method}: {count}")
        if report.by_confidence:
            click.echo("  By confidence:")
            for bucket, count in report.by_confidence.items():
                click.echo(f"    {bucket}: {count}")

    sys.exit(ExitCode.SUCCESS)


@coverage.command("missing")
@click.argument("chart_id")
@click.argument("period")
@click.option("--threshold", default=0.60, help="Minimum confidence threshold")
@click.pass_context
def coverage_missing(ctx: click.Context, chart_id: str, period: str, threshold: float) -> None:
    """Show entries missing from chart linkage."""
    from chart_binder.charts_db import ChartsDB, ChartsETL

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = ChartsDB(config.database.charts_path)
    etl = ChartsETL(db)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        click.echo(f"No chart run found for {chart_id} {period}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    missing = etl.get_missing_entries(run["run_id"], threshold=threshold)

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(missing, indent=2))
    else:
        click.echo(f"Missing Entries: {chart_id} {period}")
        click.echo("=" * 40)
        for entry in missing:
            click.echo(f"  #{entry['rank']}: {entry['artist_raw']} - {entry['title_raw']}")
            if entry.get("confidence"):
                click.echo(f"        (confidence: {entry['confidence']:.2f})")

    sys.exit(ExitCode.SUCCESS if missing else ExitCode.NO_RESULTS)


@canon.group()
def charts() -> None:
    """Manage chart data."""
    pass


@charts.command("scrape")
@click.argument("chart_type", type=click.Choice(["t40", "t40jaar", "top2000", "zwaarste"]))
@click.argument("period")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output JSON file")
@click.option("--ingest", is_flag=True, help="Automatically ingest scraped data into database")
@click.option("--strict", is_flag=True, help="Fail if entry count is below expected")
@click.option(
    "--check-continuity", is_flag=True, help="Validate overlap with previous run (weekly charts)"
)
@click.pass_context
def charts_scrape(
    ctx: click.Context,
    chart_type: str,
    period: str,
    output: Path | None,
    ingest: bool,
    strict: bool,
    check_continuity: bool,
) -> None:
    """
    Scrape chart data from web source.

    CHART_TYPE: Chart to scrape (t40, t40jaar, top2000, zwaarste)
    PERIOD: Period to scrape (YYYY-Www for weekly, YYYY for yearly)

    Examples:
      canon charts scrape t40 2024-W01
      canon charts scrape t40jaar 2023 --ingest
      canon charts scrape top2000 2024 --strict
      canon charts scrape t40 2024-W02 --check-continuity --ingest
    """
    from chart_binder.charts_db import ChartsDB, ChartsETL
    from chart_binder.http_cache import HttpCache
    from chart_binder.scrapers import SCRAPER_REGISTRY, calculate_overlap

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    cache = HttpCache(config.http_cache.directory, ttl_seconds=config.http_cache.ttl_seconds)
    db = ChartsDB(config.charts_db.path)

    # Get scraper class and DB ID from registry
    scraper_cls, chart_db_id = SCRAPER_REGISTRY[chart_type]

    with scraper_cls(cache) as scraper:
        result = scraper.scrape_with_validation(period)

    if not result.entries:
        click.echo(f"No entries found for {chart_type} {period}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    # Check entry count sanity
    if not result.is_valid:
        msg = (
            f"⚠ Entry count sanity check failed: got {result.actual_count}, "
            f"expected ~{result.expected_count} (shortage: {result.shortage})"
        )
        if strict:
            click.echo(f"✘ {msg}", err=True)
            click.echo("Possible edge case detected - check scraper for this period", err=True)
            sys.exit(ExitCode.ERROR)
        else:
            click.echo(msg, err=True)

    # Check continuity with previous run (for weekly charts)
    if check_continuity and chart_type == "t40":
        prev_period = db.get_adjacent_period(chart_db_id, period, direction=-1)
        if prev_period:
            prev_entries = db.get_entries_by_period(chart_db_id, prev_period)
            if prev_entries:
                overlap = calculate_overlap(result.entries, prev_entries)
                result.continuity_overlap = overlap
                result.continuity_reference = prev_period

                if not result.continuity_valid:
                    msg = (
                        f"⚠ Continuity check failed: only {overlap:.0%} overlap with {prev_period} "
                        f"(expected ≥50%)"
                    )
                    if strict:
                        click.echo(f"✘ {msg}", err=True)
                        click.echo("Possible scraping issue - data may be corrupted", err=True)
                        sys.exit(ExitCode.ERROR)
                    else:
                        click.echo(msg, err=True)
                elif output_format == OutputFormat.TEXT:
                    click.echo(f"✔︎ Continuity check: {overlap:.0%} overlap with {prev_period}")

    # Show warnings if any
    if result.warnings:
        for warning in result.warnings:
            click.echo(f"⚠ {warning}", err=True)

    # Convert to list of lists for JSON serialization
    entries_list = [[rank, artist, title] for rank, artist, title in result.entries]

    output_result = {
        "chart_type": chart_type,
        "chart_db_id": chart_db_id,
        "period": period,
        "entries_count": result.actual_count,
        "expected_count": result.expected_count,
        "is_valid": result.is_valid,
        "continuity_overlap": result.continuity_overlap,
        "continuity_reference": result.continuity_reference,
        "entries": entries_list,
    }

    if output:
        output.write_text(json.dumps(entries_list, indent=2, ensure_ascii=False))
        if output_format == OutputFormat.TEXT:
            click.echo(
                f"✔︎ Scraped {result.actual_count}/{result.expected_count} entries to {output}"
            )
        elif output_format == OutputFormat.JSON:
            click.echo(json.dumps({"status": "success", "output_file": str(output)}, indent=2))
    else:
        if output_format == OutputFormat.JSON:
            click.echo(json.dumps(output_result, indent=2, ensure_ascii=False))
        else:
            status = "✔︎" if result.is_valid else "⚠"
            click.echo(
                f"{status} Scraped {result.actual_count}/{result.expected_count} entries for {chart_type} {period}"
            )
            click.echo("\nFirst 10 entries:")
            for rank, artist, title in result.entries[:10]:
                click.echo(f"  {rank:3d}. {artist} - {title}")
            if len(result.entries) > 10:
                click.echo(f"  ... and {len(result.entries) - 10} more")

    # Auto-ingest if requested
    if ingest:
        etl = ChartsETL(db)

        # Ensure chart exists
        db.upsert_chart(
            chart_id=chart_db_id,
            name=f"{chart_type} chart",
            frequency="weekly" if chart_type == "t40" else "yearly",
            jurisdiction="NL",
        )

        # Ingest entries
        entries_for_ingest = [(rank, artist, title) for rank, artist, title in result.entries]
        run_id = etl.ingest_run(chart_db_id, period, entries_for_ingest)

        if output_format == OutputFormat.TEXT:
            click.echo(f"✔︎ Ingested {result.actual_count} entries (run_id: {run_id[:8]}...)")
        elif output_format == OutputFormat.JSON:
            click.echo(json.dumps({"ingested": True, "run_id": run_id}, indent=2))

    sys.exit(ExitCode.SUCCESS)


@charts.command("scrape-missing")
@click.argument("chart_type", type=click.Choice(["t40", "t40jaar", "top2000", "zwaarste"]))
@click.option("--start-year", type=int, help="Start year (default: earliest available)")
@click.option("--end-year", type=int, help="End year (default: current year)")
@click.option("--ingest", is_flag=True, help="Automatically ingest scraped data")
@click.option("--strict", is_flag=True, help="Fail on entry count sanity check failures")
@click.option("--dry-run", is_flag=True, help="Show what would be scraped without scraping")
@click.pass_context
def charts_scrape_missing(
    ctx: click.Context,
    chart_type: str,
    start_year: int | None,
    end_year: int | None,
    ingest: bool,
    strict: bool,
    dry_run: bool,
) -> None:
    """
    Scrape all missing periods for a chart type.

    Checks the database for existing runs and scrapes only missing periods.

    Examples:
      canon charts scrape-missing t40 --start-year 2020 --ingest
      canon charts scrape-missing t40jaar --dry-run
      canon charts scrape-missing top2000 --start-year 1999 --end-year 2024
    """
    import datetime

    from chart_binder.charts_db import ChartsDB, ChartsETL
    from chart_binder.http_cache import HttpCache
    from chart_binder.scrapers import SCRAPER_REGISTRY

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    # Get scraper class and DB ID from registry
    scraper_cls, chart_db_id = SCRAPER_REGISTRY[chart_type]

    # Determine year range
    current_year = datetime.datetime.now().year
    if end_year is None:
        end_year = current_year

    # Set default start years per chart type
    default_start_years = {
        "t40": 1965,  # Top 40 started in 1965
        "t40jaar": 1965,
        "top2000": 1999,  # Top 2000 started in 1999
        "zwaarste": 2020,  # Limited URL map
    }
    if start_year is None:
        start_year = default_start_years.get(chart_type, 2000)

    # Get existing runs from database
    db = ChartsDB(config.charts_db.path)
    existing_runs = db.list_runs(chart_db_id)
    existing_periods = {run["period"] for run in existing_runs}

    # Generate all expected periods
    expected_periods: list[str] = []
    if chart_type == "t40":
        # Weekly chart - generate YYYY-Www for each week
        for year in range(start_year, end_year + 1):
            # Approximately 52 weeks per year
            for week in range(1, 53):
                expected_periods.append(f"{year}-W{week:02d}")
    else:
        # Yearly charts
        for year in range(start_year, end_year + 1):
            expected_periods.append(str(year))

    # Find missing periods
    missing_periods = [p for p in expected_periods if p not in existing_periods]

    if not missing_periods:
        click.echo(f"✔︎ No missing periods for {chart_type} ({start_year}-{end_year})")
        sys.exit(ExitCode.SUCCESS)

    # Report
    if output_format == OutputFormat.JSON:
        result = {
            "chart_type": chart_type,
            "chart_db_id": chart_db_id,
            "start_year": start_year,
            "end_year": end_year,
            "total_expected": len(expected_periods),
            "existing": len(existing_periods),
            "missing": len(missing_periods),
            "missing_periods": missing_periods if dry_run else missing_periods[:10],
        }
        if dry_run:
            click.echo(json.dumps(result, indent=2))
            sys.exit(ExitCode.SUCCESS)
    else:
        click.echo(f"Chart: {chart_type} ({chart_db_id})")
        click.echo(f"Range: {start_year} - {end_year}")
        click.echo(f"Expected periods: {len(expected_periods)}")
        click.echo(f"Existing: {len(existing_periods)}")
        click.echo(f"Missing: {len(missing_periods)}")

        if dry_run:
            click.echo("\nMissing periods (first 20):")
            for period in missing_periods[:20]:
                click.echo(f"  {period}")
            if len(missing_periods) > 20:
                click.echo(f"  ... and {len(missing_periods) - 20} more")
            sys.exit(ExitCode.SUCCESS)

    # Actually scrape missing periods
    cache = HttpCache(config.http_cache.directory, ttl_seconds=config.http_cache.ttl_seconds)
    etl = ChartsETL(db) if ingest else None

    # Ensure chart exists if ingesting
    if ingest:
        db.upsert_chart(
            chart_id=chart_db_id,
            name=f"{chart_type} chart",
            frequency="weekly" if chart_type == "t40" else "yearly",
            jurisdiction="NL",
        )

    scraped = 0
    failed = 0
    skipped = 0

    click.echo(f"\nScraping {len(missing_periods)} missing periods...")

    with scraper_cls(cache) as scraper:
        for i, period in enumerate(missing_periods, 1):
            try:
                result = scraper.scrape_with_validation(period)

                if not result.entries:
                    skipped += 1
                    if output_format == OutputFormat.TEXT:
                        click.echo(f"  [{i}/{len(missing_periods)}] {period}: no data (skipped)")
                    continue

                if not result.is_valid and strict:
                    failed += 1
                    click.echo(
                        f"  [{i}/{len(missing_periods)}] {period}: ✘ sanity check failed "
                        f"({result.actual_count}/{result.expected_count})"
                    )
                    continue

                scraped += 1
                status = "✔︎" if result.is_valid else "⚠"
                if output_format == OutputFormat.TEXT:
                    click.echo(
                        f"  [{i}/{len(missing_periods)}] {period}: {status} "
                        f"{result.actual_count}/{result.expected_count} entries"
                    )

                # Ingest if requested
                if ingest and etl:
                    entries_for_ingest = [
                        (rank, artist, title) for rank, artist, title in result.entries
                    ]
                    etl.ingest_run(chart_db_id, period, entries_for_ingest)

            except Exception as e:
                failed += 1
                if output_format == OutputFormat.TEXT:
                    click.echo(f"  [{i}/{len(missing_periods)}] {period}: ✘ error: {e}")

    # Summary
    if output_format == OutputFormat.JSON:
        click.echo(
            json.dumps(
                {
                    "scraped": scraped,
                    "failed": failed,
                    "skipped": skipped,
                    "ingested": ingest,
                },
                indent=2,
            )
        )
    else:
        click.echo(f"\nSummary: {scraped} scraped, {failed} failed, {skipped} skipped")
        if ingest:
            click.echo(f"Ingested: {scraped} runs")

    sys.exit(ExitCode.SUCCESS if failed == 0 else ExitCode.ERROR)


@charts.command()
@click.argument("chart_id")
@click.argument("period")
@click.argument("source_file", type=click.Path(exists=True, path_type=Path))
@click.option("--notes", help="Notes about this chart run")
@click.pass_context
def ingest(
    ctx: click.Context, chart_id: str, period: str, source_file: Path, notes: str | None
) -> None:
    """
    Ingest chart data from a source file.

    SOURCE_FILE should be a JSON file with entries: [[rank, artist, title], ...]
    """
    from chart_binder.charts_db import ChartsDB, ChartsETL

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    # Load entries from JSON file
    try:
        data = json.loads(source_file.read_text())
        entries = [(entry[0], entry[1], entry[2]) for entry in data]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        click.echo(f"Error parsing source file: {e}", err=True)
        sys.exit(ExitCode.ERROR)

    db = ChartsDB(config.database.charts_path)
    etl = ChartsETL(db)

    # Ensure chart exists
    db.upsert_chart(chart_id, chart_id, "y")

    run_id = etl.ingest(chart_id, period, entries, notes=notes)

    result = {
        "chart_id": chart_id,
        "period": period,
        "run_id": run_id,
        "entries_count": len(entries),
    }

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"✔︎ Ingested {len(entries)} entries")
        click.echo(f"  Chart: {chart_id}")
        click.echo(f"  Period: {period}")
        click.echo(f"  Run ID: {run_id}")

    sys.exit(ExitCode.SUCCESS)


@charts.command()
@click.argument("chart_id")
@click.argument("period")
@click.option(
    "--strategy",
    default="title_artist_year",
    help="Linking strategy",
)
@click.pass_context
def link(ctx: click.Context, chart_id: str, period: str, strategy: str) -> None:
    """Link chart entries to work keys."""
    from chart_binder.charts_db import ChartsDB, ChartsETL

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = ChartsDB(config.database.charts_path)
    etl = ChartsETL(db)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        click.echo(f"No chart run found for {chart_id} {period}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    report = etl.link(run["run_id"], strategy=strategy)

    result = {
        "chart_id": chart_id,
        "period": period,
        "run_id": run["run_id"],
        "total_entries": report.total_entries,
        "linked_entries": report.linked_entries,
        "coverage_pct": round(report.coverage_pct, 2),
        "by_method": report.by_method,
    }

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"✔︎ Linked {report.linked_entries}/{report.total_entries} entries")
        click.echo(f"  Coverage: {report.coverage_pct:.1f}%")
        if report.by_method:
            click.echo("  By method:")
            for method, count in report.by_method.items():
                click.echo(f"    {method}: {count}")

    sys.exit(ExitCode.SUCCESS)


@charts.command("missing")
@click.argument("chart_id")
@click.argument("period")
@click.pass_context
def charts_missing(ctx: click.Context, chart_id: str, period: str) -> None:
    """Show unlinked chart entries."""
    # Delegate to coverage missing
    ctx.invoke(coverage_missing, chart_id=chart_id, period=period)


@charts.command("export")
@click.argument("work_key")
@click.option("--positions/--no-positions", default=False, help="Include position details")
@click.pass_context
def charts_export(ctx: click.Context, work_key: str, positions: bool) -> None:
    """Export CHARTS blob for a work key."""
    from chart_binder.charts_db import ChartsDB
    from chart_binder.charts_export import ChartsExporter

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = ChartsDB(config.database.charts_path)
    exporter = ChartsExporter(db)

    blob = exporter.export_for_work(work_key, include_positions=positions)
    json_str = blob.to_json(include_positions=positions)

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps({"work_key": work_key, "charts_blob": json_str}, indent=2))
    else:
        click.echo(f"CHARTS blob for {work_key}:")
        click.echo(json_str)

    sys.exit(ExitCode.SUCCESS)


@canon.group()
def drift() -> None:
    """Manage decision drift."""
    pass


@drift.command()
@click.pass_context
def drift_review(ctx: click.Context) -> None:
    """Review decisions that have drifted."""
    from chart_binder.decisions_db import DecisionsDB
    from chart_binder.drift import DriftDetector

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = DecisionsDB(config.database.decisions_path)
    detector = DriftDetector(db)

    stale_decisions = detector.review_drift()

    results = [
        {
            "file_id": d.file_id,
            "state": d.state.value,
            "evidence_hash": d.evidence_hash,
            "ruleset_version": d.ruleset_version,
            "mb_rg_id": d.mb_rg_id,
            "mb_release_id": d.mb_release_id,
            "work_key": d.work_key,
        }
        for d in stale_decisions
    ]

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(results, indent=2))
    else:
        if not stale_decisions:
            click.echo("✔︎ No drifted decisions found")
        else:
            click.echo(f"Found {len(stale_decisions)} drifted decisions:")
            click.echo("=" * 50)
            for d in stale_decisions:
                click.echo(f"\n  File: {d.file_id[:16]}...")
                click.echo(f"  State: {d.state.value}")
                click.echo(f"  CRG: {d.mb_rg_id}")
                click.echo(f"  RR: {d.mb_release_id}")
                click.echo(f"  Ruleset: {d.ruleset_version}")

    sys.exit(ExitCode.ERROR if stale_decisions else ExitCode.SUCCESS)


@canon.group()
def llm() -> None:
    """LLM adjudication commands (Epic 13)."""
    pass


@llm.command("status")
@click.pass_context
def llm_status(ctx: click.Context) -> None:
    """Show LLM configuration and provider status."""
    from chart_binder.llm import ProviderRegistry

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    registry = ProviderRegistry()
    provider_config = {
        "provider": config.llm.provider,
        "model_id": config.llm.model_id,
        "ollama_base_url": config.llm.ollama_base_url,
        "api_key_env": config.llm.api_key_env,
    }
    provider = registry.create_from_config(provider_config)

    result = {
        "enabled": config.llm.enabled,
        "provider": config.llm.provider,
        "model_id": config.llm.model_id,
        "provider_available": provider.is_available(),
        "auto_accept_threshold": config.llm.auto_accept_threshold,
        "review_threshold": config.llm.review_threshold,
        "timeout_s": config.llm.timeout_s,
        "max_tokens": config.llm.max_tokens,
    }

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("LLM Configuration Status")
        click.echo("=" * 40)
        click.echo(f"  Enabled: {result['enabled']}")
        click.echo(f"  Provider: {result['provider']}")
        click.echo(f"  Model: {result['model_id']}")
        status_icon = "✔︎" if result["provider_available"] else "✘"
        click.echo(f"  Provider Available: {status_icon}")
        click.echo(f"  Auto-accept Threshold: {result['auto_accept_threshold']}")
        click.echo(f"  Review Threshold: {result['review_threshold']}")
        click.echo(f"  Timeout: {result['timeout_s']}s")
        click.echo(f"  Max Tokens: {result['max_tokens']}")

    sys.exit(ExitCode.SUCCESS)


@llm.command("adjudicate")
@click.argument("file_id")
@click.option("--force", is_flag=True, help="Adjudicate even if LLM is disabled in config")
@click.pass_context
def llm_adjudicate(ctx: click.Context, file_id: str, force: bool) -> None:
    """Run LLM adjudication on a specific INDETERMINATE decision."""
    from chart_binder.decisions_db import DecisionsDB
    from chart_binder.llm import LLMAdjudicator
    from chart_binder.musicgraph import MusicGraphDB

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    if not config.llm.enabled and not force:
        click.echo("LLM adjudication is disabled. Use --force to override.", err=True)
        sys.exit(ExitCode.ERROR)

    # Load decision
    decisions_db = DecisionsDB(config.database.decisions_path)
    decision = decisions_db.get_decision(file_id)

    if not decision:
        click.echo(f"No decision found for file_id: {file_id}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    # Create adjudicator with config.llm directly (LLMConfig model)
    adjudicator = LLMAdjudicator(config=config.llm)

    # Build evidence bundle from decision and music graph data
    evidence_bundle: dict[str, Any] = {
        "artist": {"name": "Unknown"},
        "recording_candidates": [],
        "provenance": {"sources_used": ["decisions_db"]},
    }

    # Extract work_key parts (typically "artist // title" format)
    work_key = decision.get("work_key", "")
    if " // " in work_key:
        parts = work_key.split(" // ", 1)
        evidence_bundle["artist"] = {"name": parts[0]}
        if len(parts) > 1:
            evidence_bundle["recording_title"] = parts[1]
    else:
        evidence_bundle["work_key_raw"] = work_key

    # Include decision trace info if available
    trace_compact = decision.get("trace_compact", "")
    if trace_compact:
        evidence_bundle["decision_trace_compact"] = trace_compact

    # Include config snapshot from decision
    config_json = decision.get("config_snapshot_json", "{}")
    try:
        evidence_bundle["config_snapshot"] = json.loads(config_json)
    except json.JSONDecodeError:
        pass

    # Try to enrich evidence from music graph if available
    music_graph_db = MusicGraphDB(config.database.music_graph_path)

    # If we have mb_recording_id, try to get recording info
    mb_recording_id = decision.get("mb_recording_id")
    if mb_recording_id:
        recording = music_graph_db.get_recording(mb_recording_id)
        if recording:
            evidence_bundle["recording_candidates"] = [
                {
                    "title": recording.get("title", ""),
                    "mb_recording_id": mb_recording_id,
                    "rg_candidates": [],
                }
            ]
            # If we have artist info, get it
            artist_mbid = recording.get("artist_mbid")
            if artist_mbid:
                artist = music_graph_db.get_artist(artist_mbid)
                if artist:
                    evidence_bundle["artist"] = {
                        "name": artist.get("name", "Unknown"),
                        "mb_artist_id": artist_mbid,
                        "begin_area_country": artist.get("begin_area_country"),
                    }

    # Include existing decision info for context
    evidence_bundle["existing_decision"] = {
        "mb_rg_id": decision.get("mb_rg_id"),
        "mb_release_id": decision.get("mb_release_id"),
        "state": decision.get("state"),
    }

    # Build decision trace dict from compact string
    decision_trace: dict[str, Any] | None = None
    if trace_compact:
        decision_trace = {"trace_compact": trace_compact}

    result = adjudicator.adjudicate(evidence_bundle, decision_trace)

    result_dict = {
        "outcome": result.outcome.value,
        "crg_mbid": result.crg_mbid,
        "rr_mbid": result.rr_mbid,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "model_id": result.model_id,
        "error": result.error_message,
    }

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(result_dict, indent=2))
    else:
        click.echo("LLM Adjudication Result")
        click.echo("=" * 40)
        click.echo(f"  Outcome: {result.outcome.value}")
        if result.crg_mbid:
            click.echo(f"  CRG: {result.crg_mbid}")
        if result.rr_mbid:
            click.echo(f"  RR: {result.rr_mbid}")
        click.echo(f"  Confidence: {result.confidence:.2f}")
        if result.rationale:
            click.echo(f"  Rationale: {result.rationale}")
        if result.error_message:
            click.echo(f"  Error: {result.error_message}")

    sys.exit(ExitCode.SUCCESS if result.outcome != "error" else ExitCode.ERROR)


@canon.group()
def review() -> None:
    """Human review queue commands (Epic 13)."""
    pass


@review.command("list")
@click.option("--source", type=click.Choice(["indeterminate", "llm_review", "conflict"]))
@click.option("--limit", default=20, help="Maximum items to show")
@click.pass_context
def review_list(ctx: click.Context, source: str | None, limit: int) -> None:
    """List pending review items."""
    from chart_binder.llm import ReviewQueue, ReviewSource

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    source_filter = ReviewSource(source) if source else None
    items = queue.get_pending(source=source_filter, limit=limit)

    if output_format == OutputFormat.JSON:
        results = [
            {
                "review_id": item.review_id,
                "file_id": item.file_id,
                "work_key": item.work_key,
                "source": item.source.value,
                "created_at": item.created_at,
            }
            for item in items
        ]
        click.echo(json.dumps(results, indent=2))
    else:
        if not items:
            click.echo("✔︎ No pending review items")
        else:
            click.echo(f"Pending Review Items ({len(items)}):")
            click.echo("=" * 50)
            for item in items:
                click.echo(f"\n  ID: {item.review_id[:8]}...")
                click.echo(f"  Work: {item.work_key}")
                click.echo(f"  Source: {item.source.value}")

    sys.exit(ExitCode.SUCCESS)


@review.command("show")
@click.argument("review_id")
@click.pass_context
def review_show(ctx: click.Context, review_id: str) -> None:
    """Show details of a review item."""
    from chart_binder.llm import ReviewQueue

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    item = queue.get_item(review_id)

    if not item:
        click.echo(f"Review item not found: {review_id}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

    if output_format == OutputFormat.JSON:
        result = {
            "review_id": item.review_id,
            "file_id": item.file_id,
            "work_key": item.work_key,
            "source": item.source.value,
            "evidence_bundle": item.evidence_bundle,
            "decision_trace": item.decision_trace,
            "llm_suggestion": item.llm_suggestion,
            "created_at": item.created_at,
        }
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(item.to_display())

    sys.exit(ExitCode.SUCCESS)


@review.command("accept")
@click.argument("review_id")
@click.option("--crg", "crg_mbid", required=True, help="CRG MBID to accept")
@click.option("--rr", "rr_mbid", help="RR MBID to accept")
@click.option("--notes", help="Review notes")
@click.pass_context
def review_accept(
    ctx: click.Context,
    review_id: str,
    crg_mbid: str,
    rr_mbid: str | None,
    notes: str | None,
) -> None:
    """Accept a review with specific CRG/RR selection."""
    from chart_binder.llm import ReviewAction, ReviewQueue

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    success = queue.complete_review(
        review_id,
        action=ReviewAction.ACCEPT,
        action_data={"crg_mbid": crg_mbid, "rr_mbid": rr_mbid},
        reviewed_by="cli_user",
        notes=notes,
    )

    if not success:
        click.echo(f"Failed to complete review: {review_id}", err=True)
        sys.exit(ExitCode.ERROR)

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps({"status": "accepted", "review_id": review_id}))
    else:
        click.echo(f"✔︎ Accepted review {review_id[:8]}... with CRG {crg_mbid}")

    sys.exit(ExitCode.SUCCESS)


@review.command("accept-llm")
@click.argument("review_id")
@click.option("--notes", help="Review notes")
@click.pass_context
def review_accept_llm(ctx: click.Context, review_id: str, notes: str | None) -> None:
    """Accept LLM suggestion for a review item."""
    from chart_binder.llm import ReviewAction, ReviewQueue

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    item = queue.get_item(review_id)

    if not item or not item.llm_suggestion:
        click.echo(f"No LLM suggestion for review: {review_id}", err=True)
        sys.exit(ExitCode.ERROR)

    success = queue.complete_review(
        review_id,
        action=ReviewAction.ACCEPT_LLM,
        action_data=item.llm_suggestion,
        reviewed_by="cli_user",
        notes=notes,
    )

    if not success:
        click.echo(f"Failed to complete review: {review_id}", err=True)
        sys.exit(ExitCode.ERROR)

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps({"status": "accepted_llm", "review_id": review_id}))
    else:
        click.echo(f"✔︎ Accepted LLM suggestion for review {review_id[:8]}...")

    sys.exit(ExitCode.SUCCESS)


@review.command("skip")
@click.argument("review_id")
@click.option("--notes", help="Reason for skipping")
@click.pass_context
def review_skip(ctx: click.Context, review_id: str, notes: str | None) -> None:
    """Skip a review item."""
    from chart_binder.llm import ReviewAction, ReviewQueue

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    success = queue.complete_review(
        review_id,
        action=ReviewAction.SKIP,
        reviewed_by="cli_user",
        notes=notes,
    )

    if not success:
        click.echo(f"Failed to skip review: {review_id}", err=True)
        sys.exit(ExitCode.ERROR)

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps({"status": "skipped", "review_id": review_id}))
    else:
        click.echo(f"✔︎ Skipped review {review_id[:8]}...")

    sys.exit(ExitCode.SUCCESS)


@review.command("stats")
@click.pass_context
def review_stats(ctx: click.Context) -> None:
    """Show review queue statistics."""
    from chart_binder.llm import ReviewQueue

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    queue = ReviewQueue(config.llm.review_queue_path)
    stats = queue.get_stats()

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(stats, indent=2))
    else:
        click.echo("Review Queue Statistics")
        click.echo("=" * 40)
        click.echo(f"  Pending: {stats['pending']}")
        if stats.get("pending_by_source"):
            click.echo("  By Source:")
            for source, count in stats["pending_by_source"].items():
                click.echo(f"    {source}: {count}")
        click.echo(f"  Completed: {stats['completed']}")
        if stats.get("completed_by_action"):
            click.echo("  By Action:")
            for action, count in stats["completed_by_action"].items():
                click.echo(f"    {action}: {count}")

    sys.exit(ExitCode.SUCCESS)


def main() -> None:
    """Entry point for the canon CLI."""
    canon()


if __name__ == "__main__":
    main()
