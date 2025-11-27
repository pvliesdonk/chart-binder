from __future__ import annotations

import json
import sqlite3
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

import click

from chart_binder.config import Config

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


def _output_result(
    data: dict[str, Any] | list[dict[str, Any]],
    output_format: OutputFormat,
    text_formatter: Any | None = None,
) -> None:
    """Output result in specified format."""
    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(data, indent=2, default=str))
    elif text_formatter:
        text_formatter(data)
    else:
        click.echo(data)


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
@click.pass_context
def canon(
    ctx: click.Context,
    config: Path | None,
    offline: bool,
    frozen: bool,
    refresh: bool,
    output: str,
) -> None:
    """
    Chart-Binder: Charts-aware audio tagger.

    Pick the most canonical release, link MB/Discogs/Spotify IDs,
    and embed compact chart history.
    """
    cfg = Config.load(config)
    if offline or frozen:
        cfg.offline_mode = True

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

    output_format = ctx.obj["output"]
    results = []
    audio_files = _collect_audio_files(paths)

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

        except ValueError as e:
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

    output_format = ctx.obj["output"]
    results: list[dict[str, Any]] = []

    # Create resolver with config
    resolver_config = ConfigSnapshot(
        lead_window_days=90,
        reissue_long_gap_years=10,
    )
    resolver = Resolver(resolver_config)
    audio_files = _collect_audio_files(paths)

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
                if explain and decision.decision_trace.missing_facts:
                    click.echo(f"  Missing: {decision.decision_trace.missing_facts}")
                click.echo(f"  Trace: {decision.compact_tag}")

        except ValueError as e:
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

        except ValueError as e:
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

        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        result["entries"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at <= ?", (time.time(),))
        result["expired_entries"] = cursor.fetchone()[0]

        conn.close()

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
@click.option("--threshold", default=0.60, help="Minimum confidence threshold")
@click.pass_context
def coverage_chart(ctx: click.Context, chart_id: str, period: str, threshold: float) -> None:
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
def review(ctx: click.Context) -> None:
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

    sys.exit(ExitCode.SUCCESS if not stale_decisions else ExitCode.NO_RESULTS)


def main() -> None:
    """Entry point for the canon CLI."""
    canon()


if __name__ == "__main__":
    main()
