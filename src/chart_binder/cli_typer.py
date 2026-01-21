"""CLI for chart-binder using Typer and Rich.

Modern CLI with beautiful output, type-safe commands, and integrated progress tracking.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import sqlite3
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from chart_binder.config import Config
from chart_binder.console import (
    print as cprint,
)
from chart_binder.console import (
    print_error,
    print_success,
    print_warning,
    set_console,
)
from chart_binder.llm import ReviewAction, ReviewQueue, ReviewSource
from chart_binder.safe_logging import configure_rich_logging

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


# Create Typer app
app = typer.Typer(
    name="canon",
    help="Chart-Binder: charts-aware audio tagger with canonical release selection",
    no_args_is_help=True,
    add_completion=False,
)

# Create subcommands
cache_app = typer.Typer(help="HTTP cache management commands")
coverage_app = typer.Typer(help="Chart coverage analysis commands")
charts_app = typer.Typer(help="Chart data scraping and management")
review_app = typer.Typer(help="Review and resolve indeterminate decisions")
analytics_app = typer.Typer(help="Cross-chart querying and analytics")
playlist_app = typer.Typer(help="Generate playlists from chart data")

app.add_typer(cache_app, name="cache")
app.add_typer(coverage_app, name="coverage")
app.add_typer(charts_app, name="charts")
app.add_typer(review_app, name="review")
app.add_typer(analytics_app, name="analytics")
app.add_typer(playlist_app, name="playlist")


# Global state (set by callback)
class AppState:
    """Global application state passed between commands."""

    config: Config
    output_format: OutputFormat
    verbose: int


state = AppState()


def _current_state() -> tuple[Config, OutputFormat]:
    """Return the initialized config and output format or exit."""
    cfg = getattr(state, "config", None)
    output = getattr(state, "output_format", None)

    if cfg is None or output is None:
        print_error("CLI state is not initialized. Run `canon --help` first.")
        raise typer.Exit(code=ExitCode.ERROR)

    return cfg, output


def _reviewed_by() -> str:
    """Resolve reviewed_by identifier for audit logs."""
    return os.getenv("CHART_BINDER_REVIEWED_BY") or getpass.getuser() or "cli_user"


def _collect_audio_files(paths: tuple[Path, ...]) -> list[Path]:
    """Collect audio files from paths (files or directories).

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


def _get_or_calc_fingerprint(
    audio_file: Path,
    tagset: Any,  # TagSet
    logger: logging.Logger,
) -> tuple[str | None, int | None]:
    """Get fingerprint from tags (trust-on-read) or calculate it.

    Returns (fingerprint, duration_sec) tuple.
    """
    from chart_binder.fingerprint import FingerprintError, calculate_fingerprint

    # Trust-on-read: check if fingerprint is already in tags
    if tagset.compact.fingerprint and tagset.compact.fingerprint_duration:
        logger.debug(f"Using cached fingerprint from tags for {audio_file}")
        return tagset.compact.fingerprint, tagset.compact.fingerprint_duration

    # Calculate fingerprint via fpcalc
    try:
        fp_result = calculate_fingerprint(audio_file)
        logger.debug(f"Calculated fingerprint for {audio_file}: {fp_result.duration_sec}s")
        return fp_result.fingerprint, fp_result.duration_sec
    except FingerprintError as e:
        logger.warning(f"Fingerprint calculation failed for {audio_file}: {e}")
        return None, None


def _convert_evidence_bundle(
    bundle: Any, audio_file: Path, tagset: Any, musicgraph_db: Any = None
) -> dict[str, Any]:
    """Convert EvidenceBundle dataclass to dict format expected by resolver."""
    # [Implementation remains the same as original - copying from original file]
    recording_candidates = []
    rg_by_recording: dict[str, list[dict[str, Any]]] = {}

    for rg in bundle.release_groups:
        rg_mbid = rg.get("mbid")
        releases_data = []

        if musicgraph_db and rg_mbid:
            db_releases = musicgraph_db.get_releases_in_group(rg_mbid)
            for rel in db_releases:
                flags = {}
                flags_json = rel.get("flags_json")
                if flags_json:
                    try:
                        flags = json.loads(flags_json)
                    except json.JSONDecodeError:
                        pass
                if "is_official" not in flags:
                    flags["is_official"] = True

                releases_data.append(
                    {
                        "mb_release_id": rel.get("mbid"),
                        "title": rel.get("title"),
                        "date": rel.get("date"),
                        "country": rel.get("country"),
                        "label": rel.get("label"),
                        "format": rel.get("format"),
                        "barcode": rel.get("barcode"),
                        "discogs_release_id": rel.get("discogs_release_id"),
                        "flags": flags,
                    }
                )

        for rec in bundle.recordings:
            rec_mbid = rec.get("mbid", "")
            if rec_mbid not in rg_by_recording:
                rg_by_recording[rec_mbid] = []
            rg_by_recording[rec_mbid].append(
                {
                    "mb_rg_id": rg_mbid,
                    "title": rg.get("title"),
                    "primary_type": rg.get("type"),
                    "secondary_types": rg.get("secondary_types", []),
                    "first_release_date": rg.get("first_release_date"),
                    "discogs_master_id": rg.get("discogs_master_id"),
                    "releases": releases_data,
                }
            )

    for rec in bundle.recordings:
        rec_mbid = rec.get("mbid", "")
        recording_candidates.append(
            {
                "mb_recording_id": rec_mbid,
                "title": rec.get("title"),
                "rg_candidates": rg_by_recording.get(rec_mbid, []),
            }
        )

    return {
        "artifact": {"file_path": str(audio_file)},
        "artist": {
            "name": bundle.artist.get("name") if bundle.artist else tagset.artist or "Unknown",
            "mb_artist_id": bundle.artist.get("mbid") if bundle.artist else None,
            "begin_area_country": bundle.artist.get("begin_area_country")
            if bundle.artist
            else None,
            "wikidata_qid": bundle.artist.get("wikidata_qid") if bundle.artist else None,
        },
        "recording_candidates": recording_candidates,
        "timeline_facts": bundle.timeline_facts or {},
        "provenance": bundle.provenance or {"sources_used": ["MB"]},
    }


@app.callback()
def main(
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to configuration TOML file", exists=True),
    ] = None,
    offline: Annotated[
        bool, typer.Option(help="Run in offline mode (no network requests)")
    ] = False,
    frozen: Annotated[bool, typer.Option(help="Use only cached data, fail if cache miss")] = False,
    refresh: Annotated[bool, typer.Option(help="Force refresh of cached data")] = False,
    output: Annotated[
        OutputFormat,
        typer.Option("--output", "-o", help="Output format"),
    ] = OutputFormat.TEXT,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity"),
    ] = 0,
    # Database options
    db_music_graph: Annotated[
        Path | None,
        typer.Option(help="Music graph database path"),
    ] = None,
    db_charts: Annotated[
        Path | None,
        typer.Option(help="Charts database path"),
    ] = None,
    db_decisions: Annotated[
        Path | None,
        typer.Option(help="Decisions database path"),
    ] = None,
    # Cache options
    cache_dir: Annotated[Path | None, typer.Option(help="HTTP cache directory")] = None,
    cache_ttl: Annotated[int | None, typer.Option(help="Cache TTL in seconds")] = None,
    no_cache: Annotated[bool, typer.Option(help="Disable HTTP caching")] = False,
    # LLM options
    llm_provider: Annotated[
        str | None,
        typer.Option(help="LLM provider (ollama/openai)"),
    ] = None,
    llm_model: Annotated[str | None, typer.Option(help="LLM model ID")] = None,
    llm_enabled: Annotated[bool | None, typer.Option("--llm-enabled/--llm-disabled")] = None,
    llm_temperature: Annotated[
        float | None,
        typer.Option(help="LLM temperature (0.0-2.0)"),
    ] = None,
    # SearxNG options
    searxng_url: Annotated[str | None, typer.Option(help="SearxNG instance URL")] = None,
    searxng_enabled: Annotated[
        bool | None, typer.Option("--searxng-enabled/--searxng-disabled")
    ] = None,
) -> None:
    """Chart-Binder: Charts-aware audio tagger with canonical release selection."""
    logger = logging.getLogger(__name__)

    # Load config (TOML + env vars)
    cfg = Config.load(config_path)
    if config_path:
        logger.info(f"Loaded config from {config_path}")

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

    # Initialize Rich console and configure logging
    console = configure_rich_logging(
        level=log_level,
        hash_paths=cfg.logging.hash_paths,
        show_time=True,
        show_path=False,
    )
    set_console(console)

    # Suppress external library logging unless very verbose (-vvv)
    if verbose < 3:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    logger.debug(
        f"Logging configured: level={logging.getLevelName(log_level)}, hash_paths={cfg.logging.hash_paths}"
    )

    # Store in global state
    state.config = cfg
    state.output_format = output
    state.verbose = verbose


# ====================================================================
# MAIN COMMANDS
# ====================================================================


@app.command()
def scan(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to scan", exists=True),
    ],
) -> None:
    """Scan audio files and display metadata and fingerprints.

    Reads tags, calculates AcoustID fingerprints, and shows what
    chart-binder can see about each file.
    """
    from chart_binder.tagging import verify

    logger = logging.getLogger(__name__)
    logger.info(f"Running scan command on {len(paths)} path(s)")

    output_format = state.output_format
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
                cprint(f"\n[green]✔︎[/green] {audio_file}")
                cprint(f"  Title: {tagset.title or '(none)'}")
                cprint(f"  Artist: {tagset.artist or '(none)'}")
                cprint(f"  Album: {tagset.album or '(none)'}")
                cprint(f"  Year: {tagset.original_year or '(none)'}")
                if tagset.ids.mb_release_group_id:
                    cprint(f"  MB RG: {tagset.ids.mb_release_group_id}")
                if tagset.compact.decision_trace:
                    cprint(f"  Trace: {tagset.compact.decision_trace}")

        except Exception as e:
            if output_format == OutputFormat.TEXT:
                print_error(f"{audio_file}: {e}")
            results.append({"file": str(audio_file), "error": str(e)})

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(results, indent=2))

    raise typer.Exit(code=ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@app.command()
def resolve(
    artist: Annotated[str, typer.Option("--artist", "-a", help="Artist name")],
    title: Annotated[str, typer.Option("--title", "-t", help="Track title")],
    explain: Annotated[bool, typer.Option(help="Show detailed decision rationale")] = False,
) -> None:
    """Resolve artist/title to canonical release group.

    Uses the 7-rule algorithm to find the most canonical release
    for a given artist and title combination.

    Examples:
        canon resolve -a "The Beatles" -t "Hey Jude"
        canon resolve -a "Pink Floyd" -t "Another Brick In The Wall" --explain
    """
    from chart_binder.resolve import resolve_artist_title

    logger = logging.getLogger(__name__)
    logger.info(f"Resolving: {artist} - {title}")

    result = resolve_artist_title(
        artist=artist,
        title=title,
        config=state.config,
    )

    # Display results
    if state.output_format == OutputFormat.JSON:
        output_dict = {
            "artist": artist,
            "title": title,
            "state": result.state,
            "crg_mbid": result.crg_mbid,
            "rr_mbid": result.rr_mbid,
            "recording_mbid": result.recording_mbid,
            "confidence": result.confidence,
        }
        if explain:
            output_dict["trace"] = result.trace
        cprint(json.dumps(output_dict, indent=2))
    else:
        if result.state == "decided":
            cprint(f"[green]✓ Resolved: {artist} - {title}[/green]")
            cprint(f"  Release Group: {result.crg_mbid}")
            if result.rr_mbid:
                cprint(f"  Representative Release: {result.rr_mbid}")
            if result.recording_mbid:
                cprint(f"  Recording: {result.recording_mbid}")
        else:
            cprint(f"[yellow]⚠ Could not resolve: {artist} - {title}[/yellow]")

        if explain and result.trace:
            cprint("\n[bold]Decision Trace:[/bold]")
            cprint(result.trace)

    # Validate state before exit
    if result.state not in ["decided", "indeterminate"]:
        logger.warning(
            f"Unexpected state value: '{result.state}' (expected 'decided' or 'indeterminate')"
        )

    sys.exit(ExitCode.SUCCESS if result.state == "decided" else ExitCode.NO_RESULTS)


@app.command()
def decide(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to process", exists=True),
    ],
    explain: Annotated[bool, typer.Option(help="Show detailed decision rationale")] = False,
    no_persist: Annotated[bool, typer.Option(help="Skip persisting decisions to database")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Show only summary")] = False,
) -> None:
    """Decide canonical release for audio files.

    Analyzes audio files and determines the most canonical release
    using the 7-rule algorithm. Optionally persists decisions to
    the decisions database.

    Examples:
        canon decide /music/albums/
        canon decide song.mp3 --explain
        canon decide /music/ --no-persist --quiet
    """
    from chart_binder.candidates import CandidateBuilder, CandidateSet
    from chart_binder.decisions_db import DecisionsDB
    from chart_binder.decisions_db import DecisionState as DBDecisionState
    from chart_binder.fetcher import FetcherConfig, FetchMode, UnifiedFetcher
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer
    from chart_binder.resolver import (
        ConfigSnapshot,
        CRGRationale,
        DecisionState,
        Resolver,
        RRRationale,
    )
    from chart_binder.tagging import verify

    logger = logging.getLogger(__name__)
    logger.info(f"Running decide command on {len(paths)} path(s), explain={explain}")

    output_format = state.output_format
    config = state.config
    results: list[dict[str, Any]] = []

    # Initialize decisions database
    decisions_db: DecisionsDB | None = None
    if not no_persist:
        try:
            decisions_db = DecisionsDB(config.database.decisions_path)
            logger.debug(f"Initialized decisions DB at {config.database.decisions_path}")
        except Exception as e:
            logger.warning(f"Could not initialize decisions DB: {e}")

    # Create resolver with config
    resolver_config = ConfigSnapshot(
        lead_window_days=90,
        reissue_long_gap_years=10,
    )
    resolver = Resolver(resolver_config)

    # Initialize UnifiedFetcher for multi-source lookups
    cache_dir = config.http_cache.directory
    musicgraph_path = config.database.music_graph_path
    fetcher_config = FetcherConfig(
        cache_dir=cache_dir,
        db_path=musicgraph_path,
        mode=FetchMode.NORMAL,
        acoustid_api_key=config.live_sources.acoustid_api_key,
        discogs_token=config.live_sources.discogs_token,
        spotify_client_id=config.live_sources.spotify_client_id,
        spotify_client_secret=config.live_sources.spotify_client_secret,
    )

    # Initialize CandidateBuilder for evidence bundle construction
    musicgraph_db = MusicGraphDB(musicgraph_path)
    normalizer = Normalizer()
    candidate_builder = CandidateBuilder(musicgraph_db, normalizer)

    # Initialize LLM adjudicator if enabled in config
    adjudicator = None
    auto_accept_threshold = 0.85
    if config.llm.enabled:
        from chart_binder.llm.react_adjudicator import ReActAdjudicator

        # Initialize SearxNG web search if enabled
        web_search = None
        if config.llm.searxng.enabled:
            from chart_binder.llm.searxng import SearxNGSearchTool

            web_search = SearxNGSearchTool(
                base_url=config.llm.searxng.url,
                timeout=config.llm.searxng.timeout_s,
            )
            if web_search.is_available():
                logger.info(f"SearxNG web search enabled at {config.llm.searxng.url}")
            else:
                logger.warning(f"SearxNG configured but unavailable at {config.llm.searxng.url}")
                web_search = None

        adjudicator = ReActAdjudicator(config=config.llm, search_tool=web_search)
        auto_accept_threshold = config.llm.auto_accept_threshold
        logger.info(
            f"LLM adjudication enabled using ReAct pattern (auto-accept threshold: {auto_accept_threshold})"
        )

    audio_files = _collect_audio_files(paths)
    logger.debug(f"Collected {len(audio_files)} audio files")

    # Track statistics for quiet mode
    stats = {"decided": 0, "indeterminate": 0, "llm_adjudicated": 0, "errors": 0}
    total_files = len(audio_files)

    if quiet and output_format == OutputFormat.TEXT:
        cprint(f"Processing {total_files} files...")

    with UnifiedFetcher(fetcher_config) as fetcher:
        for idx, audio_file in enumerate(audio_files, 1):
            try:
                tagset = verify(audio_file)

                # Get or calculate fingerprint for stable identity
                fingerprint, duration_sec = _get_or_calc_fingerprint(audio_file, tagset, logger)

                # Source 1: Existing MB IDs from tags (as evidence, not gospel)
                if tagset.ids.mb_recording_id:
                    logger.debug(
                        f"Fetching existing MB recording from tags: {tagset.ids.mb_recording_id}"
                    )
                    try:
                        fetcher.fetch_recording(tagset.ids.mb_recording_id)
                    except Exception as e:
                        logger.debug(f"Failed to fetch tagged recording: {e}")

                # Source 2: Search by fingerprint (most reliable for audio identity)
                if fingerprint and duration_sec:
                    logger.debug(f"Searching by fingerprint ({duration_sec}s)")
                    search_results = fetcher.search_recordings(
                        fingerprint=fingerprint,
                        duration_sec=duration_sec,
                    )
                    logger.debug(f"Fingerprint search returned {len(search_results)} results")
                    for result in search_results:
                        if result.get("recording_mbid"):
                            try:
                                fetcher.fetch_recording(result["recording_mbid"])
                            except Exception as e:
                                logger.debug(f"Failed to fetch fingerprint result: {e}")

                # Source 3: Search by title + artist (MB, Discogs, and Spotify)
                if tagset.artist and tagset.title:
                    logger.debug(f"Searching for: {tagset.artist} - {tagset.title}")
                    search_results = fetcher.search_recordings(
                        title=tagset.title,
                        artist=tagset.artist,
                    )
                    logger.debug(f"Title/artist search returned {len(search_results)} results")

                    # Hydrate top 5 from each source type
                    mb_results = [r for r in search_results if r.get("recording_mbid")]
                    discogs_results = [r for r in search_results if r.get("discogs_release_id")]
                    spotify_results = [r for r in search_results if r.get("spotify_track_id")]

                    mb_count = 0
                    for result in mb_results[:5]:
                        mbid = result["recording_mbid"]
                        logger.debug(f"Hydrating MB recording {mbid}")
                        try:
                            fetcher.fetch_recording(mbid)
                            mb_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to fetch MB recording {mbid}: {e}")

                    discogs_count = 0
                    for result in discogs_results[:5]:
                        discogs_id = result["discogs_release_id"]
                        logger.debug(f"Hydrating Discogs release {discogs_id}")
                        try:
                            fetcher.fetch_discogs_release(discogs_id)
                            discogs_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to fetch Discogs release {discogs_id}: {e}")

                    spotify_count = 0
                    for result in spotify_results[:5]:
                        spotify_id = result["spotify_track_id"]
                        logger.debug(f"Hydrating Spotify track {spotify_id}")
                        try:
                            fetcher.fetch_spotify_track(spotify_id)
                            spotify_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to fetch Spotify track {spotify_id}: {e}")

                    logger.debug(
                        f"Hydrated {mb_count} MB recordings, {discogs_count} Discogs releases, "
                        f"and {spotify_count} Spotify tracks"
                    )

                # Source 4: Search by barcode (Discogs)
                if tagset.barcode:
                    logger.debug(f"Searching Discogs by barcode: {tagset.barcode}")
                    barcode_results = fetcher.search_recordings(
                        barcode=tagset.barcode,
                        title=tagset.title,
                        artist=tagset.artist,
                    )
                    logger.debug(f"Barcode search returned {len(barcode_results)} results")
                    for result in barcode_results[:5]:
                        if result.get("discogs_release_id"):
                            discogs_id = result["discogs_release_id"]
                            logger.debug(f"Hydrating Discogs barcode result {discogs_id}")
                            try:
                                fetcher.fetch_discogs_release(discogs_id)
                            except Exception as e:
                                logger.debug(f"Failed to fetch Discogs release {discogs_id}: {e}")

                # Discover all candidates from local DB (populated by fetches above)
                length_ms = duration_sec * 1000 if duration_sec else None
                candidates = candidate_builder.discover_by_title_artist_length(
                    tagset.title or "", tagset.artist or "", length_ms
                )

                # Build evidence bundle from candidates
                candidate_set = CandidateSet(
                    file_path=audio_file,
                    candidates=candidates,
                    normalized_title=tagset.title or "",
                    normalized_artist=tagset.artist or "",
                    length_ms=duration_sec * 1000 if duration_sec else None,
                )

                evidence_bundle_obj = candidate_builder.build_evidence_bundle(candidate_set)

                # Convert EvidenceBundle to dict format expected by resolver
                evidence_bundle = _convert_evidence_bundle(
                    evidence_bundle_obj, audio_file, tagset, musicgraph_db
                )

                decision = resolver.resolve(evidence_bundle)

                # If decision is INDETERMINATE and LLM adjudication is enabled, try to adjudicate
                llm_adjudicated = False
                llm_confidence = None
                llm_rationale = None
                if decision.state == DecisionState.INDETERMINATE and adjudicator:
                    logger.debug("Decision is INDETERMINATE, attempting LLM adjudication...")
                    try:
                        # Build decision trace dict from compact string
                        decision_trace_dict: dict[str, Any] | None = None
                        if decision.compact_tag:
                            decision_trace_dict = {"trace_compact": decision.compact_tag}

                        # Call LLM adjudicator
                        from chart_binder.llm.adjudicator import AdjudicationOutcome

                        adjudication_result = adjudicator.adjudicate(
                            evidence_bundle, decision_trace_dict
                        )

                        llm_confidence = adjudication_result.confidence
                        llm_rationale = adjudication_result.rationale

                        # If high confidence, auto-accept and update decision
                        if (
                            adjudication_result.outcome != AdjudicationOutcome.ERROR
                            and adjudication_result.confidence >= auto_accept_threshold
                        ):
                            decision.state = DecisionState.DECIDED
                            decision.release_group_mbid = adjudication_result.crg_mbid
                            decision.release_mbid = adjudication_result.rr_mbid
                            decision.crg_rationale = CRGRationale.LLM_ADJUDICATION
                            decision.rr_rationale = RRRationale.LLM_ADJUDICATION
                            # Update decision trace to reflect LLM adjudication
                            decision.decision_trace.crg_selection = {
                                "rule": str(CRGRationale.LLM_ADJUDICATION),
                                "confidence": adjudication_result.confidence,
                                "rationale": adjudication_result.rationale,
                            }
                            decision.decision_trace.rr_selection = {
                                "rule": str(RRRationale.LLM_ADJUDICATION),
                            }
                            llm_adjudicated = True
                            logger.info(
                                f"LLM adjudicated: CRG={adjudication_result.crg_mbid}, "
                                f"confidence={adjudication_result.confidence:.2f}"
                            )
                        else:
                            logger.debug(
                                f"LLM confidence {adjudication_result.confidence:.2f} "
                                f"below threshold {auto_accept_threshold}, keeping INDETERMINATE"
                            )
                    except Exception as e:
                        logger.warning(f"LLM adjudication failed: {e}")

                # Extract metadata from decision trace
                artist_name = evidence_bundle.get("artist", {}).get("name", "Unknown")
                artist_credits = None
                recording_title = None
                release_group_title = None
                selected_release_title = None
                discogs_master_id = None
                discogs_release_id = None

                # Get recording title and artist credits from CRG selection
                if decision.decision_trace.crg_selection:
                    crg_data = decision.decision_trace.crg_selection.get("release_group", {})
                    release_group_title = crg_data.get("title")
                    discogs_master_id = crg_data.get("discogs_master_id")
                    artist_credits = crg_data.get("artist_credit")

                    # Get recording title from the selected recording
                    recording_data = decision.decision_trace.crg_selection.get("recording", {})
                    recording_title = recording_data.get("title")

                # Extract selected release title and Discogs ID from RR selection
                if decision.decision_trace.rr_selection:
                    release_data = decision.decision_trace.rr_selection.get("release", {})
                    selected_release_title = release_data.get("title")
                    discogs_release_id = release_data.get("discogs_release_id")

                # Persist decision to database if fingerprint available
                if decisions_db and fingerprint and duration_sec:
                    file_id = DecisionsDB.generate_file_id(fingerprint, duration_sec)

                    # Determine library root and relative path
                    library_root = str(audio_file.parent)
                    relative_path = audio_file.name

                    # Upsert file artifact
                    decisions_db.upsert_file_artifact(
                        file_id=file_id,
                        library_root=library_root,
                        relative_path=relative_path,
                        duration_ms=duration_sec * 1000 if duration_sec else None,
                        fp_id=fingerprint[:32] if fingerprint else None,
                    )

                    # Map resolver state to DB state
                    db_state = (
                        DBDecisionState.DECIDED
                        if decision.state == DecisionState.DECIDED
                        else DBDecisionState.INDETERMINATE
                    )

                    # Build work_key from artist and title
                    work_key = f"{tagset.artist or 'unknown'}_{tagset.title or 'unknown'}".lower()

                    # Upsert decision
                    decisions_db.upsert_decision(
                        file_id=file_id,
                        work_key=work_key,
                        mb_rg_id=decision.release_group_mbid or "",
                        mb_release_id=decision.release_mbid or "",
                        mb_recording_id=tagset.ids.mb_recording_id,
                        ruleset_version=decision.decision_trace.ruleset_version,
                        config_snapshot={
                            "lead_window_days": resolver_config.lead_window_days,
                            "reissue_long_gap_years": resolver_config.reissue_long_gap_years,
                        },
                        evidence_hash=decision.decision_trace.evidence_hash,
                        trace_compact=decision.compact_tag,
                        state=db_state,
                    )
                    logger.debug(f"Persisted decision for {audio_file} (file_id={file_id[:16]}...)")

                result = {
                    "file": str(audio_file),
                    "state": decision.state.value,
                    "artist": artist_name,
                    "artist_credits": artist_credits,
                    "recording_title": recording_title,
                    "release_group_title": release_group_title,
                    "selected_release_title": selected_release_title,
                    "crg_mbid": decision.release_group_mbid,
                    "rr_mbid": decision.release_mbid,
                    "discogs_master_id": discogs_master_id,
                    "discogs_release_id": discogs_release_id,
                    "crg_rationale": _get_rationale_value(decision.crg_rationale),
                    "rr_rationale": _get_rationale_value(decision.rr_rationale),
                    "compact_tag": decision.compact_tag,
                    "llm_adjudicated": llm_adjudicated,
                }

                # Add LLM info if adjudication was attempted
                if llm_confidence is not None:
                    result["llm_confidence"] = llm_confidence
                if llm_rationale is not None:
                    result["llm_rationale"] = llm_rationale

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

                # Update statistics
                if decision.state.value == "decided":
                    stats["decided"] += 1
                    if llm_adjudicated:
                        stats["llm_adjudicated"] += 1
                else:
                    stats["indeterminate"] += 1

                # Show progress in quiet mode
                if quiet and output_format == OutputFormat.TEXT:
                    if idx % 10 == 0 or idx == total_files:
                        pct = (idx / total_files) * 100
                        llm_stat = f" 🤖{stats['llm_adjudicated']}" if adjudicator else ""
                        print(
                            f"  Progress: {idx}/{total_files} ({pct:.1f}%) "
                            f"[✔︎{stats['decided']}{llm_stat} ∆{stats['indeterminate']} ✘{stats['errors']}]",
                            file=sys.stderr,
                        )

                # Show detailed output if not quiet
                if not quiet and output_format == OutputFormat.TEXT:
                    state_icon = "✔︎" if decision.state.value == "decided" else "∆"
                    if llm_adjudicated:
                        state_icon = "🤖"
                    cprint(f"\n{state_icon} {audio_file}")
                    cprint(f"  Artist: {artist_name}")
                    if recording_title:
                        cprint(f"  Recording: {recording_title}")
                    cprint(f"  State: {decision.state.value}")
                    if llm_adjudicated:
                        cprint(f"  LLM Adjudicated: confidence={llm_confidence:.2f}")
                        if llm_rationale:
                            cprint(f"  LLM Rationale: {llm_rationale[:100]}")
                    if decision.release_group_mbid:
                        cprint(f"  CRG: {decision.release_group_mbid}")
                        if release_group_title:
                            cprint(f"       Title: {release_group_title}")
                        if artist_credits:
                            cprint(f"       Artist Credit: {artist_credits}")
                        if discogs_master_id:
                            cprint(f"       Discogs Master: {discogs_master_id}")
                        cprint(f"       ({decision.crg_rationale})")
                    if decision.release_mbid:
                        cprint(f"  RR:  {decision.release_mbid}")
                        if selected_release_title:
                            cprint(f"       Title: {selected_release_title}")
                        if discogs_release_id:
                            cprint(f"       Discogs Release: {discogs_release_id}")
                        cprint(f"       ({decision.rr_rationale})")
                    cprint(f"  Trace: {decision.compact_tag}")
                    if explain:
                        cprint("\n" + decision.decision_trace.to_human_readable())

                    if decision.decision_trace.missing_facts:
                        cprint("\nMissing Facts:")
                        for fact in decision.decision_trace.missing_facts:
                            cprint(f"  - {fact}")

            except Exception as e:
                stats["errors"] += 1
                if not quiet and output_format == OutputFormat.TEXT:
                    print_error(f"{audio_file}: {e}")
                results.append({"file": str(audio_file), "error": str(e)})

                # Show progress in quiet mode even for errors
                if quiet and output_format == OutputFormat.TEXT:
                    if idx % 10 == 0 or idx == total_files:
                        pct = (idx / total_files) * 100
                        print(
                            f"  Progress: {idx}/{total_files} ({pct:.1f}%) "
                            f"[✔︎{stats['decided']} ∆{stats['indeterminate']} ✘{stats['errors']}]",
                            file=sys.stderr,
                        )

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(results, indent=2))
    elif quiet and output_format == OutputFormat.TEXT:
        # Print final summary in quiet mode
        cprint(f"\n✓ Completed {total_files} files:")
        cprint(f"  Decided:       {stats['decided']}")
        if stats["llm_adjudicated"] > 0:
            cprint(f"    (via LLM):   {stats['llm_adjudicated']}")
        cprint(f"  Indeterminate: {stats['indeterminate']}")
        if stats["errors"] > 0:
            cprint(f"  Errors:        {stats['errors']}")

    raise typer.Exit(code=ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@app.command()
def write(
    paths: Annotated[
        list[Path],
        typer.Argument(help="Audio files or directories to tag", exists=True),
    ],
    dry_run: Annotated[bool, typer.Option(help="Preview changes without writing")] = False,
    apply: Annotated[bool, typer.Option(help="Apply changes (required for writes)")] = False,
) -> None:
    """Write canonical tags to audio files.

    WARNING: This command modifies your audio files. Use --dry-run first!

    Examples:
        canon write /music/ --dry-run
        canon write song.mp3 --apply
    """
    from chart_binder.fingerprint import FingerprintError, calculate_fingerprint
    from chart_binder.tagging import (
        CanonicalIDs,
        CompactFields,
        TagSet,
        verify,
        write_tags,
    )

    logger = logging.getLogger(__name__)
    output_format = state.output_format
    results = []

    if not dry_run and not apply:
        print_error("Use --dry-run to preview or --apply to write changes.")
        raise typer.Exit(code=ExitCode.ERROR)

    audio_files = _collect_audio_files(paths)

    for audio_file in audio_files:
        try:
            # Read existing tags
            existing_tagset = verify(audio_file)

            # Get or calculate fingerprint for stable identity
            fingerprint: str | None = None
            duration_sec: int | None = None

            # Trust-on-read: check if fingerprint is already in tags
            if existing_tagset.compact.fingerprint and existing_tagset.compact.fingerprint_duration:
                fingerprint = existing_tagset.compact.fingerprint
                duration_sec = existing_tagset.compact.fingerprint_duration
                logger.debug(f"Using cached fingerprint from tags for {audio_file}")
            else:
                # Calculate fingerprint via fpcalc
                try:
                    fp_result = calculate_fingerprint(audio_file)
                    fingerprint = fp_result.fingerprint
                    duration_sec = fp_result.duration_sec
                    logger.debug(f"Calculated fingerprint for {audio_file}")
                except FingerprintError as e:
                    logger.warning(f"Fingerprint calculation failed for {audio_file}: {e}")

            # Build tagset with fingerprint
            tagset = TagSet(
                ids=CanonicalIDs(),
                compact=CompactFields(
                    ruleset_version="canon-1.0",
                    fingerprint=fingerprint,
                    fingerprint_duration=duration_sec,
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
                    cprint(f"\n[red]✘[/red] {audio_file} {mode}")
                    for error in report.errors:
                        print_error(error)
                else:
                    cprint(f"\n[green]✔︎[/green] {audio_file} {mode}")
                    if report.fields_written:
                        cprint(f"  Written: {', '.join(report.fields_written)}")
                    if report.originals_stashed:
                        cprint(f"  Stashed: {', '.join(report.originals_stashed)}")

        except Exception as e:
            if output_format == OutputFormat.TEXT:
                print_error(f"{audio_file}: {e}")
            results.append({"file": str(audio_file), "error": str(e)})

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(results, indent=2))

    # Summary
    if output_format == OutputFormat.TEXT:
        total = len(results)
        errors = sum(1 for r in results if r.get("errors") or r.get("error"))
        cprint(f"\nProcessed {total} files, {errors} errors")

    raise typer.Exit(code=ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


# ====================================================================
# CACHE COMMANDS
# ====================================================================


@cache_app.command("status")
def cache_status() -> None:
    """Show HTTP cache statistics."""
    import time

    config = state.config
    output_format = state.output_format

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
        cprint(json.dumps(result, indent=2))
    else:
        cprint("Cache Status")
        cprint("=" * 40)
        cprint(f"  Directory: {result['cache_directory']}")
        cprint(f"  Enabled: {result['cache_enabled']}")
        cprint(f"  TTL: {result['ttl_seconds']} seconds")
        cprint(f"  Entries: {result['entries']}")
        cprint(f"  Expired: {result['expired_entries']}")
        total_bytes = result["total_size_bytes"]
        size_mb = int(total_bytes) / (1024 * 1024)
        cprint(f"  Size: {size_mb:.2f} MB")

    raise typer.Exit(code=ExitCode.SUCCESS)


@cache_app.command("purge")
def cache_purge(
    expired_only: Annotated[bool, typer.Option(help="Only purge expired entries")] = False,
    force: Annotated[bool, typer.Option(help="Skip confirmation prompt")] = False,
) -> None:
    """Purge HTTP cache entries."""
    from chart_binder.http_cache import HttpCache

    config = state.config
    output_format = state.output_format

    cache_dir = config.http_cache.directory

    if not cache_dir.exists():
        cprint("Cache directory does not exist.")
        raise typer.Exit(code=ExitCode.SUCCESS)

    cache = HttpCache(cache_dir, ttl_seconds=config.http_cache.ttl_seconds)

    removed = 0
    if expired_only:
        removed = cache.purge_expired()
        result = {"action": "purge_expired", "removed_entries": removed}
    else:
        if not force:
            if not typer.confirm("Are you sure you want to clear all caches?"):
                raise typer.Exit(code=ExitCode.SUCCESS)
        cache.clear()
        result = {"action": "purge_all", "removed_entries": "all"}

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        if expired_only:
            print_success(f"Purged {removed} expired entries")
        else:
            print_success("All caches cleared")

    raise typer.Exit(code=ExitCode.SUCCESS)


# ====================================================================
# COVERAGE COMMANDS
# ====================================================================


@coverage_app.command("chart")
def coverage_chart(
    chart_id: Annotated[str, typer.Argument(help="Chart ID")],
    period: Annotated[str, typer.Argument(help="Chart period")],
) -> None:
    """Show coverage report for a chart period."""
    from chart_binder.charts_db import ChartsDB

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        print_error(f"No chart run found for {chart_id} {period}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

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
        cprint(json.dumps(result, indent=2))
    else:
        cprint(f"Coverage Report: {chart_id} {period}")
        cprint("=" * 40)
        cprint(f"  Total entries: {report.total_entries}")
        cprint(f"  Linked: {report.linked_entries}")
        cprint(f"  Unlinked: {report.unlinked_entries}")
        cprint(f"  Coverage: {report.coverage_pct:.1f}%")
        if report.by_method:
            cprint("  By method:")
            for method, count in report.by_method.items():
                cprint(f"    {method}: {count}")
        if report.by_confidence:
            cprint("  By confidence:")
            for bucket, count in report.by_confidence.items():
                cprint(f"    {bucket}: {count}")

    raise typer.Exit(code=ExitCode.SUCCESS)


@coverage_app.command("missing")
def coverage_missing(
    chart_id: Annotated[str, typer.Argument(help="Chart ID")],
    period: Annotated[str, typer.Argument(help="Chart period")],
    threshold: Annotated[float, typer.Option(help="Minimum confidence threshold")] = 0.60,
) -> None:
    """Show missing or low-confidence entries in a chart."""
    from chart_binder.charts_db import ChartsDB, ChartsETL

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)
    etl = ChartsETL(db)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        print_error(f"No chart run found for {chart_id} {period}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    missing = etl.get_missing_entries(run["run_id"], threshold=threshold)

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(missing, indent=2))
    else:
        cprint(f"Missing Entries: {chart_id} {period}")
        cprint("=" * 40)
        for entry in missing:
            cprint(f"  #{entry['rank']}: {entry['artist_raw']} - {entry['title_raw']}")
            if entry.get("confidence"):
                cprint(f"        (confidence: {entry['confidence']:.2f})")

    raise typer.Exit(code=ExitCode.SUCCESS if missing else ExitCode.NO_RESULTS)


@coverage_app.command("indeterminate")
def coverage_indeterminate(
    chart_id: Annotated[
        str | None, typer.Option("--chart", "-c", help="Filter by chart ID")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Maximum results")] = 50,
) -> None:
    """List indeterminate decisions requiring review.

    Shows decisions in INDETERMINATE state that couldn't be automatically
    resolved and need manual review.

    Examples:
        canon coverage indeterminate
        canon coverage indeterminate --chart nl_top2000
        canon coverage indeterminate --limit 10
    """
    from chart_binder.decisions_db import DecisionsDB, DecisionState

    config = state.config
    output_format = state.output_format

    db = DecisionsDB(config.database.decisions_path)

    # Get all stale/indeterminate decisions
    stale = db.get_stale_decisions()

    # Filter to only INDETERMINATE
    indeterminate = [d for d in stale if d["state"] == DecisionState.INDETERMINATE]

    # Apply chart filter if specified
    if chart_id:
        # Filter by work_key prefix (chart entries have work_key starting with chart_id)
        indeterminate = [d for d in indeterminate if d.get("work_key", "").startswith(chart_id)]

    # Apply limit
    indeterminate = indeterminate[:limit]

    result = {
        "total": len(indeterminate),
        "decisions": [
            {
                "file_id": d["file_id"],
                "work_key": d["work_key"],
                "mb_rg_id": d["mb_rg_id"],
                "mb_release_id": d["mb_release_id"],
                "ruleset_version": d["ruleset_version"],
                "trace_compact": d["trace_compact"],
                "updated_at": d["updated_at"],
            }
            for d in indeterminate
        ],
    }

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        cprint("\n[bold]INDETERMINATE Decisions[/bold]")
        cprint("=" * 50)

        if not indeterminate:
            cprint("  No indeterminate decisions found.")
        else:
            for i, d in enumerate(indeterminate, 1):
                work_key = d.get("work_key", "unknown")
                trace = d.get("trace_compact", "")
                cprint(f"\n  {i}. {work_key}")
                cprint(f"     Release Group: {d['mb_rg_id']}")
                if trace:
                    cprint(f"     Trace: {trace}")

            cprint(f"\n  Total: {len(indeterminate)} indeterminate decisions")

    raise typer.Exit(code=ExitCode.SUCCESS if indeterminate else ExitCode.NO_RESULTS)


@coverage_app.command("drift")
def coverage_drift() -> None:
    """Show drift report for decisions.

    Displays summary of decisions in STALE states indicating changes
    in evidence or rules since the decision was made.

    Examples:
        canon coverage drift
    """
    from chart_binder.decisions_db import DecisionsDB
    from chart_binder.drift import DriftDetector

    config = state.config
    output_format = state.output_format

    db = DecisionsDB(config.database.decisions_path)
    detector = DriftDetector(db)

    # Get stale decisions summary
    stale_summaries = detector.review_drift()

    # Group by state
    by_state: dict[str, list] = {}
    for summary in stale_summaries:
        state_key = summary.state.value
        if state_key not in by_state:
            by_state[state_key] = []
        by_state[state_key].append(summary)

    result = {
        "total_stale": len(stale_summaries),
        "by_state": {state: len(items) for state, items in by_state.items()},
        "details": {
            state: [
                {
                    "file_id": s.file_id,
                    "work_key": s.work_key,
                    "mb_rg_id": s.mb_rg_id,
                    "ruleset_version": s.ruleset_version,
                }
                for s in items[:10]  # Limit details per category
            ]
            for state, items in by_state.items()
        },
    }

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        cprint("\n[bold]Drift Report[/bold]")
        cprint("=" * 50)

        if not stale_summaries:
            cprint("  No stale decisions found. All decisions are current.")
        else:
            for state_key, items in by_state.items():
                state_desc = {
                    "stale_evidence": "STALE_EVIDENCE: New data available",
                    "stale_rules": "STALE_RULES: Ruleset updated",
                    "stale_both": "STALE_BOTH: Evidence and rules changed",
                    "stale_nondeterministic": "STALE_NONDETERMINISTIC: Non-deterministic behavior",
                    "indeterminate": "INDETERMINATE: Requires manual review",
                }.get(state_key, state_key.upper())

                cprint(f"\n  [yellow]{state_desc}[/yellow]: {len(items)} decisions")
                for s in items[:3]:  # Show first 3 per category
                    cprint(f"    - {s.work_key}")
                if len(items) > 3:
                    cprint(f"    ... and {len(items) - 3} more")

            cprint(f"\n  Total stale decisions: {len(stale_summaries)}")

    raise typer.Exit(code=ExitCode.SUCCESS if stale_summaries else ExitCode.NO_RESULTS)


# ====================================================================
# CHARTS COMMANDS
# ====================================================================


@charts_app.command("scrape")
def charts_scrape(
    chart_type: Annotated[str, typer.Argument(help="Chart type (t40/t40jaar/top2000/zwaarste)")],
    period: Annotated[str, typer.Argument(help="Period to scrape")],
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file"),
    ] = None,
    ingest: Annotated[bool, typer.Option(help="Auto-ingest into database")] = False,
    strict: Annotated[bool, typer.Option(help="Fail if entry count below expected")] = False,
    check_continuity: Annotated[
        bool, typer.Option(help="Validate overlap with previous run")
    ] = False,
) -> None:
    """Scrape chart data from web source.

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

    config = state.config
    output_format = state.output_format

    cache = HttpCache(config.http_cache.directory, ttl_seconds=config.http_cache.ttl_seconds)
    db = ChartsDB(config.database.charts_path)

    # Get scraper class and DB ID from registry
    scraper_cls, chart_db_id = SCRAPER_REGISTRY[chart_type]

    # Get previous period's ranks for cross-reference validation if checking continuity
    db_previous_ranks: dict[tuple[str, str], int] | None = None
    prev_period: str | None = None
    if check_continuity:
        prev_period = db.get_adjacent_period(chart_db_id, period, direction=-1)
        if prev_period:
            db_previous_ranks = db.get_entries_with_ranks_by_period(chart_db_id, prev_period)

    with scraper_cls(cache) as scraper:
        result = scraper.scrape_with_validation(period, db_previous_ranks=db_previous_ranks)

    if not result.entries:
        print_error(f"No entries found for {chart_type} {period}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    # Check entry count sanity
    if not result.is_valid:
        msg = (
            f"Entry count sanity check failed: got {result.actual_count}, "
            f"expected ~{result.expected_count} (shortage: {result.shortage})"
        )
        if strict:
            print_error(msg)
            print_error("Possible edge case detected - check scraper for this period")
            raise typer.Exit(code=ExitCode.ERROR)
        else:
            print_warning(msg)

    # Check continuity with previous run
    if check_continuity and prev_period:
        prev_entries = db.get_entries_by_period(chart_db_id, prev_period)
        if prev_entries:
            overlap = calculate_overlap(result.entries, prev_entries)
            result.continuity_overlap = overlap
            result.continuity_reference = prev_period

            if not result.continuity_valid:
                msg = (
                    f"Continuity check failed: only {overlap:.0%} overlap with {prev_period} "
                    f"(expected ≥50%)"
                )
                if strict:
                    print_error(msg)
                    print_error("Possible scraping issue - data may be corrupted")
                    raise typer.Exit(code=ExitCode.ERROR)
                else:
                    print_warning(msg)
            elif output_format == OutputFormat.TEXT:
                print_success(f"Continuity check: {overlap:.0%} overlap with {prev_period}")

        # Report position mismatches from cross-reference validation
        if result.position_mismatches:
            print_warning(
                f"Position cross-reference: {len(result.position_mismatches)} mismatch(es) "
                f"with {prev_period}"
            )
            for artist, title, claimed, actual in result.position_mismatches[:5]:
                cprint(f"   {artist} - {title}: website says #{claimed}, database has #{actual}")
            if len(result.position_mismatches) > 5:
                cprint(f"   ... and {len(result.position_mismatches) - 5} more")
        elif result.rich_entries and output_format == OutputFormat.TEXT:
            print_success(f"Position cross-reference: all positions match {prev_period}")

    # Show warnings if any
    if result.warnings:
        for warning in result.warnings:
            print_warning(warning)

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
        "position_mismatches": (
            [
                {"artist": a, "title": t, "claimed": c, "actual": act}
                for a, t, c, act in result.position_mismatches
            ]
            if result.position_mismatches
            else None
        ),
        "entries": entries_list,
    }

    if output_file:
        output_file.write_text(json.dumps(entries_list, indent=2, ensure_ascii=False))
        if output_format == OutputFormat.TEXT:
            print_success(
                f"Scraped {result.actual_count}/{result.expected_count} entries to {output_file}"
            )
        elif output_format == OutputFormat.JSON:
            cprint(json.dumps({"status": "success", "output_file": str(output_file)}, indent=2))
    else:
        if output_format == OutputFormat.JSON:
            cprint(json.dumps(output_result, indent=2, ensure_ascii=False))
        else:
            status = "[green]✔︎[/green]" if result.is_valid else "[yellow]⚠[/yellow]"
            cprint(
                f"{status} Scraped {result.actual_count}/{result.expected_count} entries for {chart_type} {period}"
            )
            cprint("\nFirst 10 entries:")
            for rank, artist, title in result.entries[:10]:
                cprint(f"  {rank:3d}. {artist} - {title}")
            if len(result.entries) > 10:
                cprint(f"  ... and {len(result.entries) - 10} more")

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
        run_id = etl.ingest(chart_db_id, period, entries_for_ingest)

        if output_format == OutputFormat.TEXT:
            print_success(f"Ingested {result.actual_count} entries (run_id: {run_id[:8]}...)")
        elif output_format == OutputFormat.JSON:
            cprint(json.dumps({"ingested": True, "run_id": run_id}, indent=2))

    raise typer.Exit(code=ExitCode.SUCCESS)


@charts_app.command("scrape-missing")
def charts_scrape_missing(
    chart_type: Annotated[str, typer.Argument(help="Chart type")],
    start_year: Annotated[int | None, typer.Option(help="Start year")] = None,
    end_year: Annotated[int | None, typer.Option(help="End year")] = None,
    ingest: Annotated[bool, typer.Option(help="Auto-ingest into database")] = False,
    strict: Annotated[bool, typer.Option(help="Fail on sanity check failures")] = False,
    dry_run: Annotated[bool, typer.Option(help="Show what would be scraped")] = False,
) -> None:
    """Scrape missing periods for a chart type.

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

    config = state.config
    output_format = state.output_format

    # Get scraper class and DB ID from registry
    scraper_cls, chart_db_id = SCRAPER_REGISTRY[chart_type]

    # Determine year range
    current_year = datetime.datetime.now().year
    if end_year is None:
        end_year = current_year

    # Set default start years per chart type
    default_start_years = {
        "t40": 1965,
        "t40jaar": 1965,
        "top2000": 1999,
        "zwaarste": 2020,
    }
    if start_year is None:
        start_year = default_start_years.get(chart_type, 2000)

    # Get existing runs from database
    db = ChartsDB(config.database.charts_path)
    existing_runs = db.list_runs(chart_db_id)
    existing_periods = {run["period"] for run in existing_runs}

    # Generate all expected periods
    expected_periods: list[str] = []
    if chart_type == "t40":
        # Weekly chart - generate YYYY-Www for each week
        for year in range(start_year, end_year + 1):
            for week in range(1, 53):
                expected_periods.append(f"{year}-W{week:02d}")
    else:
        # Yearly charts
        for year in range(start_year, end_year + 1):
            expected_periods.append(str(year))

    # Find missing periods
    missing_periods = [p for p in expected_periods if p not in existing_periods]

    if not missing_periods:
        print_success(f"No missing periods for {chart_type} ({start_year}-{end_year})")
        raise typer.Exit(code=ExitCode.SUCCESS)

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
            cprint(json.dumps(result, indent=2))
            raise typer.Exit(code=ExitCode.SUCCESS)
    else:
        cprint(f"Chart: {chart_type} ({chart_db_id})")
        cprint(f"Range: {start_year} - {end_year}")
        cprint(f"Expected periods: {len(expected_periods)}")
        cprint(f"Existing: {len(existing_periods)}")
        cprint(f"Missing: {len(missing_periods)}")

        if dry_run:
            cprint("\nMissing periods (first 20):")
            for period in missing_periods[:20]:
                cprint(f"  {period}")
            if len(missing_periods) > 20:
                cprint(f"  ... and {len(missing_periods) - 20} more")
            raise typer.Exit(code=ExitCode.SUCCESS)

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

    cprint(f"\nScraping {len(missing_periods)} missing periods...")

    with scraper_cls(cache) as scraper:
        for i, period in enumerate(missing_periods, 1):
            try:
                result = scraper.scrape_with_validation(period)

                if not result.entries:
                    skipped += 1
                    if output_format == OutputFormat.TEXT:
                        cprint(f"  [{i}/{len(missing_periods)}] {period}: no data (skipped)")
                    continue

                if not result.is_valid and strict:
                    failed += 1
                    cprint(
                        f"  [{i}/{len(missing_periods)}] {period}: [red]✘[/red] sanity check failed "
                        f"({result.actual_count}/{result.expected_count})"
                    )
                    continue

                scraped += 1
                status = "[green]✔︎[/green]" if result.is_valid else "[yellow]⚠[/yellow]"
                if output_format == OutputFormat.TEXT:
                    cprint(
                        f"  [{i}/{len(missing_periods)}] {period}: {status} "
                        f"{result.actual_count}/{result.expected_count} entries"
                    )

                # Ingest if requested
                if ingest and etl:
                    entries_for_ingest = [
                        (rank, artist, title) for rank, artist, title in result.entries
                    ]
                    etl.ingest(chart_db_id, period, entries_for_ingest)

            except Exception as e:
                failed += 1
                if output_format == OutputFormat.TEXT:
                    print_error(f"  [{i}/{len(missing_periods)}] {period}: error: {e}")

    # Summary
    if output_format == OutputFormat.JSON:
        cprint(
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
        cprint(f"\nSummary: {scraped} scraped, {failed} failed, {skipped} skipped")
        if ingest:
            cprint(f"Ingested: {scraped} runs")

    raise typer.Exit(code=ExitCode.SUCCESS if failed == 0 else ExitCode.ERROR)


@charts_app.command("ingest")
def charts_ingest(
    chart_id: Annotated[str, typer.Argument(help="Chart ID")],
    period: Annotated[str, typer.Argument(help="Period")],
    source_file: Annotated[Path, typer.Argument(help="Source JSON file", exists=True)],
    notes: Annotated[str | None, typer.Option(help="Notes about this chart run")] = None,
) -> None:
    """Ingest scraped chart data into database.

    SOURCE_FILE should be a JSON file with entries: [[rank, artist, title], ...]
    """
    from chart_binder.charts_db import ChartsDB, ChartsETL

    config = state.config
    output_format = state.output_format

    # Load entries from JSON file
    try:
        data = json.loads(source_file.read_text())
        entries = [(entry[0], entry[1], entry[2]) for entry in data]
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print_error(f"Error parsing source file: {e}")
        raise typer.Exit(code=ExitCode.ERROR) from None

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
        cprint(json.dumps(result, indent=2))
    else:
        print_success(f"Ingested {len(entries)} entries")
        cprint(f"  Chart: {chart_id}")
        cprint(f"  Period: {period}")
        cprint(f"  Run ID: {run_id}")

    raise typer.Exit(code=ExitCode.SUCCESS)


@charts_app.command("link")
def charts_link(
    chart_id: Annotated[str, typer.Argument(help="Chart ID")],
    period: Annotated[str | None, typer.Argument(help="Period")] = None,
    strategy: Annotated[
        str,
        typer.Option(help="Linking strategy"),
    ] = "multi_source",
    all_periods: Annotated[bool, typer.Option(help="Process all periods")] = False,
    missing_only: Annotated[bool, typer.Option(help="Skip entries with good links")] = False,
    min_confidence: Annotated[float, typer.Option(help="Min confidence threshold")] = 0.85,
    limit: Annotated[int | None, typer.Option(help="Max entries to process")] = None,
    start_rank: Annotated[int | None, typer.Option(help="Start rank")] = None,
    end_rank: Annotated[int | None, typer.Option(help="End rank")] = None,
    prioritize_by_score: Annotated[bool, typer.Option(help="Process by score")] = False,
    batch_size: Annotated[int, typer.Option(help="Commit every N entries")] = 1,
    no_progress: Annotated[bool, typer.Option(help="Disable progress display")] = False,
) -> None:
    """Link chart entries to canonical recordings.

    Uses UnifiedFetcher for multi-source search with LLM adjudication support.

    Performance options:
    - Use --missing-only to skip entries that already have good links
    - Use --all-periods to process all periods of a chart
    - Use --limit to test with small batches (e.g., --limit 10)
    - Use --start-rank/--end-rank to process specific rank ranges
    - Use --prioritize-by-score to process most important entries first
    - Commits after each entry by default (configurable with --batch-size)

    Examples:
        canon charts link nl_top2000 2024
        canon charts link nl_top2000 2024 --missing-only
        canon charts link nl_top2000 --all-periods --missing-only
        canon charts link nl_top2000 2024 --limit 10 --prioritize-by-score
        canon charts link nl_top2000 2024 --start-rank 1 --end-rank 100
    """
    import os

    from chart_binder.charts_db import ChartsDB, ChartsETL
    from chart_binder.fetcher import FetcherConfig, FetchMode, UnifiedFetcher

    config = state.config
    output_format = state.output_format

    # Validate arguments
    if all_periods and period:
        print_error("Cannot specify both PERIOD and --all-periods")
        raise typer.Exit(code=ExitCode.ERROR)
    if not all_periods and not period:
        print_error("Must specify either PERIOD or --all-periods")
        raise typer.Exit(code=ExitCode.ERROR)

    db = ChartsDB(config.database.charts_path)

    # Get periods to process
    if all_periods:
        runs = db.list_runs(chart_id)
        if not runs:
            print_error(f"No chart runs found for {chart_id}")
            raise typer.Exit(code=ExitCode.NO_RESULTS)
        periods_to_process = [run["period"] for run in runs]
        cprint(f"Processing {len(periods_to_process)} periods for {chart_id}")
    else:
        periods_to_process = [period]  # type: ignore[list-item]

    # Create UnifiedFetcher if using multi_source strategy
    fetcher = None
    if strategy == "multi_source":
        fetcher_config = FetcherConfig(
            cache_dir=config.http_cache.directory,
            db_path=config.database.music_graph_path,
            mode=FetchMode.NORMAL,
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            spotify_client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
        )
        fetcher = UnifiedFetcher(fetcher_config)

    # Initialize LLM adjudicator if enabled for low-confidence matches
    adjudicator = None
    if config.llm.enabled and strategy == "multi_source":
        from chart_binder.llm.react_adjudicator import ReActAdjudicator

        # Initialize SearxNG web search if enabled
        web_search = None
        if config.llm.searxng.enabled:
            from chart_binder.llm.searxng import SearxNGSearchTool

            web_search = SearxNGSearchTool(
                base_url=config.llm.searxng.url,
                timeout=config.llm.searxng.timeout_s,
            )

        adjudicator = ReActAdjudicator(config=config.llm, search_tool=web_search)
        cprint("[blue]LLM adjudication enabled for low-confidence matches[/blue]")

    # Pass config to ChartsETL so it uses the shared resolver pipeline (7-rule algorithm)
    etl = ChartsETL(db, fetcher=fetcher, adjudicator=adjudicator, config=config)

    # Process each period
    all_results = []
    for idx, current_period in enumerate(periods_to_process, 1):
        if all_periods:
            cprint(f"\n[{idx}/{len(periods_to_process)}] Processing period: {current_period}")

        run = db.get_run_by_period(chart_id, current_period)
        if not run:
            print_warning(f"No chart run found for {chart_id} {current_period}")
            continue

        report = etl.link(
            run["run_id"],
            strategy=strategy,
            missing_only=missing_only,
            min_confidence=min_confidence,
            limit=limit,
            start_rank=start_rank,
            end_rank=end_rank,
            prioritize_by_score=prioritize_by_score,
            chart_id=chart_id,
            batch_size=batch_size,
            progress=not no_progress,
        )

        result = {
            "chart_id": chart_id,
            "period": current_period,
            "run_id": run["run_id"],
            "total_entries": report.total_entries,
            "linked_entries": report.linked_entries,
            "coverage_pct": round(report.coverage_pct, 2),
            "by_method": report.by_method,
        }
        all_results.append(result)

        if not all_periods or output_format == OutputFormat.JSON:
            # Show individual results
            if output_format == OutputFormat.TEXT:
                print_success(f"Linked {report.linked_entries}/{report.total_entries} entries")
                cprint(f"  Coverage: {report.coverage_pct:.1f}%")
                if report.by_method:
                    cprint("  By method:")
                    for method, count in report.by_method.items():
                        cprint(f"    {method}: {count}")

    # Close fetcher if we created one
    if fetcher:
        fetcher.close()

    # Output final results
    if output_format == OutputFormat.JSON:
        if all_periods:
            cprint(json.dumps({"chart_id": chart_id, "periods": all_results}, indent=2))
        else:
            cprint(json.dumps(all_results[0], indent=2))
    elif all_periods:
        # Summary for all periods
        total_linked = sum(r["linked_entries"] for r in all_results)
        total_entries = sum(r["total_entries"] for r in all_results)
        avg_coverage = (total_linked / total_entries * 100) if total_entries > 0 else 0
        cprint("\n[green]✔︎ All periods complete:[/green]")
        cprint(f"  Periods processed: {len(all_results)}")
        cprint(f"  Total linked: {total_linked}/{total_entries}")
        cprint(f"  Overall coverage: {avg_coverage:.1f}%")

    raise typer.Exit(code=ExitCode.SUCCESS)


@charts_app.command("missing")
def charts_missing(
    chart_id: Annotated[str, typer.Argument(help="Chart ID")],
    period: Annotated[str, typer.Argument(help="Period")],
) -> None:
    """Show entries that failed to link."""
    # Delegate to coverage missing
    coverage_missing(chart_id=chart_id, period=period)


@charts_app.command("export")
def charts_export(
    work_key: Annotated[str, typer.Argument(help="Work key")],
    positions: Annotated[bool, typer.Option(help="Include position details")] = False,
) -> None:
    """Export chart history for a work."""
    from chart_binder.charts_db import ChartsDB
    from chart_binder.charts_export import ChartsExporter

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)
    exporter = ChartsExporter(db)

    blob = exporter.export_for_work(work_key, include_positions=positions)
    json_str = blob.to_json(include_positions=positions)

    if output_format == OutputFormat.JSON:
        cprint(json.dumps({"work_key": work_key, "charts_blob": json_str}, indent=2))
    else:
        cprint(f"CHARTS blob for {work_key}:")
        cprint(json_str)

    raise typer.Exit(code=ExitCode.SUCCESS)


# ====================================================================
# REVIEW COMMANDS
# ====================================================================


@review_app.command("list")
def review_list(
    source: Annotated[
        str | None,
        typer.Option(help="Filter by source (indeterminate/llm_review/conflict)"),
    ] = None,
    limit: Annotated[int, typer.Option(help="Maximum items to show")] = 20,
) -> None:
    """List items needing review."""
    config, output_format = _current_state()
    queue = ReviewQueue(config.llm.review_queue_path)
    try:
        source_filter = ReviewSource(source) if source else None
    except ValueError:
        valid = ", ".join([s.value for s in ReviewSource])
        print_error(f"Invalid review source: '{source}'. Valid options are: {valid}")
        raise typer.Exit(code=ExitCode.ERROR) from None
    items = queue.get_pending(source=source_filter, limit=limit)

    if output_format == OutputFormat.JSON:
        payload = [
            {
                "review_id": item.review_id,
                "file_id": item.file_id,
                "work_key": item.work_key,
                "source": item.source.value,
                "created_at": item.created_at,
            }
            for item in items
        ]
        cprint(json.dumps(payload, indent=2))
    else:
        if not items:
            print_success("No pending review items.")
            raise typer.Exit(code=ExitCode.SUCCESS)

        cprint(f"Pending Review Items ({len(items)}):")
        cprint("=" * 50)
        for item in items:
            cprint(f"ID:    {item.review_id[:8]}...")
            cprint(f"Work:  {item.work_key}")
            cprint(f"Source:{item.source.value}")
            cprint("=" * 50)

    raise typer.Exit(code=ExitCode.SUCCESS)


@review_app.command("show")
def review_show(
    review_id: Annotated[str, typer.Argument(help="Review ID")],
) -> None:
    """Show details for a review item."""
    config, output_format = _current_state()
    queue = ReviewQueue(config.llm.review_queue_path)
    item = queue.get_item(review_id)

    if not item:
        print_error(f"Review item not found: {review_id}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    if output_format == OutputFormat.JSON:
        cprint(
            json.dumps(
                {
                    "review_id": item.review_id,
                    "file_id": item.file_id,
                    "work_key": item.work_key,
                    "source": item.source.value,
                    "evidence_bundle": item.evidence_bundle,
                    "decision_trace": item.decision_trace,
                    "llm_suggestion": item.llm_suggestion,
                    "created_at": item.created_at,
                },
                indent=2,
            )
        )
    else:
        cprint(item.to_display())

    raise typer.Exit(code=ExitCode.SUCCESS)


@review_app.command("accept")
def review_accept(
    review_id: Annotated[str, typer.Argument(help="Review ID")],
    crg_mbid: Annotated[str, typer.Option("--crg", help="CRG MBID to accept")],
    rr_mbid: Annotated[str | None, typer.Option("--rr", help="RR MBID to accept")] = None,
    notes: Annotated[str | None, typer.Option(help="Review notes")] = None,
) -> None:
    """Accept a specific CRG for a review item."""
    config, output_format = _current_state()
    queue = ReviewQueue(config.llm.review_queue_path)
    succeeded = queue.complete_review(
        review_id,
        action=ReviewAction.ACCEPT,
        action_data={"crg_mbid": crg_mbid, "rr_mbid": rr_mbid},
        reviewed_by=_reviewed_by(),
        notes=notes,
    )

    if not succeeded:
        print_error(f"Failed to complete review: {review_id}")
        raise typer.Exit(code=ExitCode.ERROR)

    if output_format == OutputFormat.JSON:
        cprint(json.dumps({"status": "accepted", "review_id": review_id}, indent=2))
    else:
        print_success(f"Accepted review {review_id[:8]}... with CRG {crg_mbid}")

    raise typer.Exit(code=ExitCode.SUCCESS)


@review_app.command("reject")
def review_reject(
    review_id: Annotated[str, typer.Argument(help="Review ID")],
    notes: Annotated[str | None, typer.Option(help="Review notes")] = None,
) -> None:
    """Reject/skip a review item."""
    config, output_format = _current_state()
    queue = ReviewQueue(config.llm.review_queue_path)
    succeeded = queue.complete_review(
        review_id,
        action=ReviewAction.SKIP,
        reviewed_by=_reviewed_by(),
        notes=notes,
    )

    if not succeeded:
        print_error(f"Failed to skip review: {review_id}")
        raise typer.Exit(code=ExitCode.ERROR)

    if output_format == OutputFormat.JSON:
        cprint(json.dumps({"status": "skipped", "review_id": review_id}, indent=2))
    else:
        print_success(f"Skipped review {review_id[:8]}...")

    raise typer.Exit(code=ExitCode.SUCCESS)


# ====================================================================
# ANALYTICS COMMANDS
# ====================================================================


@analytics_app.command("history")
def analytics_history(
    artist: Annotated[str, typer.Argument(help="Artist name")],
    title: Annotated[str, typer.Argument(help="Song title")],
    fuzzy_threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Fuzzy match threshold (0-1)")
    ] = 0.7,
) -> None:
    """Show chart history for a song across all charts.

    Finds a song by artist and title (with fuzzy matching) and displays
    all of its chart appearances.

    Examples:
        canon analytics history "Queen" "Bohemian Rhapsody"
        canon analytics history "Wham!" "Last Christmas" --threshold 0.8
    """
    from chart_binder.analytics import ChartAnalytics
    from chart_binder.charts_db import ChartsDB

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)
    analytics = ChartAnalytics(db)

    # Find the song
    song = analytics.get_song_by_artist_title(artist, title, threshold=fuzzy_threshold)

    if not song:
        print_error(f"No song found matching '{artist}' - '{title}'")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    # Get chart history
    history = analytics.get_song_chart_history(song.song_id)

    if not history:
        print_warning(
            f"Song found but has no chart appearances: {song.artist_canonical} - {song.title_canonical}"
        )
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    result = {
        "song_id": song.song_id,
        "artist": song.artist_canonical,
        "title": song.title_canonical,
        "recording_mbid": song.recording_mbid,
        "appearances": [
            {
                "chart": h.chart_name,
                "period": h.period,
                "rank": h.rank,
                "previous_position": h.previous_position,
                "weeks_on_chart": h.weeks_on_chart,
            }
            for h in history
        ],
    }

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        cprint(f"\n[bold]{song.artist_canonical} - {song.title_canonical}[/bold]")
        if song.recording_mbid:
            cprint(f"  MBID: {song.recording_mbid}")
        cprint("")

        # Group by chart
        from collections import defaultdict

        by_chart: dict[str, list] = defaultdict(list)
        for h in history:
            by_chart[h.chart_name].append(h)

        for chart_name, appearances in by_chart.items():
            cprint(f"[green]{chart_name}[/green]")
            for h in appearances:
                pos_info = f"#{h.rank}"
                if h.previous_position:
                    delta = h.previous_position - h.rank
                    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                    pos_info += f" ({arrow}{abs(delta)} from #{h.previous_position})"
                if h.weeks_on_chart:
                    pos_info += f" [dim]{h.weeks_on_chart}w[/dim]"
                cprint(f"  {h.period}: {pos_info}")

    raise typer.Exit(code=ExitCode.SUCCESS)


@analytics_app.command("compare")
def analytics_compare(
    run1: Annotated[str, typer.Argument(help="First chart run (chart_id:period or run_id)")],
    run2: Annotated[str, typer.Argument(help="Second chart run (chart_id:period or run_id)")],
    show_movers: Annotated[int, typer.Option("--movers", "-m", help="Show top N movers")] = 10,
    show_new: Annotated[bool, typer.Option("--new", help="Show new entries")] = False,
    show_dropped: Annotated[bool, typer.Option("--dropped", help="Show dropped entries")] = False,
) -> None:
    """Compare two chart runs for overlap and differences.

    Identifies songs common to both runs, songs unique to each run,
    and tracks that moved significantly between runs.

    Examples:
        canon analytics compare nl_top2000:2024 nl_top2000:2023
        canon analytics compare nl_top2000:2024 nl_top2000:2023 --movers 20
        canon analytics compare nl_top40:2024-W01 nl_top40:2024-W02 --new --dropped
    """
    from chart_binder.analytics import ChartAnalytics
    from chart_binder.charts_db import ChartsDB

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)
    analytics = ChartAnalytics(db)

    comparison = analytics.compare_charts(run1, run2)

    if not comparison.run1_period or not comparison.run2_period:
        print_error(f"Could not find one or both chart runs: {run1}, {run2}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    result = {
        "run1": {"id": comparison.run1_id, "period": comparison.run1_period},
        "run2": {"id": comparison.run2_id, "period": comparison.run2_period},
        "overlap_pct": round(comparison.overlap_pct, 1),
        "common_count": len(comparison.common_songs),
        "only_run1_count": len(comparison.only_in_run1),
        "only_run2_count": len(comparison.only_in_run2),
    }

    if show_movers > 0:
        result["top_movers"] = [
            {"artist": m[0], "title": m[1], "rank1": m[2], "rank2": m[3], "delta": m[4]}
            for m in comparison.movers[:show_movers]
        ]
    if show_new:
        result["new_entries"] = [
            {"artist": e[0], "title": e[1], "rank": e[2]} for e in comparison.only_in_run2
        ]
    if show_dropped:
        result["dropped_entries"] = [
            {"artist": e[0], "title": e[1], "rank": e[2]} for e in comparison.only_in_run1
        ]

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        cprint("\n[bold]Chart Comparison[/bold]")
        cprint(f"  Run 1: {comparison.run1_period}")
        cprint(f"  Run 2: {comparison.run2_period}")
        cprint("")
        cprint(f"[green]Overlap: {comparison.overlap_pct:.1f}%[/green]")
        cprint(f"  Common songs: {len(comparison.common_songs)}")
        cprint(f"  Only in {comparison.run1_period}: {len(comparison.only_in_run1)}")
        cprint(f"  Only in {comparison.run2_period}: {len(comparison.only_in_run2)}")

        if show_movers > 0 and comparison.movers:
            cprint(f"\n[bold]Top {min(show_movers, len(comparison.movers))} Movers:[/bold]")
            for artist, title, rank1, rank2, delta in comparison.movers[:show_movers]:
                direction = "[red]↓[/red]" if delta > 0 else "[green]↑[/green]"
                cprint(f"  {direction} {abs(delta):+3d}: {artist} - {title} (#{rank1} → #{rank2})")

        if show_new and comparison.only_in_run2:
            cprint(f"\n[bold]New in {comparison.run2_period}:[/bold]")
            for artist, title, rank in comparison.only_in_run2[:20]:
                cprint(f"  #{rank:3d}: {artist} - {title}")
            if len(comparison.only_in_run2) > 20:
                cprint(f"  ... and {len(comparison.only_in_run2) - 20} more")

        if show_dropped and comparison.only_in_run1:
            cprint(f"\n[bold]Dropped from {comparison.run1_period}:[/bold]")
            for artist, title, rank in comparison.only_in_run1[:20]:
                cprint(f"  #{rank:3d}: {artist} - {title}")
            if len(comparison.only_in_run1) > 20:
                cprint(f"  ... and {len(comparison.only_in_run1) - 20} more")

    raise typer.Exit(code=ExitCode.SUCCESS)


@analytics_app.command("lookup")
def analytics_lookup(
    artist: Annotated[str, typer.Argument(help="Artist name")],
    title: Annotated[str, typer.Argument(help="Song title")],
    threshold: Annotated[
        float, typer.Option("--threshold", "-t", help="Fuzzy match threshold (0-1)")
    ] = 0.7,
) -> None:
    """Look up a song in the database by artist and title.

    Uses fuzzy matching to find songs even with slight variations
    in spelling or formatting.

    Examples:
        canon analytics lookup "Beatles" "Hey Jude"
        canon analytics lookup "Wham" "Last Christmas" --threshold 0.6
    """
    from chart_binder.analytics import ChartAnalytics
    from chart_binder.charts_db import ChartsDB

    config = state.config
    output_format = state.output_format

    db = ChartsDB(config.database.charts_path)
    analytics = ChartAnalytics(db)

    song = analytics.get_song_by_artist_title(artist, title, threshold=threshold)

    if not song:
        print_error(f"No song found matching '{artist}' - '{title}'")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    result = {
        "song_id": song.song_id,
        "artist_canonical": song.artist_canonical,
        "title_canonical": song.title_canonical,
        "artist_sort": song.artist_sort,
        "work_key": song.work_key,
        "recording_mbid": song.recording_mbid,
        "release_group_mbid": song.release_group_mbid,
        "spotify_id": song.spotify_id,
        "isrc": song.isrc,
    }

    if output_format == OutputFormat.JSON:
        cprint(json.dumps(result, indent=2))
    else:
        cprint(f"\n[green]Found:[/green] {song.artist_canonical} - {song.title_canonical}")
        cprint(f"  Song ID: {song.song_id}")
        if song.work_key:
            cprint(f"  Work Key: {song.work_key}")
        if song.recording_mbid:
            cprint(f"  Recording MBID: {song.recording_mbid}")
        if song.release_group_mbid:
            cprint(f"  Release Group MBID: {song.release_group_mbid}")
        if song.spotify_id:
            cprint(f"  Spotify ID: {song.spotify_id}")
        if song.isrc:
            cprint(f"  ISRC: {song.isrc}")

    raise typer.Exit(code=ExitCode.SUCCESS)


# ====================================================================
# PLAYLIST COMMANDS
# ====================================================================


@playlist_app.command("generate")
def playlist_generate(
    chart_period: Annotated[
        str, typer.Argument(help="Chart and period (chart_id:period, e.g., 'nl_top2000:2024')")
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file path")] = Path(
        "playlist.m3u"
    ),
    format: Annotated[
        str, typer.Option("--format", "-f", help="Playlist format (m3u or m3u8)")
    ] = "m3u",
    music_library: Annotated[
        Path | None, typer.Option("--library", "-l", help="Path to music library")
    ] = None,
    beets_db: Annotated[
        Path | None, typer.Option("--beets", "-b", help="Path to beets database")
    ] = None,
    relative_paths: Annotated[
        bool, typer.Option("--relative", "-r", help="Use relative paths")
    ] = False,
    show_missing: Annotated[
        bool, typer.Option("--show-missing", help="Show missing entries report")
    ] = False,
) -> None:
    """Generate M3U/M3U8 playlist from a chart run.

    Resolves chart entries to local audio files via beets database or
    filesystem search.

    Examples:
        canon playlist generate nl_top2000:2024 --output top2000_2024.m3u
        canon playlist generate nl_top40:2024-W01 --format m3u8 --output chart.m3u8
        canon playlist generate nl_top2000:2024 --beets ~/.config/beets/library.db
        canon playlist generate nl_top2000:2024 --library /media/music --relative
    """
    from chart_binder.charts_db import ChartsDB
    from chart_binder.playlist import (
        PlaylistFormat,
        PlaylistGenerator,
        get_beets_db_path,
        get_music_library_path,
    )

    config = state.config
    output_format_cli = state.output_format

    # Parse chart_id:period
    if ":" not in chart_period:
        print_error("Invalid format. Use 'chart_id:period' (e.g., 'nl_top2000:2024')")
        raise typer.Exit(code=ExitCode.ERROR)

    chart_id, period = chart_period.split(":", 1)

    # Determine playlist format
    try:
        playlist_format = PlaylistFormat(format.lower())
    except ValueError:
        print_error(f"Invalid format '{format}'. Use 'm3u' or 'm3u8'")
        raise typer.Exit(code=ExitCode.ERROR) from None

    # Get paths (from args or environment)
    lib_path = music_library or get_music_library_path()
    beets_path = beets_db or get_beets_db_path()

    if not lib_path and not beets_path:
        print_warning("No music library or beets database configured.")
        print_warning(
            "Set MUSIC_LIBRARY or BEETS_CONFIG environment variables, or use --library/--beets options."
        )

    # Create generator
    db = ChartsDB(config.database.charts_path)
    generator = PlaylistGenerator(
        charts_db=db,
        music_library=lib_path,
        beets_db_path=beets_path,
    )

    # Generate playlist
    result = generator.generate(
        chart_id=chart_id,
        period=period,
        output=output,
        format=playlist_format,
        use_relative_paths=relative_paths,
    )

    # Build result dict for JSON output
    result_dict = {
        "chart_id": result.chart_id,
        "period": result.period,
        "output_path": str(result.output_path),
        "found": result.found,
        "total": result.total,
        "coverage_pct": round(result.coverage_pct, 1),
        "missing_count": len(result.missing),
    }

    if output_format_cli == OutputFormat.JSON:
        if show_missing:
            result_dict["missing"] = [
                {
                    "rank": m.rank,
                    "artist": m.artist,
                    "title": m.title,
                    "reason": m.reason.value,
                    "song_id": m.song_id,
                }
                for m in result.missing
            ]
        cprint(json.dumps(result_dict, indent=2))
    else:
        if result.total == 0:
            print_error(f"No chart run found for {chart_id}:{period}")
            raise typer.Exit(code=ExitCode.NO_RESULTS)

        cprint("\n[bold]Playlist Generated[/bold]")
        cprint(f"  Chart: {chart_id}:{period}")
        cprint(f"  Output: {result.output_path}")
        cprint(f"  Format: {playlist_format.value}")
        cprint("")
        cprint(
            f"  [green]Found:[/green] {result.found}/{result.total} entries ({result.coverage_pct:.1f}% coverage)"
        )

        if result.missing:
            cprint(f"  [yellow]Missing:[/yellow] {len(result.missing)} entries")

            if show_missing:
                cprint("")
                report = generator.get_missing_report(result)
                cprint(report)

    raise typer.Exit(code=ExitCode.SUCCESS)


@playlist_app.command("info")
def playlist_info(
    chart_period: Annotated[str, typer.Argument(help="Chart and period (chart_id:period)")],
) -> None:
    """Show information about a chart run for playlist generation.

    Displays entry count and linked song statistics without generating a playlist.

    Examples:
        canon playlist info nl_top2000:2024
        canon playlist info nl_top40:2024-W01
    """
    from chart_binder.charts_db import ChartsDB

    config = state.config
    output_format_cli = state.output_format

    # Parse chart_id:period
    if ":" not in chart_period:
        print_error("Invalid format. Use 'chart_id:period' (e.g., 'nl_top2000:2024')")
        raise typer.Exit(code=ExitCode.ERROR)

    chart_id, period = chart_period.split(":", 1)

    db = ChartsDB(config.database.charts_path)

    # Get run
    run = db.get_run_by_period(chart_id, period)
    if not run:
        print_error(f"No chart run found for {chart_id}:{period}")
        raise typer.Exit(code=ExitCode.NO_RESULTS)

    # Get entry stats
    conn = db._get_connection()
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Total entries
        cursor.execute("SELECT COUNT(*) as cnt FROM chart_entry WHERE run_id = ?", (run["run_id"],))
        total_entries = cursor.fetchone()["cnt"]

        # Linked entries (have at least one song link)
        cursor.execute(
            """
            SELECT COUNT(DISTINCT e.entry_id) as cnt
            FROM chart_entry e
            JOIN chart_entry_song es ON e.entry_id = es.entry_id
            WHERE e.run_id = ?
            """,
            (run["run_id"],),
        )
        linked_entries = cursor.fetchone()["cnt"]

        # Entries with MBID
        cursor.execute(
            """
            SELECT COUNT(DISTINCT e.entry_id) as cnt
            FROM chart_entry e
            JOIN chart_entry_song es ON e.entry_id = es.entry_id
            JOIN song s ON es.song_id = s.song_id
            WHERE e.run_id = ? AND s.recording_mbid IS NOT NULL
            """,
            (run["run_id"],),
        )
        mbid_entries = cursor.fetchone()["cnt"]

    finally:
        conn.close()

    link_pct = (linked_entries / total_entries * 100) if total_entries > 0 else 0.0
    mbid_pct = (mbid_entries / total_entries * 100) if total_entries > 0 else 0.0

    result_dict = {
        "chart_id": chart_id,
        "period": period,
        "run_id": run["run_id"],
        "total_entries": total_entries,
        "linked_entries": linked_entries,
        "link_pct": round(link_pct, 1),
        "mbid_entries": mbid_entries,
        "mbid_pct": round(mbid_pct, 1),
    }

    if output_format_cli == OutputFormat.JSON:
        cprint(json.dumps(result_dict, indent=2))
    else:
        cprint(f"\n[bold]{chart_id}:{period}[/bold]")
        cprint(f"  Run ID: {run['run_id']}")
        cprint("")
        cprint(f"  Total entries: {total_entries}")
        cprint(f"  Linked to songs: {linked_entries} ({link_pct:.1f}%)")
        cprint(f"  With MBIDs: {mbid_entries} ({mbid_pct:.1f}%)")
        cprint("")
        if link_pct < 50:
            print_warning("Low song linkage. Run 'canon charts link' to improve coverage.")
        elif mbid_pct < link_pct * 0.5:
            print_warning(
                "Many linked songs missing MBIDs. Run 'canon charts enrich' to add MBIDs."
            )

    raise typer.Exit(code=ExitCode.SUCCESS)


# ====================================================================
# ENTRY POINT
# ====================================================================


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
