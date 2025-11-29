from __future__ import annotations

import json
import logging
import os
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


def _convert_evidence_bundle(
    bundle: Any, audio_file: Path, tagset: Any, musicgraph_db: Any = None
) -> dict[str, Any]:
    """
    Convert EvidenceBundle dataclass to dict format expected by resolver.

    The resolver expects a specific structure with recording_candidates containing
    rg_candidates, while EvidenceBundle has a flatter structure. This bridges
    the two formats.

    Args:
        bundle: EvidenceBundle from CandidateBuilder
        audio_file: Path to the audio file
        tagset: TagSet from reading the file
        musicgraph_db: MusicGraphDB instance for fetching releases
    """
    # Build recording_candidates from the bundle's recordings and release_groups
    recording_candidates = []

    # Group release groups by recording
    rg_by_recording: dict[str, list[dict[str, Any]]] = {}
    for rg in bundle.release_groups:
        rg_mbid = rg.get("mbid")

        # Fetch releases for this release group from the database
        releases_data = []
        if musicgraph_db and rg_mbid:
            db_releases = musicgraph_db.get_releases_in_group(rg_mbid)
            for rel in db_releases:
                # Parse flags from JSON if present
                flags = {}
                flags_json = rel.get("flags_json")
                if flags_json:
                    try:
                        flags = json.loads(flags_json)
                    except json.JSONDecodeError:
                        pass

                # Default to official if no flags
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

        # Find which recording this RG is associated with
        # For now, associate all RGs with all recordings (will be refined later)
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

    # Build the evidence bundle dict
    evidence_dict: dict[str, Any] = {
        "artifact": {
            "file_path": str(audio_file),
        },
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

    return evidence_dict


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


def _get_or_calc_fingerprint(
    audio_file: Path,
    tagset: Any,  # TagSet
    logger: logging.Logger,
) -> tuple[str | None, int | None]:
    """
    Get fingerprint from tags (trust-on-read) or calculate it.

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


@canon.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--explain", is_flag=True, help="Show detailed decision rationale")
@click.option("--no-persist", is_flag=True, help="Skip persisting decisions to database")
@click.option("--quiet", "-q", is_flag=True, help="Show only summary, not per-file output")
@click.pass_context
def decide(
    ctx: click.Context,
    paths: tuple[Path, ...],
    explain: bool,
    no_persist: bool,
    quiet: bool,
) -> None:
    """
    Make canonicalization decisions for audio files.

    Resolves the Canonical Release Group (CRG) and Representative Release (RR)
    for each file based on available metadata. Uses the UnifiedFetcher to search
    multiple sources (MusicBrainz, Discogs, Spotify, AcoustID) and the
    CandidateBuilder to construct evidence bundles. Persists decisions to the
    decisions database for drift tracking.

    If LLM adjudication is enabled in config (llm.enabled=true), INDETERMINATE
    decisions are automatically adjudicated. High-confidence LLM decisions are
    auto-accepted; low-confidence decisions remain INDETERMINATE.

    Use --quiet/-q for batch processing to show only progress and summary instead
    of detailed per-file output.
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

    output_format = ctx.obj["output"]
    config: Config = ctx.obj["config"]
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
        # Pass API credentials from config
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
        from chart_binder.llm import LLMAdjudicator

        adjudicator = LLMAdjudicator(config=config.llm)
        auto_accept_threshold = config.llm.auto_accept_threshold
        logger.info(
            f"LLM adjudication enabled (auto-accept threshold: {auto_accept_threshold})"
        )

    audio_files = _collect_audio_files(paths)
    logger.debug(f"Collected {len(audio_files)} audio files")

    # Track statistics for quiet mode
    stats = {"decided": 0, "indeterminate": 0, "llm_adjudicated": 0, "errors": 0}
    total_files = len(audio_files)

    if quiet and output_format == OutputFormat.TEXT:
        click.echo(f"Processing {total_files} files...")

    with UnifiedFetcher(fetcher_config) as fetcher:
        for idx, audio_file in enumerate(audio_files, 1):
            try:
                tagset = verify(audio_file)

                # Get or calculate fingerprint for stable identity
                fingerprint, duration_sec = _get_or_calc_fingerprint(audio_file, tagset, logger)

                # Gather evidence from multiple sources - don't trust any single source
                # Each source adds to the candidate pool; resolver decides what's canonical

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
                    # Separate by source to ensure balanced evidence gathering
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
                    for result in barcode_results[:5]:  # Limit to top 5 barcode matches
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

                # Extract metadata from decision trace (decided values)
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
                    if idx % 10 == 0 or idx == total_files:  # Update every 10 files or at end
                        pct = (idx / total_files) * 100
                        llm_stat = f" 🤖{stats['llm_adjudicated']}" if adjudicator else ""
                        click.echo(
                            f"  Progress: {idx}/{total_files} ({pct:.1f}%) "
                            f"[✔︎{stats['decided']}{llm_stat} ∆{stats['indeterminate']} ✘{stats['errors']}]",
                            err=True,
                        )

                # Show detailed output if not quiet
                if not quiet and output_format == OutputFormat.TEXT:
                    state_icon = "✔︎" if decision.state.value == "decided" else "∆"
                    if llm_adjudicated:
                        state_icon = "🤖"  # Robot emoji for LLM-adjudicated
                    click.echo(f"\n{state_icon} {audio_file}")
                    click.echo(f"  Artist: {artist_name}")
                    if recording_title:
                        click.echo(f"  Recording: {recording_title}")
                    click.echo(f"  State: {decision.state.value}")
                    if llm_adjudicated:
                        click.echo(f"  LLM Adjudicated: confidence={llm_confidence:.2f}")
                        if llm_rationale:
                            click.echo(f"  LLM Rationale: {llm_rationale[:100]}")
                    if decision.release_group_mbid:
                        click.echo(f"  CRG: {decision.release_group_mbid}")
                        if release_group_title:
                            click.echo(f"       Title: {release_group_title}")
                        if artist_credits:
                            click.echo(f"       Artist Credit: {artist_credits}")
                        if discogs_master_id:
                            click.echo(f"       Discogs Master: {discogs_master_id}")
                        click.echo(f"       ({decision.crg_rationale})")
                    if decision.release_mbid:
                        click.echo(f"  RR:  {decision.release_mbid}")
                        if selected_release_title:
                            click.echo(f"       Title: {selected_release_title}")
                        if discogs_release_id:
                            click.echo(f"       Discogs Release: {discogs_release_id}")
                        click.echo(f"       ({decision.rr_rationale})")
                    click.echo(f"  Trace: {decision.compact_tag}")
                    if explain:
                        click.echo("\n" + decision.decision_trace.to_human_readable())

                    if decision.decision_trace.missing_facts:
                        click.echo("\nMissing Facts:")
                        for fact in decision.decision_trace.missing_facts:
                            click.echo(f"  - {fact}")

            except Exception as e:
                stats["errors"] += 1
                if not quiet and output_format == OutputFormat.TEXT:
                    click.echo(f"\n✘ {audio_file}: {e}", err=True)
                results.append({"file": str(audio_file), "error": str(e)})

                # Show progress in quiet mode even for errors
                if quiet and output_format == OutputFormat.TEXT:
                    if idx % 10 == 0 or idx == total_files:
                        pct = (idx / total_files) * 100
                        click.echo(
                            f"  Progress: {idx}/{total_files} ({pct:.1f}%) "
                            f"[✔︎{stats['decided']} ∆{stats['indeterminate']} ✘{stats['errors']}]",
                            err=True,
                        )

    if output_format == OutputFormat.JSON:
        click.echo(json.dumps(results, indent=2))
    elif quiet and output_format == OutputFormat.TEXT:
        # Print final summary in quiet mode
        click.echo(f"\n✓ Completed {total_files} files:")
        click.echo(f"  Decided:       {stats['decided']}")
        if stats["llm_adjudicated"] > 0:
            click.echo(f"    (via LLM):   {stats['llm_adjudicated']}")
        click.echo(f"  Indeterminate: {stats['indeterminate']}")
        if stats["errors"] > 0:
            click.echo(f"  Errors:        {stats['errors']}")

    sys.exit(ExitCode.SUCCESS if results else ExitCode.NO_RESULTS)


@canon.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
@click.option("--apply", is_flag=True, help="Apply changes (required for actual writes)")
@click.pass_context
def write(ctx: click.Context, paths: tuple[Path, ...], dry_run: bool, apply: bool) -> None:
    """
    Write canonical tags to audio files.

    Writes decision trace, canonical IDs, fingerprint, and optionally CHARTS
    blob to files. The fingerprint is stored for stable file identity across
    tag edits and file moves.

    Use --dry-run to preview changes without writing.
    Use --apply to confirm actual writes (safety feature).
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
    output_format = ctx.obj["output"]
    results = []

    if not dry_run and not apply:
        click.echo("Error: Use --dry-run to preview or --apply to write changes.", err=True)
        sys.exit(ExitCode.ERROR)

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
            # In full implementation, this would include decision data
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

    # Check continuity with previous run
    if check_continuity and prev_period:
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

        # Report position mismatches from cross-reference validation
        if result.position_mismatches:
            click.echo(
                f"⚠ Position cross-reference: {len(result.position_mismatches)} mismatch(es) "
                f"with {prev_period}",
                err=True,
            )
            for artist, title, claimed, actual in result.position_mismatches[:5]:
                click.echo(
                    f"   {artist} - {title}: website says #{claimed}, database has #{actual}",
                    err=True,
                )
            if len(result.position_mismatches) > 5:
                click.echo(
                    f"   ... and {len(result.position_mismatches) - 5} more",
                    err=True,
                )
        elif result.rich_entries and output_format == OutputFormat.TEXT:
            click.echo(f"✔︎ Position cross-reference: all positions match {prev_period}")

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
        run_id = etl.ingest(chart_db_id, period, entries_for_ingest)

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
    db = ChartsDB(config.database.charts_path)
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
                    etl.ingest(chart_db_id, period, entries_for_ingest)

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
    default="multi_source",
    type=click.Choice(["multi_source", "title_artist_year", "bundle_release"]),
    help="Linking strategy (default: multi_source)",
)
@click.option(
    "--missing-only",
    is_flag=True,
    help="Skip entries that already have links with confidence >= min-confidence",
)
@click.option(
    "--min-confidence",
    type=float,
    default=0.85,
    help="Minimum confidence threshold for missing-only filter (default: 0.85)",
)
@click.option(
    "--limit",
    type=int,
    help="Maximum number of entries to process (for testing)",
)
@click.option(
    "--start-rank",
    type=int,
    help="Start processing from this rank (1-based, inclusive)",
)
@click.option(
    "--end-rank",
    type=int,
    help="Stop processing at this rank (1-based, inclusive)",
)
@click.option(
    "--prioritize-by-score",
    is_flag=True,
    help="Process entries by score (sum of total - rank across all runs)",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Commit every N entries for checkpoint/resume (default: 100)",
)
@click.option(
    "--no-progress",
    is_flag=True,
    help="Disable progress display",
)
@click.pass_context
def link(
    ctx: click.Context,
    chart_id: str,
    period: str,
    strategy: str,
    missing_only: bool,
    min_confidence: float,
    limit: int | None,
    start_rank: int | None,
    end_rank: int | None,
    prioritize_by_score: bool,
    batch_size: int,
    no_progress: bool,
) -> None:
    """
    Link chart entries to canonical recordings using multi-source search.

    Uses UnifiedFetcher to search across MusicBrainz, Discogs, and Spotify
    with all enhanced intelligence (popularity weighting, cross-source validation, etc.).

    Performance options:
    - Use --missing-only to skip entries that already have good links
    - Use --limit to test with small batches (e.g., --limit 10)
    - Use --start-rank/--end-rank to process specific rank ranges
    - Use --prioritize-by-score to process most important entries first
    - Batch commits every 100 entries (configurable with --batch-size)

    Examples:
        canon charts link nl_top2000 2024
        canon charts link nl_top2000 2024 --missing-only
        canon charts link nl_top2000 2024 --limit 10 --prioritize-by-score
        canon charts link nl_top2000 2024 --start-rank 1 --end-rank 100
    """
    from chart_binder.charts_db import ChartsDB, ChartsETL
    from chart_binder.fetcher import FetcherConfig, FetchMode, UnifiedFetcher

    config: Config = ctx.obj["config"]
    output_format: OutputFormat = ctx.obj["output"]

    db = ChartsDB(config.database.charts_path)

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

    etl = ChartsETL(db, fetcher=fetcher)

    run = db.get_run_by_period(chart_id, period)
    if not run:
        click.echo(f"No chart run found for {chart_id} {period}", err=True)
        sys.exit(ExitCode.NO_RESULTS)

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

    # Close fetcher if we created one
    if fetcher:
        fetcher.close()

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
