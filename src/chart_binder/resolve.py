"""Shared resolution pipeline for chart-binder.

This module provides a unified resolution function that can be used by:
- The `decide` command (for audio files)
- The `link` command (for chart entries)
- The `resolve` command (for direct artist/title lookups)

All paths use the same 7-rule CRG selection algorithm implemented in resolver.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chart_binder.config import Config
    from chart_binder.llm.react_adjudicator import ReActAdjudicator

log = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    """Result of resolving an artist/title pair to canonical IDs."""

    # Decision state
    state: str  # "DECIDED" or "INDETERMINATE"

    # Canonical IDs (if resolved)
    crg_mbid: str | None = None
    rr_mbid: str | None = None
    recording_mbid: str | None = None

    # Rationale
    crg_rationale: str | None = None
    rr_rationale: str | None = None

    # Confidence and trace
    confidence: float = 0.0
    trace: str = ""

    # LLM adjudication info
    llm_adjudicated: bool = False
    llm_confidence: float | None = None
    llm_rationale: str | None = None

    # Evidence bundle for debugging
    evidence_bundle: dict[str, Any] | None = None


def resolve_artist_title(
    artist: str,
    title: str,
    config: Config,
    *,
    fingerprint: str | None = None,
    duration_sec: float | None = None,
    adjudicator: ReActAdjudicator | None = None,
    fetcher: Any | None = None,  # Optional pre-initialized UnifiedFetcher
) -> ResolutionResult:
    """
    Resolve an artist/title pair to canonical release group and representative release.

    This function implements the complete resolution pipeline:
    1. Search multiple sources (MusicBrainz, Discogs, Spotify)
    2. Hydrate local database with fetched data
    3. Build evidence bundle with candidate release groups
    4. Apply 7-rule CRG selection algorithm (Lead Window, Compilation Exclusion, etc.)
    5. Apply LLM adjudication if enabled and result is INDETERMINATE

    Args:
        artist: Artist name (raw or normalized)
        title: Track title (raw or normalized)
        config: Application configuration
        fingerprint: Optional audio fingerprint for AcoustID lookup
        duration_sec: Optional audio duration for fingerprint matching
        adjudicator: Optional pre-initialized LLM adjudicator
        fetcher: Optional pre-initialized UnifiedFetcher (avoids creating new instance per call)

    Returns:
        ResolutionResult with canonical IDs and decision trace
    """
    from chart_binder.candidates import CandidateBuilder
    from chart_binder.fetcher import FetcherConfig, FetchMode, UnifiedFetcher
    from chart_binder.llm import ReviewQueue, ReviewSource
    from chart_binder.musicgraph import MusicGraphDB
    from chart_binder.normalize import Normalizer
    from chart_binder.resolver import (
        ConfigSnapshot,
        Resolver,
    )

    # Initialize components
    resolver_config = ConfigSnapshot(
        lead_window_days=90,
        reissue_long_gap_years=10,
    )
    resolver = Resolver(resolver_config)

    fetcher_config = FetcherConfig(
        cache_dir=config.http_cache.directory,
        db_path=config.database.music_graph_path,
        mode=FetchMode.NORMAL,
        acoustid_api_key=config.live_sources.acoustid_api_key,
        discogs_token=config.live_sources.discogs_token,
        spotify_client_id=config.live_sources.spotify_client_id,
        spotify_client_secret=config.live_sources.spotify_client_secret,
    )

    musicgraph_db = MusicGraphDB(config.database.music_graph_path)
    normalizer = Normalizer()
    candidate_builder = CandidateBuilder(musicgraph_db, normalizer)

    # Initialize LLM adjudicator if enabled and not provided
    auto_accept_threshold = 0.85
    if adjudicator is None and config.llm.enabled:
        from chart_binder.llm.react_adjudicator import ReActAdjudicator

        web_search = None
        if config.llm.searxng.enabled:
            from chart_binder.llm.searxng import SearxNGSearchTool

            web_search = SearxNGSearchTool(
                base_url=config.llm.searxng.url,
                timeout=config.llm.searxng.timeout_s,
            )
            if not web_search.is_available():
                log.warning(f"SearxNG configured but unavailable at {config.llm.searxng.url}")
                web_search = None

        adjudicator = ReActAdjudicator(config=config.llm, search_tool=web_search)
        auto_accept_threshold = config.llm.auto_accept_threshold

    # Use provided fetcher or create a new one
    if fetcher is not None:
        # Use existing fetcher (more efficient for batch processing)
        return _resolve_with_fetcher(
            fetcher,
            artist,
            title,
            musicgraph_db,
            candidate_builder,
            resolver,
            adjudicator,
            auto_accept_threshold,
            fingerprint,
            duration_sec,
        )
    else:
        # Create new fetcher for one-off resolution
        with UnifiedFetcher(fetcher_config) as new_fetcher:
            return _resolve_with_fetcher(
                new_fetcher,
                artist,
                title,
                musicgraph_db,
                candidate_builder,
                resolver,
                adjudicator,
                auto_accept_threshold,
                fingerprint,
                duration_sec,
            )


def _resolve_with_fetcher(
    fetcher: Any,
    artist: str,
    title: str,
    musicgraph_db: Any,
    candidate_builder: Any,
    resolver: Any,
    adjudicator: Any | None,
    auto_accept_threshold: float,
    fingerprint: str | None,
    duration_sec: float | None,
) -> ResolutionResult:
    """
    Internal helper to resolve artist/title using a provided fetcher.

    This is extracted to avoid code duplication between the context manager
    and direct fetcher usage paths.
    """
    from chart_binder.candidates import CandidateSet
    from chart_binder.resolver import CRGRationale, DecisionState, RRRationale

    # Search by fingerprint if available
    # Check local DB for existing candidates first (avoids redundant API calls)
    candidates = candidate_builder.discover_by_title_artist_length(title, artist, None)

    # Count unique release groups to determine if we have adequate coverage
    unique_rgs = len(set(c.release_group_mbid for c in candidates if c.release_group_mbid))

    # Only search if we don't have adequate candidates (threshold: 3 release groups)
    # Popular tracks with canonical recordings are likely already in DB
    MIN_RELEASE_GROUPS = 3
    should_search = unique_rgs < MIN_RELEASE_GROUPS

    if should_search:
        log.info(
            f"Only {unique_rgs} release group(s) found in DB for {artist} - {title}, "
            f"searching external sources..."
        )

        if fingerprint and duration_sec:
            log.debug(f"Searching by fingerprint ({duration_sec}s)")
            search_results = fetcher.search_recordings(
                fingerprint=fingerprint,
                duration_sec=int(duration_sec),
            )
            for result in search_results:
                if result.get("recording_mbid"):
                    try:
                        fetcher.fetch_recording(result["recording_mbid"])
                    except Exception as e:
                        log.debug(f"Failed to fetch fingerprint result: {e}")

        # Search by title + artist
        log.debug(f"Searching for: {artist} - {title}")
        search_results = fetcher.search_recordings(title=title, artist=artist)
        log.debug(f"Title/artist search returned {len(search_results)} results")

        # Hydrate recordings from each source type
        # Use larger limits to find original recordings (can be deep in search results)
        mb_results = [r for r in search_results if r.get("recording_mbid")]
        discogs_results = [r for r in search_results if r.get("discogs_release_id")]
        spotify_results = [r for r in search_results if r.get("spotify_track_id")]

        # Hydrate top 50 MB recordings to improve chances of finding originals
        for result in mb_results[:50]:
            mbid = result["recording_mbid"]
            try:
                fetcher.fetch_recording(mbid)
            except Exception as e:
                log.debug(f"Failed to fetch MB recording {mbid}: {e}")

        for result in discogs_results[:5]:
            discogs_id = result["discogs_release_id"]
            try:
                fetcher.fetch_discogs_release(discogs_id)
            except Exception as e:
                log.debug(f"Failed to fetch Discogs release {discogs_id}: {e}")

        for result in spotify_results[:5]:
            spotify_id = result["spotify_track_id"]
            try:
                fetcher.fetch_spotify_track(spotify_id)
            except Exception as e:
                log.debug(f"Failed to fetch Spotify track {spotify_id}: {e}")

        # Re-discover candidates after hydration
        candidates = candidate_builder.discover_by_title_artist_length(title, artist, None)
    else:
        log.info(
            f"Using {unique_rgs} release group(s) from database for {artist} - {title} "
            f"(skipping search)"
        )

    if not candidates:
        log.warning(f"No candidates found for: {artist} - {title}")
        return ResolutionResult(
            state="indeterminate",  # Must be lowercase to match enum value
            trace="No candidates found in local database",
        )

    # Build evidence bundle
    candidate_set = CandidateSet(
        file_path=None,
        candidates=candidates,
        normalized_title=title,
        normalized_artist=artist,
        length_ms=int(duration_sec * 1000) if duration_sec else None,
    )
    evidence_bundle_obj = candidate_builder.build_evidence_bundle(candidate_set)

    # Convert to dict format
    evidence_bundle = _convert_evidence_bundle(evidence_bundle_obj, artist, title, musicgraph_db)

    # Resolve
    decision = resolver.resolve(evidence_bundle)

    # Log decision state for debugging
    log.info(
        f"Resolver decision for {artist} - {title}: state={decision.state.value}, "
        f"adjudicator={'present' if adjudicator else 'None'}"
    )

    # Build initial result
    result = ResolutionResult(
        state=decision.state.value,
        crg_mbid=decision.release_group_mbid,
        rr_mbid=decision.release_mbid,
        crg_rationale=str(decision.crg_rationale) if decision.crg_rationale else None,
        rr_rationale=str(decision.rr_rationale) if decision.rr_rationale else None,
        trace=decision.decision_trace.to_human_readable() if decision.decision_trace else "",
        evidence_bundle=evidence_bundle,
    )

    # Get recording MBID from evidence if we have a CRG
    if decision.release_group_mbid:
        for rec in evidence_bundle.get("recording_candidates", []):
            for rg in rec.get("rg_candidates", []):
                if rg.get("mb_rg_id") == decision.release_group_mbid:
                    result.recording_mbid = rec.get("mb_recording_mbid")
                    break
            if result.recording_mbid:
                break

    # LLM adjudication for INDETERMINATE decisions
    if decision.state == DecisionState.INDETERMINATE:
        if adjudicator:
            log.info(f"Decision INDETERMINATE for {artist} - {title}, invoking LLM adjudication...")
        else:
            log.warning(
                f"Decision INDETERMINATE for {artist} - {title}, but no adjudicator available"
            )

    if decision.state == DecisionState.INDETERMINATE and adjudicator:
        try:
            import sys

            from chart_binder.llm.adjudicator import AdjudicationOutcome

            review_queue: ReviewQueue | None = None
            if config.llm.review_queue_path:
                review_queue = ReviewQueue(config.llm.review_queue_path)

            work_key = f"{artist} // {title}"
            file_id = hashlib.sha256(work_key.encode()).hexdigest()

            # Show progress message (clear line and show LLM submission status)
            print(f"\r{'':<120}", end="", file=sys.stderr)
            print(f"  ⏳ Submitting to LLM adjudicator for {artist} - {title}...", file=sys.stderr)

            decision_trace_dict: dict[str, Any] | None = None
            if decision.compact_tag:
                decision_trace_dict = {"trace_compact": decision.compact_tag}

            adjudication_result = adjudicator.adjudicate(evidence_bundle, decision_trace_dict)

            result.llm_confidence = adjudication_result.confidence
            result.llm_rationale = adjudication_result.rationale

            if (
                adjudication_result.outcome != AdjudicationOutcome.ERROR
                and adjudication_result.confidence >= auto_accept_threshold
            ):
                result.state = "decided"
                result.crg_mbid = adjudication_result.crg_mbid
                result.rr_mbid = adjudication_result.rr_mbid
                result.crg_rationale = str(CRGRationale.LLM_ADJUDICATION)
                result.rr_rationale = str(RRRationale.LLM_ADJUDICATION)
                result.llm_adjudicated = True
                result.confidence = adjudication_result.confidence

                log.info(
                    f"✓ LLM auto-accepted: CRG={adjudication_result.crg_mbid}, "
                    f"confidence={adjudication_result.confidence:.2f}"
                )
            elif adjudication_result.outcome == AdjudicationOutcome.REVIEW and review_queue:
                suggestion = {
                    "crg_mbid": adjudication_result.crg_mbid,
                    "rr_mbid": adjudication_result.rr_mbid,
                    "confidence": adjudication_result.confidence,
                    "rationale": adjudication_result.rationale,
                    "model_id": adjudication_result.model_id,
                    "adjudication_id": adjudication_result.adjudication_id,
                }
                review_queue.add_item(
                    file_id=file_id,
                    work_key=work_key,
                    source=ReviewSource.LLM_REVIEW,
                    evidence_bundle=evidence_bundle,
                    decision_trace=decision_trace_dict,
                    llm_suggestion=suggestion,
                )
                log.info(
                    f"⚠ LLM suggestion queued for review (confidence={adjudication_result.confidence:.2f})"
                )
            else:
                log.info(
                    f"⚠ LLM adjudication below threshold: "
                    f"confidence={adjudication_result.confidence:.2f} < {auto_accept_threshold}"
                )
        except Exception as e:
            log.warning(f"✗ LLM adjudication failed: {e}")

    return result


def _convert_evidence_bundle(
    bundle: Any,
    artist: str,
    title: str,
    musicgraph_db: Any,
) -> dict[str, Any]:
    """
    Convert EvidenceBundle dataclass to dict format expected by resolver.

    This is a simplified version of cli._convert_evidence_bundle that doesn't
    require an audio file or tagset.
    """
    recording_candidates = []

    # Group release groups by recording
    rg_by_recording: dict[str, list[dict[str, Any]]] = {}
    for rg in bundle.release_groups:
        rg_mbid = rg.get("mbid")

        # Fetch releases for this release group
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

        # Associate RG with all recordings
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
        "artist": {
            "name": bundle.artist.get("name") if bundle.artist else artist,
            "mb_artist_id": bundle.artist.get("mbid") if bundle.artist else None,
            "begin_area_country": bundle.artist.get("begin_area_country")
            if bundle.artist
            else None,
        },
        "recording_title": title,
        "recording_candidates": recording_candidates,
        "timeline_facts": bundle.timeline_facts or {},
        "provenance": bundle.provenance or {"sources_used": ["MB"]},
    }
