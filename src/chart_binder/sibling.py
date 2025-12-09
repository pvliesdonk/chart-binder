"""
Work-based sibling recording expansion.

Implements A.2: When a recording is found, expand to find ALL recordings
linked via the same Work entity. This finds original album versions when
you only have a "Greatest Hits" version.

Uses the backend abstraction from A.1 for both API and DB access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chart_binder.backends.base import BackendRecording, MusicBrainzBackend

log = logging.getLogger(__name__)


@dataclass
class SiblingResult:
    """Result of sibling expansion."""

    seed_recording: BackendRecording
    work_mbid: str | None
    work_name: str | None
    siblings: list[BackendRecording] = field(default_factory=list)
    filtered_count: int = 0  # Count of siblings filtered out (e.g., covers)

    @property
    def total_found(self) -> int:
        """Total recordings found for the work (before filtering)."""
        return len(self.siblings) + self.filtered_count

    @property
    def has_work(self) -> bool:
        """Whether a work link was found."""
        return self.work_mbid is not None


class SiblingExpander:
    """
    Expands recordings to find all siblings via Work entity.

    The key insight from pymusicbrainz: when you find a recording, expand
    via Work to discover all recordings of the same composition. This is
    essential for finding original album versions.

    Usage:
        async with get_backend() as backend:
            expander = SiblingExpander(backend)
            result = await expander.expand(recording_mbid)
            for sibling in result.siblings:
                print(f"Found: {sibling.title} by {sibling.artist_name}")
    """

    def __init__(
        self,
        backend: MusicBrainzBackend,
        *,
        include_seed: bool = True,
        filter_same_artist: bool = True,
        max_siblings: int = 100,
    ):
        """
        Initialize sibling expander.

        Args:
            backend: MusicBrainz backend (API or DB)
            include_seed: Include the seed recording in results
            filter_same_artist: Only return recordings by the same artist
            max_siblings: Maximum siblings to return
        """
        self._backend = backend
        self._include_seed = include_seed
        self._filter_same_artist = filter_same_artist
        self._max_siblings = max_siblings

    async def expand(
        self,
        recording_mbid: str,
        *,
        artist_mbid: str | None = None,
    ) -> SiblingResult:
        """
        Expand a recording to find all sibling recordings via Work.

        Args:
            recording_mbid: Seed recording MBID
            artist_mbid: If provided, filter to recordings by this artist.
                         If None and filter_same_artist=True, uses the seed's artist.

        Returns:
            SiblingResult with work info and sibling recordings
        """
        # Get the seed recording
        seed = await self._backend.get_recording(recording_mbid)
        if not seed:
            log.warning(f"Seed recording not found: {recording_mbid}")
            return SiblingResult(
                seed_recording=BackendRecording(mbid=recording_mbid, title="Unknown"),
                work_mbid=None,
                work_name=None,
            )

        # Use seed's artist if not provided
        if artist_mbid is None and self._filter_same_artist:
            artist_mbid = seed.artist_mbid

        # Get the work linked to this recording
        work = await self._backend.get_work_for_recording(recording_mbid)
        if not work:
            log.debug(f"No work linked to recording {recording_mbid}")
            return SiblingResult(
                seed_recording=seed,
                work_mbid=None,
                work_name=None,
                siblings=[seed] if self._include_seed else [],
            )

        # Get all recordings for this work
        all_recordings = await self._backend.get_recordings_for_work(
            work.mbid,
            artist_mbid=artist_mbid if self._filter_same_artist else None,
            limit=self._max_siblings + 1,  # +1 to account for seed
        )

        # Build sibling list
        siblings: list[BackendRecording] = []
        filtered_count = 0

        for rec in all_recordings:
            # Skip seed recording unless include_seed is True
            if rec.mbid == recording_mbid:
                if self._include_seed:
                    siblings.append(rec)
                continue

            # Apply artist filter if needed
            if self._filter_same_artist and artist_mbid:
                if rec.artist_mbid != artist_mbid:
                    filtered_count += 1
                    continue

            siblings.append(rec)

            # Respect max limit
            if len(siblings) >= self._max_siblings:
                break

        # Ensure seed is first if included
        if self._include_seed:
            # Move seed to front if present
            seed_in_list = next((r for r in siblings if r.mbid == recording_mbid), None)
            if seed_in_list:
                siblings.remove(seed_in_list)
                siblings.insert(0, seed_in_list)
            else:
                # Seed wasn't in results (maybe artist filter), add it
                siblings.insert(0, seed)

        return SiblingResult(
            seed_recording=seed,
            work_mbid=work.mbid,
            work_name=work.name,
            siblings=siblings,
            filtered_count=filtered_count,
        )

    async def expand_multiple(
        self,
        recording_mbids: list[str],
        *,
        deduplicate: bool = True,
    ) -> list[BackendRecording]:
        """
        Expand multiple recordings and combine siblings.

        Useful for expanding a set of search results to find all
        related recordings.

        Args:
            recording_mbids: List of seed recording MBIDs
            deduplicate: Remove duplicate recordings from results

        Returns:
            Combined list of all recordings (seeds + siblings)
        """
        all_recordings: list[BackendRecording] = []
        seen_mbids: set[str] = set()

        for mbid in recording_mbids:
            result = await self.expand(mbid)

            for rec in result.siblings:
                if deduplicate:
                    if rec.mbid in seen_mbids:
                        continue
                    seen_mbids.add(rec.mbid)
                all_recordings.append(rec)

        return all_recordings


async def expand_siblings(
    backend: MusicBrainzBackend,
    recording_mbid: str,
    *,
    artist_mbid: str | None = None,
    include_seed: bool = True,
    filter_same_artist: bool = True,
    max_siblings: int = 100,
) -> SiblingResult:
    """
    Convenience function for one-shot sibling expansion.

    Args:
        backend: MusicBrainz backend
        recording_mbid: Seed recording MBID
        artist_mbid: Filter to this artist's recordings
        include_seed: Include seed in results
        filter_same_artist: Only return same-artist recordings
        max_siblings: Maximum siblings to return

    Returns:
        SiblingResult with work info and siblings
    """
    expander = SiblingExpander(
        backend,
        include_seed=include_seed,
        filter_same_artist=filter_same_artist,
        max_siblings=max_siblings,
    )
    return await expander.expand(recording_mbid, artist_mbid=artist_mbid)


## Tests


def test_sibling_result_properties():
    """Test SiblingResult dataclass properties."""
    result = SiblingResult(
        seed_recording=BackendRecording(mbid="seed-1", title="Test"),
        work_mbid="work-1",
        work_name="Test Work",
        siblings=[
            BackendRecording(mbid="rec-1", title="Recording 1"),
            BackendRecording(mbid="rec-2", title="Recording 2"),
        ],
        filtered_count=3,
    )

    assert result.has_work is True
    assert result.total_found == 5  # 2 siblings + 3 filtered
    assert len(result.siblings) == 2


def test_sibling_result_no_work():
    """Test SiblingResult when no work is linked."""
    result = SiblingResult(
        seed_recording=BackendRecording(mbid="seed-1", title="Test"),
        work_mbid=None,
        work_name=None,
    )

    assert result.has_work is False
    assert result.total_found == 0
    assert len(result.siblings) == 0
