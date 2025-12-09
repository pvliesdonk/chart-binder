"""
Abstract base class for MusicBrainz backend implementations.

This module defines the interface that both API and DB backends must implement,
ensuring consistent behavior regardless of data source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class BackendArtist:
    """Artist entity from backend."""

    mbid: str
    name: str
    sort_name: str | None = None
    country: str | None = None
    disambiguation: str | None = None


@dataclass
class BackendRecording:
    """Recording entity from backend."""

    mbid: str
    title: str
    artist_mbid: str | None = None
    artist_name: str | None = None
    length_ms: int | None = None
    isrcs: list[str] = field(default_factory=list)
    disambiguation: str | None = None


@dataclass
class BackendReleaseGroup:
    """Release group entity from backend."""

    mbid: str
    title: str
    artist_mbid: str | None = None
    artist_name: str | None = None
    primary_type: str | None = None
    secondary_types: list[str] = field(default_factory=list)
    first_release_date: str | None = None
    disambiguation: str | None = None


@dataclass
class BackendWork:
    """Work entity from backend."""

    mbid: str
    name: str
    disambiguation: str | None = None
    iswcs: list[str] = field(default_factory=list)


class MusicBrainzBackend(ABC):
    """
    Abstract base class for MusicBrainz data access.

    Implementations must provide both synchronous and asynchronous
    variants of all methods for flexibility.
    """

    # --- Recording Operations ---

    @abstractmethod
    async def get_recording(self, mbid: str) -> BackendRecording | None:
        """
        Get recording by MBID.

        Args:
            mbid: MusicBrainz recording ID

        Returns:
            BackendRecording or None if not found
        """
        ...

    @abstractmethod
    async def search_recordings(
        self,
        artist: str,
        title: str,
        *,
        strict: bool = True,
        limit: int = 25,
    ) -> list[BackendRecording]:
        """
        Search recordings by artist and title.

        Args:
            artist: Artist name
            title: Recording title
            strict: If True, use exact matching; if False, use fuzzy matching
            limit: Maximum results to return

        Returns:
            List of matching recordings
        """
        ...

    @abstractmethod
    async def search_recordings_by_isrc(self, isrc: str) -> list[BackendRecording]:
        """
        Search recordings by ISRC code.

        Args:
            isrc: ISRC code (e.g., "USRC17607839")

        Returns:
            List of matching recordings
        """
        ...

    # --- Work Operations ---

    @abstractmethod
    async def get_work(self, mbid: str) -> BackendWork | None:
        """
        Get work by MBID.

        Args:
            mbid: MusicBrainz work ID

        Returns:
            BackendWork or None if not found
        """
        ...

    @abstractmethod
    async def get_work_for_recording(self, recording_mbid: str) -> BackendWork | None:
        """
        Get the work linked to a recording.

        Args:
            recording_mbid: MusicBrainz recording ID

        Returns:
            BackendWork or None if no work linked
        """
        ...

    @abstractmethod
    async def get_recordings_for_work(
        self,
        work_mbid: str,
        *,
        artist_mbid: str | None = None,
        limit: int = 500,
    ) -> list[BackendRecording]:
        """
        Get all recordings linked to a work (sibling expansion).

        This is the core method for work-based discovery. It returns ALL
        recordings that are performances of the given work.

        Args:
            work_mbid: MusicBrainz work ID
            artist_mbid: If provided, filter to recordings by this artist only
                        (excludes covers by other artists)
            limit: Maximum recordings to return

        Returns:
            List of recordings linked to the work
        """
        ...

    # --- Release Group Operations ---

    @abstractmethod
    async def get_release_group(self, mbid: str) -> BackendReleaseGroup | None:
        """
        Get release group by MBID.

        Args:
            mbid: MusicBrainz release group ID

        Returns:
            BackendReleaseGroup or None if not found
        """
        ...

    @abstractmethod
    async def get_release_groups_for_recording(
        self,
        recording_mbid: str,
    ) -> list[BackendReleaseGroup]:
        """
        Get all release groups containing a recording.

        Args:
            recording_mbid: MusicBrainz recording ID

        Returns:
            List of release groups containing the recording
        """
        ...

    @abstractmethod
    async def get_release_groups_for_artist(
        self,
        artist_mbid: str,
        *,
        primary_type: str | None = None,
        limit: int = 100,
    ) -> list[BackendReleaseGroup]:
        """
        Get all release groups for an artist.

        Args:
            artist_mbid: MusicBrainz artist ID
            primary_type: Filter by type (Album, Single, EP, etc.)
            limit: Maximum results to return

        Returns:
            List of release groups by the artist
        """
        ...

    # --- Artist Operations ---

    @abstractmethod
    async def get_artist(self, mbid: str) -> BackendArtist | None:
        """
        Get artist by MBID.

        Args:
            mbid: MusicBrainz artist ID

        Returns:
            BackendArtist or None if not found
        """
        ...

    @abstractmethod
    async def search_artists(
        self,
        name: str,
        *,
        limit: int = 10,
    ) -> list[BackendArtist]:
        """
        Search artists by name.

        Args:
            name: Artist name to search
            limit: Maximum results to return

        Returns:
            List of matching artists
        """
        ...

    # --- Lifecycle ---

    @abstractmethod
    async def close(self) -> None:
        """Close backend connections."""
        ...

    async def __aenter__(self) -> MusicBrainzBackend:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


## Tests


def test_backend_recording_dataclass():
    """Test BackendRecording dataclass."""
    rec = BackendRecording(
        mbid="test-mbid",
        title="Test Song",
        artist_mbid="artist-mbid",
        artist_name="Test Artist",
        length_ms=180000,
        isrcs=["USTEST1234567"],
    )
    assert rec.mbid == "test-mbid"
    assert rec.title == "Test Song"
    assert len(rec.isrcs) == 1


def test_backend_release_group_dataclass():
    """Test BackendReleaseGroup dataclass."""
    rg = BackendReleaseGroup(
        mbid="rg-mbid",
        title="Test Album",
        primary_type="Album",
        secondary_types=["Compilation"],
        first_release_date="1970-01-01",
    )
    assert rg.mbid == "rg-mbid"
    assert rg.primary_type == "Album"
    assert "Compilation" in rg.secondary_types


def test_backend_work_dataclass():
    """Test BackendWork dataclass."""
    work = BackendWork(
        mbid="work-mbid",
        name="Test Work",
        iswcs=["T-123.456.789-0"],
    )
    assert work.mbid == "work-mbid"
    assert len(work.iswcs) == 1
