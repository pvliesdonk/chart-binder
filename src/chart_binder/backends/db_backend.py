"""
Database backend implementation using PostgreSQL with mbdata ORM.

This backend provides direct database access to a local MusicBrainz mirror,
offering significantly faster performance for batch operations compared to
the API backend.

Requires:
- PostgreSQL with MusicBrainz database
- mbdata package (pip install mbdata)
- psycopg2 (pip install psycopg2-binary)
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from chart_binder.backends.base import (
    BackendArtist,
    BackendRecording,
    BackendReleaseGroup,
    BackendWork,
    MusicBrainzBackend,
)

log = logging.getLogger(__name__)


class DBBackend(MusicBrainzBackend):
    """
    MusicBrainz backend using direct PostgreSQL access.

    Uses a hybrid approach:
    - mbdata ORM for simple entity lookups
    - Raw SQL for complex queries (sibling expansion, bucketing)

    This backend is much faster than the API for batch operations
    but requires a local MusicBrainz database mirror.
    """

    def __init__(
        self,
        db_url: str,
        *,
        echo: bool = False,
    ):
        """
        Initialize database backend.

        Args:
            db_url: PostgreSQL connection URL
                    e.g., "postgresql://user:pass@host:5432/musicbrainz_db"
            echo: If True, log SQL queries (for debugging)
        """
        self._engine = create_engine(db_url, echo=echo)
        self._Session = sessionmaker(bind=self._engine)
        self._session: Session | None = None

    def _get_session(self) -> Session:
        """Get or create a database session."""
        if self._session is None:
            self._session = self._Session()
        return self._session

    # --- Recording Operations ---

    async def get_recording(self, mbid: str) -> BackendRecording | None:
        """Get recording by MBID using ORM."""
        session = self._get_session()
        try:
            # Use mbdata ORM for simple lookup
            from mbdata.models import Recording

            rec = session.query(Recording).filter(Recording.gid == mbid).first()
            if not rec:
                return None

            # Get artist info from artist_credit
            artist_mbid: str | None = None
            artist_name: str | None = None
            if rec.artist_credit:
                artist_name = rec.artist_credit.name
                # Get first artist from credit
                if rec.artist_credit.artists:
                    first_credit = rec.artist_credit.artists[0]
                    if first_credit.artist:
                        artist_mbid = str(first_credit.artist.gid)

            # Get ISRCs
            isrcs: list[str] = []
            if hasattr(rec, "isrcs") and rec.isrcs:
                isrcs = [isrc.isrc for isrc in rec.isrcs]

            return BackendRecording(
                mbid=str(rec.gid),
                title=rec.name,
                artist_mbid=artist_mbid,
                artist_name=artist_name,
                length_ms=rec.length,
                isrcs=isrcs,
                disambiguation=rec.comment,
            )
        except Exception as e:
            log.warning(f"Failed to get recording {mbid}: {e}")
            return None

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

        Uses PostgreSQL full-text search for better performance.
        """
        session = self._get_session()
        try:
            # Use raw SQL for search (more control over matching)
            if strict:
                # Exact matching (case-insensitive)
                query = text("""
                    SELECT r.gid, r.name, r.length, r.comment,
                           ac.name as artist_name,
                           a.gid as artist_gid
                    FROM musicbrainz.recording r
                    JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
                    LEFT JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    LEFT JOIN musicbrainz.artist a ON acn.artist = a.id
                    WHERE LOWER(r.name) = LOWER(:title)
                      AND LOWER(ac.name) LIKE LOWER(:artist_pattern)
                    LIMIT :limit
                """)
                artist_pattern = f"%{artist}%"
            else:
                # Fuzzy matching using trigram similarity (requires pg_trgm extension)
                query = text("""
                    SELECT r.gid, r.name, r.length, r.comment,
                           ac.name as artist_name,
                           a.gid as artist_gid
                    FROM musicbrainz.recording r
                    JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
                    LEFT JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    LEFT JOIN musicbrainz.artist a ON acn.artist = a.id
                    WHERE LOWER(r.name) LIKE LOWER(:title_pattern)
                      AND LOWER(ac.name) LIKE LOWER(:artist_pattern)
                    LIMIT :limit
                """)
                artist_pattern = f"%{artist}%"

            params: dict[str, Any] = {
                "title": title,
                "title_pattern": f"%{title}%",
                "artist_pattern": artist_pattern,
                "limit": limit,
            }

            result = session.execute(query, params)
            rows = result.fetchall()

            return [
                BackendRecording(
                    mbid=str(row.gid),
                    title=row.name,
                    artist_mbid=str(row.artist_gid) if row.artist_gid else None,
                    artist_name=row.artist_name,
                    length_ms=row.length,
                    disambiguation=row.comment,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Recording search failed for '{artist}' - '{title}': {e}")
            return []

    async def search_recordings_by_isrc(self, isrc: str) -> list[BackendRecording]:
        """Search recordings by ISRC code."""
        session = self._get_session()
        try:
            query = text("""
                SELECT r.gid, r.name, r.length, r.comment,
                       ac.name as artist_name,
                       a.gid as artist_gid
                FROM musicbrainz.isrc i
                JOIN musicbrainz.recording r ON i.recording = r.id
                JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
                LEFT JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                LEFT JOIN musicbrainz.artist a ON acn.artist = a.id
                WHERE i.isrc = :isrc
            """)

            result = session.execute(query, {"isrc": isrc})
            rows = result.fetchall()

            return [
                BackendRecording(
                    mbid=str(row.gid),
                    title=row.name,
                    artist_mbid=str(row.artist_gid) if row.artist_gid else None,
                    artist_name=row.artist_name,
                    length_ms=row.length,
                    isrcs=[isrc],
                    disambiguation=row.comment,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"ISRC search failed for {isrc}: {e}")
            return []

    # --- Work Operations ---

    async def get_work(self, mbid: str) -> BackendWork | None:
        """Get work by MBID."""
        session = self._get_session()
        try:
            from mbdata.models import Work

            work = session.query(Work).filter(Work.gid == mbid).first()
            if not work:
                return None

            # Get ISWCs
            iswcs: list[str] = []
            if hasattr(work, "iswcs") and work.iswcs:
                iswcs = [iswc.iswc for iswc in work.iswcs]

            return BackendWork(
                mbid=str(work.gid),
                name=work.name,
                disambiguation=work.comment,
                iswcs=iswcs,
            )
        except Exception as e:
            log.warning(f"Failed to get work {mbid}: {e}")
            return None

    async def get_work_for_recording(self, recording_mbid: str) -> BackendWork | None:
        """Get the work linked to a recording."""
        session = self._get_session()
        try:
            # Use raw SQL for performance (avoids multiple ORM lookups)
            query = text("""
                SELECT w.gid, w.name, w.comment
                FROM musicbrainz.work w
                JOIN musicbrainz.l_recording_work lrw ON w.id = lrw.entity1
                JOIN musicbrainz.recording r ON r.id = lrw.entity0
                WHERE r.gid = :recording_mbid
                LIMIT 1
            """)

            result = session.execute(query, {"recording_mbid": recording_mbid})
            row = result.fetchone()

            if not row:
                return None

            return BackendWork(
                mbid=str(row.gid),
                name=row.name,
                disambiguation=row.comment,
            )
        except Exception as e:
            log.warning(f"Failed to get work for recording {recording_mbid}: {e}")
            return None

    async def get_recordings_for_work(
        self,
        work_mbid: str,
        *,
        artist_mbid: str | None = None,
        limit: int = 500,
    ) -> list[BackendRecording]:
        """
        Get all recordings linked to a work (sibling expansion).

        This is the KEY QUERY for work-based discovery. Uses the validated
        query pattern from A.0 spike.
        """
        session = self._get_session()
        try:
            if artist_mbid:
                # Filter by artist (excludes covers)
                query = text("""
                    SELECT DISTINCT r.gid, r.name, r.length, r.comment,
                           ac.name as artist_name,
                           a.gid as artist_gid
                    FROM musicbrainz.recording r
                    JOIN musicbrainz.l_recording_work lrw ON r.id = lrw.entity0
                    JOIN musicbrainz.work w ON w.id = lrw.entity1
                    JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
                    JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    JOIN musicbrainz.artist a ON acn.artist = a.id
                    WHERE w.gid = :work_mbid
                      AND a.gid = :artist_mbid
                    LIMIT :limit
                """)
                params: dict[str, Any] = {
                    "work_mbid": work_mbid,
                    "artist_mbid": artist_mbid,
                    "limit": limit,
                }
            else:
                # All recordings for work (may include covers)
                query = text("""
                    SELECT DISTINCT r.gid, r.name, r.length, r.comment,
                           ac.name as artist_name,
                           a.gid as artist_gid
                    FROM musicbrainz.recording r
                    JOIN musicbrainz.l_recording_work lrw ON r.id = lrw.entity0
                    JOIN musicbrainz.work w ON w.id = lrw.entity1
                    JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
                    LEFT JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    LEFT JOIN musicbrainz.artist a ON acn.artist = a.id
                    WHERE w.gid = :work_mbid
                    LIMIT :limit
                """)
                params = {"work_mbid": work_mbid, "limit": limit}

            result = session.execute(query, params)
            rows = result.fetchall()

            return [
                BackendRecording(
                    mbid=str(row.gid),
                    title=row.name,
                    artist_mbid=str(row.artist_gid) if row.artist_gid else None,
                    artist_name=row.artist_name,
                    length_ms=row.length,
                    disambiguation=row.comment,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Failed to get recordings for work {work_mbid}: {e}")
            return []

    # --- Release Group Operations ---

    async def get_release_group(self, mbid: str) -> BackendReleaseGroup | None:
        """Get release group by MBID."""
        session = self._get_session()
        try:
            from mbdata.models import ReleaseGroup

            rg = session.query(ReleaseGroup).filter(ReleaseGroup.gid == mbid).first()
            if not rg:
                return None

            # Get artist info
            artist_mbid: str | None = None
            artist_name: str | None = None
            if rg.artist_credit:
                artist_name = rg.artist_credit.name
                if rg.artist_credit.artists:
                    first_credit = rg.artist_credit.artists[0]
                    if first_credit.artist:
                        artist_mbid = str(first_credit.artist.gid)

            # Get primary type name
            primary_type: str | None = None
            if rg.type:
                primary_type = rg.type.name

            # Get secondary types via join table
            secondary_types: list[str] = []
            # Use raw SQL for secondary types (avoids complex ORM navigation)
            sec_query = text("""
                SELECT rgst.name
                FROM musicbrainz.release_group_secondary_type_join rgsj
                JOIN musicbrainz.release_group_secondary_type rgst
                    ON rgsj.secondary_type = rgst.id
                JOIN musicbrainz.release_group rg ON rg.id = rgsj.release_group
                WHERE rg.gid = :mbid
            """)
            sec_result = session.execute(sec_query, {"mbid": mbid})
            secondary_types = [row.name for row in sec_result.fetchall()]

            return BackendReleaseGroup(
                mbid=str(rg.gid),
                title=rg.name,
                artist_mbid=artist_mbid,
                artist_name=artist_name,
                primary_type=primary_type,
                secondary_types=secondary_types,
                disambiguation=rg.comment,
            )
        except Exception as e:
            log.warning(f"Failed to get release group {mbid}: {e}")
            return None

    async def get_release_groups_for_recording(
        self,
        recording_mbid: str,
    ) -> list[BackendReleaseGroup]:
        """
        Get all release groups containing a recording.

        Uses the validated query pattern from A.0 spike with release_country
        for dates.
        """
        session = self._get_session()
        try:
            query = text("""
                SELECT DISTINCT
                    rg.gid, rg.name, rg.comment,
                    rgpt.name as primary_type,
                    ac.name as artist_name,
                    a.gid as artist_gid,
                    MIN(rc.date_year) as earliest_year
                FROM musicbrainz.recording rec
                JOIN musicbrainz.track t ON t.recording = rec.id
                JOIN musicbrainz.medium m ON t.medium = m.id
                JOIN musicbrainz.release r ON m.release = r.id
                JOIN musicbrainz.release_group rg ON r.release_group = rg.id
                LEFT JOIN musicbrainz.release_group_primary_type rgpt ON rg.type = rgpt.id
                JOIN musicbrainz.artist_credit ac ON rg.artist_credit = ac.id
                LEFT JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                LEFT JOIN musicbrainz.artist a ON acn.artist = a.id
                LEFT JOIN musicbrainz.release_country rc ON r.id = rc.release
                WHERE rec.gid = :recording_mbid
                GROUP BY rg.gid, rg.name, rg.comment, rgpt.name, ac.name, a.gid
                ORDER BY earliest_year NULLS LAST
            """)

            result = session.execute(query, {"recording_mbid": recording_mbid})
            rows = result.fetchall()

            results: list[BackendReleaseGroup] = []
            for row in rows:
                # Convert year to date string
                first_release_date: str | None = None
                if row.earliest_year:
                    first_release_date = str(row.earliest_year)

                results.append(
                    BackendReleaseGroup(
                        mbid=str(row.gid),
                        title=row.name,
                        artist_mbid=str(row.artist_gid) if row.artist_gid else None,
                        artist_name=row.artist_name,
                        primary_type=row.primary_type,
                        first_release_date=first_release_date,
                        disambiguation=row.comment,
                    )
                )

            return results
        except Exception as e:
            log.warning(f"Failed to get release groups for recording {recording_mbid}: {e}")
            return []

    async def get_release_groups_for_artist(
        self,
        artist_mbid: str,
        *,
        primary_type: str | None = None,
        limit: int = 100,
    ) -> list[BackendReleaseGroup]:
        """Get all release groups for an artist."""
        session = self._get_session()
        try:
            if primary_type:
                query = text("""
                    SELECT DISTINCT rg.gid, rg.name, rg.comment,
                           rgpt.name as primary_type,
                           ac.name as artist_name
                    FROM musicbrainz.release_group rg
                    JOIN musicbrainz.artist_credit ac ON rg.artist_credit = ac.id
                    JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    JOIN musicbrainz.artist a ON acn.artist = a.id
                    LEFT JOIN musicbrainz.release_group_primary_type rgpt ON rg.type = rgpt.id
                    WHERE a.gid = :artist_mbid
                      AND LOWER(rgpt.name) = LOWER(:primary_type)
                    LIMIT :limit
                """)
                params: dict[str, Any] = {
                    "artist_mbid": artist_mbid,
                    "primary_type": primary_type,
                    "limit": limit,
                }
            else:
                query = text("""
                    SELECT DISTINCT rg.gid, rg.name, rg.comment,
                           rgpt.name as primary_type,
                           ac.name as artist_name
                    FROM musicbrainz.release_group rg
                    JOIN musicbrainz.artist_credit ac ON rg.artist_credit = ac.id
                    JOIN musicbrainz.artist_credit_name acn ON ac.id = acn.artist_credit
                    JOIN musicbrainz.artist a ON acn.artist = a.id
                    LEFT JOIN musicbrainz.release_group_primary_type rgpt ON rg.type = rgpt.id
                    WHERE a.gid = :artist_mbid
                    LIMIT :limit
                """)
                params = {"artist_mbid": artist_mbid, "limit": limit}

            result = session.execute(query, params)
            rows = result.fetchall()

            return [
                BackendReleaseGroup(
                    mbid=str(row.gid),
                    title=row.name,
                    artist_mbid=artist_mbid,
                    artist_name=row.artist_name,
                    primary_type=row.primary_type,
                    disambiguation=row.comment,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Failed to get release groups for artist {artist_mbid}: {e}")
            return []

    # --- Artist Operations ---

    async def get_artist(self, mbid: str) -> BackendArtist | None:
        """Get artist by MBID."""
        session = self._get_session()
        try:
            from mbdata.models import Artist

            artist = session.query(Artist).filter(Artist.gid == mbid).first()
            if not artist:
                return None

            # Get country from begin_area
            country: str | None = None
            if artist.begin_area:
                # Try to get ISO code
                if hasattr(artist.begin_area, "iso_3166_1_codes"):
                    codes = artist.begin_area.iso_3166_1_codes
                    if codes:
                        country = codes[0].code

            return BackendArtist(
                mbid=str(artist.gid),
                name=artist.name,
                sort_name=artist.sort_name,
                country=country,
                disambiguation=artist.comment,
            )
        except Exception as e:
            log.warning(f"Failed to get artist {mbid}: {e}")
            return None

    async def search_artists(
        self,
        name: str,
        *,
        limit: int = 10,
    ) -> list[BackendArtist]:
        """Search artists by name."""
        session = self._get_session()
        try:
            query = text("""
                SELECT a.gid, a.name, a.sort_name, a.comment
                FROM musicbrainz.artist a
                WHERE LOWER(a.name) LIKE LOWER(:name_pattern)
                   OR LOWER(a.sort_name) LIKE LOWER(:name_pattern)
                LIMIT :limit
            """)

            result = session.execute(query, {"name_pattern": f"%{name}%", "limit": limit})
            rows = result.fetchall()

            return [
                BackendArtist(
                    mbid=str(row.gid),
                    name=row.name,
                    sort_name=row.sort_name,
                    disambiguation=row.comment,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"Artist search failed for '{name}': {e}")
            return []

    # --- Lifecycle ---

    async def close(self) -> None:
        """Close database connections."""
        if self._session:
            self._session.close()
            self._session = None
        self._engine.dispose()


## Tests


def test_db_backend_creation():
    """Test DBBackend can be instantiated with a connection string."""
    # Just test that the class can be instantiated (don't actually connect)
    # Real connection tests require a PostgreSQL instance
    backend = DBBackend("postgresql://test:test@localhost/test")
    assert backend._engine is not None
