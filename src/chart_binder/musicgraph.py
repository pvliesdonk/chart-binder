from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MusicGraphDB:
    """
    SQLite database for music graph entities (artists, recordings, releases, etc.).

    Provides schema creation and upsert helpers for MusicBrainz-style entities.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        logger.info(f"Initializing MusicGraphDB at {db_path}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.debug("MusicGraphDB initialization complete")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema with proper foreign keys and indices."""
        conn = self._get_connection()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS artist (
                mbid TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                sort_name TEXT,
                begin_area_country TEXT,
                wikidata_qid TEXT,
                diacritics_signature TEXT,
                disambiguation TEXT,
                fetched_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_artist_name ON artist(name);

            CREATE TABLE IF NOT EXISTS recording (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist_mbid TEXT,
                length_ms INTEGER,
                isrcs_json TEXT,
                flags_json TEXT,
                disambiguation TEXT,
                fetched_at REAL NOT NULL,
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_recording_title ON recording(title);
            CREATE INDEX IF NOT EXISTS idx_recording_artist ON recording(artist_mbid);

            CREATE TABLE IF NOT EXISTS release_group (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist_mbid TEXT,
                type TEXT,
                secondary_types_json TEXT,
                first_release_date TEXT,
                labels_json TEXT,
                countries_json TEXT,
                discogs_master_id TEXT,
                spotify_album_id TEXT,
                disambiguation TEXT,
                fetched_at REAL NOT NULL,
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_release_group_title ON release_group(title);
            CREATE INDEX IF NOT EXISTS idx_release_group_artist ON release_group(artist_mbid);
            CREATE INDEX IF NOT EXISTS idx_release_group_firstdate ON release_group(first_release_date);

            CREATE TABLE IF NOT EXISTS release (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                release_group_mbid TEXT,
                artist_mbid TEXT,
                date TEXT,
                country TEXT,
                label TEXT,
                format TEXT,
                catno TEXT,
                barcode TEXT,
                flags_json TEXT,
                discogs_release_id TEXT,
                disambiguation TEXT,
                fetched_at REAL NOT NULL,
                FOREIGN KEY (release_group_mbid) REFERENCES release_group(mbid),
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_release_title ON release(title);
            CREATE INDEX IF NOT EXISTS idx_release_group ON release(release_group_mbid);
            CREATE INDEX IF NOT EXISTS idx_release_date_country ON release(date, country);

            CREATE TABLE IF NOT EXISTS recording_release (
                recording_mbid TEXT NOT NULL,
                release_mbid TEXT NOT NULL,
                track_position INTEGER,
                disc_number INTEGER,
                track_number TEXT,
                fetched_at REAL NOT NULL,
                PRIMARY KEY (recording_mbid, release_mbid),
                FOREIGN KEY (recording_mbid) REFERENCES recording(mbid),
                FOREIGN KEY (release_mbid) REFERENCES release(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_recording_release_recording ON recording_release(recording_mbid);
            CREATE INDEX IF NOT EXISTS idx_recording_release_release ON recording_release(release_mbid);
            """
        )

        conn.commit()
        conn.close()

    def upsert_artist(
        self,
        mbid: str,
        name: str,
        sort_name: str | None = None,
        begin_area_country: str | None = None,
        wikidata_qid: str | None = None,
        diacritics_signature: str | None = None,
        disambiguation: str | None = None,
        fetched_at: float | None = None,
    ) -> None:
        """Upsert artist record."""
        if fetched_at is None:
            fetched_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO artist (mbid, name, sort_name, begin_area_country, wikidata_qid, diacritics_signature, disambiguation, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    name = excluded.name,
                    sort_name = excluded.sort_name,
                    begin_area_country = excluded.begin_area_country,
                    wikidata_qid = excluded.wikidata_qid,
                    diacritics_signature = excluded.diacritics_signature,
                    disambiguation = excluded.disambiguation,
                    fetched_at = excluded.fetched_at
                """,
                (
                    mbid,
                    name,
                    sort_name,
                    begin_area_country,
                    wikidata_qid,
                    diacritics_signature,
                    disambiguation,
                    fetched_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_recording(
        self,
        mbid: str,
        title: str,
        artist_mbid: str | None = None,
        length_ms: int | None = None,
        isrcs_json: str | None = None,
        flags_json: str | None = None,
        disambiguation: str | None = None,
        fetched_at: float | None = None,
    ) -> None:
        """Upsert recording record."""
        if fetched_at is None:
            fetched_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO recording (mbid, title, artist_mbid, length_ms, isrcs_json, flags_json, disambiguation, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    artist_mbid = excluded.artist_mbid,
                    length_ms = excluded.length_ms,
                    isrcs_json = excluded.isrcs_json,
                    flags_json = excluded.flags_json,
                    disambiguation = excluded.disambiguation,
                    fetched_at = excluded.fetched_at
                """,
                (
                    mbid,
                    title,
                    artist_mbid,
                    length_ms,
                    isrcs_json,
                    flags_json,
                    disambiguation,
                    fetched_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_release_group(
        self,
        mbid: str,
        title: str,
        artist_mbid: str | None = None,
        type: str | None = None,  # noqa: A002
        secondary_types_json: str | None = None,
        first_release_date: str | None = None,
        labels_json: str | None = None,
        countries_json: str | None = None,
        discogs_master_id: str | None = None,
        spotify_album_id: str | None = None,
        disambiguation: str | None = None,
        fetched_at: float | None = None,
    ) -> None:
        """Upsert release_group record."""
        if fetched_at is None:
            fetched_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO release_group (mbid, title, artist_mbid, type, secondary_types_json, first_release_date, labels_json, countries_json, discogs_master_id, spotify_album_id, disambiguation, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    artist_mbid = excluded.artist_mbid,
                    type = excluded.type,
                    secondary_types_json = excluded.secondary_types_json,
                    first_release_date = excluded.first_release_date,
                    labels_json = excluded.labels_json,
                    countries_json = excluded.countries_json,
                    discogs_master_id = excluded.discogs_master_id,
                    spotify_album_id = excluded.spotify_album_id,
                    disambiguation = excluded.disambiguation,
                    fetched_at = excluded.fetched_at
                """,
                (
                    mbid,
                    title,
                    artist_mbid,
                    type,
                    secondary_types_json,
                    first_release_date,
                    labels_json,
                    countries_json,
                    discogs_master_id,
                    spotify_album_id,
                    disambiguation,
                    fetched_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_release(
        self,
        mbid: str,
        title: str,
        release_group_mbid: str | None = None,
        artist_mbid: str | None = None,
        date: str | None = None,
        country: str | None = None,
        label: str | None = None,
        format: str | None = None,  # noqa: A002
        catno: str | None = None,
        barcode: str | None = None,
        flags_json: str | None = None,
        discogs_release_id: str | None = None,
        disambiguation: str | None = None,
        fetched_at: float | None = None,
    ) -> None:
        """Upsert release record."""
        if fetched_at is None:
            fetched_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO release (mbid, title, release_group_mbid, artist_mbid, date, country, label, format, catno, barcode, flags_json, discogs_release_id, disambiguation, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    release_group_mbid = excluded.release_group_mbid,
                    artist_mbid = excluded.artist_mbid,
                    date = excluded.date,
                    country = excluded.country,
                    label = excluded.label,
                    format = excluded.format,
                    catno = excluded.catno,
                    barcode = excluded.barcode,
                    flags_json = excluded.flags_json,
                    discogs_release_id = excluded.discogs_release_id,
                    disambiguation = excluded.disambiguation,
                    fetched_at = excluded.fetched_at
                """,
                (
                    mbid,
                    title,
                    release_group_mbid,
                    artist_mbid,
                    date,
                    country,
                    label,
                    format,
                    catno,
                    barcode,
                    flags_json,
                    discogs_release_id,
                    disambiguation,
                    fetched_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_recording_release(
        self,
        recording_mbid: str,
        release_mbid: str,
        track_position: int | None = None,
        disc_number: int | None = None,
        track_number: str | None = None,
        fetched_at: float | None = None,
    ) -> None:
        """Upsert recording_release relationship."""
        if fetched_at is None:
            fetched_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO recording_release (recording_mbid, release_mbid, track_position, disc_number, track_number, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(recording_mbid, release_mbid) DO UPDATE SET
                    track_position = excluded.track_position,
                    disc_number = excluded.disc_number,
                    track_number = excluded.track_number,
                    fetched_at = excluded.fetched_at
                """,
                (
                    recording_mbid,
                    release_mbid,
                    track_position,
                    disc_number,
                    track_number,
                    fetched_at,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_artist(self, mbid: str) -> dict[str, Any] | None:
        """Get artist by MBID."""
        logger.debug(f"Querying artist: mbid={mbid}")
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM artist WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            result = dict(row) if row else None
            logger.debug(f"Artist query result: {'found' if result else 'not found'}")
            return result
        finally:
            conn.close()

    def get_recording(self, mbid: str) -> dict[str, Any] | None:
        """Get recording by MBID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recording WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def search_recordings_by_isrc(self, isrc: str) -> list[dict[str, Any]]:
        """
        Search for recordings containing a specific ISRC.

        Args:
            isrc: The ISRC code to search for

        Returns:
            List of recording dicts with all fields
        """
        logger.debug(f"Searching recordings by ISRC: {isrc}")
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Use json_each to search within the JSON array
            cursor.execute(
                """
                SELECT DISTINCT r.*
                FROM recording r, json_each(r.isrcs_json) j
                WHERE j.value = ?
                """,
                (isrc,),
            )
            results = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"ISRC search returned {len(results)} recordings")
            return results
        finally:
            conn.close()

    def get_release_group(self, mbid: str) -> dict[str, Any] | None:
        """Get release group by MBID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM release_group WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_release(self, mbid: str) -> dict[str, Any] | None:
        """Get release by MBID."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM release WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_release_groups_for_recording(self, recording_mbid: str) -> list[dict[str, Any]]:
        """
        Get all release groups containing a specific recording.

        Args:
            recording_mbid: The recording MBID to search for

        Returns:
            List of release_group dicts with artist_name added
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT rg.*, a.name as artist_name
                FROM recording_release rr
                JOIN release r ON rr.release_mbid = r.mbid
                JOIN release_group rg ON r.release_group_mbid = rg.mbid
                LEFT JOIN artist a ON rg.artist_mbid = a.mbid
                WHERE rr.recording_mbid = ?
                """,
                (recording_mbid,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_recordings_fuzzy(
        self,
        title: str,
        artist_name: str | None = None,
        length_min: int | None = None,
        length_max: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fuzzy search for recordings by title with optional filters.

        Args:
            title: Title to search for (case-insensitive LIKE)
            artist_name: Optional artist name filter (case-insensitive LIKE)
            length_min: Optional minimum length in milliseconds
            length_max: Optional maximum length in milliseconds
            limit: Maximum number of results to return

        Returns:
            List of recording dicts with artist_name from joined artist
        """
        logger.debug(
            f"Fuzzy search: title={title}, artist={artist_name}, "
            f"length={length_min}-{length_max}, limit={limit}"
        )
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query dynamically based on filters
            query = """
                SELECT r.*, a.name as artist_name
                FROM recording r
                LEFT JOIN artist a ON r.artist_mbid = a.mbid
                WHERE r.title LIKE ?
            """
            params: list[Any] = [f"%{title}%"]

            if artist_name is not None:
                query += " AND a.name LIKE ?"
                params.append(f"%{artist_name}%")

            if length_min is not None and length_max is not None:
                query += " AND r.length_ms BETWEEN ? AND ?"
                params.extend([length_min, length_max])
            elif length_min is not None:
                query += " AND r.length_ms >= ?"
                params.append(length_min)
            elif length_max is not None:
                query += " AND r.length_ms <= ?"
                params.append(length_max)

            query += " LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            logger.debug(f"Fuzzy search returned {len(results)} recordings")
            return results
        finally:
            conn.close()

    def search_artists(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for artists by name.

        Args:
            query: Search query (case-insensitive LIKE)
            limit: Maximum number of results to return

        Returns:
            List of artist dicts
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM artist WHERE name LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def search_release_groups(
        self, query: str, artist: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Search for release groups by title with optional artist filter.

        Args:
            query: Search query for release group title (case-insensitive LIKE)
            artist: Optional artist name filter (case-insensitive LIKE)
            limit: Maximum number of results to return

        Returns:
            List of release_group dicts
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if artist is not None:
                cursor.execute(
                    """
                    SELECT rg.*
                    FROM release_group rg
                    LEFT JOIN artist a ON rg.artist_mbid = a.mbid
                    WHERE rg.title LIKE ? AND a.name LIKE ?
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{artist}%", limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM release_group WHERE title LIKE ? LIMIT ?",
                    (f"%{query}%", limit),
                )

            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_releases_in_group(self, rg_mbid: str) -> list[dict[str, Any]]:
        """
        Get all releases in a release group.

        Args:
            rg_mbid: The release group MBID

        Returns:
            List of release dicts
        """
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM release WHERE release_group_mbid = ?",
                (rg_mbid,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def verify_foreign_keys(self) -> bool:
        """Verify that foreign key constraints are enabled."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            return result is not None and result[0] == 1
        finally:
            conn.close()


## Tests


def test_music_graph_db_init(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")
    assert db.db_path.exists()
    assert db.verify_foreign_keys()

    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indices = {row[0] for row in cursor.fetchall()}
    conn.close()

    expected_indices = {
        "idx_artist_name",
        "idx_recording_title",
        "idx_recording_artist",
        "idx_release_group_title",
        "idx_release_group_artist",
        "idx_release_group_firstdate",
        "idx_release_title",
        "idx_release_group",
        "idx_release_date_country",
        "idx_recording_release_recording",
        "idx_recording_release_release",
    }
    assert expected_indices.issubset(indices)


def test_artist_crud(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles", "Beatles, The")

    artist = db.get_artist("artist-1")
    assert artist is not None
    assert artist["name"] == "The Beatles"
    assert artist["sort_name"] == "Beatles, The"

    db.upsert_artist("artist-1", "Beatles", "Beatles")
    artist = db.get_artist("artist-1")
    assert artist is not None
    assert artist["name"] == "Beatles"


def test_recording_crud(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_recording("recording-1", "Yesterday", artist_mbid="artist-1", length_ms=123000)

    recording = db.get_recording("recording-1")
    assert recording is not None
    assert recording["title"] == "Yesterday"
    assert recording["artist_mbid"] == "artist-1"
    assert recording["length_ms"] == 123000


def test_release_group_crud(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_release_group(
        "rg-1",
        "Abbey Road",
        artist_mbid="artist-1",
        type="Album",
        first_release_date="1969-09-26",
    )

    conn = db._get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM release_group WHERE mbid = ?", ("rg-1",))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["title"] == "Abbey Road"
    assert row["type"] == "Album"


def test_recording_release_relationship(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_recording("recording-1", "Yesterday", artist_mbid="artist-1")
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1")
    db.upsert_release("release-1", "Help!", release_group_mbid="rg-1")
    db.upsert_recording_release("recording-1", "release-1", track_position=1, track_number="A1")

    conn = db._get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM recording_release WHERE recording_mbid = ? AND release_mbid = ?",
        ("recording-1", "release-1"),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["track_position"] == 1
    assert row["track_number"] == "A1"


def test_search_recordings_by_isrc(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_recording(
        "recording-1",
        "Yesterday",
        artist_mbid="artist-1",
        isrcs_json='["USCA12345678", "GBUM71234567"]',
    )
    db.upsert_recording(
        "recording-2",
        "Hey Jude",
        artist_mbid="artist-1",
        isrcs_json='["USCA98765432"]',
    )
    db.upsert_recording(
        "recording-3",
        "Let It Be",
        artist_mbid="artist-1",
        isrcs_json='["USCA12345678", "GBUM79876543"]',
    )

    # Search for ISRC that appears in two recordings
    results = db.search_recordings_by_isrc("USCA12345678")
    assert len(results) == 2
    titles = {r["title"] for r in results}
    assert titles == {"Yesterday", "Let It Be"}

    # Search for ISRC that appears in one recording
    results = db.search_recordings_by_isrc("USCA98765432")
    assert len(results) == 1
    assert results[0]["title"] == "Hey Jude"

    # Search for non-existent ISRC
    results = db.search_recordings_by_isrc("NONEXISTENT")
    assert len(results) == 0


def test_get_release_group(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_release_group(
        "rg-1",
        "Abbey Road",
        artist_mbid="artist-1",
        type="Album",
        first_release_date="1969-09-26",
    )

    rg = db.get_release_group("rg-1")
    assert rg is not None
    assert rg["title"] == "Abbey Road"
    assert rg["type"] == "Album"
    assert rg["artist_mbid"] == "artist-1"
    assert rg["first_release_date"] == "1969-09-26"

    # Test non-existent release group
    rg = db.get_release_group("nonexistent")
    assert rg is None


def test_get_release(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_release_group("rg-1", "Abbey Road", artist_mbid="artist-1")
    db.upsert_release(
        "release-1",
        "Abbey Road",
        release_group_mbid="rg-1",
        artist_mbid="artist-1",
        date="1969-09-26",
        country="GB",
        label="Apple Records",
    )

    release = db.get_release("release-1")
    assert release is not None
    assert release["title"] == "Abbey Road"
    assert release["release_group_mbid"] == "rg-1"
    assert release["country"] == "GB"
    assert release["label"] == "Apple Records"

    # Test non-existent release
    release = db.get_release("nonexistent")
    assert release is None


def test_get_release_groups_for_recording(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    # Set up artist
    db.upsert_artist("artist-1", "The Beatles")

    # Set up recording
    db.upsert_recording("recording-1", "Yesterday", artist_mbid="artist-1")

    # Set up multiple release groups
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-2", "Yesterday and Today", artist_mbid="artist-1", type="Album")

    # Set up releases
    db.upsert_release("release-1", "Help!", release_group_mbid="rg-1", artist_mbid="artist-1")
    db.upsert_release(
        "release-2", "Yesterday and Today", release_group_mbid="rg-2", artist_mbid="artist-1"
    )

    # Link recording to both releases
    db.upsert_recording_release("recording-1", "release-1")
    db.upsert_recording_release("recording-1", "release-2")

    # Get release groups for recording
    rgs = db.get_release_groups_for_recording("recording-1")
    assert len(rgs) == 2

    titles = {rg["title"] for rg in rgs}
    assert titles == {"Help!", "Yesterday and Today"}

    # Verify artist_name is included
    for rg in rgs:
        assert rg["artist_name"] == "The Beatles"

    # Test recording not in any release
    rgs = db.get_release_groups_for_recording("nonexistent")
    assert len(rgs) == 0


def test_search_recordings_fuzzy(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_artist("artist-2", "The Rolling Stones")

    db.upsert_recording("rec-1", "Yesterday", artist_mbid="artist-1", length_ms=125000)
    db.upsert_recording("rec-2", "Hey Jude", artist_mbid="artist-1", length_ms=431000)
    db.upsert_recording("rec-3", "Let It Be", artist_mbid="artist-1", length_ms=243000)
    db.upsert_recording("rec-4", "Paint It Black", artist_mbid="artist-2", length_ms=222000)

    # Search by title only
    results = db.search_recordings_fuzzy("hey")
    assert len(results) == 1
    assert results[0]["title"] == "Hey Jude"
    assert results[0]["artist_name"] == "The Beatles"

    # Case-insensitive search
    results = db.search_recordings_fuzzy("YESTERDAY")
    assert len(results) == 1
    assert results[0]["title"] == "Yesterday"

    # Search with artist filter
    results = db.search_recordings_fuzzy("it", artist_name="Beatles")
    assert len(results) == 1
    assert results[0]["title"] == "Let It Be"

    results = db.search_recordings_fuzzy("it", artist_name="Stones")
    assert len(results) == 1
    assert results[0]["title"] == "Paint It Black"

    # Search with length range
    results = db.search_recordings_fuzzy("", length_min=200000, length_max=250000)
    assert len(results) == 2
    titles = {r["title"] for r in results}
    assert titles == {"Let It Be", "Paint It Black"}

    # Search with length min only
    results = db.search_recordings_fuzzy("", length_min=400000)
    assert len(results) == 1
    assert results[0]["title"] == "Hey Jude"

    # Search with length max only
    results = db.search_recordings_fuzzy("", length_max=130000)
    assert len(results) == 1
    assert results[0]["title"] == "Yesterday"

    # Test limit
    results = db.search_recordings_fuzzy("", limit=2)
    assert len(results) == 2


def test_search_artists(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_artist("artist-2", "The Rolling Stones")
    db.upsert_artist("artist-3", "Beatles Revival Band")
    db.upsert_artist("artist-4", "Led Zeppelin")

    # Search for "Beatles"
    results = db.search_artists("Beatles")
    assert len(results) == 2
    names = {a["name"] for a in results}
    assert names == {"The Beatles", "Beatles Revival Band"}

    # Case-insensitive search
    results = db.search_artists("beatles")
    assert len(results) == 2

    # Partial match
    results = db.search_artists("Rolling")
    assert len(results) == 1
    assert results[0]["name"] == "The Rolling Stones"

    # Test limit
    results = db.search_artists("", limit=2)
    assert len(results) == 2

    # No matches
    results = db.search_artists("Nonexistent Artist")
    assert len(results) == 0


def test_search_release_groups(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_artist("artist-2", "The Rolling Stones")

    db.upsert_release_group("rg-1", "Abbey Road", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-2", "Let It Be", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-3", "Let It Bleed", artist_mbid="artist-2", type="Album")
    db.upsert_release_group("rg-4", "Help!", artist_mbid="artist-1", type="Album")

    # Search by title
    results = db.search_release_groups("Let It")
    assert len(results) == 2
    titles = {rg["title"] for rg in results}
    assert titles == {"Let It Be", "Let It Bleed"}

    # Case-insensitive search
    results = db.search_release_groups("ABBEY")
    assert len(results) == 1
    assert results[0]["title"] == "Abbey Road"

    # Search with artist filter
    results = db.search_release_groups("Let It", artist="Beatles")
    assert len(results) == 1
    assert results[0]["title"] == "Let It Be"

    results = db.search_release_groups("Let It", artist="Stones")
    assert len(results) == 1
    assert results[0]["title"] == "Let It Bleed"

    # Test limit
    results = db.search_release_groups("", limit=2)
    assert len(results) == 2

    # No matches
    results = db.search_release_groups("Nonexistent Album")
    assert len(results) == 0


def test_get_releases_in_group(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "The Beatles")
    db.upsert_release_group("rg-1", "Abbey Road", artist_mbid="artist-1")

    # Add multiple releases in the same release group
    db.upsert_release(
        "release-1",
        "Abbey Road",
        release_group_mbid="rg-1",
        country="GB",
        date="1969-09-26",
    )
    db.upsert_release(
        "release-2",
        "Abbey Road",
        release_group_mbid="rg-1",
        country="US",
        date="1969-10-01",
    )
    db.upsert_release(
        "release-3",
        "Abbey Road",
        release_group_mbid="rg-1",
        country="JP",
        date="1969-11-01",
    )

    # Get all releases in the group
    releases = db.get_releases_in_group("rg-1")
    assert len(releases) == 3

    countries = {r["country"] for r in releases}
    assert countries == {"GB", "US", "JP"}

    # Test with release group that has no releases
    db.upsert_release_group("rg-2", "Empty Group", artist_mbid="artist-1")
    releases = db.get_releases_in_group("rg-2")
    assert len(releases) == 0

    # Test with non-existent release group
    releases = db.get_releases_in_group("nonexistent")
    assert len(releases) == 0


def test_search_recordings_by_isrc_malformed_json(tmp_path):
    """Test ISRC search handles malformed JSON gracefully."""
    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Test Artist")
    # Insert recording with malformed JSON directly
    conn = db._get_connection()
    conn.execute(
        "INSERT INTO recording (mbid, title, isrcs_json, fetched_at) VALUES (?, ?, ?, ?)",
        ("rec-1", "Test", "not valid json", time.time()),
    )
    conn.commit()
    conn.close()

    # SQLite's json_each will raise an error for malformed JSON
    # This tests that the error properly propagates (documenting behavior)
    try:
        results = db.search_recordings_by_isrc("ANYTHING")
        # If we get here, SQLite version might handle it differently
        # Some versions return empty, others raise error
        assert len(results) == 0
    except sqlite3.OperationalError as e:
        # Expected behavior - malformed JSON raises error
        assert "malformed JSON" in str(e) or "JSON" in str(e)

    # Clean up malformed record before testing with valid data
    conn = db._get_connection()
    conn.execute("DELETE FROM recording WHERE mbid = ?", ("rec-1",))
    conn.commit()
    conn.close()

    # Add valid recording to confirm search works normally after cleanup
    db.upsert_recording("rec-2", "Valid", isrcs_json='["USCA12345678"]')
    results = db.search_recordings_by_isrc("USCA12345678")
    assert len(results) == 1
    assert results[0]["title"] == "Valid"


def test_search_recordings_fuzzy_empty_database(tmp_path):
    """Test fuzzy search on empty database."""
    db = MusicGraphDB(tmp_path / "test.sqlite")

    # Search with all parameters on empty database
    results = db.search_recordings_fuzzy("Yesterday")
    assert len(results) == 0

    results = db.search_recordings_fuzzy("Test", artist_name="Artist")
    assert len(results) == 0

    results = db.search_recordings_fuzzy("", length_min=100000, length_max=200000)
    assert len(results) == 0


def test_search_recordings_fuzzy_special_characters(tmp_path):
    """Test fuzzy search with SQL special characters (%, _, etc.)."""
    db = MusicGraphDB(tmp_path / "test.sqlite")

    db.upsert_artist("artist-1", "100% Pure")
    db.upsert_artist("artist-2", "Under_Score")

    db.upsert_recording("rec-1", "50% Love", artist_mbid="artist-1", length_ms=180000)
    db.upsert_recording("rec-2", "100% Pure", artist_mbid="artist-1", length_ms=200000)
    db.upsert_recording("rec-3", "My_Song", artist_mbid="artist-2", length_ms=220000)

    # Test % character in title search
    results = db.search_recordings_fuzzy("100%")
    assert len(results) == 1
    assert results[0]["title"] == "100% Pure"

    # Test % character in artist search
    results = db.search_recordings_fuzzy("", artist_name="100%")
    assert len(results) == 2
    titles = {r["title"] for r in results}
    assert titles == {"50% Love", "100% Pure"}

    # Test _ character (SQL wildcard for single character)
    results = db.search_recordings_fuzzy("My_")
    assert len(results) == 1
    assert results[0]["title"] == "My_Song"

    # Test underscore in artist name
    results = db.search_recordings_fuzzy("", artist_name="Under_")
    assert len(results) == 1
    assert results[0]["title"] == "My_Song"


def test_get_release_groups_for_recording_complex_chain(tmp_path):
    """Test with recording in multiple releases across multiple release groups."""
    db = MusicGraphDB(tmp_path / "test.sqlite")

    # Setup artist
    db.upsert_artist("artist-1", "The Beatles")

    # Setup one recording
    db.upsert_recording("recording-1", "Yesterday", artist_mbid="artist-1")

    # Create 3 release groups
    db.upsert_release_group("rg-1", "Help!", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-2", "Yesterday and Today", artist_mbid="artist-1", type="Album")
    db.upsert_release_group("rg-3", "Greatest Hits", artist_mbid="artist-1", type="Compilation")

    # Create multiple releases in each release group
    db.upsert_release("release-1", "Help! UK", release_group_mbid="rg-1", artist_mbid="artist-1")
    db.upsert_release("release-2", "Help! US", release_group_mbid="rg-1", artist_mbid="artist-1")
    db.upsert_release(
        "release-3", "Yesterday and Today", release_group_mbid="rg-2", artist_mbid="artist-1"
    )
    db.upsert_release(
        "release-4", "Greatest Hits Vol 1", release_group_mbid="rg-3", artist_mbid="artist-1"
    )
    db.upsert_release(
        "release-5", "Greatest Hits Vol 2", release_group_mbid="rg-3", artist_mbid="artist-1"
    )

    # Link recording to all releases
    db.upsert_recording_release("recording-1", "release-1")
    db.upsert_recording_release("recording-1", "release-2")
    db.upsert_recording_release("recording-1", "release-3")
    db.upsert_recording_release("recording-1", "release-4")
    db.upsert_recording_release("recording-1", "release-5")

    # Get release groups for recording
    rgs = db.get_release_groups_for_recording("recording-1")

    # Should return all 3 release groups despite having 5 releases
    assert len(rgs) == 3

    rg_mbids = {rg["mbid"] for rg in rgs}
    assert rg_mbids == {"rg-1", "rg-2", "rg-3"}

    # Verify types
    rg_types = {rg["type"] for rg in rgs}
    assert rg_types == {"Album", "Compilation"}

    # Verify all have artist_name
    for rg in rgs:
        assert rg["artist_name"] == "The Beatles"


def test_concurrent_database_access(tmp_path):
    """Test database handles concurrent read/write correctly."""
    import threading

    db = MusicGraphDB(tmp_path / "test.sqlite")
    db.upsert_artist("artist-1", "Test Artist")

    errors = []
    results = []

    def write_recordings(start_idx, count):
        try:
            for i in range(start_idx, start_idx + count):
                db.upsert_recording(
                    f"rec-{i}", f"Song {i}", artist_mbid="artist-1", length_ms=100000 + i
                )
        except Exception as e:
            errors.append(e)

    def read_recordings():
        try:
            for _ in range(10):
                result = db.search_recordings_fuzzy("Song", limit=100)
                results.append(len(result))
        except Exception as e:
            errors.append(e)

    # Create threads for concurrent access
    threads = []
    threads.append(threading.Thread(target=write_recordings, args=(0, 10)))
    threads.append(threading.Thread(target=write_recordings, args=(10, 10)))
    threads.append(threading.Thread(target=read_recordings))
    threads.append(threading.Thread(target=read_recordings))

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    # No errors should occur
    assert len(errors) == 0

    # Verify all recordings were written
    all_recordings = db.search_recordings_fuzzy("Song", limit=100)
    assert len(all_recordings) == 20

    # Verify reads worked (should have some results)
    assert len(results) > 0
