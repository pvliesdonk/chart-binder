from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class MusicGraphDB:
    """
    SQLite database for music graph entities (artists, recordings, releases, etc.).

    Provides schema creation and upsert helpers for MusicBrainz-style entities.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                disambiguation TEXT,
                updated_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_artist_name ON artist(name);

            CREATE TABLE IF NOT EXISTS recording (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist_mbid TEXT,
                length_ms INTEGER,
                disambiguation TEXT,
                updated_at REAL NOT NULL,
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_recording_title ON recording(title);
            CREATE INDEX IF NOT EXISTS idx_recording_artist ON recording(artist_mbid);

            CREATE TABLE IF NOT EXISTS release_group (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                artist_mbid TEXT,
                type TEXT,
                first_release_date TEXT,
                disambiguation TEXT,
                updated_at REAL NOT NULL,
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_release_group_title ON release_group(title);
            CREATE INDEX IF NOT EXISTS idx_release_group_artist ON release_group(artist_mbid);

            CREATE TABLE IF NOT EXISTS release (
                mbid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                release_group_mbid TEXT,
                artist_mbid TEXT,
                date TEXT,
                country TEXT,
                barcode TEXT,
                disambiguation TEXT,
                updated_at REAL NOT NULL,
                FOREIGN KEY (release_group_mbid) REFERENCES release_group(mbid),
                FOREIGN KEY (artist_mbid) REFERENCES artist(mbid)
            );

            CREATE INDEX IF NOT EXISTS idx_release_title ON release(title);
            CREATE INDEX IF NOT EXISTS idx_release_group ON release(release_group_mbid);

            CREATE TABLE IF NOT EXISTS recording_release (
                recording_mbid TEXT NOT NULL,
                release_mbid TEXT NOT NULL,
                track_position INTEGER,
                track_number TEXT,
                updated_at REAL NOT NULL,
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
        disambiguation: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Upsert artist record."""
        if updated_at is None:
            updated_at = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO artist (mbid, name, sort_name, disambiguation, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    name = excluded.name,
                    sort_name = excluded.sort_name,
                    disambiguation = excluded.disambiguation,
                    updated_at = excluded.updated_at
                """,
                (mbid, name, sort_name, disambiguation, updated_at),
            )
            conn.commit()

    def upsert_recording(
        self,
        mbid: str,
        title: str,
        artist_mbid: str | None = None,
        length_ms: int | None = None,
        disambiguation: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Upsert recording record."""
        if updated_at is None:
            updated_at = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO recording (mbid, title, artist_mbid, length_ms, disambiguation, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    artist_mbid = excluded.artist_mbid,
                    length_ms = excluded.length_ms,
                    disambiguation = excluded.disambiguation,
                    updated_at = excluded.updated_at
                """,
                (mbid, title, artist_mbid, length_ms, disambiguation, updated_at),
            )
            conn.commit()

    def upsert_release_group(
        self,
        mbid: str,
        title: str,
        artist_mbid: str | None = None,
        type: str | None = None,  # noqa: A002
        first_release_date: str | None = None,
        disambiguation: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Upsert release_group record."""
        if updated_at is None:
            updated_at = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO release_group (mbid, title, artist_mbid, type, first_release_date, disambiguation, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    artist_mbid = excluded.artist_mbid,
                    type = excluded.type,
                    first_release_date = excluded.first_release_date,
                    disambiguation = excluded.disambiguation,
                    updated_at = excluded.updated_at
                """,
                (mbid, title, artist_mbid, type, first_release_date, disambiguation, updated_at),
            )
            conn.commit()

    def upsert_release(
        self,
        mbid: str,
        title: str,
        release_group_mbid: str | None = None,
        artist_mbid: str | None = None,
        date: str | None = None,
        country: str | None = None,
        barcode: str | None = None,
        disambiguation: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Upsert release record."""
        if updated_at is None:
            updated_at = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO release (mbid, title, release_group_mbid, artist_mbid, date, country, barcode, disambiguation, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mbid) DO UPDATE SET
                    title = excluded.title,
                    release_group_mbid = excluded.release_group_mbid,
                    artist_mbid = excluded.artist_mbid,
                    date = excluded.date,
                    country = excluded.country,
                    barcode = excluded.barcode,
                    disambiguation = excluded.disambiguation,
                    updated_at = excluded.updated_at
                """,
                (
                    mbid,
                    title,
                    release_group_mbid,
                    artist_mbid,
                    date,
                    country,
                    barcode,
                    disambiguation,
                    updated_at,
                ),
            )
            conn.commit()

    def upsert_recording_release(
        self,
        recording_mbid: str,
        release_mbid: str,
        track_position: int | None = None,
        track_number: str | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Upsert recording_release relationship."""
        if updated_at is None:
            updated_at = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO recording_release (recording_mbid, release_mbid, track_position, track_number, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(recording_mbid, release_mbid) DO UPDATE SET
                    track_position = excluded.track_position,
                    track_number = excluded.track_number,
                    updated_at = excluded.updated_at
                """,
                (recording_mbid, release_mbid, track_position, track_number, updated_at),
            )
            conn.commit()

    def get_artist(self, mbid: str) -> dict[str, Any] | None:
        """Get artist by MBID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM artist WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_recording(self, mbid: str) -> dict[str, Any] | None:
        """Get recording by MBID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM recording WHERE mbid = ?", (mbid,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def verify_foreign_keys(self) -> bool:
        """Verify that foreign key constraints are enabled."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()
            return result is not None and result[0] == 1


## Tests


def test_music_graph_db_init(tmp_path):
    db = MusicGraphDB(tmp_path / "test.sqlite")
    assert db.db_path.exists()
    assert db.verify_foreign_keys()


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
