from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


class ChartsDB:
    """
    SQLite database for charts data and normalization aliases.

    Stores chart runs, entries, links, and the alias_norm registry.
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
        """Initialize database schema for charts and aliases."""
        conn = self._get_connection()

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chart (
                chart_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                frequency TEXT NOT NULL,
                jurisdiction TEXT,
                source_url TEXT,
                license TEXT,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS alias_norm (
                alias_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                raw TEXT NOT NULL,
                normalized TEXT NOT NULL,
                ruleset_version TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_alias_type_raw ON alias_norm(type, raw);
            CREATE INDEX IF NOT EXISTS idx_alias_normalized ON alias_norm(normalized);
            """
        )

        conn.commit()
        conn.close()

    def upsert_alias(
        self,
        alias_id: str,
        type: str,  # noqa: A002
        raw: str,
        normalized: str,
        ruleset_version: str = "norm-v1",
        created_at: float | None = None,
    ) -> None:
        """Upsert alias normalization record."""
        if created_at is None:
            created_at = time.time()

        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO alias_norm (alias_id, type, raw, normalized, ruleset_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(alias_id) DO UPDATE SET
                    type = excluded.type,
                    raw = excluded.raw,
                    normalized = excluded.normalized,
                    ruleset_version = excluded.ruleset_version
                """,
                (alias_id, type, raw, normalized, ruleset_version, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def get_alias(self, type: str, raw: str) -> dict[str, Any] | None:  # noqa: A002
        """Get alias normalization by type and raw text."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM alias_norm WHERE type = ? AND raw = ?", (type, raw))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_aliases(self, type: str | None = None) -> list[dict[str, Any]]:  # noqa: A002
        """List all aliases, optionally filtered by type."""
        conn = self._get_connection()
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if type:
                cursor.execute("SELECT * FROM alias_norm WHERE type = ? ORDER BY raw", (type,))
            else:
                cursor.execute("SELECT * FROM alias_norm ORDER BY type, raw")
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


def test_charts_db_init(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")
    assert db.db_path.exists()
    assert db.verify_foreign_keys()

    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()

    assert "chart" in tables
    assert "alias_norm" in tables


def test_alias_crud(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_alias("alias-1", "artist", "The Beatles", "beatles")

    alias = db.get_alias("artist", "The Beatles")
    assert alias is not None
    assert alias["normalized"] == "beatles"
    assert alias["type"] == "artist"

    db.upsert_alias("alias-1", "artist", "The Beatles", "the beatles")
    alias = db.get_alias("artist", "The Beatles")
    assert alias is not None
    assert alias["normalized"] == "the beatles"


def test_alias_list(tmp_path):
    db = ChartsDB(tmp_path / "charts.sqlite")

    db.upsert_alias("alias-1", "artist", "The Beatles", "beatles")
    db.upsert_alias("alias-2", "title", "Let It Be", "let it be")
    db.upsert_alias("alias-3", "artist", "Queen", "queen")

    all_aliases = db.list_aliases()
    assert len(all_aliases) == 3

    artist_aliases = db.list_aliases(type="artist")
    assert len(artist_aliases) == 2

    title_aliases = db.list_aliases(type="title")
    assert len(title_aliases) == 1
