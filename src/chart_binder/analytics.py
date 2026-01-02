"""Cross-chart querying and analytics module.

Provides functions for:
- Song chart history across all charts
- Fuzzy song lookup by artist/title
- Chart comparison between runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chart_binder.charts_db import ChartsDB, Song


@dataclass
class ChartAppearance:
    """A single chart appearance for a song."""

    chart_id: str
    chart_name: str
    period: str
    rank: int
    previous_position: int | None = None
    weeks_on_chart: int | None = None


@dataclass
class ChartComparison:
    """Result of comparing two chart runs."""

    run1_id: str
    run2_id: str
    run1_period: str
    run2_period: str
    common_songs: list[tuple[str, str, int, int]]  # (artist, title, rank1, rank2)
    only_in_run1: list[tuple[str, str, int]]  # (artist, title, rank)
    only_in_run2: list[tuple[str, str, int]]  # (artist, title, rank)
    overlap_pct: float = 0.0
    movers: list[tuple[str, str, int, int, int]] = field(
        default_factory=list
    )  # (artist, title, rank1, rank2, delta)


class ChartAnalytics:
    """Analytics for cross-chart querying."""

    def __init__(self, charts_db: ChartsDB):
        self.db = charts_db

    def get_song_chart_history(self, song_id: str) -> list[ChartAppearance]:
        """
        Get all chart appearances for a song.

        Args:
            song_id: The song's unique ID

        Returns:
            List of ChartAppearance objects sorted by chart and period
        """
        history = self.db.get_chart_history(song_id)
        return [
            ChartAppearance(
                chart_id=entry["chart_id"],
                chart_name=entry["chart_name"],
                period=entry["period"],
                rank=entry["rank"],
                previous_position=entry.get("previous_position"),
                weeks_on_chart=entry.get("weeks_on_chart"),
            )
            for entry in history
        ]

    def get_song_by_artist_title(
        self, artist: str, title: str, threshold: float = 0.7
    ) -> Song | None:
        """
        Find canonical song by fuzzy matching.

        First tries exact match on canonical fields, then falls back
        to fuzzy matching against all songs.

        Args:
            artist: Artist name to search
            title: Song title to search
            threshold: Minimum similarity threshold (0-1)

        Returns:
            Best matching Song or None if no match above threshold
        """
        from chart_binder.normalize import Normalizer

        normalizer = Normalizer()

        # Normalize input
        artist_norm = normalizer.normalize_artist(artist).normalized
        title_norm = normalizer.normalize_title(title).normalized

        # Try exact match first
        exact_match = self.db.get_song_by_canonical(artist_norm, title_norm)
        if exact_match:
            return exact_match

        # Fuzzy search: get all songs and score them
        all_songs = self._get_all_songs()
        if not all_songs:
            return None

        best_match: Song | None = None
        best_score = 0.0

        for song in all_songs:
            # Normalize song's canonical fields for comparison
            song_artist_norm = normalizer.normalize_artist(song.artist_canonical).normalized
            song_title_norm = normalizer.normalize_title(song.title_canonical).normalized

            # Calculate similarity score
            artist_sim = self._similarity(artist_norm, song_artist_norm)
            title_sim = self._similarity(title_norm, song_title_norm)

            # Combined score (weighted: title is more distinctive)
            combined = (artist_sim * 0.4) + (title_sim * 0.6)

            if combined > best_score and combined >= threshold:
                best_score = combined
                best_match = song

        return best_match

    def _get_all_songs(self) -> list[Song]:
        """Get all songs from the database."""
        from chart_binder.charts_db import Song

        conn = self.db._get_connection()
        try:
            conn.row_factory = lambda c, r: Song(
                song_id=r[0],
                artist_canonical=r[1],
                title_canonical=r[2],
                artist_sort=r[3],
                work_key=r[4],
                recording_mbid=r[5],
                release_group_mbid=r[6],
                spotify_id=r[7],
                isrc=r[8],
                created_at=r[9],
            )
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT song_id, artist_canonical, title_canonical, artist_sort,
                       work_key, recording_mbid, release_group_mbid, spotify_id,
                       isrc, created_at
                FROM song
                """
            )
            return cursor.fetchall()
        finally:
            conn.close()

    def _similarity(self, s1: str, s2: str) -> float:
        """
        Calculate similarity between two strings using Levenshtein-based ratio.

        Returns score between 0 and 1.
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Use difflib for reasonable performance
        from difflib import SequenceMatcher

        return SequenceMatcher(None, s1, s2).ratio()

    def compare_charts(self, run1_id: str, run2_id: str) -> ChartComparison:
        """
        Compare two chart runs for overlap and differences.

        Args:
            run1_id: First chart run ID (or "chart_id:period" format)
            run2_id: Second chart run ID (or "chart_id:period" format)

        Returns:
            ChartComparison with common songs, unique entries, and movers
        """
        # Resolve run IDs if in chart_id:period format
        run1 = self._resolve_run(run1_id)
        run2 = self._resolve_run(run2_id)

        if not run1 or not run2:
            return ChartComparison(
                run1_id=run1_id,
                run2_id=run2_id,
                run1_period="",
                run2_period="",
                common_songs=[],
                only_in_run1=[],
                only_in_run2=[],
            )

        # Get entries for both runs
        entries1 = self._get_run_entries(run1["run_id"])
        entries2 = self._get_run_entries(run2["run_id"])

        # Build lookup by normalized (artist, title)
        from chart_binder.normalize import Normalizer

        normalizer = Normalizer()

        def normalize_key(artist: str, title: str) -> tuple[str, str]:
            return (
                normalizer.normalize_artist(artist).normalized,
                normalizer.normalize_title(title).normalized,
            )

        # Map: normalized_key -> (artist_raw, title_raw, rank)
        map1: dict[tuple[str, str], tuple[str, str, int]] = {}
        for entry in entries1:
            key = normalize_key(entry["artist_raw"], entry["title_raw"])
            map1[key] = (entry["artist_raw"], entry["title_raw"], entry["rank"])

        map2: dict[tuple[str, str], tuple[str, str, int]] = {}
        for entry in entries2:
            key = normalize_key(entry["artist_raw"], entry["title_raw"])
            map2[key] = (entry["artist_raw"], entry["title_raw"], entry["rank"])

        # Find common and unique entries
        common_keys = set(map1.keys()) & set(map2.keys())
        only1_keys = set(map1.keys()) - set(map2.keys())
        only2_keys = set(map2.keys()) - set(map1.keys())

        common_songs = []
        movers = []
        for key in common_keys:
            artist, title, rank1 = map1[key]
            _, _, rank2 = map2[key]
            common_songs.append((artist, title, rank1, rank2))
            delta = rank1 - rank2  # positive = dropped, negative = rose
            if delta != 0:
                movers.append((artist, title, rank1, rank2, delta))

        # Sort by rank change magnitude
        movers.sort(key=lambda x: abs(x[4]), reverse=True)

        only_in_run1 = [(map1[k][0], map1[k][1], map1[k][2]) for k in only1_keys]
        only_in_run2 = [(map2[k][0], map2[k][1], map2[k][2]) for k in only2_keys]

        # Sort by rank
        common_songs.sort(key=lambda x: x[2])  # by rank in run1
        only_in_run1.sort(key=lambda x: x[2])
        only_in_run2.sort(key=lambda x: x[2])

        # Calculate overlap percentage
        total_unique = len(common_keys) + len(only1_keys) + len(only2_keys)
        overlap_pct = (len(common_keys) / total_unique * 100) if total_unique > 0 else 0.0

        return ChartComparison(
            run1_id=run1["run_id"],
            run2_id=run2["run_id"],
            run1_period=run1["period"],
            run2_period=run2["period"],
            common_songs=common_songs,
            only_in_run1=only_in_run1,
            only_in_run2=only_in_run2,
            overlap_pct=overlap_pct,
            movers=movers,
        )

    def _resolve_run(self, run_id: str) -> dict[str, Any] | None:
        """
        Resolve run identifier to run dict.

        Accepts either:
        - Direct run_id (UUID)
        - chart_id:period format (e.g., "nl_top2000:2024")
        """
        # Check if it's chart_id:period format
        if ":" in run_id:
            chart_id, period = run_id.split(":", 1)
            return self.db.get_run_by_period(chart_id, period)

        # Otherwise treat as direct run_id - look it up
        conn = self.db._get_connection()
        try:
            import sqlite3

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT run_id, chart_id, period FROM chart_run WHERE run_id = ?",
                (run_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def _get_run_entries(self, run_id: str) -> list[dict[str, Any]]:
        """Get all entries for a chart run."""
        conn = self.db._get_connection()
        try:
            import sqlite3

            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT entry_id, rank, artist_raw, title_raw
                FROM chart_entry
                WHERE run_id = ?
                ORDER BY rank
                """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_biggest_movers(
        self, chart_id: str, period1: str, period2: str, limit: int = 10
    ) -> list[tuple[str, str, int, int, int]]:
        """
        Get songs with the biggest rank changes between two periods.

        Args:
            chart_id: Chart identifier
            period1: Earlier period
            period2: Later period
            limit: Max results to return

        Returns:
            List of (artist, title, rank1, rank2, delta) sorted by absolute delta
        """
        comparison = self.compare_charts(f"{chart_id}:{period1}", f"{chart_id}:{period2}")
        return comparison.movers[:limit]

    def get_new_entries(
        self, chart_id: str, period: str, prev_period: str | None = None
    ) -> list[tuple[str, str, int]]:
        """
        Get new entries that weren't in the previous period.

        Args:
            chart_id: Chart identifier
            period: Current period
            prev_period: Previous period (auto-detected if None)

        Returns:
            List of (artist, title, rank) for new entries
        """
        if prev_period is None:
            prev_period = self.db.get_adjacent_period(chart_id, period, direction=-1)
            if not prev_period:
                return []

        comparison = self.compare_charts(f"{chart_id}:{prev_period}", f"{chart_id}:{period}")
        return comparison.only_in_run2

    def get_dropped_entries(
        self, chart_id: str, period: str, prev_period: str | None = None
    ) -> list[tuple[str, str, int]]:
        """
        Get entries that dropped out compared to previous period.

        Args:
            chart_id: Chart identifier
            period: Current period
            prev_period: Previous period (auto-detected if None)

        Returns:
            List of (artist, title, rank) for dropped entries
        """
        if prev_period is None:
            prev_period = self.db.get_adjacent_period(chart_id, period, direction=-1)
            if not prev_period:
                return []

        comparison = self.compare_charts(f"{chart_id}:{prev_period}", f"{chart_id}:{period}")
        return comparison.only_in_run1
