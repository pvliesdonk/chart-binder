from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from chart_binder.scrapers.base import ScrapedEntry

logger = logging.getLogger(__name__)


@dataclass
class KerstlijstSong:
    """A song from Kerstlijst JSON with year->position mapping."""

    artist: str
    title: str
    positions: dict[int, int]  # year -> rank


class KerstlijstImporter:
    """
    Importer for Kerstlijst (Christmas chart) JSON data.

    Unlike other scrapers, this reads from local JSON files rather than
    HTTP endpoints. The JSON format comes from hitlists.spotweb exports.

    JSON Format:
    [
      {
        "artiest": "Wham!",
        "title": "Last Christmas",
        "hitlists": {
          "spotweb": {
            "2020": 1,
            "2021": 2,
            "2022": 1
          }
        }
      }
    ]
    """

    chart_db_id = "kerstlijst"
    expected_entry_count = 100  # Typical Kerstlijst size

    def load(self, json_path: Path) -> list[KerstlijstSong]:
        """
        Load songs from Kerstlijst JSON file.

        Args:
            json_path: Path to JSON file

        Returns:
            List of KerstlijstSong objects with year->position mappings
        """
        if not json_path.exists():
            logger.warning(f"Kerstlijst JSON file not found: {json_path}")
            return []

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Kerstlijst JSON: {e}")
            return []

        songs = [song for item in data if (song := self._parse_song(item))]

        logger.info(f"Loaded {len(songs)} songs from Kerstlijst JSON")
        return songs

    def _parse_song(self, item: dict) -> KerstlijstSong | None:
        """Parse a single song entry from JSON."""
        try:
            artist = item.get("artiest", "").strip()
            title = item.get("title", "").strip()

            if not artist or not title:
                logger.warning(f"Skipping entry with missing artist/title: {item}")
                return None

            # Extract positions from hitlists.spotweb
            hitlists = item.get("hitlists", {})
            spotweb = hitlists.get("spotweb", {})

            positions: dict[int, int] = {}
            for year_str, rank in spotweb.items():
                try:
                    year = int(year_str)
                    rank_int = int(rank)
                    if 1 <= rank_int <= 500:  # Reasonable rank range
                        positions[year] = rank_int
                    else:
                        logger.warning(f"Skipping invalid rank {rank} for {artist} - {title}")
                except ValueError:
                    logger.warning(f"Invalid year/rank in {artist} - {title}: {year_str}={rank}")
                    continue

            if not positions:
                # Song has no valid positions, skip it
                return None

            return KerstlijstSong(artist=artist, title=title, positions=positions)

        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse song entry: {e}")
            return None

    def get_entries_for_year(self, songs: list[KerstlijstSong], year: int) -> list[ScrapedEntry]:
        """
        Get chart entries for a specific year.

        Args:
            songs: List of songs loaded from JSON
            year: Year to extract entries for

        Returns:
            List of ScrapedEntry sorted by rank
        """
        entries = [
            ScrapedEntry(
                rank=song.positions[year],
                artist=song.artist,
                title=song.title,
            )
            for song in songs
            if year in song.positions
        ]
        entries.sort(key=lambda e: e.rank)
        return entries

    def get_all_years(self, songs: list[KerstlijstSong]) -> set[int]:
        """Get all years present in the loaded songs."""
        years: set[int] = set()
        for song in songs:
            years.update(song.positions.keys())
        return years

    def get_entries_by_year(self, songs: list[KerstlijstSong]) -> dict[int, list[ScrapedEntry]]:
        """
        Get chart entries grouped by year.

        Args:
            songs: List of songs loaded from JSON

        Returns:
            Dict mapping year to list of ScrapedEntry (sorted by rank)
        """
        entries_by_year: dict[int, list[ScrapedEntry]] = defaultdict(list)

        for song in songs:
            for year, rank in song.positions.items():
                entries_by_year[year].append(
                    ScrapedEntry(rank=rank, artist=song.artist, title=song.title)
                )

        for entries in entries_by_year.values():
            entries.sort(key=lambda e: e.rank)

        return dict(entries_by_year)

    def validate_year(self, entries: list[ScrapedEntry], year: int) -> list[str]:
        """
        Validate entries for a specific year.

        Returns list of warning messages.
        """
        warnings: list[str] = []

        if not entries:
            warnings.append(f"No entries for year {year}")
            return warnings

        # Check for rank gaps
        ranks = sorted(e.rank for e in entries)
        expected_ranks = list(range(1, len(ranks) + 1))
        if ranks != expected_ranks:
            missing = set(expected_ranks) - set(ranks)
            if missing:
                warnings.append(f"Missing ranks for {year}: {sorted(missing)[:10]}")

        # Check for duplicate ranks
        if len(ranks) != len(set(ranks)):
            warnings.append(f"Duplicate ranks found for {year}")

        return warnings
