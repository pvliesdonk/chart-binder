from __future__ import annotations

import json
import logging
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
        "titel": "Last Christmas",
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

        songs: list[KerstlijstSong] = []

        for item in data:
            song = self._parse_song(item)
            if song:
                songs.append(song)

        logger.info(f"Loaded {len(songs)} songs from Kerstlijst JSON")
        return songs

    def _parse_song(self, item: dict) -> KerstlijstSong | None:
        """Parse a single song entry from JSON."""
        try:
            artist = item.get("artiest", "").strip()
            title = item.get("titel", "").strip()

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
        entries: list[ScrapedEntry] = []

        for song in songs:
            if year in song.positions:
                entries.append(
                    ScrapedEntry(
                        rank=song.positions[year],
                        artist=song.artist,
                        title=song.title,
                    )
                )

        # Sort by rank
        entries.sort(key=lambda e: e.rank)
        return entries

    def get_all_years(self, songs: list[KerstlijstSong]) -> set[int]:
        """Get all years present in the loaded songs."""
        years: set[int] = set()
        for song in songs:
            years.update(song.positions.keys())
        return years

    def get_entries_by_year(
        self, songs: list[KerstlijstSong]
    ) -> dict[int, list[ScrapedEntry]]:
        """
        Get chart entries grouped by year.

        Args:
            songs: List of songs loaded from JSON

        Returns:
            Dict mapping year to list of ScrapedEntry (sorted by rank)
        """
        result: dict[int, list[ScrapedEntry]] = {}

        for year in self.get_all_years(songs):
            result[year] = self.get_entries_for_year(songs, year)

        return result

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


## Tests


def test_kerstlijst_importer_load():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "kerstlijst.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "artiest": "Wham!",
                        "titel": "Last Christmas",
                        "hitlists": {"spotweb": {"2020": 1, "2021": 2}},
                    },
                    {
                        "artiest": "Mariah Carey",
                        "titel": "All I Want For Christmas Is You",
                        "hitlists": {"spotweb": {"2020": 2, "2021": 1}},
                    },
                ]
            )
        )

        importer = KerstlijstImporter()
        songs = importer.load(json_path)

        assert len(songs) == 2
        assert songs[0].artist == "Wham!"
        assert songs[0].positions[2020] == 1
        assert songs[0].positions[2021] == 2


def test_kerstlijst_get_entries_for_year():
    songs = [
        KerstlijstSong(artist="Wham!", title="Last Christmas", positions={2020: 1, 2021: 2}),
        KerstlijstSong(
            artist="Mariah Carey",
            title="All I Want For Christmas Is You",
            positions={2020: 2, 2021: 1},
        ),
    ]

    importer = KerstlijstImporter()
    entries = importer.get_entries_for_year(songs, 2020)

    assert len(entries) == 2
    assert entries[0].rank == 1
    assert entries[0].artist == "Wham!"
    assert entries[1].rank == 2
    assert entries[1].artist == "Mariah Carey"


def test_kerstlijst_get_all_years():
    songs = [
        KerstlijstSong(artist="Wham!", title="Last Christmas", positions={2020: 1, 2021: 2}),
        KerstlijstSong(artist="Other", title="Song", positions={2019: 5}),
    ]

    importer = KerstlijstImporter()
    years = importer.get_all_years(songs)

    assert years == {2019, 2020, 2021}


def test_kerstlijst_get_entries_by_year():
    songs = [
        KerstlijstSong(artist="Wham!", title="Last Christmas", positions={2020: 1, 2021: 2}),
        KerstlijstSong(
            artist="Mariah Carey",
            title="All I Want For Christmas Is You",
            positions={2020: 2, 2021: 1},
        ),
    ]

    importer = KerstlijstImporter()
    by_year = importer.get_entries_by_year(songs)

    assert 2020 in by_year
    assert 2021 in by_year
    assert len(by_year[2020]) == 2
    assert len(by_year[2021]) == 2

    # 2020: Wham! #1, Mariah #2
    assert by_year[2020][0].artist == "Wham!"
    assert by_year[2020][1].artist == "Mariah Carey"

    # 2021: Mariah #1, Wham! #2
    assert by_year[2021][0].artist == "Mariah Carey"
    assert by_year[2021][1].artist == "Wham!"


def test_kerstlijst_missing_file():
    importer = KerstlijstImporter()
    songs = importer.load(Path("/nonexistent/file.json"))
    assert songs == []


def test_kerstlijst_invalid_json():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "invalid.json"
        json_path.write_text("not valid json {{{")

        importer = KerstlijstImporter()
        songs = importer.load(json_path)
        assert songs == []


def test_kerstlijst_skips_invalid_entries():
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "kerstlijst.json"
        json_path.write_text(
            json.dumps(
                [
                    # Valid entry
                    {
                        "artiest": "Wham!",
                        "titel": "Last Christmas",
                        "hitlists": {"spotweb": {"2020": 1}},
                    },
                    # Missing artist
                    {"artiest": "", "titel": "Some Song", "hitlists": {"spotweb": {"2020": 2}}},
                    # Missing title
                    {
                        "artiest": "Some Artist",
                        "titel": "",
                        "hitlists": {"spotweb": {"2020": 3}},
                    },
                    # No positions
                    {"artiest": "No Positions", "titel": "Song", "hitlists": {"spotweb": {}}},
                    # Invalid rank (too high)
                    {
                        "artiest": "Invalid",
                        "titel": "Rank",
                        "hitlists": {"spotweb": {"2020": 9999}},
                    },
                ]
            )
        )

        importer = KerstlijstImporter()
        songs = importer.load(json_path)

        # Only the first valid entry should be loaded
        assert len(songs) == 1
        assert songs[0].artist == "Wham!"


def test_kerstlijst_validate_year():
    songs = [
        KerstlijstSong(artist="A", title="Song A", positions={2020: 1}),
        KerstlijstSong(artist="B", title="Song B", positions={2020: 2}),
        KerstlijstSong(artist="C", title="Song C", positions={2020: 4}),  # Gap at 3
    ]

    importer = KerstlijstImporter()
    entries = importer.get_entries_for_year(songs, 2020)
    warnings = importer.validate_year(entries, 2020)

    assert any("Missing ranks" in w for w in warnings)
