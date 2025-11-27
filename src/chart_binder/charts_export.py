"""CHARTS blob export for audio tags.

Implements the CHARTS field schema v1 for compact, embeddable JSON.
See: docs/appendix/charts_field_schema_v1.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChartScore:
    """Score data for a single chart."""

    chart_id: str
    score: int  # Aggregated points across all runs
    highest: int  # Best rank (lowest number)
    frequency: str  # 'y' for yearly, 'w' for weekly
    positions: dict[str, Any] | None = None  # Optional position details


@dataclass
class ChartsBlob:
    """
    CHARTS blob v1 for embedding in audio tags.

    Wire format:
    {
      "v": 1,
      "c": [
        ["<chart_id>", <score>, <highest>, "<freq>", <pos_opt?>]
      ]
    }
    """

    version: int = 1
    scores: list[ChartScore] = field(default_factory=list)

    def to_json(self, include_positions: bool = False, max_size_kb: float = 3.0) -> str:
        """
        Serialize to minified JSON.

        Args:
            include_positions: Whether to include position details
            max_size_kb: Maximum size in KB (positions omitted if exceeded)

        Returns:
            Minified JSON string
        """
        # Sort by score desc, then highest asc (deterministic)
        sorted_scores = sorted(self.scores, key=lambda s: (-s.score, s.highest))

        # Build the chart tuples
        chart_tuples = []
        for score in sorted_scores:
            if include_positions and score.positions:
                chart_tuples.append(
                    [score.chart_id, score.score, score.highest, score.frequency, score.positions]
                )
            else:
                chart_tuples.append([score.chart_id, score.score, score.highest, score.frequency])

        blob = {"v": self.version, "c": chart_tuples}
        result = json.dumps(blob, separators=(",", ":"), ensure_ascii=True)

        # Check size and remove positions if needed
        if len(result.encode("utf-8")) > max_size_kb * 1024 and include_positions:
            # Retry without positions
            return self.to_json(include_positions=False, max_size_kb=max_size_kb)

        return result

    @classmethod
    def from_json(cls, json_str: str) -> ChartsBlob:
        """Parse CHARTS blob from JSON string."""
        data = json.loads(json_str)

        if data.get("v") != 1:
            raise ValueError(f"Unsupported CHARTS version: {data.get('v')}")

        scores = []
        for chart_tuple in data.get("c", []):
            chart_id = chart_tuple[0]
            score = chart_tuple[1]
            highest = chart_tuple[2]
            frequency = chart_tuple[3]
            positions = chart_tuple[4] if len(chart_tuple) > 4 else None

            scores.append(
                ChartScore(
                    chart_id=chart_id,
                    score=score,
                    highest=highest,
                    frequency=frequency,
                    positions=positions,
                )
            )

        return cls(version=1, scores=scores)


class ChartsExporter:
    """
    Exports CHARTS blobs from chart data.

    Aggregates scores across chart runs and produces deterministic JSON.
    """

    # Chart size (N) for score calculation
    CHART_SIZES = {
        "t2000": 2000,
        "t40": 40,
        "t100": 100,
        "zwaar": 100,  # Variable, but typically 100
    }

    # Chart frequencies
    CHART_FREQUENCIES = {
        "t2000": "y",
        "t40": "w",
        "t100": "y",
        "zwaar": "y",
    }

    def __init__(self, charts_db: Any):
        """
        Initialize exporter with charts database.

        Args:
            charts_db: ChartsDB instance
        """
        self.db = charts_db

    def export_for_work(
        self,
        work_key: str,
        include_positions: bool = False,
    ) -> ChartsBlob:
        """
        Export CHARTS blob for a specific work_key.

        Aggregates all chart appearances for this work.

        Args:
            work_key: The work_key to export for
            include_positions: Include detailed position data

        Returns:
            ChartsBlob ready for serialization
        """
        conn = self.db._get_connection()
        try:
            conn.row_factory = None  # Use tuple results
            cursor = conn.cursor()

            # Get all chart links for this work_key
            cursor.execute(
                """
                SELECT l.run_id, l.rank, l.confidence, r.chart_id, r.period
                FROM chart_link l
                JOIN chart_run r ON l.run_id = r.run_id
                WHERE l.work_key = ? AND l.confidence >= 0.60
                ORDER BY r.chart_id, r.period
                """,
                (work_key,),
            )

            links = cursor.fetchall()

            # Aggregate by chart_id
            chart_data: dict[str, dict[str, Any]] = {}

            for _run_id, rank, _confidence, chart_id, period in links:
                if chart_id not in chart_data:
                    chart_data[chart_id] = {
                        "total_score": 0,
                        "best_rank": float("inf"),
                        "positions": {},
                    }

                # Calculate points: N - rank + 1
                chart_size = self.CHART_SIZES.get(chart_id, 100)
                points = chart_size - rank + 1

                chart_data[chart_id]["total_score"] += points
                chart_data[chart_id]["best_rank"] = min(chart_data[chart_id]["best_rank"], rank)

                # Store position details
                freq = self.CHART_FREQUENCIES.get(chart_id, "y")
                if freq == "y":
                    # Yearly charts: {"YYYY": rank}
                    year = period.split("-")[0] if "-" in period else period
                    chart_data[chart_id]["positions"][year] = rank
                else:
                    # Weekly charts: {"YYYY": {"WW": rank}} where WW is the week number as string
                    # Example: period "1999-W25" becomes {"1999": {"25": 4}}
                    if "-W" in period:
                        year, week = period.split("-W")
                        if year not in chart_data[chart_id]["positions"]:
                            chart_data[chart_id]["positions"][year] = {}
                        chart_data[chart_id]["positions"][year][week] = rank

            # Build ChartScore objects
            scores = []
            for chart_id, data in chart_data.items():
                freq = self.CHART_FREQUENCIES.get(chart_id, "y")
                scores.append(
                    ChartScore(
                        chart_id=chart_id,
                        score=data["total_score"],
                        highest=int(data["best_rank"]),
                        frequency=freq,
                        positions=data["positions"] if include_positions else None,
                    )
                )

            return ChartsBlob(version=1, scores=scores)

        finally:
            conn.close()

    def compute_score(self, chart_id: str, rank: int) -> int:
        """
        Compute points for a single chart entry.

        Formula: points = N - rank + 1
        """
        chart_size = self.CHART_SIZES.get(chart_id, 100)
        return chart_size - rank + 1


## Tests


def test_charts_blob_creation():
    blob = ChartsBlob()
    blob.scores = [
        ChartScore(chart_id="t2000", score=36250, highest=30, frequency="y"),
        ChartScore(chart_id="t40", score=266, highest=4, frequency="w"),
    ]

    json_str = blob.to_json()
    assert '"v":1' in json_str
    assert '"c":[' in json_str
    assert '"t2000"' in json_str


def test_charts_blob_minimal_format():
    blob = ChartsBlob()
    blob.scores = [
        ChartScore(chart_id="t2000", score=36250, highest=30, frequency="y"),
    ]

    json_str = blob.to_json()
    expected = '{"v":1,"c":[["t2000",36250,30,"y"]]}'
    assert json_str == expected


def test_charts_blob_with_positions():
    blob = ChartsBlob()
    blob.scores = [
        ChartScore(
            chart_id="t2000",
            score=36250,
            highest=30,
            frequency="y",
            positions={"2023": 48, "2024": 32},
        ),
    ]

    json_str = blob.to_json(include_positions=True)
    assert '"2023":48' in json_str
    assert '"2024":32' in json_str


def test_charts_blob_sorting():
    """Verify deterministic sorting: score desc, then highest asc."""
    blob = ChartsBlob()
    blob.scores = [
        ChartScore(chart_id="t40", score=100, highest=10, frequency="w"),
        ChartScore(chart_id="t2000", score=500, highest=5, frequency="y"),
        ChartScore(chart_id="t100", score=500, highest=3, frequency="y"),  # Same score, better rank
    ]

    json_str = blob.to_json()
    parsed = json.loads(json_str)

    # t100 (500, 3) should come before t2000 (500, 5) since lower highest is better
    assert parsed["c"][0][0] == "t100"
    assert parsed["c"][1][0] == "t2000"
    assert parsed["c"][2][0] == "t40"


def test_charts_blob_parse():
    json_str = '{"v":1,"c":[["t2000",36250,30,"y"],["t40",266,4,"w"]]}'
    blob = ChartsBlob.from_json(json_str)

    assert blob.version == 1
    assert len(blob.scores) == 2
    assert blob.scores[0].chart_id == "t2000"
    assert blob.scores[0].score == 36250
    assert blob.scores[0].highest == 30


def test_charts_blob_parse_with_positions():
    json_str = '{"v":1,"c":[["t2000",36250,30,"y",{"2023":48,"2024":32}]]}'
    blob = ChartsBlob.from_json(json_str)

    assert blob.scores[0].positions == {"2023": 48, "2024": 32}


def test_charts_blob_size_budgeting():
    """Test that positions are omitted when size exceeds limit."""
    blob = ChartsBlob()
    # Create a blob with lots of position data
    positions = {str(year): year % 100 + 1 for year in range(1980, 2025)}
    blob.scores = [
        ChartScore(chart_id="t2000", score=50000, highest=1, frequency="y", positions=positions),
    ]

    # With very small limit, positions should be omitted
    json_str = blob.to_json(include_positions=True, max_size_kb=0.1)
    parsed = json.loads(json_str)

    # Should not have positions
    assert len(parsed["c"][0]) == 4


def test_charts_exporter_score_calculation():
    """Test score calculation formula."""

    # Create mock DB (we'll test without actual DB queries)
    class MockDB:
        def _get_connection(self):
            return self

        def row_factory(self):
            pass

        def cursor(self):
            return self

        def execute(self, query, params):
            return []

        def fetchall(self):
            return []

        def close(self):
            pass

    exporter = ChartsExporter(MockDB())

    # Test score calculation: points = N - rank + 1
    assert exporter.compute_score("t2000", 1) == 2000  # 2000 - 1 + 1 = 2000
    assert exporter.compute_score("t2000", 100) == 1901  # 2000 - 100 + 1 = 1901
    assert exporter.compute_score("t40", 1) == 40  # 40 - 1 + 1 = 40
    assert exporter.compute_score("t40", 10) == 31  # 40 - 10 + 1 = 31


def test_charts_blob_version_error():
    """Test that unsupported versions raise error."""
    import pytest

    json_str = '{"v":2,"c":[]}'
    with pytest.raises(ValueError, match="Unsupported CHARTS version"):
        ChartsBlob.from_json(json_str)


def test_charts_blob_roundtrip():
    """Test JSON serialization roundtrip."""
    original = ChartsBlob()
    original.scores = [
        ChartScore(chart_id="t2000", score=1971, highest=30, frequency="y"),
        ChartScore(chart_id="t40", score=266, highest=4, frequency="w"),
    ]

    json_str = original.to_json()
    parsed = ChartsBlob.from_json(json_str)

    assert len(parsed.scores) == 2
    assert parsed.scores[0].chart_id == original.scores[0].chart_id
    assert parsed.scores[0].score == original.scores[0].score
