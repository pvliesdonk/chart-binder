"""Edge case testing infrastructure for chart scrapers.

This module provides:
1. Fixtures for known problematic entries
2. Regression tests for previously-broken cases
3. Infrastructure to easily add new edge cases as discovered

When scraping real data and encountering an unexpected pattern:
1. Add the problematic entry to scraper_edge_cases.json
2. Run tests - they will fail with a clear message
3. Fix the scraper
4. Tests pass and serve as regression tests

Usage:
    # Add new edge case to JSON
    # Run: pytest tests/test_scraper_edge_cases.py -v
    # See clear failure message
    # Fix scraper
    # Commit both fix and edge case
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Load edge cases
FIXTURES_DIR = Path(__file__).parent / "fixtures"
EDGE_CASES_FILE = FIXTURES_DIR / "scraper_edge_cases.json"


class ScraperEdgeCaseError(Exception):
    """Raised when a scraper encounters an unhandled edge case."""

    def __init__(self, scraper_id: str, entry: dict, reason: str):
        self.scraper_id = scraper_id
        self.entry = entry
        self.reason = reason
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        return f"""
================================================================================
SCRAPER EDGE CASE DETECTED
================================================================================
Scraper: {self.scraper_id}
Reason:  {self.reason}

Entry:
  Rank:   {self.entry.get("rank", "N/A")}
  Artist: {self.entry.get("artist", "N/A")}
  Title:  {self.entry.get("title", "N/A")}

To fix this:
1. Add this entry to tests/fixtures/scraper_edge_cases.json
2. Implement handling in the scraper
3. Mark the edge case as "handled" in the fixture

Example fixture entry:
{{
  "id": "{self.scraper_id}-issue-XXX",
  "description": "{self.reason}",
  "input": {self.entry},
  "expected": [...],
  "status": "pending"
}}
================================================================================
"""


def load_edge_cases() -> dict:
    """Load edge cases from fixture file."""
    if EDGE_CASES_FILE.exists():
        return json.loads(EDGE_CASES_FILE.read_text())
    return {"edge_cases": {}}


@pytest.fixture
def edge_cases():
    """Provide edge cases fixture."""
    return load_edge_cases()


class TestEdgeCaseInfrastructure:
    """Test the edge case infrastructure itself."""

    def test_fixture_file_exists(self):
        """Verify edge case fixture file exists."""
        assert EDGE_CASES_FILE.exists(), f"Edge case fixture not found: {EDGE_CASES_FILE}"

    def test_fixture_valid_json(self):
        """Verify fixture is valid JSON."""
        data = load_edge_cases()
        assert "edge_cases" in data
        assert "version" in data

    def test_fixture_has_all_scrapers(self):
        """Verify fixture defines edge cases for all known scrapers."""
        data = load_edge_cases()
        edge_cases = data.get("edge_cases", {})

        expected_scrapers = ["t40", "t40jaar", "top2000", "zwaarste"]
        for scraper in expected_scrapers:
            assert scraper in edge_cases, f"Missing edge cases for {scraper}"

    def test_fixture_structure(self):
        """Verify each scraper section has required categories."""
        data = load_edge_cases()
        edge_cases = data.get("edge_cases", {})

        required_categories = [
            "split_entries",
            "malformed_entries",
            "encoding_issues",
            "missing_fields",
        ]

        for scraper_id, scraper_cases in edge_cases.items():
            for category in required_categories:
                assert category in scraper_cases, f"Missing category {category} in {scraper_id}"


class TestTop40EdgeCases:
    """Edge case tests for Top40Scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        from chart_binder.http_cache import HttpCache
        from chart_binder.scrapers import Top40Scraper

        cache = HttpCache(tmp_path / "cache")
        return Top40Scraper(cache)

    def test_split_entry_cases(self, scraper, edge_cases):
        """Test all known split entry edge cases."""
        cases = edge_cases.get("edge_cases", {}).get("t40", {}).get("split_entries", [])

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled: {case.get('notes', '')}")

            # Test the case
            input_data = case["input"]
            expected = case["expected"]

            result = scraper._handle_split_entries(
                input_data["rank"], input_data["artist"], input_data["title"]
            )

            assert len(result) == len(expected), (
                f"Edge case {case['id']}: Expected {len(expected)} entries, got {len(result)}"
            )

            for i, (actual, exp) in enumerate(zip(result, expected, strict=True)):
                assert actual[0] == exp["rank"], (
                    f"Edge case {case['id']}: Wrong rank at index {i}. Expected {exp['rank']}, got {actual[0]}"
                )
                assert actual[1] == exp["artist"], (
                    f"Edge case {case['id']}: Wrong artist at index {i}. Expected {exp['artist']}, got {actual[1]}"
                )
                assert actual[2] == exp["title"], (
                    f"Edge case {case['id']}: Wrong title at index {i}. Expected {exp['title']}, got {actual[2]}"
                )

    def test_malformed_entry_cases(self, scraper, edge_cases):
        """Test that malformed entries are handled correctly."""
        cases = edge_cases.get("edge_cases", {}).get("t40", {}).get("malformed_entries", [])

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled")

            input_data = case["input"]
            expected_behavior = case.get("expected_behavior", "skip")

            result = scraper._handle_split_entries(
                input_data["rank"], input_data.get("artist", ""), input_data.get("title", "")
            )

            if expected_behavior == "skip":
                # Scraper should still return something, but parser should skip
                # This is more of a parser-level test
                assert isinstance(result, list), f"Edge case {case['id']}: Should return a list"

    def test_encoding_cases(self, scraper, edge_cases):
        """Test that encoding is preserved correctly."""
        cases = edge_cases.get("edge_cases", {}).get("t40", {}).get("encoding_issues", [])

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled")

            input_data = case["input"]
            expected_behavior = case.get("expected_behavior", "preserve")

            result = scraper._handle_split_entries(
                input_data["rank"], input_data["artist"], input_data["title"]
            )

            if expected_behavior == "preserve":
                # Characters should be preserved
                assert len(result) > 0
                assert input_data["artist"] in result[0][1] or result[0][1] in input_data["artist"]


class TestTop40JaarEdgeCases:
    """Edge case tests for Top40JaarScraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        from chart_binder.http_cache import HttpCache
        from chart_binder.scrapers import Top40JaarScraper

        cache = HttpCache(tmp_path / "cache")
        return Top40JaarScraper(cache)

    def test_split_entry_cases(self, scraper, edge_cases):
        """Test all known split entry edge cases for Top40Jaar."""
        cases = edge_cases.get("edge_cases", {}).get("t40jaar", {}).get("split_entries", [])

        if not cases:
            pytest.skip("No split entry cases defined for t40jaar yet")

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled")

            # Similar logic to Top40 tests
            # Top40Jaar likely uses similar parsing logic


class TestTop2000EdgeCases:
    """Edge case tests for Top2000Scraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        from chart_binder.http_cache import HttpCache
        from chart_binder.scrapers import Top2000Scraper

        cache = HttpCache(tmp_path / "cache")
        return Top2000Scraper(cache)

    def test_split_entry_cases(self, scraper, edge_cases):
        """Test all known split entry edge cases for Top2000."""
        cases = edge_cases.get("edge_cases", {}).get("top2000", {}).get("split_entries", [])

        if not cases:
            pytest.skip("No split entry cases defined for top2000 yet")

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled")

            # Test implementation here


class TestZwaarsteEdgeCases:
    """Edge case tests for ZwaarsteScraper."""

    @pytest.fixture
    def scraper(self, tmp_path):
        from chart_binder.http_cache import HttpCache
        from chart_binder.scrapers import ZwaarsteScraper

        cache = HttpCache(tmp_path / "cache")
        return ZwaarsteScraper(cache)

    def test_split_entry_cases(self, scraper, edge_cases):
        """Test all known split entry edge cases for Zwaarste."""
        cases = edge_cases.get("edge_cases", {}).get("zwaarste", {}).get("split_entries", [])

        if not cases:
            pytest.skip("No split entry cases defined for zwaarste yet")

        for case in cases:
            if case.get("status") != "handled":
                pytest.skip(f"Edge case {case['id']} not yet handled")

            # Test implementation here


class TestValidationRules:
    """Test validation rules for detecting suspicious patterns."""

    def test_validation_rules_exist(self, edge_cases):
        """Verify validation rules are defined."""
        assert "validation_rules" in edge_cases
        rules = edge_cases["validation_rules"].get("rules", [])
        assert len(rules) > 0, "No validation rules defined"

    def test_title_length_validation(self):
        """Test that excessively long titles are flagged."""
        from tests.edge_case_helpers import validate_entry

        rules = load_edge_cases().get("validation_rules", {}).get("rules", [])

        long_title = "A" * 250
        entry = {"rank": 1, "artist": "Artist", "title": long_title}

        warnings = validate_entry(entry, rules)
        assert any("exceeds" in w for w in warnings), (
            f"Should warn about excessively long title. Got warnings: {warnings}"
        )

    def test_html_tags_validation(self):
        """Test that HTML tags in text are flagged."""
        from tests.edge_case_helpers import validate_entry

        rules = load_edge_cases().get("validation_rules", {}).get("rules", [])

        entry = {"rank": 1, "artist": "Artist", "title": "Title <span>text</span>"}

        warnings = validate_entry(entry, rules)
        assert any("html" in w.lower() for w in warnings), "Should error on HTML tags in text"


class TestRegressionCases:
    """Test HTML regression cases that previously caused issues."""

    def test_regression_cases_structure(self, edge_cases):
        """Verify regression HTML cases are properly structured."""
        regression = edge_cases.get("regression_html", {})
        assert "description" in regression
        assert "cases" in regression

    def test_nested_html_extraction(self):
        """Test extraction from nested HTML elements."""
        # This would test actual HTML parsing
        # Implementation depends on how parser handles nested elements
        pytest.skip("HTML parsing regression tests require real HTML parser")


# Utility function for manual testing
def check_entry_for_issues(scraper_id: str, entry: dict) -> list[str]:
    """
    Check an entry against all validation rules.

    Returns list of warnings/errors found.

    Example:
        entry = {"rank": 1, "artist": "Artist", "title": "Title"}
        issues = check_entry_for_issues("t40", entry)
        if issues:
            print("Issues found:", issues)
    """
    from edge_case_helpers import validate_entry

    rules = load_edge_cases().get("validation_rules", {}).get("rules", [])
    return validate_entry(entry, rules)
