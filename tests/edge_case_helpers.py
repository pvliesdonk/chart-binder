"""Helper utilities for working with scraper edge cases.

This module provides utilities for:
- Adding new edge cases programmatically
- Validating entries against known rules
- Detecting suspicious patterns in scraped data
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EDGE_CASES_FILE = FIXTURES_DIR / "scraper_edge_cases.json"


def load_edge_cases() -> dict:
    """Load edge cases from fixture file."""
    if EDGE_CASES_FILE.exists():
        return json.loads(EDGE_CASES_FILE.read_text())
    return {
        "version": "1.0",
        "description": "Known scraper edge cases that require special handling",
        "edge_cases": {},
        "validation_rules": {"rules": []},
        "regression_html": {"cases": []},
    }


def save_edge_cases(data: dict) -> None:
    """Save edge cases to fixture file."""
    FIXTURES_DIR.mkdir(exist_ok=True)

    # Update metadata
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["updated"] = datetime.now().strftime("%Y-%m-%d")

    # Pretty print JSON
    EDGE_CASES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def add_edge_case(
    scraper_id: str,
    category: str,
    entry: dict,
    expected: list | dict | None = None,
    description: str = "",
    notes: str = "",
    status: str = "pending",
) -> str:
    """Add a new edge case to the fixture file.

    Call this when an unexpected pattern is discovered during scraping.

    Args:
        scraper_id: Scraper identifier (e.g., "t40", "top2000")
        category: Category of edge case (e.g., "split_entries", "malformed_entries")
        entry: The problematic entry that was encountered
        expected: Expected output or behavior (optional)
        description: Human-readable description of the issue
        notes: Additional notes about this edge case
        status: Status of the case ("pending", "handled", "wontfix")

    Returns:
        The generated edge case ID

    Example:
        >>> entry = {"rank": 1, "artist": "Artist", "title": "Title / Other Title"}
        >>> case_id = add_edge_case(
        ...     "t40",
        ...     "split_entries",
        ...     entry,
        ...     expected=[
        ...         {"rank": 1, "artist": "Artist", "title": "Title"},
        ...         {"rank": 1, "artist": "Artist", "title": "Other Title"}
        ...     ],
        ...     description="Unexpected slash in title",
        ...     status="pending"
        ... )
        >>> print(f"Added edge case: {case_id}")
    """
    data = load_edge_cases()

    # Ensure scraper exists
    if scraper_id not in data["edge_cases"]:
        data["edge_cases"][scraper_id] = {
            "split_entries": [],
            "malformed_entries": [],
            "encoding_issues": [],
            "missing_fields": [],
        }

    # Ensure category exists
    if category not in data["edge_cases"][scraper_id]:
        data["edge_cases"][scraper_id][category] = []

    # Generate unique ID
    existing_ids = [
        case["id"]
        for case in data["edge_cases"][scraper_id][category]
        if isinstance(case, dict) and "id" in case
    ]

    # Extract number from IDs like "t40-split-001"
    max_num = 0
    pattern = f"{scraper_id}-{category.replace('_', '-')}-"
    for case_id in existing_ids:
        if case_id.startswith(pattern):
            try:
                num = int(case_id.split("-")[-1])
                max_num = max(max_num, num)
            except ValueError:
                pass

    new_id = f"{pattern}{max_num + 1:03d}"

    # Build edge case object
    edge_case = {
        "id": new_id,
        "description": description or f"Edge case discovered: {category}",
        "input": entry,
        "status": status,
        "added": datetime.now().strftime("%Y-%m-%d"),
    }

    if expected is not None:
        edge_case["expected"] = expected

    if notes:
        edge_case["notes"] = notes

    # Add to list
    data["edge_cases"][scraper_id][category].append(edge_case)

    # Save
    save_edge_cases(data)

    return new_id


def add_regression_html(
    scraper_id: str,
    html_snippet: str,
    description: str,
    expected_extraction: str | None = None,
    notes: str = "",
    status: str = "pending",
) -> str:
    """Add a regression HTML case.

    Args:
        scraper_id: Scraper identifier
        html_snippet: The HTML snippet that caused issues
        description: Description of the issue
        expected_extraction: What should be extracted from the HTML
        notes: Additional notes
        status: Status of the case

    Returns:
        The generated case ID
    """
    data = load_edge_cases()

    if "regression_html" not in data:
        data["regression_html"] = {
            "description": "HTML snippets that caused parsing issues",
            "cases": [],
        }

    existing_ids = [case.get("id", "") for case in data["regression_html"]["cases"]]

    max_num = 0
    for case_id in existing_ids:
        if case_id.startswith("html-regression-"):
            try:
                num = int(case_id.split("-")[-1])
                max_num = max(max_num, num)
            except ValueError:
                pass

    new_id = f"html-regression-{max_num + 1:03d}"

    regression_case = {
        "id": new_id,
        "scraper": scraper_id,
        "description": description,
        "html_snippet": html_snippet,
        "status": status,
        "added": datetime.now().strftime("%Y-%m-%d"),
    }

    if expected_extraction:
        regression_case["expected_extraction"] = expected_extraction

    if notes:
        regression_case["notes"] = notes

    data["regression_html"]["cases"].append(regression_case)
    save_edge_cases(data)

    return new_id


def validate_entry(entry: dict, rules: list[dict] | None = None) -> list[str]:
    """Validate an entry against validation rules.

    Args:
        entry: Entry to validate (dict with rank, artist, title)
        rules: Validation rules (if None, loads from fixture)

    Returns:
        List of warning/error messages

    Example:
        >>> entry = {"rank": 1, "artist": "A" * 200, "title": "Title"}
        >>> warnings = validate_entry(entry)
        >>> if warnings:
        ...     print("Issues found:", warnings)
    """
    if rules is None:
        data = load_edge_cases()
        rules = data.get("validation_rules", {}).get("rules", [])

    warnings: list[str] = []

    for rule in rules:
        rule_type = rule.get("rule", "")
        action = rule.get("action", "warn")

        # Title too long
        if rule_type == "title_too_long":
            threshold = rule.get("threshold", 200)
            title = entry.get("title", "")
            if len(title) > threshold:
                msg = f"[{action.upper()}] Title exceeds {threshold} characters ({len(title)} chars): {title[:50]}..."
                warnings.append(msg)

        # Artist too long
        elif rule_type == "artist_too_long":
            threshold = rule.get("threshold", 150)
            artist = entry.get("artist", "")
            if len(artist) > threshold:
                msg = f"[{action.upper()}] Artist name exceeds {threshold} characters ({len(artist)} chars): {artist[:50]}..."
                warnings.append(msg)

        # Excessive slashes
        elif rule_type == "excessive_slashes":
            threshold = rule.get("threshold", 5)
            title = entry.get("title", "")
            artist = entry.get("artist", "")
            slash_count = title.count("/") + artist.count("/")
            if slash_count > threshold:
                msg = f"[{action.upper()}] Excessive slashes detected ({slash_count} total) - possible parsing issue"
                warnings.append(msg)

        # Numeric-only title
        elif rule_type == "numeric_only_title":
            pattern = rule.get("pattern", r"^\d+$")
            title = entry.get("title", "")
            if re.match(pattern, title):
                msg = (
                    f"[{action.upper()}] Title is numeric-only: '{title}' - likely a parsing error"
                )
                warnings.append(msg)

        # HTML tags in text
        elif rule_type == "html_tags_in_text":
            pattern = rule.get("pattern", r"<[^>]+>")
            title = entry.get("title", "")
            artist = entry.get("artist", "")
            if re.search(pattern, title) or re.search(pattern, artist):
                msg = f"[{action.upper()}] HTML tags detected in text - parsing failed"
                warnings.append(msg)

    return warnings


def check_for_common_issues(entries: list[tuple[int, str, str]]) -> dict[str, list]:
    """Check a list of entries for common issues.

    Args:
        entries: List of (rank, artist, title) tuples

    Returns:
        Dict mapping issue types to lists of problematic entries

    Example:
        >>> entries = [(1, "Artist", "Title" * 100), (2, "Artist2", "Title2")]
        >>> issues = check_for_common_issues(entries)
        >>> if issues["long_titles"]:
        ...     print(f"Found {len(issues['long_titles'])} entries with long titles")
    """
    data = load_edge_cases()
    rules = data.get("validation_rules", {}).get("rules", [])

    issues: dict[str, list] = {
        "long_titles": [],
        "long_artists": [],
        "excessive_slashes": [],
        "numeric_titles": [],
        "html_in_text": [],
        "duplicates": [],
    }

    seen = set()

    for rank, artist, title in entries:
        entry = {"rank": rank, "artist": artist, "title": title}

        # Check for duplicates
        key = (rank, artist, title)
        if key in seen:
            issues["duplicates"].append(entry)
        seen.add(key)

        # Validate against rules
        warnings = validate_entry(entry, rules)

        for warning in warnings:
            if "Title exceeds" in warning:
                issues["long_titles"].append(entry)
            elif "Artist name exceeds" in warning:
                issues["long_artists"].append(entry)
            elif "slashes" in warning:
                issues["excessive_slashes"].append(entry)
            elif "numeric-only" in warning:
                issues["numeric_titles"].append(entry)
            elif "HTML tags" in warning:
                issues["html_in_text"].append(entry)

    return issues


def suggest_edge_case_category(entry: dict) -> str:
    """Suggest which category an edge case should belong to.

    Args:
        entry: The problematic entry

    Returns:
        Suggested category name

    Example:
        >>> entry = {"rank": 1, "artist": "", "title": "Title"}
        >>> category = suggest_edge_case_category(entry)
        >>> print(category)  # "missing_fields"
    """
    artist = entry.get("artist", "")
    title = entry.get("title", "")
    rank = entry.get("rank")

    # Missing fields
    if not artist or not title or rank is None:
        return "missing_fields"

    # Encoding issues (non-ASCII)
    if any(ord(c) > 127 for c in artist + title):
        return "encoding_issues"

    # Split entries (contains separators)
    if "/" in title or "/" in artist or ";" in title:
        return "split_entries"

    # Default to malformed
    return "malformed_entries"


def format_edge_case_error(scraper_id: str, entry: dict, reason: str) -> str:
    """Format a clear error message for an edge case.

    This is used by scrapers to generate helpful error messages.

    Args:
        scraper_id: Scraper identifier
        entry: The problematic entry
        reason: Why this is an issue

    Returns:
        Formatted error message
    """
    suggested_category = suggest_edge_case_category(entry)

    return f"""
================================================================================
SCRAPER EDGE CASE DETECTED
================================================================================
Scraper: {scraper_id}
Reason:  {reason}

Entry:
  Rank:   {entry.get("rank", "N/A")}
  Artist: {entry.get("artist", "N/A")}
  Title:  {entry.get("title", "N/A")}

Suggested Category: {suggested_category}

To fix this:
1. Add this entry to tests/fixtures/scraper_edge_cases.json
2. Implement handling in the scraper
3. Mark the edge case as "handled" in the fixture

Quick add with Python:
    from tests.edge_case_helpers import add_edge_case
    add_edge_case(
        "{scraper_id}",
        "{suggested_category}",
        {entry},
        description="{reason}",
        status="pending"
    )

Or manually add to tests/fixtures/scraper_edge_cases.json:
{{
  "id": "{scraper_id}-{suggested_category.replace("_", "-")}-XXX",
  "description": "{reason}",
  "input": {entry},
  "expected": [...],  # Define expected behavior
  "status": "pending"
}}
================================================================================
"""


# Convenience exports for common operations
__all__ = [
    "add_edge_case",
    "add_regression_html",
    "validate_entry",
    "check_for_common_issues",
    "suggest_edge_case_category",
    "format_edge_case_error",
    "load_edge_cases",
    "save_edge_cases",
]
