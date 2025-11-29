#!/usr/bin/env python3
"""Example usage of the scraper edge case testing infrastructure.

This script demonstrates how to:
1. Add edge cases programmatically
2. Validate scraped entries
3. Check for common issues
4. Use the helper functions

Run this to understand the edge case system before using it in production.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.edge_case_helpers import (
    check_for_common_issues,
    format_edge_case_error,
    load_edge_cases,
    suggest_edge_case_category,
    validate_entry,
)


def example_1_add_split_entry():
    """Example: Adding a split entry edge case."""
    print("=" * 80)
    print("Example 1: Adding a Split Entry Edge Case")
    print("=" * 80)

    # You discovered this problematic entry while scraping
    problematic_entry = {
        "rank": 1,
        "artist": "Queen",
        "title": "Bohemian Rhapsody / You're My Best Friend",
    }

    # What you expect the scraper to produce
    expected_output = [
        {"rank": 1, "artist": "Queen", "title": "Bohemian Rhapsody"},
        {"rank": 1, "artist": "Queen", "title": "You're My Best Friend"},
    ]

    # Add to edge case fixtures (commented out to avoid modifying fixtures)
    # case_id = add_edge_case(
    #     scraper_id="t40",
    #     category="split_entries",
    #     entry=problematic_entry,
    #     expected=expected_output,
    #     description="Classic double A-side with slash separator",
    #     notes="Very common in 1970s charts",
    #     status="pending"  # Will change to "handled" once fixed
    # )
    # print(f"Added edge case: {case_id}")

    print(f"Entry: {problematic_entry}")
    print(f"Expected: {expected_output}")
    print("\nThis would be added with status='pending' until the scraper is fixed.")
    print()


def example_2_validate_single_entry():
    """Example: Validating a single entry."""
    print("=" * 80)
    print("Example 2: Validating a Single Entry")
    print("=" * 80)

    # Some problematic entries
    test_entries = [
        {
            "name": "Normal entry",
            "entry": {"rank": 1, "artist": "The Beatles", "title": "Hey Jude"},
        },
        {
            "name": "Suspiciously long title",
            "entry": {"rank": 2, "artist": "Artist", "title": "T" * 250},
        },
        {
            "name": "HTML in text",
            "entry": {"rank": 3, "artist": "Artist", "title": "Title <span>text</span>"},
        },
        {
            "name": "Too many slashes",
            "entry": {
                "rank": 4,
                "artist": "A / B / C / D",
                "title": "1 / 2 / 3 / 4 / 5",
            },
        },
    ]

    for test in test_entries:
        print(f"\nValidating: {test['name']}")
        print(f"Entry: {test['entry']}")

        warnings = validate_entry(test["entry"])

        if warnings:
            print(f"  Issues found ({len(warnings)}):")
            for warning in warnings:
                print(f"    - {warning}")
        else:
            print("  ✓ No issues found")

    print()


def example_3_check_batch_of_entries():
    """Example: Checking a batch of entries for common issues."""
    print("=" * 80)
    print("Example 3: Checking a Batch of Entries")
    print("=" * 80)

    # Simulated scraper output
    scraped_entries = [
        (1, "The Beatles", "Hey Jude"),
        (2, "Queen", "Bohemian Rhapsody"),
        (3, "Artist", "Title" * 100),  # Suspiciously long
        (4, "Artist", "123"),  # Numeric-only title
        (5, "The Beatles", "Hey Jude"),  # Duplicate
        (6, "Artist <span>", "Title"),  # HTML in artist
        (7, "A / B / C / D / E / F", "Song"),  # Too many slashes
    ]

    print(f"Checking {len(scraped_entries)} entries...\n")

    issues = check_for_common_issues(scraped_entries)

    # Report findings
    for issue_type, problematic_entries in issues.items():
        if problematic_entries:
            print(f"{issue_type}: {len(problematic_entries)} found")
            for rank, artist, title in problematic_entries[:3]:  # Show first 3
                print(f"  - Rank {rank}: {artist} - {title[:50]}")

    total_issues = sum(len(v) for v in issues.values())
    print(f"\nTotal issues found: {total_issues}")
    print()


def example_4_suggest_category():
    """Example: Auto-suggesting the right category for an edge case."""
    print("=" * 80)
    print("Example 4: Auto-Suggesting Edge Case Categories")
    print("=" * 80)

    test_cases = [
        {"rank": 1, "artist": "", "title": "Title"},
        {"rank": 2, "artist": "Artist", "title": "Title / Other"},
        {"rank": 3, "artist": "Café Tacvba", "title": "Ërës Tú"},
        {"rank": 4, "artist": "Normal", "title": "Normal"},
    ]

    for entry in test_cases:
        category = suggest_edge_case_category(entry)
        print(f"Entry: {entry}")
        print(f"  Suggested category: {category}\n")

    print()


def example_5_format_error_message():
    """Example: Formatting a helpful error message."""
    print("=" * 80)
    print("Example 5: Formatting Error Messages")
    print("=" * 80)

    entry = {"rank": 1, "artist": "", "title": "Some Title"}

    error_msg = format_edge_case_error(
        scraper_id="t40", entry=entry, reason="Artist field is empty"
    )

    print(error_msg)
    print()


def example_6_add_regression_html():
    """Example: Adding an HTML regression case."""
    print("=" * 80)
    print("Example 6: Adding HTML Regression Cases")
    print("=" * 80)

    html_snippet = """
    <div class="top40-list__title">
        Title
        <span class="metadata">feat. Artist</span>
    </div>
    """

    # This would add to the fixtures (commented out)
    # case_id = add_regression_html(
    #     scraper_id="t40",
    #     html_snippet=html_snippet.strip(),
    #     description="Nested span element in title",
    #     expected_extraction="Title feat. Artist",
    #     notes="Parser should extract all text from nested elements",
    #     status="pending"
    # )
    # print(f"Added HTML regression case: {case_id}")

    print("HTML snippet:")
    print(html_snippet)
    print("\nExpected extraction: 'Title feat. Artist'")
    print("This documents HTML parsing edge cases for regression testing.")
    print()


def example_7_load_and_inspect():
    """Example: Loading and inspecting current edge cases."""
    print("=" * 80)
    print("Example 7: Inspecting Current Edge Cases")
    print("=" * 80)

    data = load_edge_cases()

    print(f"Edge case fixture version: {data.get('version', 'unknown')}")
    print(f"Description: {data.get('description', 'N/A')}\n")

    print("Scrapers with edge cases:")
    for scraper_id, cases in data.get("edge_cases", {}).items():
        total_cases = sum(len(v) if isinstance(v, list) else 0 for v in cases.values())
        print(f"  {scraper_id}: {total_cases} cases")

        for category, items in cases.items():
            if isinstance(items, list) and items:
                handled = sum(1 for item in items if item.get("status") == "handled")
                pending = sum(1 for item in items if item.get("status") == "pending")
                if handled or pending:
                    print(
                        f"    - {category}: {len(items)} total ({handled} handled, {pending} pending)"
                    )

    print("\nValidation rules:")
    rules = data.get("validation_rules", {}).get("rules", [])
    for rule in rules[:5]:  # Show first 5
        print(f"  - {rule.get('id')}: {rule.get('rule')} (action: {rule.get('action')})")

    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("SCRAPER EDGE CASE TESTING INFRASTRUCTURE - EXAMPLES")
    print("=" * 80 + "\n")

    examples = [
        example_1_add_split_entry,
        example_2_validate_single_entry,
        example_3_check_batch_of_entries,
        example_4_suggest_category,
        example_5_format_error_message,
        example_6_add_regression_html,
        example_7_load_and_inspect,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error running {example.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 80)
    print("Examples complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review tests/fixtures/scraper_edge_cases.json")
    print("2. Run: pytest tests/test_scraper_edge_cases.py -v")
    print("3. Add your own edge cases as you discover them")
    print("4. Read: tests/fixtures/README_EDGE_CASES.md for full documentation")
    print()


if __name__ == "__main__":
    main()
