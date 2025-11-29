# Edge Case Testing - Quick Start Guide

## What is this?

A testing infrastructure that helps you:
- Capture problematic scraping patterns as fixtures
- Get clear error messages when unexpected patterns occur
- Prevent regressions by converting bugs into tests
- Easily add new edge cases as they're discovered

## Quick Start

### 1. When you find a problematic entry while scraping:

```python
from tests.edge_case_helpers import add_edge_case

# The problematic entry you discovered
entry = {
    "rank": 1,
    "artist": "Queen",
    "title": "We Will Rock You / We Are The Champions"
}

# What you expect the scraper to produce
expected = [
    {"rank": 1, "artist": "Queen", "title": "We Will Rock You"},
    {"rank": 1, "artist": "Queen", "title": "We Are The Champions"}
]

# Add it to the fixture
case_id = add_edge_case(
    scraper_id="t40",              # Which scraper?
    category="split_entries",       # What kind of issue?
    entry=entry,
    expected=expected,
    description="Double A-side with slash",
    status="pending"               # Change to "handled" once fixed
)

print(f"Added: {case_id}")
```

### 2. Run the tests (they will fail):

```bash
pytest tests/test_scraper_edge_cases.py -v
```

### 3. Fix your scraper to handle the edge case

### 4. Update the fixture status to "handled":

Edit `tests/fixtures/scraper_edge_cases.json` and change:
```json
"status": "pending"  →  "status": "handled"
```

### 5. Tests now pass!

```bash
pytest tests/test_scraper_edge_cases.py -v
# ✓ All tests pass
```

## Common Tasks

### Check if an entry has issues:

```python
from tests.edge_case_helpers import validate_entry

entry = {"rank": 1, "artist": "Artist", "title": "Title"}
warnings = validate_entry(entry)

if warnings:
    for warning in warnings:
        print(f"⚠ {warning}")
```

### Analyze a batch of scraped entries:

```python
from tests.edge_case_helpers import check_for_common_issues

entries = [
    (1, "Artist", "Title"),
    (2, "Artist 2", "Title 2"),
    # ... more entries
]

issues = check_for_common_issues(entries)

print(f"Long titles: {len(issues['long_titles'])}")
print(f"HTML in text: {len(issues['html_in_text'])}")
print(f"Duplicates: {len(issues['duplicates'])}")
```

### Add validation to your scraper:

```python
from chart_binder.scrapers.base import ChartScraper

class MyScraper(ChartScraper):
    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        html = self._fetch_url(url)
        entries = self._parse_html(html)

        # Validate all entries
        issues = self._validate_entries(entries, strict=False)

        # Check for critical problems
        if issues["html_in_text"]:
            logger.error("HTML parsing failed!")

        return entries
```

## Edge Case Categories

| Category | Description | Example |
|----------|-------------|---------|
| `split_entries` | Entries that should be split | "Song 1 / Song 2" |
| `malformed_entries` | Structural issues | Missing fields |
| `encoding_issues` | Character encoding | Special chars: é, ñ |
| `missing_fields` | Required data absent | Empty artist/title |

## Running Tests

```bash
# All edge case tests
pytest tests/test_scraper_edge_cases.py -v

# Specific scraper
pytest tests/test_scraper_edge_cases.py::TestTop40EdgeCases -v

# Infrastructure tests only
pytest tests/test_scraper_edge_cases.py::TestEdgeCaseInfrastructure -v

# See example usage
python tests/example_edge_case_usage.py
```

## File Structure

```
tests/
├── fixtures/
│   ├── scraper_edge_cases.json     # All edge cases stored here
│   └── README_EDGE_CASES.md        # Full documentation
├── test_scraper_edge_cases.py      # Test suite
├── edge_case_helpers.py            # Helper functions
├── example_edge_case_usage.py      # Examples
└── EDGE_CASE_QUICKSTART.md         # This file
```

## Helper Functions Reference

```python
from tests.edge_case_helpers import (
    add_edge_case,              # Add a new edge case
    add_regression_html,        # Add HTML snippet that broke
    validate_entry,             # Check one entry
    check_for_common_issues,    # Check batch of entries
    format_edge_case_error,     # Format helpful error message
    suggest_edge_case_category  # Auto-suggest category
)
```

## Validation in ChartScraper Base Class

All scrapers inherit these methods from `ChartScraper`:

```python
# Validate one entry
warnings = self._validate_entry(rank, artist, title, strict=False)

# Validate all entries
issues = self._validate_entries(entries, strict=False)
# Returns dict with: html_in_text, long_titles, duplicates, etc.
```

## Examples

See `tests/example_edge_case_usage.py` for:
- Adding split entry cases
- Validating entries
- Checking batches
- Adding regression HTML
- And more!

## Getting Help

1. **Full documentation**: Read `tests/fixtures/README_EDGE_CASES.md`
2. **Examples**: Run `python tests/example_edge_case_usage.py`
3. **Current edge cases**: Check `tests/fixtures/scraper_edge_cases.json`

## Workflow Summary

```
Discover issue → Add to fixtures → Run tests (fail) →
Fix scraper → Mark as "handled" → Tests pass → Commit
```

That's it! You now have a systematic way to handle scraper edge cases.
