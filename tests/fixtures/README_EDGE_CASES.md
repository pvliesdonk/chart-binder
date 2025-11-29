# Scraper Edge Case Testing Infrastructure

This directory contains fixtures and infrastructure for testing scraper edge cases.

## Overview

The edge case testing system helps you:
1. **Capture** problematic scraping patterns as they're discovered
2. **Document** why certain patterns are problematic
3. **Test** that scrapers handle edge cases correctly
4. **Prevent regressions** by turning bugs into tests

## Files

- **`scraper_edge_cases.json`** - Fixture file containing all known edge cases
- **`README_EDGE_CASES.md`** - This documentation

## Quick Start

### When you encounter a new edge case:

```python
# Option 1: Add manually to scraper_edge_cases.json
# See structure below

# Option 2: Add programmatically
from tests.edge_case_helpers import add_edge_case

entry = {"rank": 1, "artist": "Artist", "title": "Title / Other Title"}
case_id = add_edge_case(
    scraper_id="t40",
    category="split_entries",
    entry=entry,
    expected=[
        {"rank": 1, "artist": "Artist", "title": "Title"},
        {"rank": 1, "artist": "Artist", "title": "Other Title"}
    ],
    description="Unexpected slash separator",
    notes="Common in 1970s charts",
    status="pending"  # Change to "handled" once fixed
)
```

### Run edge case tests:

```bash
# Run all edge case tests
pytest tests/test_scraper_edge_cases.py -v

# Run tests for specific scraper
pytest tests/test_scraper_edge_cases.py::TestTop40EdgeCases -v

# Run only tests for handled cases
pytest tests/test_scraper_edge_cases.py -k "handled"
```

## Edge Case Categories

### 1. Split Entries
Entries that should be split into multiple songs:
- Double A-sides: "Song 1 / Song 2"
- Multiple artists: "Artist A / Artist B"
- Semicolon separators: "Song 1; Song 2"

**Status:**
- `handled` - Scraper correctly handles this case
- `pending` - Known issue, not yet fixed
- `wontfix` - Won't handle (document why)

### 2. Malformed Entries
Entries with structural issues:
- Missing artist or title
- Invalid rank values
- Truncated text

### 3. Encoding Issues
Character encoding problems:
- Special characters: é, ñ, ü
- Non-Latin scripts
- Emoji in titles

### 4. Missing Fields
Entries missing required data:
- Null/None values
- Empty strings
- Type mismatches

## JSON Structure

```json
{
  "edge_cases": {
    "t40": {
      "split_entries": [
        {
          "id": "t40-split-001",
          "description": "Human-readable description",
          "input": {
            "rank": 1,
            "artist": "Artist",
            "title": "Title"
          },
          "expected": [
            {"rank": 1, "artist": "Artist", "title": "Title 1"},
            {"rank": 1, "artist": "Artist", "title": "Title 2"}
          ],
          "status": "handled",
          "notes": "Additional context"
        }
      ]
    }
  }
}
```

## Validation Rules

Scrapers automatically validate entries against rules defined in `validation_rules`:

```json
{
  "validation_rules": {
    "rules": [
      {
        "id": "validation-001",
        "rule": "title_too_long",
        "threshold": 200,
        "action": "warn"
      }
    ]
  }
}
```

**Actions:**
- `warn` - Log warning but continue
- `error` - Fail validation and stop

## Using Validation in Scrapers

```python
from chart_binder.scrapers.base import ChartScraper

class MyScraper(ChartScraper):
    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        entries = self._parse_html(html)

        # Validate all entries
        issues = self._validate_entries(entries, strict=False)

        # Check for critical issues
        if issues["html_in_text"]:
            logger.error(f"HTML parsing failed: {issues['html_in_text']}")

        return entries
```

## Regression HTML Cases

Track HTML snippets that caused parsing failures:

```python
from tests.edge_case_helpers import add_regression_html

html = '<div class="title">Title <span>metadata</span></div>'
case_id = add_regression_html(
    scraper_id="t40",
    html_snippet=html,
    description="Nested span in title element",
    expected_extraction="Title metadata",
    notes="Parser should extract all text content"
)
```

## Helper Functions

### `add_edge_case()`
Add a new edge case to the fixture file.

```python
from tests.edge_case_helpers import add_edge_case

case_id = add_edge_case(
    scraper_id="t40",           # Scraper identifier
    category="split_entries",   # Category
    entry={...},                # Problematic entry
    expected=[...],             # Expected output
    description="...",          # What's wrong
    notes="...",               # Additional context
    status="pending"           # pending/handled/wontfix
)
```

### `validate_entry()`
Check a single entry against validation rules.

```python
from tests.edge_case_helpers import validate_entry

entry = {"rank": 1, "artist": "A" * 200, "title": "Title"}
warnings = validate_entry(entry)

if warnings:
    for warning in warnings:
        print(warning)
```

### `check_for_common_issues()`
Analyze a batch of entries for common problems.

```python
from tests.edge_case_helpers import check_for_common_issues

entries = [(1, "Artist", "Title"), ...]
issues = check_for_common_issues(entries)

print(f"Found {len(issues['long_titles'])} entries with long titles")
print(f"Found {len(issues['duplicates'])} duplicate entries")
```

### `format_edge_case_error()`
Generate a helpful error message for manual reporting.

```python
from tests.edge_case_helpers import format_edge_case_error

entry = {"rank": 1, "artist": "", "title": "Title"}
error_msg = format_edge_case_error("t40", entry, "Missing artist")
print(error_msg)  # Formatted message with instructions
```

## Workflow Example

### Discovering a New Edge Case

1. **Run scraper and encounter issue:**
   ```python
   scraper = Top40Scraper(cache)
   entries = scraper.scrape("2024-W01")
   # Notice weird entry: (1, "Artist / Other", "Title / Other Title")
   ```

2. **Add to fixtures:**
   ```python
   from tests.edge_case_helpers import add_edge_case

   problematic_entry = {
       "rank": 1,
       "artist": "Artist / Other",
       "title": "Title / Other Title"
   }

   add_edge_case(
       scraper_id="t40",
       category="split_entries",
       entry=problematic_entry,
       expected=[
           {"rank": 1, "artist": "Artist", "title": "Title"},
           {"rank": 1, "artist": "Other", "title": "Other Title"}
       ],
       description="Multiple artists and titles with slash separators",
       status="pending"
   )
   ```

3. **Run tests (they fail):**
   ```bash
   pytest tests/test_scraper_edge_cases.py::TestTop40EdgeCases::test_split_entry_cases -v
   # FAILED - case not handled yet
   ```

4. **Fix the scraper:**
   ```python
   # Update Top40Scraper._handle_split_entries() to handle this case
   ```

5. **Update fixture status:**
   ```json
   {
     "id": "t40-split-004",
     "status": "handled"  // Changed from "pending"
   }
   ```

6. **Tests pass:**
   ```bash
   pytest tests/test_scraper_edge_cases.py::TestTop40EdgeCases::test_split_entry_cases -v
   # PASSED
   ```

7. **Commit both fix and test:**
   ```bash
   git add src/chart_binder/scrapers/top40.py
   git add tests/fixtures/scraper_edge_cases.json
   git commit -m "fix(scraper): handle multiple artist/title splits"
   ```

## Best Practices

1. **Add cases before fixing** - Document the problem first
2. **Use descriptive IDs** - `t40-split-001` is better than `case1`
3. **Include notes** - Explain context and why it matters
4. **Update status** - Mark cases as `handled` once fixed
5. **Test edge cases** - Run tests before committing
6. **Keep fixtures updated** - Review and clean up periodically

## Maintenance

### Reviewing Edge Cases

```bash
# List all pending cases
python -c "
import json
from pathlib import Path

data = json.loads(Path('tests/fixtures/scraper_edge_cases.json').read_text())
for scraper, cases in data['edge_cases'].items():
    for category, items in cases.items():
        for item in items:
            if isinstance(item, dict) and item.get('status') == 'pending':
                print(f\"{item['id']}: {item.get('description', 'No description')}\")
"
```

### Adding Validation Rules

Edit `scraper_edge_cases.json`:

```json
{
  "validation_rules": {
    "rules": [
      {
        "id": "validation-new",
        "rule": "contains_year",
        "pattern": "\\b(19|20)\\d{2}\\b",
        "action": "warn",
        "notes": "Year in title might indicate metadata"
      }
    ]
  }
}
```

## Troubleshooting

### Tests are skipped
- Check `status` field - only `"handled"` cases run
- Verify fixture file is valid JSON
- Ensure scraper has necessary methods

### Validation not working
- Check logger level: `logging.basicConfig(level=logging.WARNING)`
- Verify `_validate_entries()` is called in scraper
- Confirm rules are properly formatted in JSON

### Can't add edge case
- Ensure `tests/fixtures/` directory exists
- Check file permissions on `scraper_edge_cases.json`
- Verify scraper ID matches one of: t40, t40jaar, top2000, zwaarste

## See Also

- `tests/test_scraper_edge_cases.py` - Test implementation
- `tests/edge_case_helpers.py` - Helper functions
- `src/chart_binder/scrapers/base.py` - Validation methods in ChartScraper
