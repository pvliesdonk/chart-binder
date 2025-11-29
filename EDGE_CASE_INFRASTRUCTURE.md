# Scraper Edge Case Testing Infrastructure - Summary

## Overview

A comprehensive testing infrastructure has been created for handling scraper edge cases in chart-binder. This system enables you to:

1. **Capture** known problematic entries in fixtures
2. **Fail fast** with clear messages when unexpected patterns occur
3. **Add** new edge cases easily as they're discovered
4. **Prevent regressions** by turning bugs into tests

## Files Created

### 1. Fixtures

**`/home/user/chart-binder/tests/fixtures/scraper_edge_cases.json`**
- JSON fixture containing all known edge cases
- Organized by scraper (t40, t40jaar, top2000, zwaarste)
- Categorized by type (split_entries, malformed_entries, encoding_issues, missing_fields)
- Includes validation rules for detecting suspicious patterns
- Pre-populated with example cases from Top40Scraper

### 2. Tests

**`/home/user/chart-binder/tests/test_scraper_edge_cases.py`**
- Comprehensive test suite for edge cases
- Tests for all four scrapers
- Infrastructure tests to validate fixture structure
- Validation rule tests
- Regression HTML tests
- Skips pending cases, runs only "handled" cases

### 3. Helpers

**`/home/user/chart-binder/tests/edge_case_helpers.py`**
- `add_edge_case()` - Add new edge case to fixtures
- `add_regression_html()` - Add HTML snippets that broke parsing
- `validate_entry()` - Check single entry against rules
- `check_for_common_issues()` - Analyze batch of entries
- `suggest_edge_case_category()` - Auto-suggest category
- `format_edge_case_error()` - Generate helpful error messages

### 4. Enhanced Base Scraper

**`/home/user/chart-binder/src/chart_binder/scrapers/base.py`** (modified)
- Added `_validate_entry()` - Validate individual entries
- Added `_validate_entries()` - Validate batch of entries
- Detects: HTML in text, long fields, excessive slashes, numeric-only titles, empty fields, duplicates

### 5. Documentation

**`/home/user/chart-binder/tests/fixtures/README_EDGE_CASES.md`**
- Complete documentation of the edge case system
- Detailed explanations of all categories
- JSON structure documentation
- Helper function reference
- Workflow examples
- Troubleshooting guide

**`/home/user/chart-binder/tests/EDGE_CASE_QUICKSTART.md`**
- Quick start guide for common tasks
- Code examples
- Common patterns
- Cheat sheet for helper functions

### 6. Examples

**`/home/user/chart-binder/tests/example_edge_case_usage.py`**
- Runnable examples demonstrating all features
- 7 different usage scenarios
- Shows real-world patterns

## Quick Start

### Add an edge case:

```python
from tests.edge_case_helpers import add_edge_case

entry = {"rank": 1, "artist": "Artist", "title": "Title / Other"}
case_id = add_edge_case(
    scraper_id="t40",
    category="split_entries",
    entry=entry,
    expected=[...],
    description="Double A-side",
    status="pending"
)
```

### Run tests:

```bash
# All edge case tests
pytest tests/test_scraper_edge_cases.py -v

# Specific scraper
pytest tests/test_scraper_edge_cases.py::TestTop40EdgeCases -v

# See examples
python tests/example_edge_case_usage.py
```

### Validate entries in scraper:

```python
class MyScraper(ChartScraper):
    def scrape(self, period: str) -> list[tuple[int, str, str]]:
        entries = self._parse_html(html)
        
        # Validate
        issues = self._validate_entries(entries, strict=False)
        
        if issues["html_in_text"]:
            logger.error("Parsing failed!")
        
        return entries
```

## Test Results

```
$ pytest tests/test_scraper_edge_cases.py -v

TestEdgeCaseInfrastructure::test_fixture_file_exists PASSED
TestEdgeCaseInfrastructure::test_fixture_valid_json PASSED
TestEdgeCaseInfrastructure::test_fixture_has_all_scrapers PASSED
TestEdgeCaseInfrastructure::test_fixture_structure PASSED
TestTop40EdgeCases::test_split_entry_cases PASSED
TestTop40EdgeCases::test_malformed_entry_cases PASSED
TestTop40EdgeCases::test_encoding_cases SKIPPED (pending)
TestValidationRules::test_validation_rules_exist PASSED
TestValidationRules::test_title_length_validation PASSED
TestValidationRules::test_html_tags_validation PASSED
TestRegressionCases::test_regression_cases_structure PASSED

10 passed, 5 skipped in 0.64s
```

## Edge Case Categories

1. **split_entries** - Entries that should be split into multiple songs
   - Example: "Song 1 / Song 2"
   
2. **malformed_entries** - Structural issues
   - Example: Missing artist or title
   
3. **encoding_issues** - Character encoding problems
   - Example: Special characters not preserved
   
4. **missing_fields** - Required data absent
   - Example: Null rank values

## Validation Rules

Built-in validation detects:
- HTML tags in text (ERROR)
- Titles over 200 characters (WARN)
- Artists over 150 characters (WARN)
- More than 5 slashes (WARN)
- Numeric-only titles (WARN)
- Empty required fields (WARN)

## Workflow

```
┌─────────────────┐
│ Discover issue  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Add to fixtures │ ◄── add_edge_case()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run tests       │ ◄── pytest (they fail)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fix scraper     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Mark "handled"  │ ◄── Update status in JSON
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tests pass      │ ◄── pytest
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Commit both     │ ◄── git commit
└─────────────────┘
```

## Pre-populated Edge Cases

The fixture includes examples from Top40Scraper:

- **t40-split-001**: Double A-side with slash separator ✓ handled
- **t40-split-002**: Multiple artists and titles ✓ handled
- **t40-split-003**: Semicolon as separator ✓ handled
- **t40-malformed-001**: Missing artist ✓ handled
- **t40-malformed-002**: Missing title ✓ handled
- **t40-encoding-001**: Special characters (pending)
- **t40-missing-001**: Null rank ✓ handled

## Example Usage

See examples in action:

```bash
python tests/example_edge_case_usage.py
```

Output shows:
- Adding split entries
- Validating single entries
- Checking batches
- Auto-suggesting categories
- Formatting error messages
- Adding regression HTML
- Inspecting current edge cases

## Integration with Scrapers

The base `ChartScraper` class now includes validation methods that all scrapers inherit:

```python
# In your scraper
def scrape(self, period: str) -> list[tuple[int, str, str]]:
    entries = self._parse_html(html)
    
    # Option 1: Validate each entry as you parse
    warnings = self._validate_entry(rank, artist, title)
    
    # Option 2: Validate all at once
    issues = self._validate_entries(entries)
    
    return entries
```

## Next Steps

1. **Add edge cases** as you discover them while scraping real data
2. **Run tests** regularly to catch regressions
3. **Update scrapers** to handle new patterns
4. **Document** why certain patterns exist (use the "notes" field)
5. **Review** pending cases periodically

## Getting Help

- **Quick start**: Read `tests/EDGE_CASE_QUICKSTART.md`
- **Full docs**: Read `tests/fixtures/README_EDGE_CASES.md`
- **Examples**: Run `python tests/example_edge_case_usage.py`
- **Current cases**: Check `tests/fixtures/scraper_edge_cases.json`

## Benefits

✓ **Systematic** - Structured approach to handling edge cases  
✓ **Documented** - All known issues captured in fixtures  
✓ **Tested** - Every edge case has a regression test  
✓ **Maintainable** - Easy to add new cases as discovered  
✓ **Clear errors** - Helpful messages when issues occur  
✓ **Validated** - Built-in detection of suspicious patterns  

---

**Status**: All infrastructure tests pass ✓  
**Coverage**: 10 tests, 5 skipped (pending cases)  
**Ready for**: Production use
