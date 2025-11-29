# CLI Reference

Complete documentation for the `canon` command-line interface.

## Global Options

All commands support these global options:

```bash
canon [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|--------|-------------|
| `--config PATH` | Path to configuration TOML file |
| `--offline` | Run in offline mode (no network requests) |
| `--frozen` | Use only cached data, fail if cache miss |
| `--refresh` | Force refresh of cached data |
| `-o, --output [text\|json]` | Output format (default: text) |
| `--help` | Show help message |

## Commands Overview

| Command Group | Description |
|---------------|-------------|
| `scan` | Scan audio files and discover metadata |
| `decide` | Make canonicalization decisions |
| `write` | Write canonical tags to files |
| `cache` | Manage HTTP and entity caches |
| `charts` | Manage chart data |
| `coverage` | Generate coverage reports |
| `drift` | Manage decision drift |
| `llm` | LLM adjudication commands |
| `review` | Human review queue commands |

---

## canon scan

Scan audio files and discover existing metadata.

```bash
canon scan [OPTIONS] PATHS...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | One or more file or directory paths to scan (required) |

### Description

Reads existing tags from audio files and prints metadata information. Supports MP3, FLAC, OGG, and MP4/M4A files. When given directories, recursively searches for audio files.

### Examples

```bash
# Scan a single file
canon scan song.mp3

# Scan multiple files
canon scan track1.mp3 track2.flac

# Scan a directory recursively
canon scan /path/to/music/

# Output as JSON
canon -o json scan song.mp3
```

### Output Fields

| Field | Description |
|-------|-------------|
| `title` | Track title |
| `artist` | Artist name |
| `album` | Album name |
| `original_year` | Original release year |
| `mb_recording_id` | MusicBrainz Recording ID |
| `mb_release_group_id` | MusicBrainz Release Group ID |
| `mb_release_id` | MusicBrainz Release ID |
| `charts_blob` | Embedded chart data (if present) |
| `decision_trace` | Previous decision trace (if present) |

---

## canon decide

Make canonicalization decisions for audio files.

```bash
canon decide [OPTIONS] PATHS...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | One or more file or directory paths (required) |

### Options

| Option | Description |
|--------|-------------|
| `--explain` | Show detailed decision rationale |

### Description

Analyzes audio files and resolves the Canonical Release Group (CRG) and Representative Release (RR) based on available metadata and chart data.

### Decision States

| State | Description |
|-------|-------------|
| `decided` | Clear canonical release selected |
| `indeterminate` | Ambiguous case requiring review |
| `blocked` | Missing required information |

### Examples

```bash
# Make decisions
canon decide song.mp3

# Show detailed explanation
canon decide --explain song.mp3

# Process a directory
canon decide /path/to/music/

# JSON output for scripting
canon -o json decide song.mp3
```

### Output Fields

| Field | Description |
|-------|-------------|
| `state` | Decision state |
| `crg_mbid` | Canonical Release Group MBID |
| `rr_mbid` | Representative Release MBID |
| `crg_rationale` | Reason for CRG selection |
| `rr_rationale` | Reason for RR selection |
| `compact_tag` | Compact decision trace |

---

## canon write

Write canonical tags to audio files.

```bash
canon write [OPTIONS] PATHS...
```

### Arguments

| Argument | Description |
|----------|-------------|
| `PATHS` | One or more file or directory paths (required) |

### Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview changes without writing |
| `--apply` | Apply changes (required for actual writes) |

### Description

Writes decision trace, canonical IDs, and optionally CHARTS blob to audio files. One of `--dry-run` or `--apply` must be specified as a safety feature.

### Examples

```bash
# Preview changes
canon write --dry-run song.mp3

# Apply changes
canon write --apply song.mp3

# Process directory
canon write --apply /path/to/music/
```

### Output Fields

| Field | Description |
|-------|-------------|
| `fields_written` | List of fields that were written |
| `fields_skipped` | List of fields that were skipped |
| `originals_stashed` | Original values that were backed up |
| `errors` | Any errors that occurred |

---

## canon cache

Manage HTTP and entity caches.

### canon cache status

Show cache status and statistics.

```bash
canon cache status
```

### Output Fields

| Field | Description |
|-------|-------------|
| `cache_directory` | Path to cache directory |
| `cache_enabled` | Whether caching is enabled |
| `ttl_seconds` | Time-to-live for cache entries |
| `entries` | Number of cached entries |
| `expired_entries` | Number of expired entries |
| `total_size_bytes` | Total cache size |

### canon cache purge

Clear caches.

```bash
canon cache purge [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--expired-only` | Only purge expired entries |
| `--force` | Skip confirmation prompt |

### Examples

```bash
# Show cache status
canon cache status

# Purge expired entries
canon cache purge --expired-only

# Clear all caches
canon cache purge --force
```

---

## canon charts

Manage chart data.

### canon charts scrape

Scrape chart data from web sources.

```bash
canon charts scrape [OPTIONS] CHART_TYPE PERIOD
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CHART_TYPE` | Chart to scrape: `t40`, `t40jaar`, `top2000`, `zwaarste` |
| `PERIOD` | Period to scrape (format depends on chart type) |

### Chart Types

| Type | Description | Period Format | Example |
|------|-------------|---------------|---------|
| `t40` | Dutch Top 40 weekly chart | `YYYY-Www` | `2024-W01` |
| `t40jaar` | Dutch Top 40 year-end chart | `YYYY` | `2023` |
| `top2000` | NPO Radio 2 Top 2000 | `YYYY` | `2024` |
| `zwaarste` | 538 De Zwaarste Lijst | `YYYY` | `2024` |

### Options

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Output JSON file (optional) |
| `--ingest` | Automatically ingest scraped data into database |
| `--strict` | Fail if entry count is below expected (sanity check) |

### Entry Count Validation

Each chart type has an expected entry count. The scraper validates results and warns if counts are low:

| Chart | Expected Entries |
|-------|------------------|
| `t40` | 40 |
| `t40jaar` | 100 |
| `top2000` | 2000 |
| `zwaarste` | ~150 |

With `--strict`, the command fails if entries are >10% below expected (possible edge case).

### Examples

```bash
# Scrape Dutch Top 40 week 1 of 2024
canon charts scrape t40 2024-W01

# Scrape and automatically ingest into database
canon charts scrape t40 2024-W01 --ingest

# Scrape with strict validation (fail on low count)
canon charts scrape top2000 2024 --strict

# Scrape year-end chart to file
canon charts scrape t40jaar 2023 -o top40_2023.json

# Scrape Top 2000 with JSON output
canon -o json charts scrape top2000 2024
```

### Output

Without `-o` flag, displays first 10 entries with validation status. With `-o` flag, saves full results to JSON file.

Output shows actual/expected entry counts:
```
✔︎ Scraped 40/40 entries for t40 2024-W01
```

Warning if below expected:
```
⚠ Entry count sanity check failed: got 35, expected ~40 (shortage: 5)
```

---

### canon charts scrape-missing

Scrape all missing periods for a chart type.

```bash
canon charts scrape-missing [OPTIONS] CHART_TYPE
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CHART_TYPE` | Chart to scrape: `t40`, `t40jaar`, `top2000`, `zwaarste` |

### Options

| Option | Description |
|--------|-------------|
| `--start-year INT` | Start year (default: earliest available for chart) |
| `--end-year INT` | End year (default: current year) |
| `--ingest` | Automatically ingest scraped data |
| `--strict` | Fail on entry count sanity check failures |
| `--dry-run` | Show what would be scraped without scraping |

### Examples

```bash
# Show what's missing for Top 40 weekly (dry run)
canon charts scrape-missing t40 --start-year 2020 --dry-run

# Scrape all missing year-end charts and ingest
canon charts scrape-missing t40jaar --ingest

# Scrape missing Top 2000 years from 1999-2024
canon charts scrape-missing top2000 --start-year 1999 --end-year 2024 --ingest
```

### Output

Shows progress for each period:
```
Chart: t40 (nl_top40)
Range: 2020 - 2024
Expected periods: 260
Existing: 150
Missing: 110

Scraping 110 missing periods...
  [1/110] 2020-W01: ✔︎ 40/40 entries
  [2/110] 2020-W02: ✔︎ 40/40 entries
  ...

Summary: 110 scraped, 0 failed, 0 skipped
Ingested: 110 runs
```

---

### canon charts ingest

Ingest chart data from a source file.

```bash
canon charts ingest [OPTIONS] CHART_ID PERIOD SOURCE_FILE
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CHART_ID` | Chart identifier (e.g., nl_top40, uk_singles) |
| `PERIOD` | Chart period (e.g., 2024-W01, 2024-01) |
| `SOURCE_FILE` | JSON file with chart entries |

### Options

| Option | Description |
|--------|-------------|
| `--notes TEXT` | Notes about this chart run |

### Source File Format

JSON array of entries: `[[rank, artist, title], ...]`

```json
[
  [1, "Artist One", "Song Title"],
  [2, "Artist Two", "Another Song"]
]
```

### Example

```bash
canon charts ingest nl_top40 2024-W01 chart_data.json --notes "Week 1 data"
```

### canon charts link

Link chart entries to work keys.

```bash
canon charts link [OPTIONS] CHART_ID PERIOD
```

### Arguments

| Argument | Description |
|----------|-------------|
| `CHART_ID` | Chart identifier |
| `PERIOD` | Chart period |

### Options

| Option | Description |
|--------|-------------|
| `--strategy TEXT` | Linking strategy (default: title_artist_year) |

### Example

```bash
canon charts link nl_top40 2024-W01
```

### canon charts missing

Show unlinked chart entries.

```bash
canon charts missing CHART_ID PERIOD
```

### canon charts export

Export CHARTS blob for a work key.

```bash
canon charts export [OPTIONS] WORK_KEY
```

### Arguments

| Argument | Description |
|----------|-------------|
| `WORK_KEY` | Work key (format: "artist // title") |

### Options

| Option | Description |
|--------|-------------|
| `--positions / --no-positions` | Include position details (default: no) |

### Example

```bash
canon charts export "Billie Eilish // Bad Guy"
canon charts export --positions "Billie Eilish // Bad Guy"
```

---

## canon coverage

Generate coverage reports.

### canon coverage chart

Show coverage report for a chart run.

```bash
canon coverage chart CHART_ID PERIOD
```

### Output Fields

| Field | Description |
|-------|-------------|
| `total_entries` | Total number of chart entries |
| `linked_entries` | Number of linked entries |
| `unlinked_entries` | Number of unlinked entries |
| `coverage_pct` | Coverage percentage |
| `by_method` | Breakdown by linking method |
| `by_confidence` | Breakdown by confidence level |

### canon coverage missing

Show entries missing from chart linkage.

```bash
canon coverage missing [OPTIONS] CHART_ID PERIOD
```

### Options

| Option | Description |
|--------|-------------|
| `--threshold FLOAT` | Minimum confidence threshold (default: 0.60) |

---

## canon drift

Manage decision drift detection.

### canon drift review

Review decisions that have drifted.

```bash
canon drift review
```

Shows decisions where:
- Evidence has changed since the decision was made
- Ruleset version has been updated
- External data sources have new information

---

## canon llm

LLM adjudication commands for handling ambiguous cases.

### canon llm status

Show LLM configuration and provider status.

```bash
canon llm status
```

### Output Fields

| Field | Description |
|-------|-------------|
| `enabled` | Whether LLM is enabled |
| `provider` | Provider type (ollama/openai) |
| `model_id` | Model identifier |
| `provider_available` | Whether provider is reachable |
| `auto_accept_threshold` | Confidence threshold for auto-accept |
| `review_threshold` | Confidence threshold for review queue |

### canon llm adjudicate

Run LLM adjudication on a specific decision.

```bash
canon llm adjudicate [OPTIONS] FILE_ID
```

### Arguments

| Argument | Description |
|----------|-------------|
| `FILE_ID` | File identifier from decisions database |

### Options

| Option | Description |
|--------|-------------|
| `--force` | Adjudicate even if LLM is disabled |

### Output Fields

| Field | Description |
|-------|-------------|
| `outcome` | Adjudication outcome |
| `crg_mbid` | Suggested CRG MBID |
| `rr_mbid` | Suggested RR MBID |
| `confidence` | Confidence score |
| `rationale` | Explanation for the decision |
| `model_id` | Model used |
| `error` | Error message (if any) |

---

## canon review

Human review queue commands for handling edge cases.

### canon review list

List pending review items.

```bash
canon review list [OPTIONS]
```

### Options

| Option | Description |
|--------|-------------|
| `--source [indeterminate\|llm_review\|conflict]` | Filter by source |
| `--limit INT` | Maximum items to show (default: 20) |

### canon review show

Show details of a review item.

```bash
canon review show REVIEW_ID
```

### canon review accept

Accept a review with specific CRG/RR selection.

```bash
canon review accept [OPTIONS] REVIEW_ID
```

### Options

| Option | Description |
|--------|-------------|
| `--crg MBID` | CRG MBID to accept (required) |
| `--rr MBID` | RR MBID to accept |
| `--notes TEXT` | Review notes |

### Example

```bash
canon review accept abc123 --crg b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d
```

### canon review accept-llm

Accept LLM suggestion for a review item.

```bash
canon review accept-llm [OPTIONS] REVIEW_ID
```

### Options

| Option | Description |
|--------|-------------|
| `--notes TEXT` | Review notes |

### canon review skip

Skip a review item.

```bash
canon review skip [OPTIONS] REVIEW_ID
```

### Options

| Option | Description |
|--------|-------------|
| `--notes TEXT` | Reason for skipping |

### canon review stats

Show review queue statistics.

```bash
canon review stats
```

### Output Fields

| Field | Description |
|-------|-------------|
| `pending` | Number of pending reviews |
| `pending_by_source` | Breakdown by source |
| `completed` | Number of completed reviews |
| `completed_by_action` | Breakdown by action |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error occurred |
| 2 | No results found |

---

## Examples

### Complete Workflow

```bash
# 1. Scan your music library
canon scan /path/to/music/ --output json > scan.json

# 2. Make decisions
canon decide /path/to/music/

# 3. Review any indeterminate cases
canon review list

# 4. Handle reviews
canon review show abc123
canon review accept abc123 --crg <mbid>

# 5. Preview tag changes
canon write --dry-run /path/to/music/

# 6. Apply changes
canon write --apply /path/to/music/
```

### Working with Charts

```bash
# Scrape chart data from web sources
canon charts scrape t40 2024-W01 -o week1.json
canon charts scrape t40 2024-W02 -o week2.json

# Or scrape year-end lists
canon charts scrape t40jaar 2023 -o top40_2023.json
canon charts scrape top2000 2024 -o top2000_2024.json

# Import scraped data into database
canon charts ingest nl_top40 2024-W01 week1.json
canon charts ingest nl_top40 2024-W02 week2.json

# Link all entries to MusicBrainz
canon charts link nl_top40 2024-W01
canon charts link nl_top40 2024-W02

# Check coverage
canon coverage chart nl_top40 2024-W01
canon coverage missing nl_top40 2024-W01 --threshold 0.5
```

### Offline Operations

```bash
# First, build up cache online
canon scan /path/to/music/

# Later, work offline
canon --offline decide /path/to/music/

# Or be strict about cache
canon --frozen scan /path/to/music/
```
