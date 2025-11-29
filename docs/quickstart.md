# Quick Start Guide

This guide will help you get started with Chart-Binder in just a few minutes.

## Prerequisites

Before you begin, ensure you have:

- Python 3.11 or higher installed
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- Some audio files (MP3, FLAC, OGG, or M4A) to work with

## Installation

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/pvliesdonk/chart-binder.git
cd chart-binder

# Install dependencies (includes beets plugin support)
uv sync --all-extras

# Verify installation
uv run canon --help
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/pvliesdonk/chart-binder.git
cd chart-binder

# Install in editable mode
pip install -e .

# Verify installation
canon --help
```

## Your First Scan

Let's start by scanning an audio file to see what metadata it contains:

```bash
# Scan a single file
canon scan path/to/song.mp3
```

Example output:

```
✔︎ path/to/song.mp3
  Title: Bad Guy
  Artist: Billie Eilish
  Album: WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?
  Year: 2019
  MB RG: b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d
```

### Scanning Multiple Files

```bash
# Scan all files in a directory
canon scan /path/to/music/

# Scan with JSON output for scripting
canon scan --output json /path/to/music/
```

## Making Decisions

The `decide` command analyzes your files and determines the canonical release group and representative release:

```bash
# Basic decision
canon decide path/to/song.mp3

# With detailed explanation
canon decide --explain path/to/song.mp3
```

Example output:

```
✔︎ path/to/song.mp3
  State: decided
  CRG: b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d
       (album_with_chart_entry)
  RR:  a1b2c3d4-e5f6-7890-abcd-ef1234567890
       (preferred_country_nl)
  Trace: CB/1.0/d/ACE/RRnl
```

### Understanding Decision States

| State | Meaning |
|-------|---------|
| `decided` | A clear canonical release was selected |
| `indeterminate` | Ambiguous case, may need LLM or human review |
| `blocked` | Missing required information |

## Writing Tags

After making decisions, you can write the canonical tags to your files:

```bash
# Preview changes without writing (always do this first!)
canon write --dry-run path/to/song.mp3

# Actually write the changes
canon write --apply path/to/song.mp3
```

Example dry-run output:

```
✔︎ path/to/song.mp3 (dry run)
  Written: CANON_CRG_MBID, CANON_RR_MBID, CANON_TRACE
  Stashed: none

Processed 1 files, 0 errors
```

## Working with Charts

Chart data enhances decision-making by providing historical context about when songs charted and in what positions.

### Step 1: Scrape or Prepare Chart Data

**Option A: Scrape from web sources (recommended)**

```bash
# Dutch Top 40 weekly chart
canon charts scrape t40 2024-W01 -o chart_data.json

# Year-end charts
canon charts scrape t40jaar 2023 -o top40_2023.json

# Other supported charts
canon charts scrape top2000 2024 -o top2000.json     # NPO Radio 2 Top 2000
canon charts scrape zwaarste 2024 -o zwaarste.json   # 538 De Zwaarste Lijst
```

**Option B: Create JSON manually**

Create a JSON file with your chart data:

```json
[
  [1, "Billie Eilish", "Bad Guy"],
  [2, "Lil Nas X", "Old Town Road"],
  [3, "Ed Sheeran", "I Don't Care"]
]
```

Each entry is: `[rank, artist, title]`

### Step 2: Ingest the Chart

```bash
canon charts ingest nl_top40 2024-W01 chart_data.json
```

Output:

```
✔︎ Ingested 3 entries
  Chart: nl_top40
  Period: 2024-W01
  Run ID: 12345
```

### Step 3: Link Chart Entries

```bash
canon charts link nl_top40 2024-W01
```

Output:

```
✔︎ Linked 3/3 entries
  Coverage: 100.0%
  By method:
    exact_match: 2
    fuzzy_match: 1
```

### Step 4: Check Coverage

```bash
canon coverage chart nl_top40 2024-W01
```

## Configuration

For basic usage, Chart-Binder works with defaults. For more control, create a `config.toml`:

```toml
# config.toml

[http_cache]
directory = ".cache/http"
ttl_seconds = 86400
enabled = true

[database]
music_graph_path = "musicgraph.sqlite"
charts_path = "charts.sqlite"
decisions_path = "decisions.sqlite"
```

Use it with:

```bash
canon --config config.toml scan path/to/music/
```

## Common Workflows

### Batch Processing a Music Library

```bash
# 1. Scan everything first
canon scan /path/to/music/ --output json > scan_results.json

# 2. Make decisions
canon decide /path/to/music/

# 3. Preview changes
canon write --dry-run /path/to/music/

# 4. Apply changes
canon write --apply /path/to/music/
```

### Offline Mode

If you want to work without network requests:

```bash
# Use only cached data
canon --offline scan /path/to/music/

# Fail if cache miss (stricter)
canon --frozen scan /path/to/music/
```

### Force Refresh Cached Data

```bash
canon --refresh scan /path/to/music/
```

## Next Steps

Now that you have the basics:

1. **Read the [CLI Reference](cli-reference.md)** for complete command documentation
2. **Configure [LLM adjudication](configuration.md#llm-configuration)** for handling ambiguous cases
3. **Learn about [normalization rules](appendix/normalization_ruleset_v1.md)** to understand how text is processed
4. **Explore the [Python API](api-reference.md)** for programmatic access
5. **Set up the [beets plugin](beets-plugin.md)** if you use beets

## Troubleshooting

### "No results" when scanning

- Ensure the file path is correct
- Check that the file is a supported format (MP3, FLAC, OGG, M4A)
- Verify the file has readable metadata

### "Cache miss" in frozen mode

This means the requested data isn't cached. Either:
- Run without `--frozen` to fetch the data
- Or use `--refresh` to force fetch fresh data

### Decision shows "indeterminate"

This means Chart-Binder couldn't make a confident decision. Options:
- Use `--explain` to see why
- Enable LLM adjudication in config
- Use the review queue to handle manually

### Rate limiting errors

If you're hitting API rate limits:
- Enable HTTP caching (default)
- Run in offline mode with cached data
- Adjust rate limits in configuration

## Getting Help

- Run `canon --help` for command help
- Check [GitHub Issues](https://github.com/pvliesdonk/chart-binder/issues) for known issues
- Read the [technical specification](spec.md) for detailed documentation
