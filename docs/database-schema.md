# Database Schema Documentation

Chart-Binder uses SQLite databases to store chart data, MusicBrainz entities, and canonicalization decisions.

## Database Files

| File | Purpose |
|------|---------|
| `charts.sqlite` | Chart runs, entries, links, and normalization aliases |
| `musicgraph.sqlite` | MusicBrainz entities (artists, recordings, releases) |
| `decisions.sqlite` | Canonicalization decisions and traces (future) |

---

## charts.sqlite

The main database for chart data ingestion and linking.

### chart

Chart metadata and configuration.

```sql
CREATE TABLE chart (
    chart_id TEXT PRIMARY KEY,      -- e.g., "nl_top40", "nl_top2000"
    name TEXT NOT NULL,             -- Human-readable name
    frequency TEXT NOT NULL,        -- "weekly", "yearly"
    jurisdiction TEXT,              -- Country code, e.g., "NL"
    source_url TEXT,                -- Source website URL
    license TEXT,                   -- License information
    created_at REAL NOT NULL        -- Unix timestamp
);
```

**Current charts:**
| chart_id | name | frequency |
|----------|------|-----------|
| `nl_top40` | Dutch Top 40 weekly | weekly |
| `nl_top40_jaar` | Dutch Top 40 year-end | yearly |
| `nl_top2000` | NPO Radio 2 Top 2000 | yearly |
| `nl_538_zwaarste` | 538 De Zwaarste Lijst | yearly |

### chart_run

A single chart instance (one week or one year).

```sql
CREATE TABLE chart_run (
    run_id TEXT PRIMARY KEY,        -- UUID
    chart_id TEXT NOT NULL,         -- FK to chart
    period TEXT NOT NULL,           -- "2024-W01" or "2024"
    scraped_at REAL NOT NULL,       -- Unix timestamp
    source_hash TEXT NOT NULL,      -- Hash of source data
    notes TEXT,                     -- Optional notes
    UNIQUE(chart_id, period)
);
```

**Period formats:**
- Weekly charts: `YYYY-Www` (e.g., `2024-W01`, `2024-W52`)
- Yearly charts: `YYYY` (e.g., `2024`, `1999`)

### chart_entry

Individual entries (songs) in a chart run.

```sql
CREATE TABLE chart_entry (
    run_id TEXT NOT NULL,           -- FK to chart_run
    rank INTEGER NOT NULL,          -- Position in chart (1-based)
    artist_raw TEXT NOT NULL,       -- Artist name as scraped
    title_raw TEXT NOT NULL,        -- Title as scraped
    entry_unit TEXT NOT NULL,       -- "recording", "single_release", "medley"
    extra_raw TEXT,                 -- Extra info (remixer, version, etc.)
    artist_normalized TEXT,         -- Normalized artist name
    title_normalized TEXT,          -- Normalized title
    title_tags_json TEXT,           -- JSON: extracted tags (remix, live, etc.)
    PRIMARY KEY (run_id, rank)
);
```

**Entry units:**
- `recording`: Standard song/track
- `single_release`: Physical single release
- `medley`: Multiple songs combined
- `unknown`: Unclassified

### chart_link

Links between chart entries and work keys (for matching to MusicBrainz).

```sql
CREATE TABLE chart_link (
    run_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    work_key TEXT,                  -- "artist // title" format
    link_method TEXT NOT NULL,      -- How the link was made
    confidence REAL NOT NULL,       -- 0.0 to 1.0
    release_anchor_id TEXT,         -- Specific release MBID
    side_designation TEXT,          -- A-side, B-side, etc.
    PRIMARY KEY (run_id, rank)
);
```

**Link methods:**
- `isrc`: Matched by ISRC code
- `title_artist_year`: Fuzzy match on title/artist/year
- `bundle_release`: Linked via release bundle
- `manual`: Human-assigned link

### alias_norm

Normalization exception registry for special cases.

```sql
CREATE TABLE alias_norm (
    alias_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,             -- "artist" or "title"
    raw TEXT NOT NULL,              -- Original string
    normalized TEXT NOT NULL,       -- Normalized form
    ruleset_version TEXT NOT NULL,  -- e.g., "norm-v1"
    created_at REAL NOT NULL
);
```

**Example aliases:**
| type | raw | normalized |
|------|-----|------------|
| artist | The The | the the |
| artist | De Dijk | de dijk |

---

## musicgraph.sqlite

MusicBrainz entity cache for offline operation.

### artist

MusicBrainz artist entities.

```sql
CREATE TABLE artist (
    mbid TEXT PRIMARY KEY,          -- MusicBrainz ID
    name TEXT NOT NULL,             -- Artist name
    sort_name TEXT,                 -- Sort name
    type TEXT,                      -- Person, Group, etc.
    area TEXT,                      -- Country/region
    disambiguation TEXT,            -- Disambiguation comment
    fetched_at REAL NOT NULL        -- When data was fetched
);
```

### recording

MusicBrainz recording entities (songs/tracks).

```sql
CREATE TABLE recording (
    mbid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist_credit TEXT NOT NULL,    -- Full artist credit string
    length_ms INTEGER,              -- Duration in milliseconds
    disambiguation TEXT,
    first_release_date TEXT,        -- YYYY-MM-DD
    isrcs_json TEXT,                -- JSON array of ISRCs
    fetched_at REAL NOT NULL
);
```

### release_group

Album/EP/Single groupings.

```sql
CREATE TABLE release_group (
    mbid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist_credit TEXT NOT NULL,
    primary_type TEXT,              -- Album, Single, EP, etc.
    secondary_types_json TEXT,      -- JSON array: ["Compilation", "Live"]
    first_release_date TEXT,
    fetched_at REAL NOT NULL
);
```

### release

Specific releases (format, region, label).

```sql
CREATE TABLE release (
    mbid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist_credit TEXT NOT NULL,
    release_group_mbid TEXT,        -- FK to release_group
    date TEXT,                      -- Release date
    country TEXT,                   -- Release country
    status TEXT,                    -- Official, Bootleg, etc.
    barcode TEXT,
    fetched_at REAL NOT NULL
);
```

### recording_release

Many-to-many: recordings appearing on releases.

```sql
CREATE TABLE recording_release (
    recording_mbid TEXT NOT NULL,
    release_mbid TEXT NOT NULL,
    track_number INTEGER,
    disc_number INTEGER,
    PRIMARY KEY (recording_mbid, release_mbid)
);
```

---

## Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Scraper   │────▶│   charts    │────▶│   Link &    │
│  (web data) │     │   .sqlite   │     │   Match     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐             │
                    │ musicgraph  │◀────────────┘
                    │  .sqlite    │
                    └─────────────┘
```

1. **Scrape**: `canon charts scrape t40 2024-W01 --ingest`
   - Fetches data from web source
   - Validates entry count and continuity
   - Stores in `chart_run` and `chart_entry`

2. **Link**: `canon charts link nl_top40 2024-W01`
   - Matches entries to work keys
   - Stores in `chart_link`

3. **Resolve**: Uses MusicBrainz data from `musicgraph.sqlite`
   - Finds canonical release groups
   - Selects representative releases

---

## Querying Examples

### List all chart runs
```sql
SELECT chart_id, period, scraped_at,
       (SELECT COUNT(*) FROM chart_entry WHERE run_id = chart_run.run_id) as entry_count
FROM chart_run
ORDER BY chart_id, period DESC;
```

### Get entries for a specific week
```sql
SELECT rank, artist_raw, title_raw
FROM chart_entry
WHERE run_id = (
    SELECT run_id FROM chart_run
    WHERE chart_id = 'nl_top40' AND period = '2024-W01'
)
ORDER BY rank;
```

### Find unlinked entries
```sql
SELECT e.rank, e.artist_raw, e.title_raw
FROM chart_entry e
LEFT JOIN chart_link l ON e.run_id = l.run_id AND e.rank = l.rank
WHERE e.run_id = ? AND l.work_key IS NULL;
```

### Coverage report
```sql
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN l.work_key IS NOT NULL THEN 1 ELSE 0 END) as linked,
    ROUND(100.0 * SUM(CASE WHEN l.work_key IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as coverage_pct
FROM chart_entry e
LEFT JOIN chart_link l ON e.run_id = l.run_id AND e.rank = l.rank
WHERE e.run_id = ?;
```

---

## Validation Rules

### Entry Count
| Chart | Expected | Tolerance |
|-------|----------|-----------|
| nl_top40 | 40 | ≥36 (90%) |
| nl_top40_jaar | 100 | ≥90 (90%) |
| nl_top2000 | 2000 | ≥1800 (90%) |
| nl_538_zwaarste | ~150 | ≥135 (90%) |

### Continuity (weekly charts)
- Expect ≥50% overlap with previous week
- Songs don't typically all change at once
- Low overlap indicates possible scraping error
