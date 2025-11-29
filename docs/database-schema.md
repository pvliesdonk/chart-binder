# Database Schema Documentation

Chart-Binder uses SQLite databases to store chart data, MusicBrainz entities, and canonicalization decisions.

## Database Files

| File | Purpose |
|------|---------|
| `charts.sqlite` | Chart runs, entries, song registry, and normalization aliases |
| `musicgraph.sqlite` | MusicBrainz entities (artists, recordings, releases) |
| `decisions.sqlite` | Canonicalization decisions and traces (future) |

---

## Three-Layer Data Model

Chart entries flow through three layers, preserving raw data while avoiding duplication:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: RAW (chart_entry) - immutable after scrape             │
│   Stores exactly what the website said                          │
│   "The Beatles - Penny Lane / Strawberry Fields Forever"        │
│   rank=1, previous_position=3, weeks_on_chart=5                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓ split/normalize/link
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: LINK (chart_entry_song) - editable join table          │
│   Links raw entries to songs (no text duplication)              │
│   entry_id=xyz → song_id=abc (song_idx=0)                       │
│   entry_id=xyz → song_id=def (song_idx=1)  [double A-side]      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: SONG (song) - canonical registry, stored ONCE          │
│   song_id=abc: "Beatles" / "Penny Lane" + MBID, Spotify, etc.  │
│   song_id=def: "Beatles" / "Strawberry Fields Forever"          │
│   (same song appearing in 500 runs = 1 row, not 500)            │
└─────────────────────────────────────────────────────────────────┘
```

**Key benefits:**
- Raw data never lost - can always re-process or audit
- Double A-sides properly handled - one raw entry → multiple songs
- **No text duplication** - song name stored once, linked many times
- Song identity persists across all chart runs
- Reverse query: find all chart appearances for a song
- Corrections update link table, not raw data

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

### chart_entry (Layer 1: Raw)

Raw scraped entries - **immutable after ingestion**.

```sql
CREATE TABLE chart_entry (
    entry_id TEXT PRIMARY KEY,      -- Deterministic hash(run_id, rank)
    run_id TEXT NOT NULL,           -- FK to chart_run
    rank INTEGER NOT NULL,          -- Position in chart (1-based)
    artist_raw TEXT NOT NULL,       -- Artist name exactly as scraped
    title_raw TEXT NOT NULL,        -- Title exactly as scraped
    previous_position INTEGER,      -- Position last week (from website)
    weeks_on_chart INTEGER,         -- Weeks on chart (from website)
    entry_unit TEXT NOT NULL,       -- "recording", "single_release", "medley"
    extra_raw TEXT,                 -- Extra info (remixer, version, etc.)
    scraped_at REAL NOT NULL,       -- When this entry was scraped
    UNIQUE(run_id, rank)
);
```

**Entry units:**
- `recording`: Standard song/track
- `single_release`: Physical single release (may contain multiple songs)
- `medley`: Multiple songs combined
- `unknown`: Unclassified

### chart_entry_song (Layer 2: Link Table)

Links raw entries to songs - **editable for corrections**. No text duplication.

```sql
CREATE TABLE chart_entry_song (
    id TEXT PRIMARY KEY,            -- UUID
    entry_id TEXT NOT NULL,         -- FK to chart_entry (raw)
    song_idx INTEGER NOT NULL,      -- 0, 1, 2... for multi-song entries
    song_id TEXT NOT NULL,          -- FK to song (required)
    link_method TEXT,               -- 'auto', 'manual', 'alias'
    link_confidence REAL,           -- 0.0 to 1.0
    FOREIGN KEY (entry_id) REFERENCES chart_entry(entry_id),
    FOREIGN KEY (song_id) REFERENCES song(song_id),
    UNIQUE(entry_id, song_idx)
);
```

**Key design:**
- Song name stored ONCE in `song` table, not repeated here
- This is a pure link table with metadata about how the link was made
- One raw entry can link to multiple songs (double A-sides)

**Link methods:**
- `auto`: Automatically matched by normalization
- `manual`: Human-assigned link
- `alias`: Matched via alias_norm exception

### song (Layer 3: Canonical)

Canonical song registry - persistent song identities.

```sql
CREATE TABLE song (
    song_id TEXT PRIMARY KEY,       -- Stable internal ID (UUID)
    artist_canonical TEXT NOT NULL, -- Canonical artist name
    title_canonical TEXT NOT NULL,  -- Canonical title
    artist_sort TEXT,               -- Sort name for artist
    work_key TEXT,                  -- MusicBrainz work key
    recording_mbid TEXT,            -- MusicBrainz recording MBID
    release_group_mbid TEXT,        -- MusicBrainz release group MBID
    spotify_id TEXT,                -- Spotify track ID
    isrc TEXT,                      -- ISRC code
    created_at REAL NOT NULL,       -- When first seen
    UNIQUE(artist_canonical, title_canonical)
);
```

### chart_link (Legacy compatibility)

Links between chart entries and work keys. *Deprecated - use song table instead.*

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
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Scraper   │────▶│ chart_entry │────▶│chart_entry_ │────▶│    song     │
│  (web data) │     │   (raw)     │     │   song      │     │ (canonical) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                    ┌─────────────┐                                 │
                    │ musicgraph  │◀────────────────────────────────┘
                    │  .sqlite    │      (MBID linking)
                    └─────────────┘
```

1. **Scrape**: `canon charts scrape t40 2024-W01 --ingest`
   - Fetches data from web source
   - Stores raw data in `chart_entry` (immutable)
   - Validates entry count and continuity

2. **Normalize**: Automatic during ingest
   - Splits double A-sides into separate `chart_entry_song` rows
   - Normalizes artist/title text
   - Links to existing `song` entries or creates new ones

3. **Link**: `canon charts link nl_top40 2024-W01`
   - Matches songs to MusicBrainz recordings
   - Updates `song` table with MBIDs

---

## Querying Examples

### Get all chart appearances for a song

```sql
-- Find all positions for "Bohemian Rhapsody" by Queen
SELECT
    r.chart_id,
    r.period,
    e.rank,
    e.previous_position,
    e.weeks_on_chart
FROM song s
JOIN chart_entry_song es ON es.song_id = s.song_id
JOIN chart_entry e ON e.entry_id = es.entry_id
JOIN chart_run r ON r.run_id = e.run_id
WHERE s.artist_canonical = 'Queen'
  AND s.title_canonical = 'Bohemian Rhapsody'
ORDER BY r.chart_id, r.period;
```

### Get chart history by MBID (for beets tagging)

```sql
SELECT
    r.chart_id,
    r.period,
    e.rank
FROM song s
JOIN chart_entry_song es ON es.song_id = s.song_id
JOIN chart_entry e ON e.entry_id = es.entry_id
JOIN chart_run r ON r.run_id = e.run_id
WHERE s.recording_mbid = '12345-abcd-...'
ORDER BY r.period DESC;
```

### List all chart runs

```sql
SELECT chart_id, period, scraped_at,
       (SELECT COUNT(*) FROM chart_entry WHERE run_id = chart_run.run_id) as entry_count
FROM chart_run
ORDER BY chart_id, period DESC;
```

### Get entries for a specific week

```sql
SELECT e.rank, e.artist_raw, e.title_raw, e.previous_position
FROM chart_entry e
JOIN chart_run r ON e.run_id = r.run_id
WHERE r.chart_id = 'nl_top40' AND r.period = '2024-W01'
ORDER BY e.rank;
```

### Find double A-sides

```sql
SELECT e.artist_raw, e.title_raw, COUNT(*) as song_count
FROM chart_entry e
JOIN chart_entry_song es ON es.entry_id = e.entry_id
GROUP BY e.entry_id
HAVING COUNT(*) > 1;
```

### Find raw entries not yet linked to songs

```sql
SELECT e.artist_raw, e.title_raw, r.chart_id, r.period
FROM chart_entry e
JOIN chart_run r ON r.run_id = e.run_id
LEFT JOIN chart_entry_song es ON es.entry_id = e.entry_id
WHERE es.id IS NULL
ORDER BY r.period DESC;
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

### Cross-Reference (previous position)
- Website's claimed "previous position" checked against database
- Mismatches indicate data quality issues
