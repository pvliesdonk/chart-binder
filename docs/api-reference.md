# API Reference

Python API documentation for Chart-Binder.

## Installation

```python
# After installing chart-binder
from chart_binder import tagging, normalizer, resolver
from chart_binder.config import Config
```

---

## Tagging Module

Read and write audio file metadata.

### verify

Read existing tags from an audio file.

```python
from chart_binder.tagging import verify

tagset = verify("path/to/song.mp3")

# Access basic metadata
print(tagset.title)       # Track title
print(tagset.artist)      # Artist name
print(tagset.album)       # Album name
print(tagset.original_year)  # Original release year
print(tagset.track_number)   # Track number
print(tagset.disc_number)    # Disc number

# Access MusicBrainz IDs
print(tagset.ids.mb_recording_id)
print(tagset.ids.mb_release_group_id)
print(tagset.ids.mb_release_id)
print(tagset.ids.mb_artist_id)

# Access Chart-Binder specific fields
print(tagset.compact.charts_blob)
print(tagset.compact.decision_trace)
print(tagset.compact.ruleset_version)
```

**Parameters:**
- `file_path: str | Path` - Path to audio file

**Returns:** `TagSet` - Object containing all tag data

**Raises:** `Exception` if file cannot be read

### write_tags

Write tags to an audio file.

```python
from chart_binder.tagging import write_tags, TagSet, CanonicalIDs, CompactFields

# Create a tagset with data to write
tagset = TagSet(
    ids=CanonicalIDs(
        mb_release_group_id="b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d",
        mb_release_id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    ),
    compact=CompactFields(
        ruleset_version="canon-1.0",
        decision_trace="CB/1.0/d/ACE/RRnl",
        charts_blob='{"nl40":{"peak":1,"wks":12}}',
    ),
)

# Preview changes (dry run)
report = write_tags(
    "path/to/song.mp3",
    tagset,
    authoritative=False,  # Augment-only mode
    dry_run=True,
)

# Check what would be written
print(report.fields_written)     # List of fields to write
print(report.fields_skipped)     # List of skipped fields
print(report.originals_stashed)  # Original values backed up
print(report.dry_run)            # True if dry run

# Actually write
report = write_tags(
    "path/to/song.mp3",
    tagset,
    authoritative=False,
    dry_run=False,
)
```

**Parameters:**
- `file_path: str | Path` - Path to audio file
- `tagset: TagSet` - Tags to write
- `authoritative: bool` - If True, overwrite existing values; if False, only add missing
- `dry_run: bool` - If True, don't actually write

**Returns:** `WriteReport` - Report of what was written

### TagSet

Data class containing tag information.

```python
from chart_binder.tagging import TagSet, CanonicalIDs, CompactFields

tagset = TagSet(
    title="Bad Guy",
    artist="Billie Eilish",
    album="WHEN WE ALL FALL ASLEEP, WHERE DO WE GO?",
    original_year=2019,
    track_number=2,
    disc_number=1,
    country="US",
    label="Interscope",
    ids=CanonicalIDs(
        mb_recording_id="abc-123",
        mb_release_group_id="def-456",
        mb_release_id="ghi-789",
        mb_artist_id="jkl-012",
    ),
    compact=CompactFields(
        ruleset_version="canon-1.0",
        decision_trace="CB/1.0/d/ACE/RRnl",
        charts_blob='{"nl40":{"peak":1}}',
    ),
)
```

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `title` | str \| None | Track title |
| `artist` | str \| None | Artist name |
| `album` | str \| None | Album name |
| `original_year` | int \| None | Original release year |
| `track_number` | int \| None | Track number |
| `disc_number` | int \| None | Disc number |
| `country` | str \| None | Release country |
| `label` | str \| None | Record label |
| `ids` | CanonicalIDs | MusicBrainz IDs |
| `compact` | CompactFields | Chart-Binder specific fields |

---

## Normalizer Module

Text normalization for matching and comparison.

### normalize

Normalize artist or title text.

```python
from chart_binder.normalizer import normalize

# Basic normalization
result = normalize("Artist Name feat. Guest Artist")
print(result.normalized)  # Normalized form for matching
print(result.core)        # Core form without editions
print(result.tags)        # Extracted edition tags

# Title with editions
result = normalize("Song Title (Radio Edit) [Remastered 2020]")
print(result.core)   # "song title"
print(result.tags)   # [{"kind": "edit", "sub": "radio"}, {"kind": "remaster", "year": 2020}]

# Access original and signature
print(result.original)              # Original input
print(result.diacritics_signature)  # Signature for diacritic matching
print(result.ruleset_version)       # "norm-v1"
```

**Parameters:**
- `text: str` - Text to normalize

**Returns:** `NormalizeResult` - Normalized text with extracted tags

### NormalizeResult

Result of normalization.

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `normalized` | str | Normalized form for matching |
| `core` | str | Core form without edition tags |
| `tags` | list[dict] | Extracted edition/descriptor tags |
| `original` | str | Original input text |
| `diacritics_signature` | str | Signature for diacritic matching |
| `ruleset_version` | str | Ruleset version used |

### Edition Tags

Extracted tags have structured format:

```python
# Radio edit
{"kind": "edit", "sub": "radio"}

# Live recording
{"kind": "live", "place": "Madison Square Garden", "date": "2020"}

# Remaster
{"kind": "remaster", "year": 2020}

# Remix
{"kind": "remix", "mixer": "DJ Name"}

# Acoustic version
{"kind": "acoustic"}

# OST/Soundtrack
{"kind": "ost", "work": "Movie Name"}

# Content rating
{"kind": "content", "sub": "explicit"}
```

---

## Resolver Module

Make canonicalization decisions.

### Resolver

Main class for making CRG/RR decisions.

```python
from chart_binder.resolver import Resolver, ConfigSnapshot

# Create resolver with config
config = ConfigSnapshot(
    lead_window_days=90,       # Days to consider for lead single
    reissue_long_gap_years=10, # Years to consider a reissue
)
resolver = Resolver(config)

# Build evidence bundle
evidence_bundle = {
    "artifact": {
        "file_path": "/path/to/song.mp3",
    },
    "artist": {
        "name": "Billie Eilish",
        "mb_artist_id": "abc-123",
    },
    "recording_candidates": [
        {
            "mb_recording_id": "rec-456",
            "title": "Bad Guy",
            "rg_candidates": [
                {
                    "mb_rg_id": "rg-789",
                    "title": "WHEN WE ALL FALL ASLEEP",
                    "primary_type": "Album",
                    "first_release_date": "2019-03-29",
                    "releases": [
                        {
                            "mb_release_id": "rel-012",
                            "date": "2019-03-29",
                            "country": "US",
                            "label": "Interscope",
                            "title": "WHEN WE ALL FALL ASLEEP",
                            "flags": {"is_official": True},
                        }
                    ],
                }
            ],
        }
    ],
    "timeline_facts": {
        "chart_entries": [
            {"chart_id": "nl_top40", "peak": 1, "weeks": 12}
        ],
    },
    "provenance": {
        "sources_used": ["musicbrainz", "charts_db"],
    },
}

# Make decision
decision = resolver.resolve(evidence_bundle)

# Access decision results
print(decision.state)              # DecisionState enum
print(decision.release_group_mbid) # Selected CRG
print(decision.release_mbid)       # Selected RR
print(decision.crg_rationale)      # Reason for CRG selection
print(decision.rr_rationale)       # Reason for RR selection
print(decision.compact_tag)        # Compact trace string

# Access decision trace
trace = decision.decision_trace
print(trace.evidence_hash)
print(trace.considered_candidates)
print(trace.crg_selection)
print(trace.rr_selection)
print(trace.missing_facts)
print(trace.to_human_readable())
```

### Decision

Result of resolver.resolve().

**Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `state` | DecisionState | Decision state |
| `release_group_mbid` | str \| None | Selected CRG MBID |
| `release_mbid` | str \| None | Selected RR MBID |
| `crg_rationale` | CRGRationale \| None | CRG selection reason |
| `rr_rationale` | RRRationale \| None | RR selection reason |
| `compact_tag` | str | Compact decision trace |
| `decision_trace` | DecisionTrace | Detailed trace |

### DecisionState

Enum for decision states.

```python
from chart_binder.resolver import DecisionState

DecisionState.DECIDED        # Clear decision made
DecisionState.INDETERMINATE  # Ambiguous, needs review
DecisionState.BLOCKED        # Missing required information
```

---

## Config Module

Configuration management.

### Config

Load and access configuration.

```python
from chart_binder.config import Config
from pathlib import Path

# Load from file
config = Config.load(Path("config.toml"))

# Or use defaults
config = Config()

# Access sections
print(config.offline_mode)

# HTTP cache settings
print(config.http_cache.directory)
print(config.http_cache.ttl_seconds)
print(config.http_cache.enabled)

# Database paths
print(config.database.music_graph_path)
print(config.database.charts_path)
print(config.database.decisions_path)

# Live sources
print(config.live_sources.musicbrainz_rate_limit)
print(config.live_sources.acoustid_api_key)

# LLM settings
print(config.llm.enabled)
print(config.llm.provider)
print(config.llm.model_id)
print(config.llm.auto_accept_threshold)
```

### Config.load()

Load configuration from file with environment overrides.

```python
# From file
config = Config.load(Path("config.toml"))

# With defaults (no file)
config = Config.load(None)

# Non-existent file uses defaults
config = Config.load(Path("/nonexistent.toml"))
```

---

## Charts Module

Work with chart data.

### ChartsDB

Database for chart data.

```python
from chart_binder.charts_db import ChartsDB, ChartsETL

# Open database
db = ChartsDB("charts.sqlite")

# Upsert a chart definition
db.upsert_chart("nl_top40", "Dutch Top 40", "w")  # weekly

# Get chart info
chart = db.get_chart("nl_top40")

# Get chart run by period
run = db.get_run_by_period("nl_top40", "2024-W01")

# Get coverage report
report = db.get_coverage_report(run["run_id"])
print(report.total_entries)
print(report.linked_entries)
print(report.coverage_pct)
```

### ChartsETL

ETL operations for chart data.

```python
from chart_binder.charts_db import ChartsDB, ChartsETL

db = ChartsDB("charts.sqlite")
etl = ChartsETL(db)

# Ingest chart data
entries = [
    (1, "Artist One", "Song One"),
    (2, "Artist Two", "Song Two"),
]
run_id = etl.ingest("nl_top40", "2024-W01", entries)

# Link entries to work keys
report = etl.link(run_id, strategy="title_artist_year")

# Get missing entries
missing = etl.get_missing_entries(run_id, threshold=0.6)
```

### ChartsExporter

Export chart data for embedding.

```python
from chart_binder.charts_db import ChartsDB
from chart_binder.charts_export import ChartsExporter

db = ChartsDB("charts.sqlite")
exporter = ChartsExporter(db)

# Export chart blob for a work
blob = exporter.export_for_work("Artist // Title", include_positions=False)

# Convert to JSON
json_str = blob.to_json(include_positions=False)
print(json_str)  # {"nl40":{"peak":1,"wks":12}}
```

---

## HTTP Cache Module

Caching for API responses.

```python
from chart_binder.http_cache import HttpCache
from pathlib import Path

# Create cache
cache = HttpCache(Path(".cache/http"), ttl_seconds=86400)

# Store a response
cache.set("cache_key", b"response data")

# Retrieve
data = cache.get("cache_key")  # Returns bytes or None

# Check if cached
if cache.has("cache_key"):
    data = cache.get("cache_key")

# Purge expired entries
removed_count = cache.purge_expired()

# Clear all
cache.clear()
```

---

## LLM Module

LLM adjudication for ambiguous cases.

### LLMAdjudicator

```python
from chart_binder.llm import LLMAdjudicator
from chart_binder.config import Config

config = Config.load()
adjudicator = LLMAdjudicator(config=config.llm)

# Build evidence bundle (same format as resolver)
evidence_bundle = {
    "artist": {"name": "Artist Name"},
    "recording_candidates": [...],
}

# Decision trace from resolver
decision_trace = {
    "trace_compact": "CB/1.0/i/TIE",
}

# Adjudicate
result = adjudicator.adjudicate(evidence_bundle, decision_trace)

print(result.outcome)      # AdjudicationOutcome enum
print(result.crg_mbid)     # Suggested CRG
print(result.rr_mbid)      # Suggested RR
print(result.confidence)   # Confidence score (0.0-1.0)
print(result.rationale)    # Explanation
print(result.model_id)     # Model used
print(result.error_message) # Error if any
```

### ReviewQueue

Human review queue management.

```python
from chart_binder.llm import ReviewQueue, ReviewSource, ReviewAction

queue = ReviewQueue("review_queue.sqlite")

# Get pending items
items = queue.get_pending(source=ReviewSource.INDETERMINATE, limit=20)

# Get specific item
item = queue.get_item("review-id-123")
print(item.to_display())  # Human-readable format

# Complete a review
success = queue.complete_review(
    review_id="review-id-123",
    action=ReviewAction.ACCEPT,
    action_data={"crg_mbid": "abc-123", "rr_mbid": "def-456"},
    reviewed_by="username",
    notes="Verified release dates",
)

# Get statistics
stats = queue.get_stats()
print(stats["pending"])
print(stats["completed"])
```

---

## Decisions Module

Decision persistence and drift detection.

```python
from chart_binder.decisions_db import DecisionsDB
from chart_binder.drift import DriftDetector

# Open decisions database
db = DecisionsDB("decisions.sqlite")

# Store a decision
db.upsert_decision(
    file_id="file-hash-123",
    state="decided",
    mb_rg_id="rg-abc",
    mb_release_id="rel-def",
    work_key="Artist // Title",
    evidence_hash="ev-hash",
    ruleset_version="canon-1.0",
    trace_compact="CB/1.0/d/ACE/RRnl",
)

# Get a decision
decision = db.get_decision("file-hash-123")

# Drift detection
detector = DriftDetector(db)
stale_decisions = detector.review_drift()

for d in stale_decisions:
    print(f"Drifted: {d.file_id} - {d.state}")
```

---

## MusicGraph Module

Entity relationship database.

```python
from chart_binder.musicgraph import MusicGraphDB

db = MusicGraphDB("musicgraph.sqlite")

# Get artist info
artist = db.get_artist("artist-mbid")
print(artist["name"])
print(artist["begin_area_country"])

# Get recording info
recording = db.get_recording("recording-mbid")
print(recording["title"])
print(recording["artist_mbid"])

# Store entities
db.upsert_artist(
    mbid="artist-mbid",
    name="Artist Name",
    sort_name="Name, Artist",
    begin_area_country="US",
)

db.upsert_recording(
    mbid="recording-mbid",
    title="Song Title",
    artist_mbid="artist-mbid",
    length_ms=210000,
)
```

---

## Type Hints

All public APIs are fully typed. Use your IDE's type hints for detailed signatures:

```python
from chart_binder.tagging import TagSet, verify
from chart_binder.normalizer import NormalizeResult
from chart_binder.resolver import Decision, DecisionState, Resolver

# IDE will show full type information
tagset: TagSet = verify("song.mp3")
```

---

## Error Handling

```python
from chart_binder.tagging import verify

try:
    tagset = verify("path/to/song.mp3")
except FileNotFoundError:
    print("File not found")
except Exception as e:
    print(f"Error reading file: {e}")
```

Most functions raise standard Python exceptions. Check function docstrings for specific exception types.
