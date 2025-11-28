# Beets Plugin Guide

Integrate Chart-Binder with [beets](https://beets.io/), the media library management system.

## Overview

The Chart-Binder beets plugin provides canonicalization during the beets import process. It can:

- Make canonicalization decisions during import
- Attach MusicBrainz IDs and CHARTS data
- Show side-by-side comparison with beets' candidates
- Override beets' album/track selection when appropriate

## Installation

The plugin is included with Chart-Binder. Install Chart-Binder:

```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

## Configuration

Add the plugin to your beets configuration (`~/.config/beets/config.yaml`):

```yaml
plugins:
  - chart_binder

chart_binder:
  # Mode: advisory, authoritative, or augment
  mode: advisory
  
  # Show detailed explanations during import
  explain: false
  
  # Preview changes without writing
  dry_run: false
  
  # Offline mode (no network requests)
  offline: false
  
  # Auto-accept confidence threshold (0.0-1.0)
  accept_threshold: 0.85
  
  # CRG selection parameters
  # Lead window: days to consider for lead single vs album decision
  lead_window_days: 90
  # Reissue gap: years after which a release is considered a reissue
  reissue_long_gap_years: 10
  
  # Enable CHARTS blob attachment
  charts_enabled: true
  
  # Database paths (optional, uses defaults if not specified)
  charts_db_path: /path/to/charts.sqlite
  decisions_db_path: /path/to/decisions.sqlite
```

## Operating Modes

The plugin supports three modes of operation:

### Advisory Mode (Default)

Shows Chart-Binder's decision alongside beets' candidate. You choose which to use.

```yaml
chart_binder:
  mode: advisory
  explain: true  # Show detailed comparison
```

During import, you'll see:

```
--- Chart-Binder Decision ---
File: /path/to/song.mp3

Beets candidate:
  Album: Some Album
  Date: 2019
  Label: Record Label

Chart-Binder decision:
  State: decided
  CRG: b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d
       (album_with_chart_entry)
  RR: a1b2c3d4-e5f6-7890-abcd-ef1234567890
       (preferred_country_nl)
  Trace: CB/1.0/d/ACE/RRnl
```

### Authoritative Mode

Chart-Binder's decision automatically replaces beets' candidate.

```yaml
chart_binder:
  mode: authoritative
```

Use this when you trust Chart-Binder's decisions completely.

### Augment-Only Mode

Never changes core fields (album, artist, title, date). Only adds IDs, CHARTS blob, and decision trace.

```yaml
chart_binder:
  mode: augment
```

This is the safest mode for preserving your existing metadata.

## CLI Commands

The plugin adds several beets commands:

### canon-explain

Show decision trace for items in your library:

```bash
# Explain a specific track
beet canon-explain path:/path/to/song.mp3

# Explain all tracks by an artist
beet canon-explain artist:Artist

# Explain albums
beet canon-explain -a albumartist:Artist
```

### canon-pin

Pin the current decision to prevent changes on re-import:

```bash
# Pin specific tracks
beet canon-pin path:/path/to/song.mp3

# Pin all tracks by artist
beet canon-pin artist:Artist

# Pin albums
beet canon-pin -a albumartist:Artist
```

### canon-unpin

Remove pin from items:

```bash
beet canon-unpin path:/path/to/song.mp3
```

### canon-override

Override the canonicalization for specific items:

```bash
# Override to a specific release group
beet canon-override --rg=b8aef8f3-8e47-4e9e-b9a6-2d6c9e3a7c1d path:/path/to/song.mp3

# Also specify a release
beet canon-override --rg=<rg_mbid> --release=<release_mbid> path:/path/to/song.mp3
```

## Import Workflow

### Basic Import

```bash
# Standard import with advisory mode
beet import /path/to/music/

# Import with explanations
beet import /path/to/music/
```

### Batch Import

For non-interactive batch imports:

```yaml
chart_binder:
  mode: augment  # Or authoritative
```

```bash
# Non-interactive import
beet import -q /path/to/music/
```

### Dry Run

Preview what would be changed:

```yaml
chart_binder:
  dry_run: true
```

Or temporarily:

```bash
# Modify config at runtime (beets feature)
beet -c 'chart_binder.dry_run: true' import /path/to/music/
```

## Field Mappings

The plugin writes these fields to beets items:

### MusicBrainz IDs

| Beets Field | Description |
|-------------|-------------|
| `mb_releasegroupid` | Canonical Release Group MBID |
| `mb_albumid` | Representative Release MBID |
| `mb_trackid` | Recording MBID |

### Chart-Binder Fields

| Field | Description |
|-------|-------------|
| `canon_state` | Decision state (decided/indeterminate) |
| `canon_crg_rationale` | CRG selection reason |
| `canon_rr_rationale` | RR selection reason |
| `canon_trace` | Compact decision trace |
| `charts` | CHARTS blob (minified JSON) |

## Decision States

| State | Meaning |
|-------|---------|
| `decided` | Clear canonical release selected |
| `indeterminate` | Ambiguous case, may need review |
| `blocked` | Missing required information |

## Charts Integration

When `charts_enabled: true`, the plugin attaches chart history to items:

```yaml
chart_binder:
  charts_enabled: true
  charts_db_path: /path/to/charts.sqlite
```

The CHARTS blob contains peak positions and weeks on chart:

```json
{"nl40":{"peak":1,"wks":12},"uk":{"peak":3,"wks":8}}
```

## Offline Mode

For imports without network access:

```yaml
chart_binder:
  offline: true
```

In offline mode:
- Only uses cached data and local tags
- Falls back to augment-only if data is missing
- Marks decisions as INDETERMINATE when uncertain

## Conflict Handling

### Compilation Albums

If beets picks a Various Artists compilation but Chart-Binder selects the original album:

- In **advisory** mode: Shows comparison, you choose
- In **authoritative** mode: Chart-Binder's choice wins
- In **augment** mode: No change, just adds metadata

### Promo Singles vs Albums

When a track appears on both a single and album:

- The `delta_days` field shows the time difference
- Chart-Binder uses chart data to determine which was the "hit"

### Reissues and Remasters

Chart-Binder detects reissues (10+ year gap) and typically prefers the original:

- Shows when beets picked a later remaster
- Offers the original release as alternative

## Import Summary

At the end of import, the plugin shows a summary:

```
Chart-Binder Import Summary:
  Total items: 100
  Canonized: 75
  Augmented: 20
  Skipped: 3
  Indeterminate: 2
  Cache: 85 hits, 15 misses
```

## Troubleshooting

### Plugin Not Loading

Check that chart_binder is in your plugins list:

```bash
beet config
```

Look for:
```yaml
plugins:
  - chart_binder
```

### Database Errors

Ensure database paths are correct and writable:

```yaml
chart_binder:
  charts_db_path: /absolute/path/to/charts.sqlite
  decisions_db_path: /absolute/path/to/decisions.sqlite
```

### Slow Imports

Enable caching and offline mode for faster re-imports:

```yaml
chart_binder:
  offline: true  # Use cached data
```

### Unexpected Decisions

Use `canon-explain` to understand decisions:

```bash
beet canon-explain path:/path/to/problematic/song.mp3
```

If the decision is wrong, use `canon-override`:

```bash
beet canon-override --rg=<correct_rg_mbid> path:/path/to/song.mp3
```

## Example Configurations

### Conservative Setup

Only add metadata, never change existing fields:

```yaml
plugins:
  - chart_binder

chart_binder:
  mode: augment
  charts_enabled: true
  offline: false
```

### Full Automation

Trust Chart-Binder for all decisions:

```yaml
plugins:
  - chart_binder

chart_binder:
  mode: authoritative
  accept_threshold: 0.90
  charts_enabled: true
  offline: false
```

### Review-Focused

Show everything, make manual decisions:

```yaml
plugins:
  - chart_binder

chart_binder:
  mode: advisory
  explain: true
  dry_run: false
  charts_enabled: true
```
