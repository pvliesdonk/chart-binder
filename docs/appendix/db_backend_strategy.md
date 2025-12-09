# Database Backend Strategy

**Issue**: [#16 - A.0: Database access strategy spike](https://github.com/pvliesdonk/chart-binder/issues/16)
**Date**: 2024-12-09
**Status**: Decided

## Problem Statement

pymusicbrainz uses `mbdata` for PostgreSQL access, but there's complexity:
- `pvliesdonk/mbdata` - Fork with ARRAY support, unmaintained and schema outdated
- `metabrainz/mbdata` - Upstream, matches current schema, ARRAY support unclear

We need to determine the best approach for chart-binder's DB backend.

## Options Evaluated

1. **Use upstream `metabrainz/mbdata`** - Check if ARRAY support works
2. **Use pure SQL without ORM** - More verbose but no dependency issues
3. **Create minimal SQLAlchemy models** - Just the tables we need
4. **Hybrid**: Use mbdata where it works, raw SQL for ARRAY columns

## Investigation Results

### Test Environment

- PostgreSQL 16.3 at `192.168.50.212:5432`
- Database: `musicbrainz_db`
- Schema: 377 tables in `musicbrainz` schema

### ARRAY Column Analysis

ARRAY columns exist only in **materialized views**, not core entity tables:

| Table | Column | Type |
|-------|--------|------|
| `artist_release_group` | `secondary_types` | `int2[]` |
| `artist_release` | `catalog_numbers` | `text[]` |
| `cdtoc` | `track_offset` | `int4[]` |

**Core entity tables** (artist, recording, release_group, work, release) use only simple types.

### Upstream mbdata Testing

Tested with `mbdata==30.0.1`:

```python
from mbdata.models import Artist, Recording, ReleaseGroup, Work
```

**Result**: ✅ All core entity queries work perfectly

- `Artist.query.filter(Artist.name == 'Queen')` → Works
- `Recording.query.filter(...)` → Works
- `ReleaseGroup.query.filter(...)` → Works
- Relationship tables (`LinkRecordingWork`) → Works

### Raw SQL for Complex Queries

For complex joins and ARRAY column access, raw SQL via SQLAlchemy works:

```python
session.execute(text('''
    SELECT rg.name, rgpt.name as primary_type, MIN(rc.date_year)
    FROM musicbrainz.work w
    JOIN musicbrainz.l_recording_work lrw ON w.id = lrw.entity1
    JOIN musicbrainz.recording rec ON rec.id = lrw.entity0
    -- ... complex joins ...
    GROUP BY rg.gid, rg.name, rgpt.name
'''))
```

**Result**: ✅ Works for sibling expansion and bucketing queries

### ARRAY Column Workaround

ARRAY columns in materialized views can be accessed via:
1. Raw SQL with `session.execute(text(...))` - ARRAY returned as Python list
2. Join approach via `release_group_secondary_type_join` - avoids ARRAY entirely

## Decision

**Use hybrid approach: upstream mbdata + raw SQL**

### Rationale

1. **Upstream mbdata is sufficient** for core entity models
2. **No forked mbdata needed** - ARRAY columns are only in views we rarely need
3. **Raw SQL is cleaner** for complex discovery queries anyway
4. **Maintainability** - upstream mbdata stays in sync with MusicBrainz schema

### Implementation Pattern

```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from mbdata.models import Artist, Recording, ReleaseGroup, Work

# Simple queries: use ORM
artist = session.query(Artist).filter(Artist.gid == mbid).first()

# Complex queries: use raw SQL
siblings = session.execute(text('''
    SELECT r.gid, r.name, ac.name as artist
    FROM musicbrainz.recording r
    JOIN musicbrainz.l_recording_work lrw ON r.id = lrw.entity0
    WHERE lrw.entity1 = :work_id
'''), {'work_id': work_id}).fetchall()
```

## Key Queries Validated

### 1. Sibling Expansion (Recording → Work → Recordings)

```sql
-- Find work for recording
SELECT w.id, w.gid, w.name
FROM musicbrainz.work w
JOIN musicbrainz.l_recording_work lrw ON w.id = lrw.entity1
WHERE lrw.entity0 = :recording_id

-- Find all recordings for work
SELECT r.gid, r.name, ac.name as artist
FROM musicbrainz.recording r
JOIN musicbrainz.l_recording_work lrw ON r.id = lrw.entity0
JOIN musicbrainz.artist_credit ac ON r.artist_credit = ac.id
WHERE lrw.entity1 = :work_id
```

### 2. Bucketed Candidates (Release Groups by Type)

```sql
SELECT DISTINCT
    rg.gid, rg.name,
    rgpt.name as primary_type,
    MIN(rc.date_year) as earliest_year
FROM musicbrainz.work w
JOIN musicbrainz.l_recording_work lrw ON w.id = lrw.entity1
JOIN musicbrainz.recording rec ON rec.id = lrw.entity0
JOIN musicbrainz.track t ON t.recording = rec.id
JOIN musicbrainz.medium m ON t.medium = m.id
JOIN musicbrainz.release r ON m.release = r.id
JOIN musicbrainz.release_group rg ON r.release_group = rg.id
LEFT JOIN musicbrainz.release_group_primary_type rgpt ON rg.type = rgpt.id
LEFT JOIN musicbrainz.release_country rc ON r.id = rc.release
JOIN musicbrainz.artist_credit ac ON rg.artist_credit = ac.id
WHERE w.name = :work_name
GROUP BY rg.gid, rg.name, rgpt.name
ORDER BY earliest_year NULLS LAST
```

### 3. Secondary Types (via Join, no ARRAY)

```sql
SELECT rg.name, rgst.name as secondary_type
FROM musicbrainz.release_group rg
JOIN musicbrainz.release_group_secondary_type_join rgsj
    ON rg.id = rgsj.release_group
JOIN musicbrainz.release_group_secondary_type rgst
    ON rgsj.secondary_type = rgst.id
WHERE rg.id = :release_group_id
```

## Schema Notes

### Key Tables for Discovery

| Table | Purpose |
|-------|---------|
| `artist` | Artist entities |
| `recording` | Track/song recordings |
| `release_group` | Album/EP/Single groupings |
| `release` | Specific releases (no dates!) |
| `release_country` | Release dates by country |
| `work` | Musical works (compositions) |
| `l_recording_work` | Recording ↔ Work links |
| `artist_credit` | Credit name |
| `artist_credit_name` | Credit → Artist mapping |
| `release_group_primary_type` | Album, EP, Single, etc. |
| `release_group_secondary_type` | Compilation, Soundtrack, Live, etc. |
| `release_group_secondary_type_join` | RG ↔ Secondary type mapping |

### Important: Dates are NOT on `release`

Release dates are in `release_country` table, not `release` table:

```sql
-- Correct: join to release_country for dates
LEFT JOIN musicbrainz.release_country rc ON r.id = rc.release
-- Then use: rc.date_year, rc.date_month, rc.date_day
```

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    "mbdata>=30.0.0",
    "psycopg2-binary>=2.9.0",
    "sqlalchemy>=2.0.0",
]
```

## Next Steps

1. ✅ Decision documented
2. Implement backend abstraction layer (Issue A.1)
3. Create `DBBackend` class using this pattern
4. Create `APIBackend` class wrapping existing `musicbrainz.py`
