# Epic 6 - Decision Store & Drift

## Summary

This PR implements **Epic 6 - Decision Store & Drift**, completing the decision storage infrastructure and drift detection system per the specification. This epic provides the foundation for tracking canonicalization decisions over time and detecting when upstream data or rule changes affect previous decisions.

Implements all acceptance criteria from `docs/roadmap.md` Epic 6 and `docs/appendix/drift_and_determinism_contract_v1.md`.

## Changes

### Decisions Database (`decisions_db.py`)

**DecisionsDB** - SQLite database for decision tracking (966 lines total)

**Schema (5 tables)**:

1. **`file_artifact`** - File signature tracking
   - Stable `file_id` from SHA256(path:size:mtime)
   - Library root + relative path storage
   - Duration, fingerprint ID, original tags hash
   - Created timestamp

2. **`decision`** - Current decisions with state machine
   - Unique per file_id (one active decision per file)
   - Links to file_artifact via foreign key
   - Stores: CRG, RR, recording MBID
   - Metadata: ruleset_version, config_snapshot_json, evidence_hash
   - Compact trace string for tag embedding
   - State: DECIDED | STALE-* | INDETERMINATE | UNDECIDED
   - Pin flag for manual overrides
   - Created/updated timestamps

3. **`decision_history`** - Archived decisions
   - All decision columns + superseded_at, superseded_reason
   - Composite PK: (decision_id, superseded_at)
   - Tracks: refresh | ruleset_change | manual_override | pin
   - Full audit trail of all decision changes

4. **`override_rule`** - Manual overrides (future use)
   - Scope: artist | recording | release_group | file
   - Directive storage for rule exceptions
   - Creator tracking and notes

5. **`schema_meta`** - Version tracking
   - Stores `db_version_decisions = 1`

**Indexes**:
- `idx_file_path` on (library_root, relative_path)
- `idx_decision_file`, `idx_decision_state`, `idx_decision_rg`
- `idx_history_file`, `idx_history_superseded`
- `idx_override_scope`

**CRUD Operations**:

```python
# File tracking
generate_file_id(path, size, mtime) -> str  # Deterministic
upsert_file_artifact(...)
get_file_artifact(file_id)

# Decision management
upsert_decision(...)  # Auto-archives existing decision
get_decision(file_id)
update_decision_state(file_id, new_state)

# Drift review
get_stale_decisions() -> list  # All STALE-* or INDETERMINATE
get_decision_history(file_id) -> list
```

**State Machine Transitions**:
- `UNDECIDED → DECIDED` - First acceptance
- `DECIDED → STALE-EVIDENCE` - Evidence hash changed
- `DECIDED → STALE-RULES` - Ruleset version changed
- `DECIDED → STALE-BOTH` - Both changed
- `Any → INDETERMINATE` - Resolver cannot decide
- `STALE-* → DECIDED` - User accepts/applies new decision

**Key Features**:
- Automatic archival on upsert (old decision → history table)
- WAL mode for better concurrency
- Foreign key constraints enforced
- Enum-based state machine for type safety

### Drift Detection (`drift.py`)

**DriftDetector** - Evidence comparison and drift classification

**Evidence Hash Computation**:
```python
compute_evidence_hash(evidence_bundle) -> str
```
- Deterministic SHA256 of canonical JSON
- Sorts all keys recursively
- Strips volatile fields: `fetched_at`, `created_at`, `updated_at`, `fetched_at_utc`
- Ensures reproducible hashing across cache refreshes

**Drift Detection Algorithm**:
```python
detect_drift(file_id, new_evidence_bundle, new_decision, new_ruleset_version) -> DriftResult
```

1. Retrieve stored decision (if exists)
2. Compute new evidence hash from fresh data
3. Compare old vs new:
   - Evidence hash (data drift)
   - Ruleset version (rule changes)
   - Decision outcome (CRG/RR)
4. Classify drift:
   - **STALE-EVIDENCE**: Evidence changed + decision differs
   - **STALE-RULES**: Rules changed + decision differs
   - **STALE-BOTH**: Both changed + decision differs
   - **No drift**: Same decision outcome

**DriftResult** dataclass:
- `has_drift: bool`
- `drift_category: str | None` - STALE-EVIDENCE | STALE-RULES | STALE-BOTH
- Old vs new comparisons: state, evidence_hash, ruleset_version, CRG, RR
- Full before/after context for user review

**Review Interface**:
```python
review_drift() -> list[DriftResult]
```
- Queries all STALE-* and INDETERMINATE decisions
- Returns DriftResult for each
- Foundation for `canon drift review` CLI command

### Configuration Updates

**DatabaseConfig** extended:
```python
class DatabaseConfig(BaseModel):
    music_graph_path: Path = Path("musicgraph.sqlite")
    charts_path: Path = Path("charts.sqlite")
    decisions_path: Path = Path("decisions.sqlite")  # NEW
```

**Environment variable**:
- `CHART_BINDER_DATABASE_DECISIONS_PATH`

### Module Exports

Updated `__init__.py`:
```python
__all__ = (
    ...,
    "DecisionsDB",      # NEW
    "DriftDetector",    # NEW
    ...
)
```

## Test Coverage

**14 tests passing** (5 DecisionsDB + 5 DriftDetector + 4 Config)

### DecisionsDB Tests

1. **`test_decisions_db_schema`** - Verify all 5 tables created
2. **`test_file_id_generation`** - Deterministic file signatures
3. **`test_file_artifact_crud`** - File tracking operations
4. **`test_decision_crud`** - Decision storage and retrieval
5. **`test_decision_history`** - Automatic archival on update

### DriftDetector Tests

1. **`test_evidence_hash_deterministic`** - Sorted keys produce same hash
2. **`test_evidence_hash_strips_volatile`** - Timestamps excluded from hash
3. **`test_detect_drift_no_stored_decision`** - UNDECIDED → DECIDED flow
4. **`test_detect_drift_evidence_changed`** - STALE-EVIDENCE detection
5. **`test_detect_drift_rules_changed`** - STALE-RULES detection

### Quality Gates

- ✅ All tests passing (100%)
- ✅ Ruff linting clean
- ✅ Type checking clean (basedpyright)

## Implementation Notes

### Design Decisions

1. **Automatic Archival**
   - `upsert_decision()` checks for existing decision
   - Archives old decision to `decision_history` before update
   - Ensures complete audit trail without manual steps
   - Sets `superseded_reason = REFRESH` by default

2. **Evidence Hash Stability**
   - Volatile field removal prevents spurious drift
   - Sorted JSON ensures determinism across Python versions
   - SHA256 chosen for collision resistance

3. **File Signature**
   - Uses path:size:mtime for file_id
   - More stable than content hash (fast, no file read)
   - Handles renames (new file_id = new decision)
   - Detects file modifications via mtime

4. **State Machine Explicitness**
   - `DecisionState` enum prevents typos
   - Database stores string for human readability
   - Type safety in Python code

5. **SQLite Optimizations**
   - WAL mode: better concurrency, crash recovery
   - Foreign keys ON: referential integrity
   - Strategic indexes on common query patterns

### Drift Detection Logic

**Scenario 1: Upstream data changed**
```
Old: Evidence hash abc123, CRG=rg1
New: Evidence hash def456, CRG=rg2
→ STALE-EVIDENCE (Discogs backfilled premiere date)
```

**Scenario 2: Rules updated**
```
Old: Ruleset canon-1.0, CRG=rg1
New: Ruleset canon-2.0, CRG=rg2 (same evidence)
→ STALE-RULES (Lead window changed from 90→120 days)
```

**Scenario 3: Both changed**
```
Old: Evidence abc, ruleset 1.0, CRG=rg1
New: Evidence def, ruleset 2.0, CRG=rg2
→ STALE-BOTH
```

### Future Integration (Out of Scope)

**Epic 10 - CLI UX** will add:
```bash
canon drift review              # List all STALE-* decisions
canon drift review --apply      # Accept all new decisions
canon drift review --pin <file> # Pin current decision
canon decide --frozen           # Error on drift instead of recompute
```

**Epic 11 - Explainability** will use:
- `decision.trace_compact` for tag embedding
- Full structured trace storage in decisions
- Side-by-side comparison in drift review UI

## Acceptance Criteria

Per `docs/roadmap.md` Epic 6:

✅ **`file_artifact` signature**
- Stable hash from path+size+mtime
- `generate_file_id()` deterministic
- Tracks library location, duration, fingerprint

✅ **`decision` row with state machine**
- Full schema with CRG, RR, recording MBID
- `ruleset_version`, `config_snapshot`, `evidence_hash`
- Compact trace string
- State transitions implemented
- Pin flag supported

✅ **`decision_history` with reasons**
- Archives on every decision change
- `superseded_at`, `superseded_reason` tracked
- Full audit trail preserved

✅ **State transitions accurate**
- UNDECIDED → DECIDED ✅
- DECIDED → STALE-* ✅
- STALE-* → DECIDED ✅
- Enum-based safety ✅

✅ **Drift categories (STALE-*) correct**
- STALE-EVIDENCE: evidence hash differs ✅
- STALE-RULES: ruleset version differs ✅
- STALE-BOTH: both differ ✅
- Controlled test scenarios pass ✅

✅ **Evidence hash deterministic**
- Sorted keys ✅
- Volatile fields stripped ✅
- Tests verify stability ✅

## Dependencies & Order

**Builds on**:
- ✅ Epic 2: MusicGraphDB for entity storage
- ✅ Epic 3: Normalization for work_key generation
- ✅ Epic 5: Resolver for CRG/RR selection

**Enables**:
- Epic 10: CLI `canon drift review` command
- Epic 11: Explainability with trace embedding
- Epic 12: Beets plugin decision persistence

## Testing Instructions

```bash
# Run all Epic 6 tests
uv run pytest src/chart_binder/decisions_db.py src/chart_binder/drift.py -v

# Run all tests
uv run pytest

# Lint check
uv run ruff check src/chart_binder/

# Type check
uv run basedpyright src/chart_binder/
```

### Example Usage

```python
from pathlib import Path
from chart_binder import DecisionsDB, DriftDetector

# Initialize
db = DecisionsDB(Path("decisions.sqlite"))
detector = DriftDetector(db)

# Track a file
file_id = DecisionsDB.generate_file_id(
    Path("/music/song.mp3"),
    size=1024000,
    mtime=1234567890.0
)

db.upsert_file_artifact(
    file_id=file_id,
    library_root="/music",
    relative_path="artist/album/song.mp3",
    duration_ms=180000
)

# Store decision
db.upsert_decision(
    file_id=file_id,
    work_key="artist_songtitle",
    mb_rg_id="rg-abc-123",
    mb_release_id="rel-def-456",
    mb_recording_id="rec-ghi-789",
    ruleset_version="canon-1.0",
    config_snapshot={"lead_window_days": 90},
    evidence_hash="evidence_abc123",
    trace_compact="CRG:ALBUM|RR:ORIGIN",
)

# Later: detect drift
evidence_bundle = {...}  # From live sources
new_decision = {"mb_rg_id": "rg-xyz-999", "mb_release_id": "rel-uvw-888"}

drift = detector.detect_drift(
    file_id=file_id,
    new_evidence_bundle=evidence_bundle,
    new_decision=new_decision,
    new_ruleset_version="canon-1.0"
)

if drift.has_drift:
    print(f"Drift detected: {drift.drift_category}")
    print(f"Old CRG: {drift.old_crg} → New CRG: {drift.new_crg}")
```

## Related

- **Spec**: `docs/spec.md` Section 2 (System Overview)
- **Roadmap**: `docs/roadmap.md` Epic 6
- **Drift Contract**: `docs/appendix/drift_and_determinism_contract_v1.md`
- **DB Schema**: `docs/appendix/db_schema_sketch_v1.md` Section 2

---

**Ready for review!** All acceptance criteria met, comprehensive test coverage, and full drift detection infrastructure in place.
