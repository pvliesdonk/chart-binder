# Roadmap (v1)

This roadmap turns the specification into small, implementable units suitable for a mini LLM (fixtures first, no live network). Each epic lists focused features and crisp acceptance criteria.

## Milestones

- M0: Foundations (config, DBs, HTTP cache, normalization core)
- M1: Canonicalization (candidates + CRG/RR + decisions + drift)
- M2: Tagging + CLI + Charts ETL (fixtures) + CHARTS export
- M3: Beets plugin adapter + coverage reports

## Epics, Features, Acceptance

### Epic 1 — Project Skeleton & Config (M0)

- Features
  - Config loader: TOML + env override; typed `Config` with defaults and validation.
  - CLI entry `canon` with subcommand scaffold only (no business logic yet).
- Acceptance
  - `canon --help` renders subcommands.
  - Config merging unit tests (file+env). Defaults match spec.

### Epic 2 — HTTP Cache & Entity Cache (M0)

- Features
  - File-backed HTTP cache with SQLite index (ETag/Last-Modified/TTL).
  - `musicgraph.sqlite` schema creation; upsert helpers for `artist`, `recording`, `release_group`, `release`, `recording_release`.
- Acceptance
  - Cached GET respects TTL and validators.
  - CRUD round-trip verified; indices present; PRAGMA FK on.

### Epic 3 — Normalization v1 (M0)

- Features
  - Deterministic pipeline: NFC/space, casefold, punctuation/diacritics canonicalization, guests normalization, edition/descriptor extraction with guardrails, canonical separators.
  - Alias registry storage and matcher (`alias_norm`).
- Acceptance
  - All Normalization QA pack cases pass; idempotence holds.

### Epic 4 — Candidate Builder (M1)

- Features
  - Candidate discovery by (ISRC | AcoustID | title+artist_core+length bucket).
  - Evidence bundle v1 construction (deterministic canonical JSON; hash function).
- Acceptance
  - Fixture-based inputs produce expected candidate sets and evidence_hash stability.

### Epic 5 — Canonical Resolver: CRG & RR (M1)

- Features
  - Implement Canonicalization Rule Table: CRG selection (ordered rules) and RR selection with tie-breakers.
  - INDETERMINATE path emits `missing_facts` and challenge candidates.
- Acceptance
  - Golden decisions produce expected CRG/RR and rationale codes; ambiguous fixtures return INDETERMINATE with correct missing facts.

### Epic 6 — Decision Store & Drift (M1)

- Features
  - `file_artifact` signature; `decision` row with `ruleset_version`, `config_snapshot`, `evidence_hash`, compact trace; `decision_history` with reasons; state machine.
  - CLI `canon drift review` compares stored vs recomputed decisions.
- Acceptance
  - State transitions accurate; history rows archived; drift categories (STALE-*) correct under controlled changes.

### Epic 7 — Tag Assembly & Writers (M2)

- Features
  - Assemble canonical tagset; writers for ID3v2.4 (MP3), Vorbis/FLAC, MP4; compact fields (CHARTS/TRACE/RULESET/EVIDENCE); non-destructive first write.
  - Round-trip verify reader.
- Acceptance
  - After write+read, core fields/compact fields match; ORIG_* stashed once; minimal write set works across formats.

### Epic 8 — Charts ETL Core (M2)

- Features
  - `charts.sqlite` schema; `charts ingest` loads fixture pages; normalize `chart_entry` via alias registry.
  - Linker implements `title_artist_year` and `bundle_release` with confidence model and thresholds; coverage report.
- Acceptance
  - Charts ETL Test Pack: runs produce `chart_run`, `chart_entry`, `chart_link` with expected confidences; coverage % computed.

### Epic 9 — CHARTS Blob Export (M2)

- Features
  - Aggregate per-work scores/highest; deterministic minified JSON v1; optional positions embedding; size budgeting.
- Acceptance
  - Outputs match examples; sort/order/stability proven; blob size < ~3 KB without positions.

### Epic 10 — CLI UX & Modes (M2)

- Features
  - Implement `scan`, `decide [--explain]`, `write [--dry-run]`, `cache status|purge`, `coverage`, and `charts` commands; `--offline|--frozen|--refresh|--apply|--pin|--unpin`.
- Acceptance
  - Deterministic outputs with JSON/text modes; exit codes; progress summaries.

### Epic 11 — Explainability (M2)

- Features
  - Compact Decision Trace string + full structured trace persisted; normalization breadcrumbs and rule paths included.
- Acceptance
  - `canon decide --explain` prints rationale codes and key tipping facts; trace stored with decisions.

### Epic 12 — Beets Plugin Adapter (M3)

- Features
  - Thin adapter calling library API; modes (advisory/authoritative/augment-only); flexible attributes for custom fields; CLI flags; overrides/pin commands.
- Acceptance
  - Import flow shows side-by-side; writes respect mode; summary includes canonized/augmented/skipped/indeterminate and cache stats.

### Epic 13 — Human-in-the-Loop (Phase 2)

- Features
  - Minimal review queue for confidence 0.60–0.85 and ambiguous CRG; accept/keep/augment/skip; persist overrides with provenance.
- Acceptance
  - Queue resolves deterministically; overrides applied and auditable.

### Epic 14 — Quality, Performance, Security (Continuous)

- Features
  - Batch sizing; token-bucket rate limits; cache-only mode; PII-safe logging; lint/type/test gates; property tests.
- Acceptance
  - Lint/type clean; all tests green; logs contain no PII; rate limits exercised in tests.

## Dependencies & Order

1) Config → 2) Caches/DBs → 3) Normalization → 4) Candidates → 5) Resolver → 6) Decisions/Drift → 7) Tags → 8) Charts ETL → 9) CHARTS export → 10) CLI polish → 11) Explainability → 12) Beets → 13) Human-in-the-Loop.

## Mini‑LLM Implementation Guidance

- Prefer fixtures and stubs; disallow live network in CI.
- Keep tasks ≤ ~100–150 LOC; expose pure functions with typed inputs/outputs.
- Deterministic ordering everywhere (sorted keys, stable hashing, fixed rounding).
- Add acceptance tests per feature before wiring into CLI.

## Risks & Mitigations

- Upstream data drift: evidence hashing + drift states; cache-only mode for reproduction.
- Over-stripping titles/artists: exception-driven alias table and guardrails; QA pack coverage.
- Tag portability: minimal write set + round-trip verification + container guardrails.

## Deliverables per Milestone

- M0: Config module, DB schemas + helpers, normalization with QA tests.
- M1: Candidate builder, resolver, decision store + drift CLI, golden decisions.
- M2: Tag writers + verify, CLI subcommands, Charts ETL with fixtures, CHARTS export.
- M3: Beets plugin adapter and coverage reports.

