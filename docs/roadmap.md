# Roadmap (v1)

This roadmap turns the specification into small, implementable units suitable for a mini LLM (fixtures first, no live network). Each epic lists focused features and crisp acceptance criteria.

## Milestones

- M0: Foundations (config, DBs, HTTP cache, normalization core, live source ingestors)
- M1: Canonicalization (candidates + CRG/RR + decisions + drift)
- M2: Tagging + CLI + Charts ETL + CHARTS export
- M3: Beets plugin adapter + coverage reports
- Phase 2: LLM adjudication + human review queue

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

### Epic 3.5 — Live Source Ingestors (M0/M1 boundary)

- Features
  - **MusicBrainz API client**: recording/release-group/release lookups by MBID or search query; batch lookup optimization; URL relationships parsing; rate limiting (1 req/sec, configurable).
  - **Discogs API client**: master/release lookups by ID; OAuth authentication; marketplace data filtering; rate limiting (60/min auth, 25/min unauth).
  - **Spotify Web API client**: track/album metadata by ID or search; client credentials flow; preview URL and popularity extraction; rate limiting per ToS.
  - **Wikidata SPARQL client**: artist country/origin queries (P27, P495, P740); result caching; timeout handling.
  - **AcoustID client**: fingerprint submission and lookup; duration+fingerprint corroboration; confidence thresholds.
  - **Unified fetcher interface**: `fetch_recording(mbid)`, `search_recordings(isrc|title_artist)` with fallback chain; cache-aware with TTL/ETag respect; `--offline|--refresh|--frozen` mode support.
  - **Entity hydration**: parse API responses into `musicgraph.sqlite` entities; maintain external ID linkages; timestamp provenance (`fetched_at_utc`).
- Acceptance
  - Live API calls succeed with valid credentials (use VCR cassettes for CI); rate limits enforced (token bucket verification); cache hit/miss logic correct; `--offline` mode never hits network; entity CRUD round-trips with all fields populated; fixture-based tests continue passing via mocked responses.
- Notes
  - Tests use recorded VCR cassettes (pytest-vcr) to avoid live network in CI while validating response parsing.
  - Fixtures remain primary for deterministic golden tests; live sources augment with real-world variability testing.

### Epic 4 — Candidate Builder (M1)

- Features
  - Candidate discovery by (ISRC | AcoustID | title+artist_core+length bucket) using live sources from Epic 3.5.
  - Evidence bundle v1 construction (deterministic canonical JSON; hash function).
- Acceptance
  - Fixture-based inputs produce expected candidate sets and evidence_hash stability.
  - Live source integration tests with VCR cassettes validate end-to-end discovery flow.

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

### Epic 13 — LLM Adjudication & Human-in-the-Loop (Phase 2)

- Features
  - **LLM Adjudication for True Ties**:
    - Structured prompt template: evidence bundle (artist, recording candidates, RG candidates with releases, timeline facts) + config snapshot + rationale for INDETERMINATE state + missing facts.
    - Prompt format: System instruction defining role ("You are a music metadata expert..."), evidence as JSON with minimal explanation, specific question ("Which release group is the canonical premiere for this recording?"), output constraints (must return valid CRG MBID + confidence 0-1 + brief rationale).
    - Model invocation: configurable `llm.model_id` (e.g., `claude-3.5-sonnet`, `gpt-4o`), `llm.timeout_s` (default 30), `llm.max_tokens` (default 1024).
    - Response parsing: extract CRG/RR selection, confidence score, and rationale from structured JSON or natural language output; validate MBID exists in evidence bundle; log full prompt+response for audit.
    - Confidence thresholds: auto-accept ≥ 0.85, escalate to human review 0.60–0.85, reject < 0.60 (keep INDETERMINATE).
    - Fallback on LLM failure: timeout, parse error, or low confidence → escalate to human queue or keep INDETERMINATE.
  - **Human Review Queue**:
    - Queue population: INDETERMINATE decisions after all deterministic rules exhausted; LLM suggestions with confidence 0.60–0.85; conflicting sources (MB vs Discogs dates disagree > 1 year).
    - Review UI (CLI-first, web later): display evidence bundle, decision trace, LLM suggestion (if any), side-by-side candidate comparison.
    - Actions: `accept <crg_mbid> <rr_mbid>` (manual selection), `accept-llm` (accept LLM suggestion), `keep-indeterminate` (defer), `add-alias <from> <to>` (create normalization override), `skip` (ignore file).
    - Provenance tracking: `{user_id, timestamp, action, rationale_text, llm_suggestion_id}` stored in `decision_history` or `override_rule`.
  - **Override Rules**:
    - Per-artist or per-track scoped overrides: "For artist X, always prefer RG Y over Z" or "For recording R, canonical is always RG W".
    - Stored in `override_rule` table: `{scope, artist_mbid?, recording_mbid?, target_rg_mbid, target_rr_mbid?, created_by, created_at, reason}`.
    - Applied before resolver runs (hard override) or as tie-breaker (soft preference).
  - **Audit Trail**:
    - Full LLM prompt + response logged to `llm_adjudication_log` table: `{decision_id, model_id, prompt_template_version, prompt_json, response_json, confidence, accepted, created_at}`.
    - Decision history tracks source: `{source: "deterministic" | "llm_auto" | "llm_reviewed" | "manual"}`.
- Acceptance
  - LLM adjudication resolves 80%+ of INDETERMINATE cases with confidence ≥ 0.60; prompt+response logged for all invocations; timeout/error handling graceful; confidence thresholds enforced; human review queue displays all fields correctly; overrides applied and reflected in drift review; audit trail complete for compliance/debugging.
- Config
  ```toml
  [llm]
  enabled = false  # Set to true to enable LLM adjudication
  model_id = "claude-3.5-sonnet"  # or "gpt-4o", "gpt-4o-mini", etc.
  api_key_env = "ANTHROPIC_API_KEY"  # or "OPENAI_API_KEY"
  timeout_s = 30
  max_tokens = 1024
  auto_accept_threshold = 0.85  # Auto-accept if LLM confidence ≥ this
  review_threshold = 0.60  # Escalate to human if confidence < this
  prompt_template_version = "v1"  # Track prompt iterations for A/B testing
  ```

### Epic 14 — Quality, Performance, Security (Continuous)

- Features
  - Batch sizing; token-bucket rate limits; cache-only mode; PII-safe logging; lint/type/test gates; property tests.
- Acceptance
  - Lint/type clean; all tests green; logs contain no PII; rate limits exercised in tests.

## Dependencies & Order

1) Config → 2) Caches/DBs → 3) Normalization → 3.5) Live Sources → 4) Candidates → 5) Resolver → 6) Decisions/Drift → 7) Tags → 8) Charts ETL → 9) CHARTS export → 10) CLI polish → 11) Explainability → 12) Beets → 13) LLM Adjudication & Human Review.

## Mini‑LLM Implementation Guidance

- **Fixtures first for golden tests**: All deterministic logic (normalization, resolver rules) tested with fixtures. Use VCR cassettes (pytest-vcr) for live source tests to record/replay API responses.
- **Live sources for integration**: After fixtures validate core logic, live sources provide real-world variability testing and end-to-end validation.
- **CI remains deterministic**: Disallow live network in CI; use recorded cassettes or mocked responses. `--offline` mode enforced for reproducibility.
- Keep tasks ≤ ~100–150 LOC; expose pure functions with typed inputs/outputs.
- Deterministic ordering everywhere (sorted keys, stable hashing, fixed rounding).
- Add acceptance tests per feature before wiring into CLI.

## Risks & Mitigations

- Upstream data drift: evidence hashing + drift states; cache-only mode for reproduction.
- Over-stripping titles/artists: exception-driven alias table and guardrails; QA pack coverage.
- Tag portability: minimal write set + round-trip verification + container guardrails.

## Deliverables per Milestone

- M0: Config module, DB schemas + helpers, normalization with QA tests, live source ingestors with VCR-based tests.
- M1: Candidate builder (using live sources), resolver, decision store + drift CLI, golden decisions.
- M2: Tag writers + verify, CLI subcommands, Charts ETL, CHARTS export.
- M3: Beets plugin adapter and coverage reports.
- Phase 2: LLM adjudication module, human review CLI, override rules, audit trail.

