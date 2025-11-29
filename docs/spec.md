# Chart Binder Specification v1

## 1. Purpose & Scope

Provide a deterministic Python library with a thin CLI to canonicalize audio files to a Canonical Release Group (CRG) and a Representative Release (RR), embed portable canonical IDs and a compact CHARTS blob, and expose clear explainability and drift detection. A Beets plugin will reuse the library.

Non-goals for v1: GUIs, Postgres migrations. Live source integration and LLM adjudication are included in the roadmap but fixtures-first for deterministic testing (live sources via VCR cassettes; LLM kept disabled by default until Phase 2).

## 2. System Overview

Components (deterministic, testable, decoupled):

- **Multi-Source Ingestors**: MusicBrainz, Discogs, Spotify, Wikidata, AcoustID (live API clients with VCR-based tests for CI), chart scrapers (fixtures). **All sources treated as equals** — no single source is canonical. Entities from any source can participate in canonicalization decisions using synthetic IDs (e.g., `discogs-release-{id}`, `spotify-track-{id}`).
- **Cross-Source Validation**: Results appearing in multiple sources receive confidence boosts (+0.10 for 2 sources, +0.15 for 3+ sources), rewarding cross-platform validation.
- Normalizer: Normalization Ruleset v1 to produce artist/title cores, guests, and edition tags; maintains alias registry.
- Candidate Builder: Expands candidate graph (recording ↔ release_group ↔ release) using ISRC/AcoustID/title+duration corroboration from all sources.
- Resolver: Canonicalization Rule Table selects CRG first, then RR, with explicit rationale codes and safe INDETERMINATE outcomes. Can select entities from any source based on evidence quality.
- Charts ETL: Scrape → normalize → link → snapshot; exports compact CHARTS JSON blob v1.
- Tagging: Assembles canonical tagset and writes/reads tags across ID3v2.4, Vorbis/FLAC, MP4. Verifies round-trip.
- Stores & Cache: HTTP cache (per-source with TTLs), entity cache (`musicgraph.sqlite`), charts (`charts.sqlite`), decisions (`decisions.sqlite`).
- Explainability & Drift: Evidence bundle hashing, compact Decision Trace string, structured trace, drift classification.

## 3. Public Library API (stable for v1)

Conceptual signatures (Python):

```python
identify(file: Path) -> CandidateSet
decide(candidates: CandidateSet, config: Config) -> CanonicalDecision
assemble_tags(decision: CanonicalDecision, charts_blob: ChartsBlob | None, external_ids: dict[str, str] | None) -> TagSet
write_tags(file: Path, tagset: TagSet) -> WriteReport
verify(file: Path) -> TagSet
explain(target: CanonicalDecision | Path) -> HumanReadableTrace

# Charts
charts.ingest(chart_id: str, period: str) -> str  # run_id
charts.link(run_id: str, strategy: str = "default") -> LinkReport
charts.export(work_key: str) -> ChartsBlob

# Coverage
coverage.missing_from_chart(chart_id: str, period: str, threshold: float) -> Report
coverage.missing_in_library(chart_id: str, period: str | None = None) -> Report
```

Key data objects are immutable dataclasses with JSON codecs. All operations are deterministic under `--offline` or fixed fixtures.

## 4. CLI (thin, deterministic)

- `canon scan <paths…>`: fingerprint (optional) + read existing tags; print facts.
- `canon decide <paths…> [--explain] [--offline]`.
- `canon write <paths…> [--dry-run]`.
- `canon charts ingest <chart_id> <period>`.
- `canon charts link <chart_id> <period>`.
- `canon charts missing <chart_id> <period>`.
- `canon coverage missing-in-library <chart_id> [period]`.
- `canon cache status|purge`.
- `canon drift review`.

Modes: `--offline`, `--frozen`, `--refresh`, `--apply`, `--pin`, `--unpin`.

## 5. Config (TOML; env-overridable)

- `lead_window_days = 90`
- `reissue_long_gap_years = 10`
- `reissue_terms = ["remaster", "deluxe", …]`
- `artist_origin_fallback = ["MB_BEGIN_AREA_COUNTRY","WIKIDATA_P27_OR_P495","EARLIEST_COMMERCIAL_RELEASE_COUNTRY"]`
- `charts.enabled = ["t2000","t40","t100","zwaar"]`
- `charts.link.thresholds = {auto=0.85, review=0.60}`
- `cache.http.ttl = {mb=3600, discogs=86400, spotify=7200, wikidata=604800}`
- `cache.paths = {http="/cache/http", db="/db"}`
- `labels.authority_order = ["Island","EMI","Columbia","Warner", …]` (optional)
- LLM adjudication (Phase 2):
  - `llm.enabled = false` (set to `true` for INDETERMINATE tie-breaking)
  - `llm.model_id = "claude-3.5-sonnet"` or `"gpt-4o"`, `"gpt-4o-mini"`
  - `llm.api_key_env = "ANTHROPIC_API_KEY"` or `"OPENAI_API_KEY"`
  - `llm.timeout_s = 30`
  - `llm.max_tokens = 1024`
  - `llm.auto_accept_threshold = 0.85` (auto-accept if confidence ≥ this)
  - `llm.review_threshold = 0.60` (escalate to human review if < this)
  - `llm.prompt_template_version = "v1"` (for A/B testing prompts)

## 6. Storage & Data Model (SQLite-first)

Files:

- `/cache/http/{source}/…` — raw responses (ETag-aware) + SQLite index.
- `/db/musicgraph.sqlite` — `artist`, `recording`, `release_group`, `release`, `recording_release`.
- `/db/charts.sqlite` — `chart`, `chart_run`, `chart_entry`, `alias_norm`, `chart_link`.
- `/db/decisions.sqlite` — `file_artifact`, `decision`, `decision_history`, `override_rule`.
- `/snapshots/charts/{chart_id}/{period}.json` — optional immutable export.

See docs/appendix/db_schema_sketch_v1.md for column-level details.

## 7. Normalization Ruleset v1 (summary)

Deterministic pipeline to produce:

- `artist_core`, `artist_guests[]` (guests standardized to " feat. ").
- `title_core`, `title_guests[]`.
- `tags[]`: edition/intent extracted (radio edit, live, remix, remaster, acoustic, OST, demo, mono/stereo, content, re_recording, medley, language).
- `diacritics_signature` for preference boosting without affecting keys.

Key steps: NFC/whitespace, casefold, punctuation canonicalization, diacritics-insensitive comparisons, guests normalization, descriptor extraction with guardrails (retain identity-bearing parts like Part II, catalog numbers, classical forms), canonical artist separators (internal token " • "), locale-aware alias rules (EN/NL). Idempotent; versioned as `norm-v1`.

Work key for fuzzy links: `hash(artist_key + " // " + title_key [+ length_bucket])`.

## 8. Canonicalization Rule Table (summary)

CRG selection (first match wins):

1) Soundtrack origin earliest; 2) Album vs Promo-Single lead-window resolution; 3) Live-only origin; 4) Intent match (remix/DJ-mix); 5) Compilation exclusion unless proven premiere; 6) Fallback earliest official date with tie-breakers (artist origin country presence → label authority → country precedence); else INDETERMINATE + `missing_facts`.

RR selection in CRG: filter official; prefer artist origin country earliest; else earliest worldwide; reissue/remaster guard (long-gap and term-based) unless first official availability of that version; tie-breakers: label authority → original-era format → lowest catno → deterministic ID.

Outputs: `RELEASE_GROUP_MBID`, `RELEASE_MBID`, optional recording ID; rationale codes: `CRG:*`, `RR:*`.

## 9. Evidence, Trace, and Drift

- Evidence bundle v1: minimal fields the rules rely on; canonical JSON hashed to `evidence_hash` (timestamps/cache ages removed).
- Decision Trace: compact tag string `evh=…;crg=…;rr=…;src=…;cfg=…` + full structured JSON trace stored in DB.
- States: UNDECIDED, DECIDED, STALE-EVIDENCE, STALE-RULES, STALE-BOTH, INDETERMINATE.
- `canon drift review` recomputes with current rules/config and classifies outcomes.

## 10. Charts ETL & CHARTS Blob

Charts ETL:

- `chart_run` + raw `chart_entry` → normalize via alias rules → link to `work_key` using methods `isrc|title_artist_year|bundle_release|manual` with confidence.
- Singles-as-bundles (Top 40): side-mapping confidence model with thresholds: auto ≥0.85; review 0.60–0.85; reject < 0.60. Double A-side if two sides auto-qualify within 0.05.
- Coverage: `% linked` and missing reports per run.

CHARTS tag blob v1 (minified JSON): `{"v":1,"c":[["<chart_id>",score,highest,"y|w",positions?],…]}`. Positions are optional for size budgeting; deterministic ordering by score desc, then highest asc.

Registry seed: `t2000`, `t40`, `t100`, `zwaar`.

## 11. Tagging Map (portable)

Core fields (overwrite only if authoritative/accepted): Title, Artist (no guests), Album (RG title by default), Album Artist, Original Year, Track/Disc No/Total, Label, Country, Media/Format, Canonical Release Type.

Canonical IDs: MB recording/RG/release, Discogs master/release, Spotify track/album, Wikidata QID.

Compact fields: CHARTS, TAG_DECISION_TRACE, CANON_RULESET_VERSION, CANON_EVIDENCE_HASH.

Round-trip verification after writes via same reader implementation; non-destructive first write (stash ORIG_* once). Container-specific guardrails: ID3v2.3 fallback (`TDOR` mirrored via custom if needed), MP4 `\xa9day` freeform, CHARTS size target ≤ ~3 KB.

## 12. Beets Plugin (adapter)

Modes: advisory (default), authoritative, augment-only. Hooks at import_begin, per-file analysis (identify), decision stage (decide), optional charts attach, user decision, write stage, finalize summary. Field mappings align to Beets Item/Album + flexible attributes for custom fields. CLI flags: `--canon`, `--canon-explain`, `--canon-dry-run`. Overrides and pinning commands persist to `override_rule`.

## 13. Determinism, Security, Operations

- Deterministic runs via `--offline` and fixed fixtures; cache-only vs live-refresh modes.
- Evidence built solely from caches + logged live fetches; ruleset and config snapshots stored with decisions.
- Logs avoid PII; file paths hashed or relative to library root. License/ToS-aware chart ingestion.
- Performance envelopes: identify/decide in batches, write/verify in smaller batches; per-source token buckets; SQLite with WAL, FK on.

## 14. Testing & QA

- Normalization QA pack v1: inputs → expected cores/tags; idempotence checks; collision audit.
- Charts ETL Test Pack v1: 10 focused runs covering yearly, weekly single bundles, medley, karaoke trap, retrospective labeling, OST cues, collaboration separators.
- Golden decisions for ~200 tracks with expected CRG/RR outcomes and rationale codes.
- Property checks: adding VA compilation must not flip canonicalization without true premiere evidence.

## 15. Versioning

- `ruleset_version` combines normalization/canonicalization/side-mapping versions (e.g., `canon-1.0.norm-v1.tags-v1.side-v1`).
- CHARTS wire format `v=1` is stable unless the on-tag schema changes.
- Database schema versioned via `schema_meta` with additive migrations.

## 16. Glossary

- CRG — Canonical Release Group; RR — Representative Release; Work key — normalized artist+title hash; Evidence bundle — minimal facts used by rules; Drift — decision changes due to rules/config/facts changes.
