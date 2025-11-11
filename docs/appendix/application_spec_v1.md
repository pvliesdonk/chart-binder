# Application Spec v1 (Library + CLI)

## Purpose

A Python library (clean API) with a thin CLI for batch ops. Beets plugin later reuses the library.

## Components

1. **Ingestors**: MusicBrainz, Discogs, Spotify, Wikidata/Wikipedia, Chromaprint/AcoustID, Charts scrapers.
2. **Normalizer**: applies Normalization Ruleset v1 + Truth Table.
3. **Resolver**: builds candidate graph (recording↔RG↔release), applies Canonicalization Rule Table, picks CRG + RR.
4. **Charts ETL**: scrape → normalize → link → snapshot; exposes coverage queries.
5. **Tagger**: assembles compact tag set, writes ID3/Vorbis/MP4, then verifies.
6. **Cache & Store**: HTTP cache, entity cache, charts DB, decision store.
7. **Explainability**: Decision Trace, evidence bundle hashing, drift detection.

## Public Library API (concept)

* `identify(file) -> CandidateSet`
* `decide(candidates, config) -> CanonicalDecision`
* `charts.ingest(chart_id, period) -> run_id`
* `charts.link(run_id, strategy) -> LinkReport`
* `charts.export(work_key) -> ChartsBlob` (the compact JSON)
* `coverage.missing_from_chart(chart_id, period, threshold) -> Report`
* `coverage.missing_in_library(chart_id[, period]) -> Report`
* `assemble_tags(decision, charts_blob, external_ids) -> TagSet`
* `write_tags(file, tagset) -> WriteReport`
* `verify(file) -> TagSet`
* `explain(decision|file) -> HumanReadableTrace`

## CLI (thin, deterministic)

* `canon scan <paths…>` — fingerprint + extract existing tags; print facts.
* `canon decide <paths…> [--explain]` — run resolver; show CRG/RR and rationale.
* `canon write <paths…> [--dry-run]` — write tags + verify round-trip.
* `canon charts ingest <chart_id> <period>` — run scraper into snapshot.
* `canon charts link <chart_id> <period>` — normalize + link.
* `canon charts missing <chart_id> <period>` — show missing links.
* `canon coverage missing-in-library <chart_id> [period]` — gaps vs your library.
* `canon cache status|purge` — view/clear caches.
* `canon drift review` — detect and show decisions that would change under current data/rules.

## Config (single TOML block, env-overridable)

* `lead_window_days` (default 90)
* `reissue_long_gap_years` (default 10)
* `reissue_terms` (list)
* `artist_origin_fallback = ["MB_BEGIN_AREA_COUNTRY","WIKIDATA_P27_OR_P495","EARLIEST_COMMERCIAL_RELEASE_COUNTRY"]`
* `charts.enabled = ["t2000","t40","t100","zwaar"]`
* `charts.link.thresholds = {auto=0.85, review=0.60}`
* `cache.http.ttl = {mb=3600, discogs=86400, spotify=7200, wikidata=604800}`
* `cache.paths = {...}`
* `llm.enabled=false` (tie-break only), `llm.model_id?`, `llm.timeout_s?`
* `labels.authority_order = ["Island","EMI","Columbia","Warner",…]` (optional)

## Storage Layout (single file or split DBs)

* `/cache/http/{source}/…` — raw responses (ETag aware).
* `/db/musicgraph.sqlite` — entities (artists, recordings, RGs, releases, joins).
* `/db/charts.sqlite` — `chart`, `chart_run`, `chart_entry`, `alias_norm`, `chart_link`.
* `/db/decisions.sqlite` — `decision` (CRG/RR, evidence_hash, trace, cfg snapshot).
* `/snapshots/charts/{chart_id}/{period}.json` — immutable export (optional).

## Data Model (essentials)

* **Entities**: `artist`, `recording`, `release_group`, `release` (flattened fields used by rules).
* **Joins**: `recording_release`, `release_release_group`, external IDs tables.
* **Decisions**: `{file_id, work_key, crg_id, rr_id, ruleset_version, evidence_hash, trace_compact, created_at}`
* **Charts**: as in ETL test pack.

## Determinism & Drift

* `evidence_hash = SHA256(canonical_json(evidence))` (excluding volatile timestamps).
* Decision stores ruleset version + config snapshot.
* `drift review` compares stored decision vs recomputed; outcomes: `UNCHANGED | WOULD_CHANGE | INDETERMINATE`.

## Human-in-the-Loop

* Queue: ambiguities (album vs single), side-mapping 0.60–0.85, conflicts (MB vs Discogs dates > 365d).
* Actions: accept A, accept B, mark INDETERMINATE, add alias/override (scoped per artist or track).
* Provenance: who/when/why stored with decision.

## Tagging Philosophy

* Standard frames reflect canonical decision (Album/Artist/Title/Date).
* Power-user data in compact fields: `RELEASE_GROUP_MBID`, `RELEASE_MBID`, `RECORDING_MBID`, Discogs/Spotify IDs, `CHARTS`, `TAG_DECISION_TRACE`.
* Non-destructive first write: stash `ORIG_*` once; batch undo via pre-write snapshots.

## Performance

* Two-pass pipeline: (A) identify/decide (parallel, read-heavy), (B) write/verify (batched).
* Rate-limit per source with token buckets; batch MB lookups by IDs.
* Cache-only mode for long runs.

## Testing & QA

* Use **Normalization QA pack** and **Charts ETL test pack** as acceptance fixtures.
* Golden decisions for ~200 tracks with known CRG/RR outcomes.
* Property tests: adding a VA compilation candidate must not flip results unless true premiere.

## Security & Privacy

* Avoid PII in logs (hash file paths); allow library-root relative paths.
* Honor robots/ToS; store licenses per chart source in `chart` table.
* API keys via env; never embed in snapshots.
