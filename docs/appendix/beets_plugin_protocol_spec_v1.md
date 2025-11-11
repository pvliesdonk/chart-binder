# Beets Plugin Protocol Spec (v1, conceptual)

## Purpose

Let Beets use your **canonicalizer** instead of/default alongside Beets’ native matching, and attach your compact **CHARTS** + IDs to items/albums. The plugin must be a thin adapter over your library API.

## Scope & Modes

* **Advisory mode (default):** show your decision/rationale; user can accept to overwrite Beets’ candidate, or just augment with IDs/CHARTS.
* **Authoritative mode:** your CRG/RR replaces Beets’ album/track metadata automatically (with rollback).
* **Augment-only mode:** never change core fields (album/artist/title/date); only add IDs/CHARTS + trace.

## Hooks & Data Flow

1. **Importer start (`import_begin`)**

   * Load canonicalizer config, ruleset versions, caches status.
   * Expose plugin options in CLI flags (`--canon=advisory|authoritative|augment`, `--canon-explain`, `--canon-dry-run`).

2. **Per-file analysis (`import_task_files`)**

   * Extract duration + existing tags from Beets `Item`.
   * Call your library: `identify(file) -> CandidateSet`.
   * If configured, batch fingerprinting (Chromaprint) outside the Beets critical path.

3. **Decision stage (`import_task_match`)**

   * Call `decide(candidates, config) -> CanonicalDecision {CRG, RR, recording?}`.
   * Build `Decision Summary`:

     * `release_group`: title, primary type, first date
     * `representative_release`: country/date/label
     * `confidence` & rule path (`CRG:…`, `RR:…`)
     * drift indicator if re-importing

4. **Charts attach (optional step)**

   * Resolve `work_key` → `charts.export(work_key) -> CHARTS blob`.
   * Keep this read-only if charts DB isn’t present.

5. **User decision (advisory)**

   * If Beets found a different candidate: show side-by-side

     * Beets vs Canonizer: album, date, label, track count
     * Your rationale code + shortest explanation (1–2 lines)
   * Actions: **Accept Canon**, **Keep Beets**, **Augment Only**, **Skip/Review Later**.

6. **Write stage (`import_task_apply`)**

   * If **authoritative** or **accepted**:

     * Update Beets `Item` fields (see Tag Writer map’s “Core fields”).
     * Add custom fields for IDs/CHARTS/trace (Beets supports flexible attributes).
   * If **augment-only**:

     * Only set IDs/CHARTS/trace.

7. **Finalize (`import_end`)**

   * Emit import summary:

     * #items canonized, #augmented, #skipped, #indeterminate
     * top 10 conflicts & why
     * cache hits/misses

## Beets Field Mappings (logical → Beets Item/Album fields)

* Core album-level (when you take over): `album`, `albumartist`, `original_year`, `label`, `country` (custom), `catalognum` (custom), `media` (format)
* Core track-level: `title`, `artist`, `track`, `disc`, `year`, `tracktotal`, `disctotal`, `length`
* Custom (all strings unless noted):

  * `mb_recording_id`
  * `mb_release_group_id`
  * `mb_release_id`
  * `discogs_master_id`
  * `discogs_release_id`
  * `spotify_track_id`
  * `spotify_album_id`
  * `wikidata_qid`
  * `canonical_release_type` (album|single|ep|soundtrack|live)
  * `charts` (minified JSON)
  * `tag_decision_trace` (compact string)
  * `canon_ruleset_version`
  * `canon_evidence_hash`
  * `canon_state` (DECIDED|INDETERMINATE|STALE-*)

## Conflict Policy

* **Compilation exclusion**: if Beets picked a VA comp and your decision says “not canonical,” present that clearly; default to your decision in authoritative mode.
* **Promo-single window**: surface `delta_days` so users understand album vs single choices.
* **Reissue guard**: show when Beets picked a 30-year remaster; offer the origin-country earliest release.

## Overrides & Pinning

* Per-artist and per-item overrides (stored in your `override_rule` table) surfaced as Beets commands:

  * `beet canon-pin` (pin current decision)
  * `beet canon-unpin`
  * `beet canon-override rg=<mb_rg_id>`
  * `beet canon-explain` (print full trace)

## Failure & Offline Modes

* If your caches are missing or API fails: fall back to **augment-only** with whatever IDs you can infer; mark `canon_state=INDETERMINATE`.
* `--canon-offline`: operate solely from local caches/decisions; never hit the network.

## UX Notes

* Keep prompts terse; one-line rationale + codes.
* Provide a `--no-prompt` batch mode (advisory but non-interactive) with a `--accept-threshold` (e.g., accept only when your outcome isn’t INDETERMINATE).
