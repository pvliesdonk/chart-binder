# Canonicalization Rule Table (machine-checkable, no code)

This is the deterministic checklist your engine applies. It selects:

* a **Canonical Release Group (CRG)**, and
* a **Representative Release (RR)** within that CRG.

Tie-breakers and knobs are explicit. LLM adjudication is used only when rules end in a true tie.

## A) Inputs (evidence bundle keys the rules rely on)

* `artist`

  * `mb_artist_id`, `name`, `begin_area_country?`, `wikidata_country?`
* `recording` (per candidate)

  * `mb_recording_id`, `title`, `length_ms?`, `isrcs[]?`, `disambig?`, `flags` (`is_live?`, `is_remix?`, `is_acoustic?`, `is_mono?`, `is_demo?`)
* `release_group` (per candidate RG)

  * `mb_rg_id`, `title`, `primary_type ∈ {Album, Single, EP, Soundtrack, Live, Other}`, `secondary_types[]`,
  * `first_release_date?`, `labels[]`, `countries[]`, `discogs_master_id?`, `spotify_album_id?`
* `releases` (within each candidate RG)

  * list of: `mb_release_id`, `date?`, `country?`, `label?`, `format?`, `catno?`, `packaging?`, `title`, `flags` (`is_official?`, `is_promo?`, `is_bootleg?`, `is_reissue_hint?`, `is_remaster_hint?`)
* `timeline_facts` (computed)

  * `earliest_soundtrack_date?`, `earliest_album_date?`, `earliest_single_ep_date?`, `earliest_live_date?`, each with `source ∈ {MB, Discogs, Both}`
  * `lead_single_window_days` (config)
  * `reissue_long_gap_years` (config)
* `linkage`

  * `discogs_release_ids?[]`, `discogs_master_id?`, `spotify_track_id?`, `wikidata_qid?`
* `quality_meta`

  * `source_completeness_score` per candidate (heuristic), `http_cache_age`, `entity_cache_age`

> All fields marked `?` may be missing; rules specify fallbacks.

---

## B) Canonical Release Group selection (CRG)

**Preconditions:**

* Build candidate set of `(recording, release_group)` pairs that plausibly contain *the same recording* (AcoustID/ISRC/length/title-norm corroboration).

**Rule order (first match wins):**

1. **Soundtrack Origin**

   * Predicate: exists candidate RG with `primary_type == Soundtrack` and its `first_release_date` ≤ min of all other types for this recording.
   * Action: **CRG = that Soundtrack RG** (earliest by date; tie → label authority → artist origin country presence).
   * Rationale code: `CRG:SOUNDTRACK_PREMIERE`.

2. **Album vs Promo-Single (Lead-Window)**

   * Predicate A (album-first within window):

     * An Album RG exists with `earliest_album_date = D_album`.
     * A Single/EP RG exists with `earliest_single_ep_date = D_single`.
     * `0 < (D_album - D_single) ≤ LEAD_WINDOW_DAYS` **and** single is flagged promo/lead (from MB secondary type, or Discogs notes, or pattern match like “Promo”, “Advance”).

   * Action A: **CRG = Album RG** (earliest by date).

   * Rationale: `CRG:ALBUM_LEAD_WINDOW`.

   * Predicate B (single truly first):

     * A Single/EP RG exists with `D_single` and either `D_album` is null or `(D_album - D_single) > LEAD_WINDOW_DAYS`.

   * Action B: **CRG = Single/EP RG** (earliest by date).

   * Rationale: `CRG:SINGLE_TRUE_PREMIERE`.

3. **Live Origin**

   * Predicate: All non-Live candidates are absent **or** verified later; at least one Live RG contains this recording and `earliest_live_date` is the global earliest.
   * Action: **CRG = earliest Live RG**.
   * Rationale: `CRG:LIVE_ONLY_PREMIERE`.

4. **Remix / DJ-mix / Other Intent**

   * Predicate: Recording is a remix/alt version (flags or title-norm, e.g., “Remix”, “Radio Edit”, “Extended Mix”) and the earliest occurrence is on a Single/EP or specialized RG (Remix EP, DJ Mix).
   * Action: **CRG = earliest RG matching that intent**.
   * Rationale: `CRG:INTENT_MATCH`.

5. **Compilation Exclusion**

   * Predicate: A Various-Artists compilation is among candidates but no evidence it premiered there.
   * Action: **DO NOT choose a VA compilation**. Drop such candidates unless they are strictly earliest *and* there is explicit premiere evidence.
   * Rationale: `CRG:COMPILATION_EXCLUDED` or `CRG:COMPILATION_AS_PREMIERE` (rare).

6. **Fallback by Earliest Official Date**

   * Predicate: No earlier rule fired; at least one candidate has `first_release_date`.
   * Action: **CRG = RG with earliest confirmed official first_release_date**.
   * Tie-breakers:

     1. Presence of artist origin country in RG’s releases.
     2. Label authority (curated imprint priority list; else lexicographic).
     3. Country precedence (artist origin → global markets → others).
   * Rationale: `CRG:EARLIEST_OFFICIAL`.

7. **Ambiguous / Insufficient**

   * Predicate: Dates missing or conflicting such that ≥2 RGs remain tied after tie-breakers.
   * Action: **INDETERMINATE** → emit `missing_facts[]` (e.g., “need single street date”, “confirm OST release liner”). Optionally escalate to LLM adjudicator with the structured table.
   * Rationale: `CRG:INDETERMINATE`.

---

## C) Representative Release selection (RR) within CRG

From **CRG**’s releases:

1. **Filter official**

   * Keep `is_official == true`; drop promos/bootlegs unless *only promos exist* (then keep best-evidenced promo).

2. **Artist origin country preference**

   * If any release has `country == ARTIST_ORIGIN_COUNTRY`, take the **earliest date** among those.

3. **Else earliest worldwide**

   * Take the earliest `date` among remaining official releases.

4. **Reissue / Remaster guard**

   * If chosen release matches `REISSUE_TERMS` or `year_gap ≥ REISSUE_LONG_GAP_YEARS` **and** there exists an earlier non-reissue official release, prefer the earlier one.
   * Exception: if this RR is the **first official availability** of this specific recording/version (evidence flag), keep it.

5. **Tie-breakers**

   * Label authority list → original‐era format (7" for classic singles, LP for albums) → lowest catalog number → deterministic lexicographic on `mb_release_id`.

Output both IDs:

* `RELEASE_GROUP_MBID = CRG.mb_rg_id`
* `RELEASE_MBID = RR.mb_release_id`

Rationale code: `RR:ORIGIN_COUNTRY_EARLIEST` | `RR:WORLD_EARLIEST` | `RR:REISSUE_FILTER_APPLIED`.

---

## D) Alternate versions registry (non-canonical but notable)

When the selected recording has known re-recordings or later culturally dominant editions:

* Emit `ALT_VERSIONS[]` entries with `{type: "re_recording"|"remaster"|"edit", recording_id?, rg_id?, release_id?, note}`.
* Canonical remains anchored to the original-era CRG/RR.

---

## E) Decision Trace (compact, auditable)

A minimal string (for tag embedding) plus a full structured trace (for logs).

* **Compact tag string (example):**
  `evh=9b1f2c…;crg=SINGLE_TRUE_PREMIERE;rr=ORIGIN_COUNTRY_EARLIEST;src=mb,dc;cfg=lw90,rg10`

  Where:

  * `evh` = evidence_hash (stable digest over sorted evidence fields),
  * `crg`/`rr` = rationale codes,
  * `src` = sources used (mb=MusicBrainz, dc=Discogs, sp=Spotify, wd=Wikidata),
  * `cfg` = knobs snapshot (`lw` lead window days, `rg` reissue gap years).

* **Full trace (stored in your DB/cache):**

  ```
  {
    "ruleset_version": "1.0",
    "evidence_hash": "…",
    "artist_origin_country": "UK",
    "considered_candidates": [
      { "rg": "...", "type": "Album", "first_date": "1991-02-04", "sources": ["MB"] },
      { "rg": "...", "type": "Single", "first_date": "1990-11-12", "promo_hint": true, "sources": ["MB","Discogs"] }
    ],
    "crg_selection": {
      "rule": "CRG:ALBUM_LEAD_WINDOW",
      "lead_window_days": 90,
      "delta_days": 84
    },
    "rr_selection": {
      "rule": "RR:ORIGIN_COUNTRY_EARLIEST",
      "release": {"id":"…","date":"1991-02-04","country":"UK","label":"…"}
    },
    "missing_facts": []
  }
  ```

---

## F) Config knobs (externally settable; snapshot in trace)

* `LEAD_WINDOW_DAYS` (default 90)
* `REISSUE_LONG_GAP_YEARS` (default 10)
* `REISSUE_TERMS` (case/locale-insensitive patterns; curated list)
* `LABEL_AUTHORITY_ORDER` (optional list; else lexicographic)
* `ARTIST_ORIGIN_FALLBACK = [MB_BEGIN_AREA_COUNTRY, WIKIDATA_P27_OR_P495, EARLIEST_COMMERCIAL_RELEASE_COUNTRY]`

---

## G) Error handling & safe outcomes

* If **no candidate RG** has any date evidence: return `INDETERMINATE` with `missing_facts = ["first_release_date"]`.
* If **conflicting timelines** (e.g., Discogs vs MB disagree by > 1 year): mark `conflict_sources`, lower confidence, and prefer the source with higher completeness score; keep the loser as `challenge_candidate` in the trace.
* If **fingerprint only** with weak corroboration: downgrade candidate weight; require at least one of (ISRC match, length±2s, strong title-norm) to proceed; else `INDETERMINATE`.

---

# Evidence Bundle Schema (frozen v1, concise)

This is the minimal field set you persist, hash, and present to an adjudicator (human or LLM) when needed.

```
version: 1
artifact:
  file_sig: { duration_ms, fp_id?, fp_confidence?, orig_tags_hash }
artist:
  mb_artist_id
  name
  begin_area_country?
  wikidata_country?
recording_candidates:  # array
  - mb_recording_id
    title
    length_ms?
    isrcs?[]
    flags: { is_live?, is_remix?, is_demo?, is_radio_edit? }
    rg_candidates:  # array (release groups containing this recording)
      - mb_rg_id
        title
        primary_type
        secondary_types?[]
        first_release_date?
        discogs_master_id?
        spotify_album_id?
        labels?[]       # {name, mb_label_id?}
        countries?[]    # from constituent releases
        releases:       # trimmed list needed for RR
          - mb_release_id
            date?
            country?
            label?
            format?
            catno?
            title
            flags: { is_official?, is_promo?, is_bootleg?, is_reissue_hint?, is_remaster_hint? }
timeline_facts:
  earliest_soundtrack_date?
  earliest_album_date?
  earliest_single_ep_date?
  earliest_live_date?
provenance:
  sources_used: ["MB","Discogs","Spotify","Wikidata"]
  fetched_at_utc
  cache_age_s
config_snapshot:
  lead_window_days
  reissue_long_gap_years
  reissue_terms[]
```

**Hashing rule:** `evidence_hash = SHA256( canonical_json(evidence_bundle without fetched_at/cache_age) )`
