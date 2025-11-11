# DB Schema Sketch (v1)

Design favors clarity and indexability. SQLite-friendly; migrate easily to Postgres.

## 1) Music entity cache (`musicgraph.sqlite`)

### `artist`

* `mb_artist_id TEXT PRIMARY KEY`
* `name TEXT`
* `begin_area_country TEXT NULL`   — ISO-3166-1 alpha-2
* `wikidata_qid TEXT NULL`
* `diacritics_signature TEXT NULL`
* `fetched_at_utc TEXT`

**Index:** primary key.

### `recording`

* `mb_recording_id TEXT PRIMARY KEY`
* `mb_artist_id TEXT NOT NULL REFERENCES artist(mb_artist_id)`
* `title TEXT`
* `length_ms INTEGER NULL`
* `isrcs_json TEXT NULL`           — JSON array
* `flags_json TEXT NULL`           — `{is_live,is_remix,is_demo,is_radio_edit,...}`
* `fetched_at_utc TEXT`

**Index:** `recording_artist_idx` on `(mb_artist_id)`.

### `release_group`

* `mb_rg_id TEXT PRIMARY KEY`
* `title TEXT`
* `primary_type TEXT`              — enum-ish
* `secondary_types_json TEXT NULL`
* `first_release_date TEXT NULL`   — `YYYY[-MM[-DD]]`
* `labels_json TEXT NULL`
* `countries_json TEXT NULL`
* `discogs_master_id TEXT NULL`
* `spotify_album_id TEXT NULL`
* `fetched_at_utc TEXT`

**Index:** `rg_firstdate_idx` on `(first_release_date)`.

### `release`

* `mb_release_id TEXT PRIMARY KEY`
* `mb_rg_id TEXT NOT NULL REFERENCES release_group(mb_rg_id)`
* `title TEXT`
* `date TEXT NULL`
* `country TEXT NULL`
* `label TEXT NULL`
* `format TEXT NULL`
* `catno TEXT NULL`
* `flags_json TEXT NULL`           — `{is_official,is_promo,is_bootleg,is_reissue_hint,is_remaster_hint}`
* `discogs_release_id TEXT NULL`
* `fetched_at_utc TEXT`

**Index:**

* `rel_rg_date_idx` on `(mb_rg_id, date)`
* `rel_country_idx` on `(country, date)`

### `recording_release` (join)

* `mb_recording_id TEXT REFERENCES recording(mb_recording_id)`
* `mb_release_id TEXT REFERENCES release(mb_release_id)`
* `track_position INTEGER NULL`
* `disc_number INTEGER NULL`
* PRIMARY KEY (`mb_recording_id`,`mb_release_id`)

**Index:** `rec_rel_release_idx` on `(mb_release_id)`.

---

## 2) Decisions & files (`decisions.sqlite`)

### `file_artifact`

* `file_id TEXT PRIMARY KEY`       — stable hash of path+size+mtime (or your chosen signature)
* `library_root TEXT`
* `relative_path TEXT`
* `duration_ms INTEGER NULL`
* `fp_id TEXT NULL`                — fingerprint id
* `orig_tags_hash TEXT NULL`
* `created_at TEXT`

**Index:** `file_path_idx` on `(library_root, relative_path)`.

### `decision`

* `decision_id TEXT PRIMARY KEY`
* `file_id TEXT NOT NULL REFERENCES file_artifact(file_id)`
* `work_key TEXT NOT NULL`         — normalized composite key
* `mb_rg_id TEXT NOT NULL`         — CRG
* `mb_release_id TEXT NOT NULL`    — RR
* `mb_recording_id TEXT NULL`
* `ruleset_version TEXT NOT NULL`
* `config_snapshot_json TEXT NOT NULL`
* `evidence_hash TEXT NOT NULL`
* `trace_compact TEXT NOT NULL`    — compact tag string
* `state TEXT NOT NULL`            — `DECIDED | STALE-* | INDETERMINATE`
* `pinned BOOLEAN NOT NULL DEFAULT 0`
* `created_at TEXT NOT NULL`
* `updated_at TEXT NOT NULL`

**Indexes:**

* `decision_file_idx` on `(file_id)`
* `decision_state_idx` on `(state)`
* `decision_rg_idx` on `(mb_rg_id)`
* Unique constraint `(file_id)` active only for current rows (history in separate table).

### `decision_history`

* All columns from `decision` **plus**:

  * `superseded_at TEXT NOT NULL`
  * `superseded_reason TEXT NOT NULL`  — `refresh|ruleset_change|manual_override|pin`
* Primary key: `(decision_id, superseded_at)`

### `override_rule` (optional, human-in-the-loop)

* `override_id TEXT PRIMARY KEY`
* `scope TEXT`            — `artist|recording|release_group|file`
* `scope_id TEXT`
* `directive TEXT`        — e.g., `prefer_rg=<mb_rg_id>`, `treat_re_recording_as_canonical=1`
* `note TEXT`
* `created_by TEXT`
* `created_at TEXT`

**Index:** `override_scope_idx` on `(scope, scope_id)`.

---

## 3) Charts ETL (`charts.sqlite`)

### `chart`

* `chart_id TEXT PRIMARY KEY`      — `t2000|t40|t100|zwaar|...`
* `name TEXT`
* `frequency TEXT`                 — `y|w`
* `jurisdiction TEXT`
* `source_url TEXT`
* `license TEXT`
* `created_at TEXT`

### `chart_run`

* `run_id TEXT PRIMARY KEY`
* `chart_id TEXT NOT NULL REFERENCES chart(chart_id)`
* `period TEXT NOT NULL`           — `YYYY` or `YYYY-WNN`
* `scraped_at TEXT NOT NULL`
* `source_hash TEXT NOT NULL`      — hash of raw page/API payload
* `notes TEXT NULL`

**Unique:** `(chart_id, period)`.

### `chart_entry`

* `run_id TEXT REFERENCES chart_run(run_id)`
* `rank INTEGER NOT NULL`
* `artist_raw TEXT NOT NULL`
* `title_raw TEXT NOT NULL`
* `entry_unit TEXT NOT NULL`       — `recording|single_release|medley|unknown`
* `extra_raw TEXT NULL`
* PRIMARY KEY (`run_id`,`rank`)

**Index:** `entry_artist_title_idx` on `(artist_raw, title_raw)`.

### `alias_norm`

* `alias_id TEXT PRIMARY KEY`
* `type TEXT NOT NULL`             — `artist|title`
* `raw TEXT NOT NULL`
* `normalized TEXT NOT NULL`
* `ruleset_version TEXT NOT NULL`
* `created_at TEXT`

**Index:** `alias_raw_idx` on `(type, raw)`.

### `chart_link`

* `run_id TEXT NOT NULL`
* `rank INTEGER NOT NULL`
* `work_key TEXT NULL`
* `link_method TEXT NOT NULL`      — `isrc|title_artist_year|bundle_release|manual`
* `confidence REAL NOT NULL`
* `release_anchor_id TEXT NULL`    — release ID for single bundles
* `side_designation TEXT NULL`     — `A|B|AA|null`
* PRIMARY KEY (`run_id`,`rank`)

**Indexes:**

* `link_work_idx` on `(work_key)`
* `link_conf_idx` on `(confidence)`

---

## 4) HTTP & Entity cache (files + indexes)

* Raw HTTP responses stored on disk under `/cache/http/{source}/…`, with a small SQLite index:

  * `http_cache_index(source TEXT, key TEXT PRIMARY KEY, etag TEXT, last_modified TEXT, fetched_at TEXT, ttl_s INTEGER, path TEXT)`

* Semantic entity cache is the `musicgraph.sqlite` itself (above). Eviction via TTL and schema-version bump.

---

## 5) Minimal Views / Reports

* **`v_drift_candidates`** (decisions needing attention):
  Join `decision` where `state LIKE 'STALE-%'` OR `state='INDETERMINATE'`.

* **`v_missing_from_chart(chart_id, period, threshold)`**:
  Left join `chart_entry` → `chart_link` where `work_key IS NULL OR confidence < :threshold`.

* **`v_missing_in_library(chart_id[, period])`**:
  Join `chart_link` (conf ≥ auto) to `file_artifact` via `work_key`; invert to find links with no files.

---

## 6) Migration and versioning

* Schema version table: `schema_meta(key TEXT PRIMARY KEY, value TEXT)` with:

  * `db_version_musicgraph`, `db_version_charts`, `db_version_decisions`
  * `ruleset_version_current`
* Migrations use additive changes + backfilled defaults; no destructive changes without version bump and export path.
* Export/import: NDJSON lines per table for safe archival.

---

## 7) Operational envelopes

* Batch size defaults: identify/decide in chunks of 200 files; write/verify in 50 to limit tag writer contention.
* Rate limits: per-source token buckets, persisted to disk to survive restarts.
* Integrity:

  * Foreign keys **on**.
  * `PRAGMA synchronous=NORMAL`, `journal_mode=WAL` (SQLite).
  * Periodic `VACUUM` for charts DB after large ingests.

