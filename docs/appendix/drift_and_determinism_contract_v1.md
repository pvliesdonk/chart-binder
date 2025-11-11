# Drift & Determinism Contract (v1)

## Purpose

Guarantee that the same inputs yield the same decision (CRG + RR) unless:

* upstream facts changed,
* your rules/config changed, or
* you explicitly override.

## Core identifiers

* **`ruleset_version`** — semantic version of your decision logic (normalization + canonicalization + truth tables + side-mapping weights). Example: `canon-1.0.norm-v1.tags-v1.side-v1`.
* **`config_snapshot`** — the exact knobs used (`lead_window_days`, `reissue_*`, thresholds, label authority order, chart IDs enabled…).
* **`evidence_hash`** — SHA256 over a canonical JSON of the **evidence bundle** (entities + facts) with volatile fields removed (timestamps, HTTP headers, cache ages).
* **`decision_id`** — stable UUID per file’s accepted decision; changes only when a new decision replaces it.

## States

* **`UNDECIDED`** — no decision stored.
* **`DECIDED`** — decision stored; includes `crg`, `rr`, `recording_id?`, `ruleset_version`, `config_snapshot`, `evidence_hash`, compact trace.
* **`STALE-EVIDENCE`** — recomputation with the *same* rules/config yields a *different* `evidence_hash` (upstream data changed).
* **`STALE-RULES`** — recomputation with a *different* rules/config yields the *same* evidence hash but a *different* outcome.
* **`STALE-BOTH`** — both evidence and rules changed and result differs.
* **`INDETERMINATE`** — decision engine cannot select uniquely; needs review or more facts.

## Transitions

* `UNDECIDED → DECIDED` — first acceptance.
* `DECIDED → STALE-*` — `canon drift review` detects change.
* `STALE-* → DECIDED` — user accepts recomputed outcome (or pins original).
* Any → `INDETERMINATE` — recomputation yields tie/insufficient facts.
* `DECIDED → DECIDED` — user applies **override** (documented rationale).

## Modes (CLI/library)

* `--frozen` — never change existing decisions; only report drift.
* `--refresh` — recompute; if outcome differs, **do not write** unless `--apply` is given.
* `--apply` — adopt new decision; writes tags & stores new decision (old decision archived).
* `--pin` — force keep current decision even if drift detected (stores `pin_reason`).
* `--unpin` — allow future updates.

## Drift detection algorithm

1. Rebuild **evidence bundle** with current caches and sources.
2. Compute `evidence_hash_new`.
3. Re-run decision logic with current `ruleset_version + config`.
4. Compare:

   * If `evidence_hash_new != evidence_hash_old` and decision differs → `STALE-EVIDENCE` (or `STALE-BOTH` if rules also changed).
   * If `evidence_hash_new == evidence_hash_old` and decision differs → `STALE-RULES`.
   * If same decision → remain `DECIDED` (no-op).

**Note:** The *evidence bundle* includes normalized artist/title cores, candidate RGs, first dates, release lists, live/remix flags, origin country resolution, and chart linkage facts—everything your rules use.

## What’s archived on change

* Prior `decision` row moved to `decision_history` with `superseded_at`, `reason ∈ {"refresh","ruleset_change","manual_override","pin"}`.
* Full structured trace stored once per decision (compressed JSON).

## Human-in-the-loop policy

* When entering any `STALE-*` state, present side-by-side:

  * `old` vs `new` CRG/RR,
  * rule paths (`CRG:SINGLE_TRUE_PREMIERE` → `CRG:ALBUM_LEAD_WINDOW`, etc.),
  * what fact tipped it (e.g., “Discogs master backfilled original 1977 single date”).
* Required user action: **Apply / Keep / Pin / Inspect evidence**.

## Cache determinism

* Evidence is built **only** from entity cache content plus explicit live fetches logged in the trace.
* To re-run deterministically, you can use:

  * **cache-only mode** (no live calls) to reproduce a past run precisely,
  * **live-refresh mode** to accept upstream drift.

## Tag determinism

* Tag payloads (IDs + CHARTS + compact trace) are a pure function of the `decision` row + configured exporters.
* Round-trip verify immediately after writes to ensure container quirks didn’t mutate values.

