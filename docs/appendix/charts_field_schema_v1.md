# CHARTS field schema v1 (compact, embeddable)

**Goal:** one self-contained JSON value you can stash in a single Vorbis comment or ID3v2 TXXX. Short by default; expandable when you want positions.

## 1) Wire format (v1)

```json
{
  "v": 1,
  "c": [
    ["<chart_id>", <score>, <highest>, "<freq>", <pos_opt?>]
  ]
}
```

* `"v"` — schema version (int).
* `"c"` — list of chart tuples (ordered by descending score, then lowest `highest`).
* Chart tuple fields:

  1. `<chart_id>` — short ID from registry below (string).
  2. `<score>` — aggregated integer points across all runs (see §2).
  3. `<highest>` — best rank (lowest integer) ever achieved in that chart (int).
  4. `<freq>` — `"y"` for yearly charts, `"w"` for weekly charts (string).
  5. `<pos_opt?>` — optional compact positions:

     * For yearly: `{ "YYYY": rank, ... }`
     * For weekly: `{ "YYYY": { "W": rank, ... }, ... }` where `"W"` is unpadded week number string (e.g., `"7"`).

**Minified JSON** recommended when embedding. Order is deterministic.

### Size budgeting (practical)

* Keep under ~2–3 KB when possible. ID3v2 TXXX can handle far more, but some tools choke on very large custom frames.
* Omit `<pos_opt>` by default. Include it only when you want a provenance-rich export.

---

## 2) Aggregation (final)

Let a chart run have size **N** and a rank **r** (1 is best). The per-run points are:

```
points = N - r + 1
```

* **Weekly charts** (e.g., Top 40): total `score` is the **sum across all weeks**.
* **Yearly charts** (e.g., Top 2000, Top 100, Zwaarste Lijst): total `score` is the **sum across all years**.
* `highest` is the **minimum** rank observed across all runs.

**No decay, no normalization** across years. Scores are additive.

---

## 3) Chart ID registry (seed)

| id      | name                          | freq | jurisdiction | N (typical) |
| ------- | ----------------------------- | ---- | ------------ | ----------- |
| `t2000` | NPO Radio 2 Top 2000          | y    | NL           | 2000        |
| `t40`   | Nederlandse Top 40            | w    | NL           | 40          |
| `t100`  | Nederlandse Top 100 (yearly)  | y    | NL           | 100         |
| `zwaar` | Studio Brussel Zwaarste Lijst | y    | BE           | 100 (var.)  |

* Registry lives outside the tag (docs/config). You can add more IDs without changing the schema.
* Display names, licenses, and URLs stay out of the tag to keep it small.

---

## 4) Singles-as-units in older weekly charts (Top 40)

Older entries often represent the **single release** (bundle) rather than a specific recording. We handle this at link time; the tag stays the same.

### 4.1 Side mapping model (conceptual)

When a chart entry is a *single bundle*, map it to specific track(s) via a scored decision. Evidence signals:

* **E1: Catalog side designation** from Discogs/MB release tracklist (A/B/AA markers).
* **E2: Title match** between file’s recording title and the single’s listed sides (after normalization).
* **E3: Duration proximity** between file and listed side(s) (±2 s default).
* **E4: Market notes** (press, sleeves) indicating a *double A-side* (both sides promoted).
* **E5: Later database entries** that retrospectively assign the charted title to a specific side.

### 4.2 Confidence scoring (0–1)

```
conf = 0.40*E1 + 0.25*E2 + 0.20*E3 + 0.10*E4 + 0.05*E5
```

Each Ei is 0..1 based on evidence presence/quality. Tunable weights, but freeze them per ruleset.

### 4.3 Thresholds (deterministic)

* **Auto-link** side(s) if `conf ≥ 0.85`.

  * If a **double A-side** is detected, attribute the bundle’s points to **both** sides.
* **Needs review** if `0.60 ≤ conf < 0.85`. Queue for human adjudication; don’t tag until resolved (or tag the bundle to RG only, not recording).
* **Reject** if `conf < 0.60`. Attribute points only at the **release group** level for analytics; do **not** attach CHARTS to a specific recording.

**Tie rule:** If two sides have `conf` within 0.05 of each other and either qualifies for auto-link, treat as a **double A-side** unless the release explicitly marks A/B.

**Deterministic fallback:** If everything fails but you must emit a tag, it’s valid to leave CHARTS intact (scores & highest) while the Decision Trace records that linkage is at the single bundle level.

---

## 5) Title & artist normalization (for linking, not for tags)

* Unicode NFC, case-fold, trim punctuation.
* Collapse featuring tokens to a canonical `" feat. "` form; ignore guests in primary match, then bonus-boost if guests match.
* Title edition stripping: remove suffixes like `- Remastered 20xx`, `(Radio Edit)`, `(Mono)`, `(Live at …)`, `- 2011 Mix`, `(From "…")`. Preserve these as metadata for side inference, not for the core match.

Keep a **ruleset version** for normalization to avoid silent drift.

---

## 6) Examples

### 6.1 Minimal (scores only)

```json
{"v":1,"c":[["t2000",36250,30,"y"],["t40",266,4,"w"]]}
```

### 6.2 With positions (yearly + weekly)

```json
{
  "v":1,
  "c":[
    ["t2000",36250,30,"y",{"2006":393,"2007":117,"2008":470,"2009":96,"2010":101,"2011":63,"2012":54,"2013":44,"2014":44,"2015":38,"2016":35,"2017":38,"2018":30,"2019":39,"2020":46,"2021":42,"2022":39,"2023":48,"2024":32}],
    ["t40",266,4,"w",{"1991":{"4":17,"5":11,"6":5,"7":4,"8":4,"9":7,"10":11,"11":15,"12":29}}]
  ]
}
```

### 6.3 Yearly Top 100 only

```json
{"v":1,"c":[["t100",59,42,"y",{"1991":42}]]}
```

---

## 7) Determinism, provenance, and drift

* The **Decision Trace** (separate tag) should state whether any CHARTS linkage came **via single bundle** and, if so, which release ID and side(s) were chosen, including the `conf` score and evidences used.
* When chart sources reissue data or you refine normalization, bump a **ruleset version** in your internal DB; the CHARTS `v` stays `1` unless the wire format changes.

---

## 8) Failure modes & guardrails

* If a chart run’s size **N** is unknown (scrape failure), skip that run (don’t guess points).
* If ranks are suspect (ties, duplicates), keep the **best** rank per work per run.
* If an entry clearly refers to a **medley** or a **mash-up**, treat it as its own work; don’t distribute points to constituent songs unless the chart explicitly did.

---

## 9) What to implement next (concept spec)

* **Normalization ruleset v1**: exact token list and regexes for title/artist canon + examples where they *must not* fire (to avoid over-stripping).
* **Alias table structure**: how you store and evolve per-artist/per-title aliases without breaking past links.
* **Confidence calibration set**: 20–30 historical Top 40 single bundles with known side outcomes to tune `E1..E5` weights and thresholds.

Pick which of these you want to pin down first, and I’ll draft the spec in the same crisp style.
