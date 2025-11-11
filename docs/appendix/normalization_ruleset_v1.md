# Normalization Ruleset v1 (concept)

## 0) Goals

* Produce **core forms** used for matching, not display.
* Extract **edition tags** (live, remix, radio edit, remaster year, OST, mono, acoustic, etc.) without losing the core title/artist.
* Be **locale-aware** (NL/EN focus) and **exception-driven** to avoid breaking “weird” artist names.

Outputs for any string S:

* `normalized`: canonical string used for equality/edit-distance.
* `core`: editionless form used for “same-work” matching.
* `tags[]`: extracted edition semantics (structured).
* `ruleset_version`: `"norm-v1"`.

---

## 1) Order of operations (deterministic)

Apply in this order; each step records which rule fired.

1. **Unicode & whitespace**

   * Normalize to NFC.
   * Collapse all whitespace runs to single spaces; trim ends.
   * Replace non-breaking spaces with spaces; remove zero-widths.

2. **Casefold (matching only)**

   * Lowercase for `normalized` and `core`. Keep original casing separately for display and provenance.

3. **Punctuation canonicalization**

   * Map curly quotes → straight, en/em dashes → hyphen, ellipsis → three dots.
   * Canonicalize separators: `&`, `+`, `x`, `X`, `×` → a canonical token (see §4.2).

4. **Diacritics (matching only)**

   * Strip diacritics for `normalized`/`core` comparisons, **but** store a `diacritics_signature` (so later you can boost exact-diacritic matches).

5. **Feature/guest canonicalization (title & artist)**

   * Standardize to `" feat. "`. Accepted forms (both EN/NL): `feat`, `featuring`, `ft`, `met` (NL contexts), `con` (ES/IT contexts—low priority), `with`.
   * For **matching**, ignore guests in `artist_core`; keep them in `artist_guests[]`.
   * For **title**, move any trailing “feat.” blocks to a `title_guests[]` tag and remove from `title_core`.

6. **Edition/descriptor extraction (title)**
   Parse and **remove** only when clearly editional; add structured tags. Repeat until stable (max N passes to avoid loops).

   **Parenthetical/bracketed suffixes** (case-insensitive):

   * `(Radio Edit|Single Edit|7" Edit|Edit)` → `{"kind":"edit","sub":"radio"|"single"|"7in"|"edit"}`
   * `(Live|Live at …|Live from …|Live [YYYY])` → `{"kind":"live","place/date":"…"}`
   * `(Remaster(ed)?( \d{4})?)` → `{"kind":"remaster","year":YYYY?}`
   * `(Mono|Stereo)` → `{"kind":"mix","sub":"mono"|"stereo"}`
   * `(Acoustic|Unplugged|Piano Version)` → `{"kind":"acoustic"}`
   * `(Remix|[A-Z0-9 .\-]+ Remix)` → `{"kind":"remix","mixer":"…"?}`
   * `(Extended( Mix)?|12" Mix|Club Mix|Dub|Instrumental|Karaoke Version)` → kind accordingly
   * `(From "[^"]+"|From ‘[^’]+’|From the Motion Picture …)` → `{"kind":"ost","work":"…"}`
   * `(Demo|Early Version|Rough Mix)` → `{"kind":"demo"}`
   * `(Clean|Explicit)` → `{"kind":"content","sub":"clean"|"explicit"}`
   * `(Re-recorded|Taylor’s Version|New Recording)` → `{"kind":"re_recording","note":"…"}`
   * `(Medley)` → `{"kind":"medley"}`
   * `(Part [IVX]+|Pt\. ?\d+|Vol\. ?\d+)` → **do not remove**; **retain** in `core` as it changes identity.

   **Dash-style suffixes** (when clearly editional):
   ` - Remastered 2009`, ` - Live at …`, ` - Radio Edit`, etc. → same tags as above, then remove from `core`.

   **OST prefixes**:
   `From "…"` at the start → `{"kind":"ost"}` and strip.

   **Guardrails (do not strip):**

   * If the bracketed phrase contains **only** the main title (e.g., translation info) and removing would make an empty string, keep it.
   * If removing all descriptors would reduce title length by >70% and what remains is a single stopword (e.g., “the”), back off and keep original.
   * If phrase matches known **identity-bearing** patterns (`Part II`, `Op. 27, No. 2`, `BWV 565`) → never strip.

7. **Conjunction/separator normalization (artist)**

   * Canonicalize separators between main artists (see §4.2). For matching, treat `Artist A & Artist B` equivalent to `Artist A and Artist B`.

8. **Articles/stopwords (artist)**

   * Only drop leading English “the” **if multi-word and not in exception list** (see §4.3).
   * Do **not** drop Dutch articles (“De”, “Het”, “’t”, “Den”) by default; treat via **per-artist alias** (§4.3).

9. **Title colon/suffix split**

   * Split `Title: Subtitle` into `core="Title"` and add tag `{"kind":"subtitle","value":"Subtitle"}` **only** if subtitle matches edition patterns (e.g., `Title: Remastered 2011`). Otherwise keep as part of core.

10. **Numeric & roman numerals**

* Normalize roman numerals to uppercase, collapse unicode variants; keep them (identity-bearing).

11. **Final compaction**

* Remove duplicate spaces, leading/trailing punctuation, trailing hyphens left by stripping.

---

## 2) Artist normalization details

### 2.1 Primary vs guests

* `artist_core` = main credited artist(s) after removing guests (`feat.` et al.).
* Store guests into `artist_guests[]` (normalized forms).

### 2.2 Separator normalization

* Accept `&`, `and`, `+`, `x`, `×`, `/`, `,` and map to a **canonical neutral token** for matching (e.g., `•` internally). Keep the original tokenization for display.
* Treat `Artist A x Artist B` ≈ `Artist A & Artist B` for matching.

### 2.3 Exception list (must not “helpfully” change)

Maintain an **alias/exception registry** (editable):

* Do **not** drop articles for: **The The**, **The 1975**, **The Weeknd** (article is identity).
* Preserve punctuation: **P!nk**, **fun.**, **blink-182**, **will.i.am**, **!!!**, **(hed) p.e.**, **MØ** (also keep diacritics signature), **Björk**.
* Multi-language/locale cases: **De Dijk**, **De Jeugd van Tegenwoordig**, **Het Goede Doel**—do not strip “De/Het”.
* DJ/producer prefixes: **DJ Paul Elstak**, **DJ Tiësto** → treat `DJ` as part of name.

Store both **normalized** and **display** aliases; prefer MB/Wikidata canonical when present.

---

## 3) Title normalization details

### 3.1 Guests inside title

* Move trailing or parenthetical feat/with/ met to `title_guests[]`, remove from `title_core`.

### 3.2 Edition tags taxonomy (examples)

Each extracted tag is a small object; multiple tags may exist.

* `{"kind":"live","place":"Brixton Academy","date":"1997"}`
* `{"kind":"remix","mixer":"Armand Van Helden"}`
* `{"kind":"edit","sub":"radio"}`
* `{"kind":"ost","work":"Trainspotting"}`
* `{"kind":"remaster","year":"2011"}`
* `{"kind":"re_recording","note":"Taylor’s Version"}`
* `{"kind":"mix","sub":"mono"}`
* `{"kind":"content","sub":"explicit"}`
* `{"kind":"demo"}`

These tags feed the **intent rules** (e.g., remix vs album).

### 3.3 Don’t-strip patterns (identity)

* Movement/work numbers: `Op.`, `BWV`, `K.`, `Hob.`, `D.`, `RV`, `Catalogue` tags.
* Parts: `Part I/II/III`, `Pt. 2`, `Vol. 1`—these define distinct works.
* Language variants in parentheses that are **real alternate titles** (e.g., `(English Version)`)—keep in core but also tag `{"kind":"lang","value":"en"}` if detected.

---

## 4) Locale & tokenization

### 4.1 Locale cues (NL/EN)

* Treat **“met”** (NL) as a guest marker only when surrounded by spaces and followed by a capitalized artist token; else keep.
* OST cues in NL: `(Uit "…")`, `(Van de Film "…")` → `{"kind":"ost"}`.

### 4.2 Conjunctions (canonical token)

* Internal canonical token for **matching**: `•` (three characters: space, bullet, space).
* Map: `&|and|\+|x|×|/|,` → `•`
* This avoids ambiguity without committing to a display choice.

### 4.3 Alias table

* A persistent `alias_norm` table supports:

  * **Per-artist**: preferred canonical, disallowed drops (articles), punctuation to preserve.
  * **Per-title**: special-case keep/strip rules (e.g., “Live” is part of band name for *Live*; never treat as edition when artist is **Live** and it appears as a bare token).

---

## 5) Matching keys (what you actually compare)

For **artist**:

* `artist_key` = tokenized set of `artist_core` names joined by canonical token (diacritics-stripped, casefolded).

For **title**:

* `title_key` = `title_core` after edition stripping (diacritics-stripped, casefolded).

Composite **work key** for fuzzy links:

* `work_key = hash( artist_key + " // " + title_key )`
* Optionally include `length_bucket` (duration rounded to nearest 2s) for stronger matches.

---

## 6) Confidence signals built on normalization (informative)

These aren’t normalization per se, but they **flow from it**:

* **Exact core match** (artist_key + title_key) → strong.
* **Guest-agnostic match** (ignoring feat.) → strong.
* **Edition-agnostic match** (radio edit vs album) → strong for same recording.
* **Remix/live tag present** → shifts candidate to remix/live intent paths.
* **OST tag** → boosts soundtrack origin rule.

---

## 7) Idempotence & drift control

* Running the ruleset twice must not change the result.
* Every removal/tag extraction leaves a **breadcrumb** in the Decision Trace (e.g., `strip:[(Radio Edit),(Remastered 2011)]`).
* Version the ruleset: `ruleset_version="norm-v1"`. If any regex/token list changes, bump to `"norm-v2"`; re-normalize only when you choose to migrate.

---

## 8) Edge-case gallery (do/don’t fire)

1. **“One” — U2 (Radio Edit)**

   * `title_core = "one"`; tags: `{"kind":"edit","sub":"radio"}`.

2. **“Paranoid Android (Remastered 2009)” — Radiohead**

   * `title_core = "paranoid android"`; tags: `{"kind":"remaster","year":"2009"}`.

3. **“Under Pressure (feat. David Bowie)” — Queen**

   * `artist_core = "queen"`; `artist_guests=["david bowie"]`.
   * `title_core = "under pressure"`; `title_guests=["david bowie"]`.

4. **“I Want You (She’s So Heavy) [Remix]” — The Beatles**

   * `title_core = "i want you (she’s so heavy)"` (square-bracket tag removed); tag: `{"kind":"remix"}`.
   * Preserve inner parentheses—they’re part of identity.

5. **“Smells Like Teen Spirit (Live at Reading 1992)” — Nirvana**

   * Tag: `{"kind":"live","place":"reading","date":"1992"}`; `title_core = "smells like teen spirit"`.

6. **“Blue Monday ’88” — New Order**

   * Recognize `'88` as **re-recording/edit marker** → tag `{"kind":"re_recording","note":"’88"}`;
   * `title_core = "blue monday"`.

7. **“The The — This Is the Day”**

   * Keep leading article (exception); `artist_core = "the the"`.

8. **“De Dijk — Binnen zonder kloppen”**

   * Keep “De” (NL proper name); no article drop.

9. **Artist = “Live” — Title = “Live”**

   * Do **not** tag as `kind:live`; this is identity collision; keep both intact.

10. **Classical: “Piano Sonata No. 14 in C-sharp minor, Op. 27, No. 2: I. Adagio sostenuto (Mono)”**

* Keep work/movement numbers; strip `(Mono)` tag only.

11. **Karaoke artifacts: “Bohemian Rhapsody (Originally Performed by Queen) [Karaoke Version]”**

* Strip `(Originally Performed …)` and `[Karaoke Version]`; tags: `{"kind":"karaoke"}`;
* Title core becomes `"bohemian rhapsody"`, but you should separately mark **performance type** to avoid linking to the original recording (outside normalization scope but enabled by tags).

12. **Top 40 single bundle: “Double A-Side: Song A / Song B”**

* Keep both sides in `title_core` if it’s the official single title; extract a structured tag: `{"kind":"single_bundle","sides":["song a","song b"]}` to help side mapping.

---

## 9) Configurable bits (pull from a single config object)

* `FEAT_TOKENS = ["feat.", "featuring", "ft.", "met", "with"]`
* `OST_TOKENS = ["from \"", "uit \"", "van de film"]`
* `EDITION_PATTERNS` (list of regexes and the tag they emit)
* `ARTIST_ARTICLE_DROPS = {"en":["the"],"nl":[]}` (NL drops disabled by default)
* `EXCEPTION_ARTISTS` (names → directives: keep articles, preserve punctuation, etc.)
* `CANON_SEP_TOKEN = " • "`
* `LENGTH_BUCKET_SECS = 2`
* `RULESET_VERSION = "norm-v1"`

---

## 10) Acceptance checks (quick)

* **Idempotence**: `normalize(normalize(S)) == normalize(S)`.
* **Minimality**: core never empty unless the original was empty.
* **Reversibility of meaning**: removing tags and reattaching them yields the original (modulo punctuation spacing).
* **Collision audit**: bands with special punctuation (!!!, fun., P!nk) remain distinguishable.
