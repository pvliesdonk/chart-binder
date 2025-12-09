# Charts ETL Test Pack v1 (10 focused runs)

Each “run” is a concrete scrape snapshot you can mock. For every entry: provide the raw line as scraped, the expected normalization, unit type, and the linking outcome target (not specific IDs).

**Legend**

* `entry_unit`: `"recording"` | `"single_release"` | `"medley"` | `"unknown"`
* `norm`: `{artist_core, title_core, tags[]}` (per Normalization Ruleset v1)
* `link`: `{method: isrc|title_artist_year|bundle_release|manual, confidence: high|med|low}`
* `expect`: what your linker should resolve to (work_key dimension)
* `notes`: ambiguity you must surface

---

## R1 — Top 2000 (Year: 2024, NL — yearly)

1. Raw: *Queen — Bohemian Rhapsody* (rank 1)
   norm: artist_core=“queen”, title_core=“bohemian rhapsody”
   entry_unit=recording
   link: {method:title_artist_year, confidence: high}
   expect: recording-level work_key
   notes: none

2. Raw: *Nirvana — Smells Like Teen Spirit (Live at Reading 1992)*
   tags=[live:Reading 1992] → title_core=“smells like teen spirit”
   entry_unit=recording
   link: {method:title_artist_year, confidence: med}
   expect: **live recording** (not studio recording)
   notes: requires live tag to steer to live RG

3. Raw: *New Order — Blue Monday ’88*
   tags=[re_recording:’88], title_core=“blue monday”
   entry_unit=recording
   link: {method:title_artist_year, confidence: med}
   expect: the **’88 re-recording** work_key
   notes: ensure ’88 doesn’t collapse to 1983

4. Raw: *Björk — Jóga* (diacritics)
   norm preserves diacritics signature
   entry_unit=recording
   link: {method:title_artist_year, confidence: high}
   expect: recording-level work_key
   notes: verify diacritics-agnostic match still prefers exact

---

## R2 — Top 2000 (Year: 2006, NL — yearly)

5. Raw: *The The — This Is the Day*
   keep “The” (exception)
   entry_unit=recording
   link: {method:title_artist_year, confidence: high}
   expect: recording-level
   notes: article exception test

6. Raw: *BLØF — Aan de kust*
   entry_unit=recording
   link: {method:title_artist_year, confidence: high}
   expect: recording-level
   notes: Ø handling

---

## R3 — Nederlandse Top 40 (Week: 1991-W07, NL — weekly; **single bundles**)

7. Raw: *Queen — We Will Rock You / We Are the Champions (Double A-Side)* (rank 4)
   tags=[single_bundle:[…]]; title_core keeps both
   entry_unit=single_release
   link: {method:bundle_release, confidence: high}
   expect: **attribute to both sides** (double A-side)
   notes: side-mapping must auto-link both (conf ≥ 0.85)

8. Raw: *The Beatles — Strawberry Fields Forever / Penny Lane*
   entry_unit=single_release
   link: {method:bundle_release, confidence: med}
   expect: **attribute to both sides** unless A/B explicitly indicated in the market’s catalog
   notes: if catalog marks A/B, prefer designated A-side

9. Raw: *U2 — One (Radio Edit)*
   tags=[edit:radio]
   entry_unit=recording
   link: {method:title_artist_year, confidence: high}
   expect: same recording as album track unless known edit re-rec
   notes: promo-single vs album test (feeds canonicalizer later)

---

## R4 — Nederlandse Top 100 (Year: 1991, NL — yearly)

10. Raw: *R.E.M. — Losing My Religion (Single Version)*
    tags=[version:single]
    entry_unit=recording
    link: {method:title_artist_year, confidence: high}
    expect: album recording (single version is same take)
    notes: ensure “version” doesn’t create a new work unless duration differs significantly

---

## R5 — Studio Brussel Zwaarste Lijst (Year: 2023, BE — yearly)

11. Raw: *Metallica — Battery (Live)*
    tags=[live]
    entry_unit=recording
    link: {method:title_artist_year, confidence: med}
    expect: live recording work_key
    notes: multiple official lives exist → pick the one that aligns with common reference (duration ±2s)

12. Raw: *Rammstein — Mein Herz brennt (Piano Version)*
    tags=[acoustic/piano]
    entry_unit=recording
    link: {method:title_artist_year, confidence: high}
    expect: acoustic/piano re-recording work_key
    notes: not the album original

---

## R6 — Top 40 (Week: 1979-W12, NL — weekly; **tricky medley**)

13. Raw: *Stars on 45 — Beatles Medley*
    tags=[medley]
    entry_unit=medley
    link: {method:title_artist_year, confidence: high}
    expect: **medley work_key** (not distributed to constituent Beatles songs)
    notes: ensure no leakage of points to components

---

## R7 — Top 40 (Week: 1999-W25, NL — weekly; **cover/karaoke trap**)

14. Raw: *Unknown Artist — Bohemian Rhapsody (Karaoke Version)*
    tags=[karaoke]
    entry_unit=recording
    link: {method:title_artist_year, confidence: low}
    expect: **reject linkage** to Queen; mark as cover/karaoke
    notes: coverage report should treat as resolved but non-linkable to original

---

## R8 — Top 40 (Week: 2002-W34, NL — weekly; **retrospective labeling**)

15. Raw: *The Rolling Stones — (I Can’t Get No) Satisfaction – 2002 Remaster*
    tags=[remaster:2002]
    entry_unit=recording
    link: {method:title_artist_year, confidence: high}
    expect: original recording work_key (remaster tag ignored for identity)
    notes: don’t create a new work

---

## R9 — Top 2000 (Year: 2011, NL — yearly; **OST cue**)

16. Raw: *Berlin — Take My Breath Away (From "Top Gun")*
    tags=[ost:“Top Gun”]
    entry_unit=recording
    link: {method:title_artist_year, confidence: high}
    expect: album/single recording linked, with OST tag informing canonicalizer (potential soundtrack origin)
    notes: ensures OST extraction feeds policy

---

## R10 — Top 40 (Week: 2013-W16, NL — weekly; **x/× separator**)

17. Raw: *Armin van Buuren x Trevor Guthrie — This Is What It Feels Like*
    separator normalized to canonical token
    entry_unit=recording
    link: {method:title_artist_year, confidence: high}
    expect: recording-level
    notes: `x`/`×` treated as collaboration, not remix

---

### Acceptance for the ETL:

* For each run, produce: `chart_run`, normalized `chart_entry` rows, `chart_link` rows with `method`+`confidence`, and a coverage report (% linked).
* R3/R6 must demonstrate **single bundle** and **medley** handling.
* R14 must **not** link to originals.
* R3/8 tie rules: if two sides auto-qualify within 0.05 confidence, treat as **double A-side**.

---

