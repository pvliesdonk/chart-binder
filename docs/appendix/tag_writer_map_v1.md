# Tag Writer Map (v1, conceptual)

Goal: consistent, portable tagging across **ID3v2.4 (MP3)**, **Vorbis/FLAC**, and **MP4**. Core fields mirror canonical decisions; advanced data goes into compact/custom fields.

## Encodings & Limits (practical)

* **ID3v2.4**: UTF-8; TXXX frame names ≤ 64 chars recommended; multiple values via repeated frames (prefer single JSON TXXX for CHARTS).
* **Vorbis/FLAC**: UTF-8 key/value pairs; multi-values allowed; keys uppercase.
* **MP4**: atoms; UTF-8; custom via `----:com.apple.iTunes:<name>`.

## Core Canonical Fields (overwrite only if authoritative/accepted)

| Logical                                | ID3v2.4                       | Vorbis/FLAC                | MP4                             | Notes                                             |        |    |            |      |
| -------------------------------------- | ----------------------------- | -------------------------- | ------------------------------- | ------------------------------------------------- | ------ | -- | ---------- | ---- |
| Title (canonical)                      | `TIT2`                        | `TITLE`                    | `\xa9nam`                       | From canonical recording                          |        |    |            |      |
| Artist (canonical)                     | `TPE1`                        | `ARTIST`                   | `\xa9ART`                       | No guests in core                                 |        |    |            |      |
| Album (canonical RG title / RR title*) | `TALB`                        | `ALBUM`                    | `\xa9alb`                       | Prefer RG title; RR title if materially different |        |    |            |      |
| Album Artist                           | `TPE2`                        | `ALBUMARTIST`              | `aART`                          |                                                   |        |    |            |      |
| Original Year                          | `TDOR` (or `TDRC` if absent)  | `ORIGINALYEAR`/`DATE`      | `\xa9day`                       | Use YYYY or YYYY-MM-DD                            |        |    |            |      |
| Track No / Total                       | `TRCK`                        | `TRACKNUMBER`/`TRACKTOTAL` | `trkn`                          | From RR tracklist if known                        |        |    |            |      |
| Disc No / Total                        | `TPOS`                        | `DISCNUMBER`/`DISCTOTAL`   | `disk`                          |                                                   |        |    |            |      |
| Label                                  | `TPUB`                        | `LABEL`                    | `----:com.apple.iTunes:LABEL`   | From RR                                           |        |    |            |      |
| Country                                | `TXXX:COUNTRY`                | `COUNTRY`                  | `----:…:COUNTRY`                | Origin country for RR                             |        |    |            |      |
| Media/Format                           | `TMED`                        | `MEDIA`                    | `----:…:MEDIA`                  | Optional                                          |        |    |            |      |
| Release Type                           | `TXXX:CANONICAL_RELEASE_TYPE` | `CANONICAL_RELEASE_TYPE`   | `----:…:CANONICAL_RELEASE_TYPE` | album                                             | single | ep | soundtrack | live |

* If RG vs RR titles differ subtly (e.g., “Album (Deluxe)”), prefer RG for conceptual identity; store RR specifics in custom fields.

## Canonical IDs (always safe to add)

| Logical                   | ID3v2.4                                                     | Vorbis/FLAC           | MP4                          |
| ------------------------- | ----------------------------------------------------------- | --------------------- | ---------------------------- |
| MusicBrainz Recording     | `UFID:http://musicbrainz.org` **or** `TXXX:MB_RECORDING_ID` | `MB_RECORDING_ID`     | `----:…:MB_RECORDING_ID`     |
| MusicBrainz Release Group | `TXXX:MB_RELEASE_GROUP_ID`                                  | `MB_RELEASE_GROUP_ID` | `----:…:MB_RELEASE_GROUP_ID` |
| MusicBrainz Release       | `TXXX:MB_RELEASE_ID`                                        | `MB_RELEASE_ID`       | `----:…:MB_RELEASE_ID`       |
| Discogs Master            | `TXXX:DISCOGS_MASTER_ID`                                    | `DISCOGS_MASTER_ID`   | `----:…:DISCOGS_MASTER_ID`   |
| Discogs Release           | `TXXX:DISCOGS_RELEASE_ID`                                   | `DISCOGS_RELEASE_ID`  | `----:…:DISCOGS_RELEASE_ID`  |
| Spotify Track             | `TXXX:SPOTIFY_TRACK_ID`                                     | `SPOTIFY_TRACK_ID`    | `----:…:SPOTIFY_TRACK_ID`    |
| Spotify Album             | `TXXX:SPOTIFY_ALBUM_ID`                                     | `SPOTIFY_ALBUM_ID`    | `----:…:SPOTIFY_ALBUM_ID`    |
| Wikidata QID              | `TXXX:WIKIDATA_QID`                                         | `WIKIDATA_QID`        | `----:…:WIKIDATA_QID`        |

**Note:** For MB Recording, some ecosystems prefer **UFID** with owner as MB URL; others prefer TXXX. Supporting both is fine.

## Compact JSON Fields

| Logical                  | ID3v2.4                      | Vorbis/FLAC             | MP4                            | Notes                          |
| ------------------------ | ---------------------------- | ----------------------- | ------------------------------ | ------------------------------ |
| CHARTS blob              | `TXXX:CHARTS`                | `CHARTS`                | `----:…:CHARTS`                | Use minified JSON v1           |
| Decision Trace (compact) | `TXXX:TAG_DECISION_TRACE`    | `TAG_DECISION_TRACE`    | `----:…:TAG_DECISION_TRACE`    | `evh=…;crg=…;rr=…;src=…;cfg=…` |
| Ruleset Version          | `TXXX:CANON_RULESET_VERSION` | `CANON_RULESET_VERSION` | `----:…:CANON_RULESET_VERSION` | For drift forensics            |
| Evidence Hash            | `TXXX:CANON_EVIDENCE_HASH`   | `CANON_EVIDENCE_HASH`   | `----:…:CANON_EVIDENCE_HASH`   |                                |

## Optional Enrichment

| Logical                 | ID3v2.4                             | Vorbis/FLAC                   | MP4                   |
| ----------------------- | ----------------------------------- | ----------------------------- | --------------------- |
| ISRCs (multi)           | `TSRC` (repeat) or `TXXX:ISRC_LIST` | `ISRC` (multi) or `ISRC_LIST` | `----:…:ISRC_LIST`    |
| Work/Composition MBID   | `TXXX:MB_WORK_ID`                   | `MB_WORK_ID`                  | `----:…:MB_WORK_ID`   |
| Alternate Versions JSON | `TXXX:ALT_VERSIONS`                 | `ALT_VERSIONS`                | `----:…:ALT_VERSIONS` |

## Multi-Value Strategy

* **ID3v2.4**: prefer repeated frames (e.g., multiple `TSRC`), but for portability store a JSON list in a single `TXXX:*_LIST` as well.
* **Vorbis**: true multi-values supported; set multiple fields with same key.
* **MP4**: arrays supported for some atoms; for customs, keep JSON in a single atom.

## Round-Trip Verification (must-do)

* After writes, read tags through **the same library** and compare:

  * Equality for canonical IDs, CHARTS, trace, and core fields.
  * Tolerate minor date formatting differences (normalize on read to compare).

## Container Edge Cases & Guardrails

* **ID3v2.3 back-compat**: Some tools downconvert 2.4→2.3 and lose `TDOR`. Mirror original year into `TXXX:ORIGINALYEAR` for resilience.
* **MP4 date quirks**: `\xa9day` is freeform; store ISO `YYYY` or `YYYY-MM-DD`.
* **Long JSON**: keep CHARTS under ~3 KB; omit positions if needed.
* **Non-destructive first write**: stash originals once:

  * `TXXX:ORIG_ALBUM`, `ORIG_TITLE`, `ORIG_ARTIST`, `ORIG_DATE` (or Vorbis equivalents).

## Field Authority Rules

* If **augment-only** mode: never alter `TIT2/TALB/TPE1/TPE2/TDRC/TDOR`; only set IDs/CHARTS/trace.
* If **authoritative**:

  * `TALB` becomes **RG title**; if RR has a distinct historical title (not “Deluxe/Remaster”), you may prefer **RR title**—but record which choice in trace (`album=RG|RR`).
  * `TDRC/TDOR`: `TDOR` = original release year, `TDRC` = RR date if you want exactness; otherwise keep only `TDOR` and put full date into a custom field.

## Minimal Write Set (safe everywhere)

* **IDs:** MB recording/RG/release; Discogs master/release; Spotify track/album; Wikidata QID.
* **CHARTS:** compact JSON v1 without positions.
* **TRACE:** compact decision string.
* **TYPE:** canonical release type.

This set is unlikely to break other tag readers and is enough to reconstruct decisions.

