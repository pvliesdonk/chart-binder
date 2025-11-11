# Normalization QA pack v1 (inputs → expected cores & tags)

Legend:

* `A_in`/`T_in` = input artist/title as found.
* `A_core`/`T_core` = normalized cores (case-folded, diacritics-agnostic for matching).
* `A_guests`/`T_guests` = extracted guests.
* `tags` = extracted edition/intent tags (summary).
* Notes call out guardrails.

## A. Edits, remasters, mixes

1.

A_in: Queen
T_in: Under Pressure (Radio Edit)
→ A_core: queen | A_guests: []
→ T_core: under pressure | T_guests: []
→ tags: {edit:radio}

2.

A_in: Radiohead
T_in: Paranoid Android - Remastered 2009
→ A_core: radiohead
→ T_core: paranoid android
→ tags: {remaster:2011?}  **Note:** if exact year is 2009 from string, set 2009.

3.

A_in: New Order
T_in: Blue Monday ’88
→ A_core: new order
→ T_core: blue monday
→ tags: {re_recording:"’88"}
Note: strip year variant from core; keep for display.

4.

A_in: Donna Summer
T_in: I Feel Love (12" Mix)
→ A_core: donna summer
→ T_core: i feel love
→ tags: {edit:"12in"}

5.

A_in: Massive Attack
T_in: Teardrop (Club Mix)
→ A_core: massive attack
→ T_core: teardrop
→ tags: {mix:club}

## B. Live, acoustic, sessions

6.

A_in: Nirvana
T_in: Smells Like Teen Spirit (Live at Reading 1992)
→ A_core: nirvana
→ T_core: smells like teen spirit
→ tags: {live:{place/date:"reading 1992"}}

7.

A_in: Eric Clapton
T_in: Layla (Unplugged)
→ A_core: eric clapton
→ T_core: layla
→ tags: {acoustic}

8.

A_in: The Smiths
T_in: What Difference Does It Make? (Peel Session)
→ A_core: the smiths
→ T_core: what difference does it make?
→ tags: {session:peel}

## C. Featurings (artist vs title)

9.

A_in: Queen feat. David Bowie
T_in: Under Pressure
→ A_core: queen | A_guests: [david bowie]
→ T_core: under pressure | T_guests: []
→ tags: {}

10.

A_in: Queen
T_in: Under Pressure (feat. David Bowie)
→ A_core: queen | A_guests: []
→ T_core: under pressure | T_guests: [david bowie]
→ tags: {}

11.

A_in: Suzan & Freek
T_in: Blauwe Dag (met Snelle)
→ A_core: suzan • freek | A_guests: []
→ T_core: blauwe dag | T_guests: [snelle]
→ tags: {}
Note: “met” recognized as guest marker (NL).

## D. OST signals

12.

A_in: Berlin
T_in: Take My Breath Away (From "Top Gun")
→ A_core: berlin
→ T_core: take my breath away
→ tags: {ost:"top gun"}

13.

A_in: Kenny Loggins
T_in: From "Top Gun" – Danger Zone
→ A_core: kenny loggins
→ T_core: danger zone
→ tags: {ost:"top gun"}
Note: prefix stripped.

14.

A_in: Falco
T_in: Jeanny (Uit "Einzelhaft")
→ A_core: falco
→ T_core: jeanny
→ tags: {ost:"einzelhaft"} (album-as-work tolerated; still OST-like signal)

## E. Singles as bundles (Top 40 edge)

15.

A_in: The Beatles
T_in: Strawberry Fields Forever / Penny Lane
→ A_core: the beatles
→ T_core: strawberry fields forever / penny lane
→ tags: {single_bundle:["strawberry fields forever","penny lane"]}
Note: core keeps both; side mapping decides attribution.

16.

A_in: Queen
T_in: We Will Rock You / We Are The Champions
→ A_core: queen
→ T_core: we will rock you / we are the champions
→ tags: {single_bundle:[…]}

## F. Language versions, content flags

17.

A_in: Nena
T_in: 99 Luftballons (English Version)
→ A_core: nena
→ T_core: 99 luftballons
→ tags: {lang:"en"}
Note: core keeps base title; language in tags.

18.

A_in: Eminen
T_in: Without Me (Clean)
→ A_core: eminen
→ T_core: without me
→ tags: {content:"clean"}

## G. Don’t-strip identity parts

19.

A_in: The Beatles
T_in: I Want You (She’s So Heavy) [Remix]
→ A_core: the beatles
→ T_core: i want you (she’s so heavy)
→ tags: {remix}

20.

A_in: Beethoven
T_in: Piano Sonata No. 14 in C-sharp minor, Op. 27, No. 2: I. Adagio sostenuto (Mono)
→ A_core: beethoven
→ T_core: piano sonata no. 14 in c-sharp minor, op. 27, no. 2: i. adagio sostenuto
→ tags: {mix:"mono"}
Note: catalog/movement retained.

## H. Article/diacritics exceptions (NL/EN)

21.

A_in: The The
T_in: This Is the Day
→ A_core: the the (keep article)
→ T_core: this is the day
→ tags: {}

22.

A_in: De Dijk
T_in: Binnen zonder kloppen
→ A_core: de dijk (keep “De”)
→ T_core: binnen zonder kloppen

23.

A_in: BLØF
T_in: Aan de kust
→ A_core: bløf (diacritics signature preserved)
→ T_core: aan de kust

24.

A_in: Tiësto
T_in: Adagio For Strings (Radio Edit)
→ A_core: tiësto
→ T_core: adagio for strings
→ tags: {edit:radio}

## I. Remix naming & generic “mix”

25.

A_in: Armand van Helden
T_in: Professional Widow (Armand’s Star Trunk Funkin’ Mix)
→ A_core: armand van helden
→ T_core: professional widow
→ tags: {remix:"armand’s star trunk funkin’"}
Note: normalize mixer name.

26.

A_in: Moby
T_in: Porcelain (Mix)
→ A_core: moby
→ T_core: porcelain
→ tags: {mix}

## J. Demos, sessions, versions

27.

A_in: Oasis
T_in: Wonderwall (Demo)
→ A_core: oasis
→ T_core: wonderwall
→ tags: {demo}

28.

A_in: David Bowie
T_in: Heroes (Album Version)
→ A_core: david bowie
→ T_core: heroes
→ tags: {version:"album"}

## K. Trick cases

29.

A_in: Live
T_in: Live
→ A_core: live (band)
→ T_core: live (title)
→ tags: {}
Note: do **not** infer performance tag.

30.

A_in: fun.
T_in: Some Nights (Radio Edit)
→ A_core: fun. (keep punctuation)
→ T_core: some nights
→ tags: {edit:radio}

31.

A_in: Mr. Probz
T_in: Waves (Robin Schulz Remix)
→ A_core: mr. probz
→ T_core: waves
→ tags: {remix:"robin schulz"}

32.

A_in: The Weeknd
T_in: Save Your Tears (Live)
→ A_core: the weeknd (keep “The”)
→ T_core: save your tears
→ tags: {live}

33.

A_in: Queen
T_in: Bohemian Rhapsody (Originally Performed by Queen) [Karaoke Version]
→ A_core: queen
→ T_core: bohemian rhapsody
→ tags: {karaoke}
Note: guard against linking to original.

34.

A_in: Coldplay
T_in: Fix You (From “Music of the Spheres”)
→ A_core: coldplay
→ T_core: fix you
→ tags: {ost:"music of the spheres"}
Note: album-style “From …” still treated as OSTish cue for intent.

35.

A_in: Armin van Buuren x Trevor Guthrie
T_in: This Is What It Feels Like
→ A_core: armin van buuren • trevor guthrie (separator normalized)
→ T_core: this is what it feels like
→ tags: {}

36.

A_in: De Jeugd van Tegenwoordig
T_in: Sterrenstof (Explicit)
→ A_core: de jeugd van tegenwoordig
→ T_core: sterrenstof
→ tags: {content:"explicit"}

37.

A_in: Beyoncé
T_in: Halo (Live 2009)
→ A_core: beyoncé
→ T_core: halo
→ tags: {live:"2009"}

38.

A_in: The Rolling Stones
T_in: (I Can’t Get No) Satisfaction – 2002 Remaster
→ A_core: the rolling stones
→ T_core: (i can’t get no) satisfaction
→ tags: {remaster:2002}

39.

A_in: Queen
T_in: We Are The Champions / We Will Rock You (Double A-Side)
→ A_core: queen
→ T_core: we are the champions / we will rock you
→ tags: {single_bundle:[…]}

40.

A_in: Bazart
T_in: Goud (Piano Version)
→ A_core: bazart
→ T_core: goud
→ tags: {acoustic}
