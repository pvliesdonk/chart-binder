from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class TagKind(StrEnum):
    """Edition tag types extracted during normalization."""

    edit = "edit"
    live = "live"
    remaster = "remaster"
    mono = "mono"
    stereo = "stereo"
    acoustic = "acoustic"
    remix = "remix"
    extended = "extended"
    instrumental = "instrumental"
    karaoke = "karaoke"
    ost = "ost"
    demo = "demo"
    content = "content"
    re_recording = "re_recording"
    medley = "medley"
    lang = "lang"
    mix = "mix"
    session = "session"


@dataclass
class EditionTag:
    """Structured edition tag extracted from title or artist."""

    kind: TagKind
    sub: str | None = None
    value: str | None = None
    note: str | None = None


@dataclass
class NormalizedResult:
    """
    Result of normalization pipeline.

    Contains both core forms for matching and extracted metadata.
    """

    normalized: str
    core: str
    guests: list[str]
    tags: list[EditionTag]
    diacritics_signature: str
    ruleset_version: str = "norm-v1"


class Normalizer:
    """
    Deterministic text normalizer for artist/title strings.

    Implements the normalization ruleset v1 from the spec.
    """

    FEAT_TOKENS = ["feat.", "featuring", "ft.", "met", "with", "con"]
    OST_PREFIX_PATTERNS = [
        r'^from\s+"([^"]+)"',
        r"^from\s+'([^']+)'",
        r"^uit\s+\"([^\"]+)\"",
        r"^van de film\s+\"([^\"]+)\"",
    ]
    GUEST_SEPARATOR_PATTERN = r"\s+(?:feat\.?|featuring|ft\.?|met|with|con)\s+"

    CANONICAL_SEPARATOR = " • "

    EXCEPTION_ARTISTS = {
        "the the",
        "the 1975",
        "the weeknd",
        "de dijk",
        "de jeugd van tegenwoordig",
        "het goede doel",
    }

    PRESERVE_PUNCTUATION_ARTISTS = {
        "p!nk",
        "fun.",
        "blink-182",
        "will.i.am",
        "!!!",
        "(hed) p.e.",
        "mø",
        "björk",
    }

    def __init__(self):
        pass

    def normalize_artist(self, artist: str) -> NormalizedResult:
        """
        Normalize artist string.

        Extracts guests and applies artist-specific rules.
        """
        original = artist
        s = self._apply_unicode_whitespace(artist)
        s_lower = s.lower()

        diacritics_sig = self._extract_diacritics_signature(s)

        if s_lower in self.PRESERVE_PUNCTUATION_ARTISTS:
            core = s_lower
            return NormalizedResult(
                normalized=core,
                core=core,
                guests=[],
                tags=[],
                diacritics_signature=diacritics_sig,
            )

        core, guests = self._extract_guests(s)

        core = self._apply_casefold(core)
        core = self._apply_punctuation_canonicalization(core)
        core = self._strip_diacritics(core)
        core = self._normalize_artist_separators(core)
        core = self._strip_leading_article(core, original)
        core = self._final_compaction(core)

        normalized_guests = [
            self._final_compaction(self._strip_diacritics(self._apply_casefold(g))) for g in guests
        ]

        return NormalizedResult(
            normalized=core,
            core=core,
            guests=normalized_guests,
            tags=[],
            diacritics_signature=diacritics_sig,
        )

    def normalize_title(self, title: str) -> NormalizedResult:
        """
        Normalize title string.

        Extracts edition tags, guests, and produces matchable core form.
        """
        s = self._apply_unicode_whitespace(title)
        diacritics_sig = self._extract_diacritics_signature(s)

        s, ost_tags = self._extract_ost_prefix(s)
        s, suffix_tags = self._extract_edition_suffixes(s)
        s, title_guests = self._extract_title_guests(s)

        core = self._apply_casefold(s)
        core = self._apply_punctuation_canonicalization(core)
        core = self._strip_diacritics(core)
        core = self._final_compaction(core)

        all_tags = ost_tags + suffix_tags

        normalized_guests = [
            self._final_compaction(self._strip_diacritics(self._apply_casefold(g)))
            for g in title_guests
        ]

        return NormalizedResult(
            normalized=core,
            core=core,
            guests=normalized_guests,
            tags=all_tags,
            diacritics_signature=diacritics_sig,
        )

    def _apply_unicode_whitespace(self, s: str) -> str:
        """Normalize to NFC and collapse whitespace."""
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"\s+", " ", s)
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[\u200b-\u200d\ufeff]", "", s)
        return s.strip()

    def _apply_casefold(self, s: str) -> str:
        """Convert to lowercase for matching."""
        return s.lower()

    def _apply_punctuation_canonicalization(self, s: str) -> str:
        """Canonicalize quotes, dashes, and ellipsis."""
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        s = s.replace("\u2026", "...")
        return s

    def _strip_diacritics(self, s: str) -> str:
        """Remove diacritics for matching."""
        nfd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def _extract_diacritics_signature(self, s: str) -> str:
        """Extract positions and types of diacritics for boosting."""
        nfd = unicodedata.normalize("NFD", s)
        diacritics = []
        for i, c in enumerate(nfd):
            if unicodedata.category(c) == "Mn":
                diacritics.append(f"{i}:{c}")
        return ",".join(diacritics)

    def _extract_guests(self, s: str) -> tuple[str, list[str]]:
        """Extract featuring/with guests from artist string."""
        parts = re.split(self.GUEST_SEPARATOR_PATTERN, s, flags=re.IGNORECASE)
        if len(parts) == 1:
            return s, []

        core = parts[0].strip()
        guests = [g.strip() for g in parts[1:] if g.strip()]
        return core, guests

    def _extract_title_guests(self, s: str) -> tuple[str, list[str]]:
        """Extract featuring/with guests from title suffix."""
        pattern = r"\s*[\(\[]\s*(?:feat\.?|featuring|ft\.?|met|with)\s+([^\)\]]+)[\)\]]"
        matches = list(re.finditer(pattern, s, re.IGNORECASE))

        if not matches:
            return s, []

        guests = []
        for match in reversed(matches):
            guests.append(match.group(1).strip())
            s = s[: match.start()] + s[match.end() :]

        return s.strip(), list(reversed(guests))

    def _extract_ost_prefix(self, s: str) -> tuple[str, list[EditionTag]]:
        """Extract OST prefix markers."""
        for pattern in self.OST_PREFIX_PATTERNS:
            match = re.match(pattern, s, re.IGNORECASE)
            if match:
                work = match.group(1)
                tag = EditionTag(kind=TagKind.ost, value=work)
                remaining = s[match.end() :].strip()
                return remaining, [tag]
        return s, []

    STOPWORDS = {"the", "a", "an", "and", "or", "but"}

    def _extract_edition_suffixes(self, s: str) -> tuple[str, list[EditionTag]]:
        """
        Extract edition tags from parenthetical/bracketed suffixes.

        Applies guardrails to avoid over-stripping.
        """
        tags: list[EditionTag] = []
        original = s
        original_len = len(s)
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            s_before = s

            s, new_tags = self._try_extract_one_suffix(s)
            tags.extend(new_tags)

            if s == s_before:
                break

        words = s.strip().lower().split()
        if len(s) < original_len * 0.3 and len(words) == 1 and words[0] in self.STOPWORDS:
            return original, []

        return s.strip(), tags

    def _try_extract_one_suffix(self, s: str) -> tuple[str, list[EditionTag]]:
        """Try to extract one edition suffix from end of string."""
        patterns = [
            (
                r'\s*[\(\[]\s*(radio\s+edit|single\s+edit|7"\s*edit|edit)\s*[\)\]]$',
                TagKind.edit,
                "sub",
                self._parse_edit_sub,
            ),
            (r"\s*[\(\[]\s*live(?:\s+at\s+([^\)\]]+))?\s*[\)\]]$", TagKind.live, "value", None),
            (
                r"\s*[\(\[]\s*remaster(?:ed)?(?:\s+(\d{4}))?\s*[\)\]]$",
                TagKind.remaster,
                "value",
                None,
            ),
            (r"\s*[\(\[]\s*(mono|stereo)\s*[\)\]]$", TagKind.mix, "sub", None),
            (
                r"\s*[\(\[]\s*(acoustic|unplugged|piano\s+version)\s*[\)\]]$",
                TagKind.acoustic,
                "note",
                None,
            ),
            (r"\s*[\(\[]\s*([^)\]]*remix)\s*[\)\]]$", TagKind.remix, "value", None),
            (
                r'\s*[\(\[]\s*(extended(?:\s+mix)?|12"\s*mix|club\s+mix|dub|instrumental|karaoke\s+version)\s*[\)\]]$',
                TagKind.extended,
                "sub",
                self._parse_extended_sub,
            ),
            (r"\s*[\(\[]\s*(demo|early\s+version|rough\s+mix)\s*[\)\]]$", TagKind.demo, None, None),
            (r"\s*[\(\[]\s*(clean|explicit)\s*[\)\]]$", TagKind.content, "sub", None),
            (
                r"\s*[\(\[]\s*(re-recorded|taylor\'?s\s+version|new\s+recording)\s*[\)\]]$",
                TagKind.re_recording,
                "note",
                None,
            ),
            (r"\s*[\(\[]\s*medley\s*[\)\]]$", TagKind.medley, None, None),
            (
                r"\s*[\(\[]\s*(?:originally\s+performed\s+by\s+[^\)\]]+)\s*[\)\]]$",
                TagKind.karaoke,
                None,
                None,
            ),
            (r"\s*[\(\[]\s*peel\s+session\s*[\)\]]$", TagKind.session, "sub", None),
        ]

        for pattern, kind, value_field, parser in patterns:
            match = re.search(pattern, s, re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex and match.lastindex >= 1 else None
                tag_kwargs: dict[str, str | TagKind | None] = {"kind": kind}
                if value_field and value:
                    if parser:
                        tag_kwargs[value_field] = parser(value)  # pyright: ignore[reportArgumentType]
                    else:
                        tag_kwargs[value_field] = value.lower().strip()  # pyright: ignore[reportArgumentType]
                tag = EditionTag(**tag_kwargs)  # pyright: ignore[reportArgumentType]
                new_s = s[: match.start()].strip()
                return new_s, [tag]

        dash_pattern = (
            r"\s*-\s*(remaster(?:ed)?\s+\d{4}|live\s+at\s+[^$]+|radio\s+edit|single\s+version)$"
        )
        match = re.search(dash_pattern, s, re.IGNORECASE)
        if match:
            suffix = match.group(1).lower()
            if "remaster" in suffix:
                year_match = re.search(r"(\d{4})", suffix)
                tag = EditionTag(
                    kind=TagKind.remaster, value=year_match.group(1) if year_match else None
                )
            elif "live" in suffix:
                tag = EditionTag(kind=TagKind.live, value=suffix.replace("live at", "").strip())
            elif "edit" in suffix:
                tag = EditionTag(kind=TagKind.edit, sub="radio" if "radio" in suffix else "single")
            else:
                tag = EditionTag(kind=TagKind.edit, sub="version")
            new_s = s[: match.start()].strip()
            return new_s, [tag]

        return s, []

    def _parse_edit_sub(self, value: str) -> str:
        """Parse edit sub-type from matched value."""
        value_lower = value.lower()
        if "radio" in value_lower:
            return "radio"
        elif "single" in value_lower:
            return "single"
        elif '7"' in value_lower or "7in" in value_lower:
            return "7in"
        else:
            return "edit"

    def _parse_extended_sub(self, value: str) -> str:
        """Parse extended mix sub-type from matched value."""
        value_lower = value.lower()
        if "12" in value_lower:
            return "12in"
        elif "club" in value_lower:
            return "club"
        elif "dub" in value_lower:
            return "dub"
        elif "instrumental" in value_lower:
            return "instrumental"
        elif "karaoke" in value_lower:
            return "karaoke"
        else:
            return "extended"

    def _normalize_artist_separators(self, s: str) -> str:
        """Normalize separators between artists to canonical token."""
        s = re.sub(r"\s*[&\+x×/,]\s*", self.CANONICAL_SEPARATOR, s, flags=re.IGNORECASE)
        return s

    def _strip_leading_article(self, s: str, original: str) -> str:
        """Strip leading 'the' unless in exception list."""
        s_lower = s.lower()
        orig_lower = original.lower()

        if orig_lower in self.EXCEPTION_ARTISTS:
            return s

        if s_lower.startswith("the ") and len(s_lower.split()) > 1:
            return s[4:]

        return s

    def _final_compaction(self, s: str) -> str:
        """Remove duplicate spaces and trailing punctuation."""
        s = re.sub(r"\s+", " ", s)
        s = s.strip()
        s = re.sub(r"[-\s]+$", "", s)
        return s


## Tests


def test_unicode_whitespace():
    norm = Normalizer()
    result = norm._apply_unicode_whitespace("hello\u00a0world  \u200btest")
    assert result == "hello world test"


def test_casefold():
    norm = Normalizer()
    assert norm._apply_casefold("HELLO World") == "hello world"


def test_diacritics_stripping():
    norm = Normalizer()
    assert norm._strip_diacritics("café") == "cafe"
    assert norm._strip_diacritics("Björk") == "Bjork"


def test_diacritics_signature():
    norm = Normalizer()
    sig = norm._extract_diacritics_signature("café")
    assert len(sig) > 0


def test_guest_extraction_artist():
    norm = Normalizer()
    result = norm.normalize_artist("Queen feat. David Bowie")
    assert result.core == "queen"
    assert "david bowie" in result.guests


def test_guest_extraction_title():
    norm = Normalizer()
    result = norm.normalize_title("Under Pressure (feat. David Bowie)")
    assert result.core == "under pressure"
    assert "david bowie" in result.guests


def test_edition_tag_extraction_radio_edit():
    norm = Normalizer()
    result = norm.normalize_title("One (Radio Edit)")
    assert result.core == "one"
    assert any(tag.kind == TagKind.edit for tag in result.tags)


def test_edition_tag_extraction_remaster():
    norm = Normalizer()
    result = norm.normalize_title("Paranoid Android (Remastered 2009)")
    assert result.core == "paranoid android"
    assert any(tag.kind == TagKind.remaster and tag.value == "2009" for tag in result.tags)


def test_edition_tag_extraction_live():
    norm = Normalizer()
    result = norm.normalize_title("Smells Like Teen Spirit (Live at Reading 1992)")
    assert result.core == "smells like teen spirit"
    assert any(tag.kind == TagKind.live for tag in result.tags)


def test_exception_artist_the_the():
    norm = Normalizer()
    result = norm.normalize_artist("The The")
    assert result.core == "the the"


def test_exception_artist_de_dijk():
    norm = Normalizer()
    result = norm.normalize_artist("De Dijk")
    assert result.core == "de dijk"


def test_preserve_punctuation_pink():
    norm = Normalizer()
    result = norm.normalize_artist("P!nk")
    assert result.core == "p!nk"


def test_idempotence_artist():
    norm = Normalizer()
    result1 = norm.normalize_artist("The Beatles")
    result2 = norm.normalize_artist(result1.core)
    assert result1.core == result2.core


def test_idempotence_title():
    norm = Normalizer()
    result1 = norm.normalize_title("Yesterday (Remastered 2009)")
    result2 = norm.normalize_title(result1.core)
    assert result1.core == result2.core
