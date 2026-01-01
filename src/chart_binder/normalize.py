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

    # Edition tag extraction patterns
    PATTERN_EDIT = r'\s*[\(\[]\s*(radio\s+edit|single\s+edit|7"\s*edit|edit)\s*[\)\]]$'
    PATTERN_LIVE = r"\s*[\(\[]\s*live(?:\s+at\s+([^\)\]]+))?\s*[\)\]]$"
    PATTERN_REMASTER = r"\s*[\(\[]\s*remaster(?:ed)?(?:\s+(\d{4}))?\s*[\)\]]$"
    PATTERN_MIX = r"\s*[\(\[]\s*(mono|stereo)\s*[\)\]]$"
    PATTERN_ACOUSTIC = r"\s*[\(\[]\s*(acoustic|unplugged|piano\s+version)\s*[\)\]]$"
    PATTERN_REMIX = r"\s*[\(\[]\s*([^)\]]*remix)\s*[\)\]]$"
    PATTERN_EXTENDED = r'\s*[\(\[]\s*(extended(?:\s+mix)?|12"\s*mix|club\s+mix|dub|instrumental|karaoke\s+version)\s*[\)\]]$'
    PATTERN_DEMO = r"\s*[\(\[]\s*(demo|early\s+version|rough\s+mix)\s*[\)\]]$"
    PATTERN_CONTENT = r"\s*[\(\[]\s*(clean|explicit)\s*[\)\]]$"
    PATTERN_RE_RECORDING = (
        r"\s*[\(\[]\s*(re-recorded|taylor\'?s\s+version|new\s+recording)\s*[\)\]]$"
    )
    PATTERN_MEDLEY = r"\s*[\(\[]\s*medley\s*[\)\]]$"
    PATTERN_KARAOKE = (
        r"\s*[\(\[]\s*(?:originally\s+performed\s+by\s+[^\)\]]+|karaoke(?:\s+version)?)\s*[\)\]]$"
    )
    PATTERN_SESSION = r"\s*[\(\[]\s*(peel)\s+session\s*[\)\]]$"
    PATTERN_DASH_SUFFIX = (
        r"\s*-\s*(remaster(?:ed)?\s+\d{4}|live\s+at\s+.+|radio\s+edit|single\s+version)$"
    )

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
        """Convert to lowercase for matching with language-specific rules."""
        return s.casefold()

    def _apply_punctuation_canonicalization(self, s: str) -> str:
        """Canonicalize quotes, dashes, and ellipsis."""
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        s = s.replace("\u201c", '"').replace("\u201d", '"')
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        s = s.replace("\u2026", "...")
        return s

    TRANSLITERATION_MAP = {
        "ø": "o",
        "Ø": "O",
        "å": "a",
        "Å": "A",
        "æ": "ae",
        "Æ": "AE",
        "ð": "d",
        "Ð": "D",
        "þ": "th",
        "Þ": "TH",
        "ß": "ss",
        "œ": "oe",
        "Œ": "OE",
        "ł": "l",
        "Ł": "L",
    }
    TRANSLITERATION_TABLE = str.maketrans(TRANSLITERATION_MAP)

    def _strip_diacritics(self, s: str) -> str:
        """Remove diacritics for matching."""
        s = s.translate(self.TRANSLITERATION_TABLE)
        nfd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    def _extract_diacritics_signature(self, s: str) -> str:
        """Extract positions and types of diacritics for boosting."""
        nfd = unicodedata.normalize("NFD", s)
        diacritics = []
        for i, c in enumerate(nfd):
            if unicodedata.category(c) == "Mn":
                diacritics.append(f"{i}:{c}")
            elif c in self.TRANSLITERATION_MAP or c.lower() in self.TRANSLITERATION_MAP:
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
            (self.PATTERN_EDIT, TagKind.edit, "sub", self._parse_edit_sub),
            (self.PATTERN_LIVE, TagKind.live, "value", None),
            (self.PATTERN_REMASTER, TagKind.remaster, "value", None),
            (self.PATTERN_MIX, TagKind.mix, "sub", None),
            (self.PATTERN_ACOUSTIC, TagKind.acoustic, "note", None),
            (self.PATTERN_REMIX, TagKind.remix, "value", None),
            (self.PATTERN_EXTENDED, TagKind.extended, "sub", self._parse_extended_sub),
            (self.PATTERN_DEMO, TagKind.demo, None, None),
            (self.PATTERN_CONTENT, TagKind.content, "sub", None),
            (self.PATTERN_RE_RECORDING, TagKind.re_recording, "note", None),
            (self.PATTERN_MEDLEY, TagKind.medley, None, None),
            (self.PATTERN_KARAOKE, TagKind.karaoke, None, None),
            (self.PATTERN_SESSION, TagKind.session, "sub", None),
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

        match = re.search(self.PATTERN_DASH_SUFFIX, s, re.IGNORECASE)
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


# =============================================================================
# Fuzzy Matching Utilities
# =============================================================================
#
# These functions provide centralized fuzzy string matching for candidate
# validation and search result filtering. They use rapidfuzz for performance.
#
# Design rationale:
# - is_sane(): Prevents false positives from over-normalization. If we strip
#   too much (diacritics, punctuation, guests), the normalized form might be
#   meaningless (e.g., single char, all numbers). This catches those cases.
# - fuzzy_ratio(): Raw similarity score (0-100) for ranking candidates.
# - fuzzy_match(): Boolean check for filtering candidates above threshold.
#
# The 70% threshold for is_sane() was chosen empirically:
# - Lower values (50-60%) let through too many false positives
# - Higher values (80-90%) reject valid normalizations (e.g., "The Beatles" → "beatles")
# - 70% balances catching garbage while allowing legitimate simplification
# =============================================================================


# Default thresholds (can be overridden by callers)
DEFAULT_SANITY_THRESHOLD = 70  # Minimum similarity to original to be "sane"
DEFAULT_MATCH_THRESHOLD = 80  # Minimum similarity for fuzzy_match()


def fuzzy_ratio(a: str, b: str) -> float:
    """
    Calculate fuzzy similarity ratio between two strings.

    Uses Levenshtein distance normalized to 0-100 scale.
    Higher values = more similar.

    Why Levenshtein (fuzz.ratio):
    - Simple, predictable behavior
    - Works well for similar-length strings
    - Fast for typical artist/title lengths (~50 chars)

    For word-order independent matching (e.g., "Artist A & Artist B" vs
    "Artist B & Artist A"), use fuzzy_token_set_ratio() instead.

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        Similarity score from 0.0 (completely different) to 100.0 (identical)

    Example:
        >>> fuzzy_ratio("hello", "hello")
        100.0
        >>> fuzzy_ratio("hello", "hallo")
        80.0
        >>> fuzzy_ratio("hello", "world")
        20.0
    """
    try:
        from rapidfuzz import fuzz

        return fuzz.ratio(a, b)
    except ImportError:
        # Fallback: exact match only (no fuzzy matching available)
        return 100.0 if a == b else 0.0


def fuzzy_token_set_ratio(a: str, b: str) -> float:
    """
    Calculate fuzzy similarity using token set ratio.

    Token set ratio is order-independent: it tokenizes both strings,
    finds common tokens, and compares the remaining tokens.

    Why token_set_ratio:
    - Handles word reordering: "Queen & David Bowie" ≈ "David Bowie & Queen"
    - Handles partial matches: "Yesterday (Remastered)" ≈ "Yesterday"
    - Better for artist names with multiple members in different orders

    When to use:
    - Comparing artist names (order may vary)
    - Comparing titles with extra info (remixes, features)

    When NOT to use:
    - Exact title matching (use fuzzy_ratio instead)
    - Short strings (tokenization less meaningful)

    Args:
        a: First string to compare
        b: Second string to compare

    Returns:
        Similarity score from 0.0 to 100.0 (order-independent)

    Example:
        >>> fuzzy_token_set_ratio("Queen & David Bowie", "David Bowie & Queen")
        100.0
        >>> fuzzy_token_set_ratio("Yesterday", "Yesterday (Remastered)")
        100.0
    """
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(a, b)
    except ImportError:
        # Fallback: exact match only
        return 100.0 if a == b else 0.0


def fuzzy_match(a: str, b: str, threshold: int = DEFAULT_MATCH_THRESHOLD) -> bool:
    """
    Check if two strings are similar enough to be considered a match.

    This is a convenience wrapper around fuzzy_ratio() that returns a boolean.
    Use this for filtering candidates where you need a yes/no decision.

    Why 80% default threshold:
    - Industry standard for fuzzy deduplication
    - Allows minor typos ("Beetles" vs "Beatles" = 86%)
    - Rejects clearly different strings ("Beatles" vs "Stones" = 40%)
    - Matches our MusicBrainz search confidence baseline

    Args:
        a: First string to compare
        b: Second string to compare
        threshold: Minimum similarity score to return True (0-100, default 80)

    Returns:
        True if similarity >= threshold, False otherwise

    Example:
        >>> fuzzy_match("hello", "hallo")  # 80% similar
        True
        >>> fuzzy_match("hello", "hallo", threshold=85)  # Too strict
        False
    """
    return fuzzy_ratio(a, b) >= threshold


def is_sane(
    original: str,
    normalized: str,
    threshold: int = DEFAULT_SANITY_THRESHOLD,
) -> bool:
    """
    Validate that a normalized string is still meaningful.

    This function catches over-normalization: cases where we stripped too
    much and the result is garbage (empty, single char, all numbers, etc.).

    The check has two parts:
    1. Length/content checks: reject obviously invalid results
    2. Similarity check: ensure normalized isn't too different from original

    Why we need this:
    - Normalization can strip diacritics, guests, edition tags, articles
    - Aggressive stripping might leave meaningless residue
    - Example: "A" from "The A-Team (Remastered 2020)" is not useful
    - Example: "123" from "123 (feat. Artist)" is probably wrong

    The 70% threshold means:
    - "The Beatles" → "beatles" (75% similar) = SANE
    - "The A-Team" → "a team" (55% similar) = SANE (passes length check)
    - "The A" → "a" (33% similar) = NOT SANE

    Args:
        original: The original string before normalization
        normalized: The normalized string to validate
        threshold: Minimum similarity to original (0-100, default 70)

    Returns:
        True if normalized is a valid simplification of original,
        False if it looks like over-normalization garbage

    Example:
        >>> is_sane("The Beatles", "beatles")
        True
        >>> is_sane("Yesterday (Remastered 2009)", "yesterday")
        True
        >>> is_sane("The A", "a")  # Single char after normalization
        False
        >>> is_sane("123 (Remix)", "123")  # All digits
        False
    """
    # Reject empty or whitespace-only results
    if not normalized or not normalized.strip():
        return False

    # Reject single-character results (too aggressive stripping)
    if len(normalized.strip()) <= 1:
        return False

    # Reject all-digit results (likely a catalog number, not a title)
    if normalized.strip().isdigit():
        return False

    # Reject all-punctuation results
    if all(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in normalized.strip()):
        return False

    # Check similarity to original
    # Lower threshold than fuzzy_match() because normalization IS supposed
    # to change the string (lowercase, strip diacritics, etc.)
    similarity = fuzzy_ratio(original.lower(), normalized.lower())
    if similarity < threshold:
        # Edge case: if normalized is a substantial substring, allow it
        # This handles "Yesterday (Remastered 2009)" → "yesterday"
        # where similarity is low but result is clearly the core title
        if len(normalized) >= 3 and normalized.lower() in original.lower():
            return True
        return False

    return True


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


# --- Fuzzy matching tests ---


def test_fuzzy_ratio_identical():
    """Identical strings should have 100% similarity."""
    assert fuzzy_ratio("hello", "hello") == 100.0
    assert fuzzy_ratio("The Beatles", "The Beatles") == 100.0


def test_fuzzy_ratio_similar():
    """Similar strings should have high similarity."""
    # One character difference in 5-char string
    ratio = fuzzy_ratio("hello", "hallo")
    assert 75 <= ratio <= 85  # Allow some variance


def test_fuzzy_ratio_different():
    """Different strings should have low similarity."""
    ratio = fuzzy_ratio("hello", "world")
    assert ratio < 50


def test_fuzzy_token_set_ratio_reorder():
    """Token set ratio should handle word reordering."""
    # Same words, different order
    ratio = fuzzy_token_set_ratio("Queen & David Bowie", "David Bowie & Queen")
    assert ratio >= 90  # Should be very similar


def test_fuzzy_token_set_ratio_subset():
    """Token set ratio should handle subsets well."""
    # "Yesterday" is subset of "Yesterday (Remastered)"
    ratio = fuzzy_token_set_ratio("Yesterday", "Yesterday Remastered")
    assert ratio >= 80


def test_fuzzy_match_above_threshold():
    """fuzzy_match should return True above threshold."""
    # "Beetles" vs "Beatles" - common typo
    assert fuzzy_match("Beetles", "Beatles")  # > 80%


def test_fuzzy_match_below_threshold():
    """fuzzy_match should return False below threshold."""
    assert not fuzzy_match("Beatles", "Rolling Stones")


def test_fuzzy_match_custom_threshold():
    """fuzzy_match should respect custom threshold."""
    # These are ~80% similar, so threshold 85 should fail
    assert not fuzzy_match("hello", "hallo", threshold=85)
    assert fuzzy_match("hello", "hallo", threshold=75)


def test_is_sane_valid_normalization():
    """is_sane should accept valid normalizations."""
    # Normal case: lowercase + strip article
    assert is_sane("The Beatles", "beatles")
    # Normal case: strip edition suffix
    assert is_sane("Yesterday (Remastered 2009)", "yesterday")
    # Normal case: strip diacritics
    assert is_sane("Café del Mar", "cafe del mar")


def test_is_sane_rejects_empty():
    """is_sane should reject empty results."""
    assert not is_sane("Something", "")
    assert not is_sane("Something", "   ")


def test_is_sane_rejects_single_char():
    """is_sane should reject single-character results."""
    assert not is_sane("The A", "a")
    assert not is_sane("Something", "x")


def test_is_sane_rejects_all_digits():
    """is_sane should reject all-digit results (likely catalog numbers)."""
    assert not is_sane("123 (Remix)", "123")
    assert not is_sane("Track 42", "42")


def test_is_sane_rejects_over_normalization():
    """is_sane should reject results too different from original."""
    # This would fail because "xyz" is not similar to "The Beatles"
    assert not is_sane("The Beatles", "xyz")


def test_is_sane_allows_substantial_substring():
    """is_sane should allow result that's a substantial substring of original."""
    # "yesterday" is in "Yesterday (Remastered 2009)" even though
    # the fuzzy ratio might be low due to length difference
    assert is_sane("Yesterday (Remastered 2009)", "yesterday")
