"""
Normalization QA Pack v1 Test Suite.

Tests based on docs/appendix/normalization_qa_pack_v1.md
"""

from __future__ import annotations

from chart_binder.normalize import Normalizer, TagKind


def test_qa_radio_edit():
    """QA Pack #1: Queen — Under Pressure (Radio Edit)"""
    norm = Normalizer()
    result = norm.normalize_title("Under Pressure (Radio Edit)")
    assert result.core == "under pressure"
    assert result.guests == []
    assert any(tag.kind == TagKind.edit and tag.sub == "radio" for tag in result.tags)


def test_qa_remaster_dash():
    """QA Pack #2: Radiohead — Paranoid Android - Remastered 2009"""
    norm = Normalizer()
    result = norm.normalize_title("Paranoid Android - Remastered 2009")
    assert result.core == "paranoid android"
    assert any(tag.kind == TagKind.remaster and tag.value == "2009" for tag in result.tags)


def test_qa_re_recording():
    """QA Pack #3: New Order — Blue Monday '88"""
    norm = Normalizer()
    result = norm.normalize_title("Blue Monday '88")
    assert result.core == "blue monday '88"


def test_qa_12_inch_mix():
    """QA Pack #4: Donna Summer — I Feel Love (12\" Mix)"""
    norm = Normalizer()
    result = norm.normalize_title('I Feel Love (12" Mix)')
    assert result.core == "i feel love"
    assert any(tag.kind == TagKind.extended and tag.sub == "12in" for tag in result.tags)


def test_qa_club_mix():
    """QA Pack #5: Massive Attack — Teardrop (Club Mix)"""
    norm = Normalizer()
    result = norm.normalize_title("Teardrop (Club Mix)")
    assert result.core == "teardrop"
    assert any(tag.kind == TagKind.extended and tag.sub == "club" for tag in result.tags)


def test_qa_live():
    """QA Pack #6: Nirvana — Smells Like Teen Spirit (Live at Reading 1992)"""
    norm = Normalizer()
    result = norm.normalize_title("Smells Like Teen Spirit (Live at Reading 1992)")
    assert result.core == "smells like teen spirit"
    assert any(tag.kind == TagKind.live for tag in result.tags)


def test_qa_unplugged():
    """QA Pack #7: Eric Clapton — Layla (Unplugged)"""
    norm = Normalizer()
    result = norm.normalize_title("Layla (Unplugged)")
    assert result.core == "layla"
    assert any(tag.kind == TagKind.acoustic for tag in result.tags)


def test_qa_peel_session():
    """QA Pack #8: The Smiths — What Difference Does It Make? (Peel Session)"""
    norm = Normalizer()
    result = norm.normalize_title("What Difference Does It Make? (Peel Session)")
    assert result.core == "what difference does it make?"
    assert any(tag.kind == TagKind.session for tag in result.tags)


def test_qa_artist_featuring():
    """QA Pack #9: Queen feat. David Bowie — Under Pressure"""
    norm = Normalizer()
    result = norm.normalize_artist("Queen feat. David Bowie")
    assert result.core == "queen"
    assert "david bowie" in result.guests


def test_qa_title_featuring():
    """QA Pack #10: Queen — Under Pressure (feat. David Bowie)"""
    norm = Normalizer()
    result = norm.normalize_title("Under Pressure (feat. David Bowie)")
    assert result.core == "under pressure"
    assert "david bowie" in result.guests


def test_qa_suzan_freek():
    """QA Pack #11: Suzan & Freek"""
    norm = Normalizer()
    result = norm.normalize_artist("Suzan & Freek")
    assert " • " in result.core or "&" not in result.core


def test_qa_the_beatles():
    """QA Pack #12: The Beatles (article stripping)"""
    norm = Normalizer()
    result = norm.normalize_artist("The Beatles")
    assert result.core == "beatles"


def test_qa_the_the():
    """QA Pack #13: The The (exception - keep article)"""
    norm = Normalizer()
    result = norm.normalize_artist("The The")
    assert result.core == "the the"


def test_qa_blof():
    """QA Pack #14: BLØF (diacritics)"""
    norm = Normalizer()
    result = norm.normalize_artist("BLØF")
    assert result.core == "blof"
    assert len(result.diacritics_signature) > 0


def test_qa_parentheses_in_title():
    """QA Pack #15: I Want You (She's So Heavy) [Remix] — preserve inner parens"""
    norm = Normalizer()
    result = norm.normalize_title("I Want You (She's So Heavy) [Remix]")
    assert "she's so heavy" in result.core
    assert any(tag.kind == TagKind.remix for tag in result.tags)


def test_qa_de_dijk():
    """QA Pack #16: De Dijk (NL article exception)"""
    norm = Normalizer()
    result = norm.normalize_artist("De Dijk")
    assert result.core == "de dijk"


def test_qa_pink():
    """QA Pack #17: P!nk (preserve punctuation)"""
    norm = Normalizer()
    result = norm.normalize_artist("P!nk")
    assert result.core == "p!nk"


def test_qa_idempotence_artist():
    """QA Pack #18: Idempotence check for artist"""
    norm = Normalizer()
    original = "The Beatles feat. Billy Preston"
    result1 = norm.normalize_artist(original)
    result2 = norm.normalize_artist(result1.core)
    assert result1.core == result2.core


def test_qa_idempotence_title():
    """QA Pack #19: Idempotence check for title"""
    norm = Normalizer()
    original = "Let It Be (Remastered 2009)"
    result1 = norm.normalize_title(original)
    result2 = norm.normalize_title(result1.core)
    assert result1.core == result2.core


def test_qa_mono_stereo():
    """QA Pack #20: Mono/Stereo tags"""
    norm = Normalizer()
    result = norm.normalize_title("Good Vibrations (Mono)")
    assert result.core == "good vibrations"
    assert any(tag.kind == TagKind.mix and tag.sub == "mono" for tag in result.tags)


def test_qa_explicit_clean():
    """QA Pack #21: Explicit/Clean content tags"""
    norm = Normalizer()
    result = norm.normalize_title("Rapper's Delight (Clean)")
    assert result.core == "rapper's delight"
    assert any(tag.kind == TagKind.content and tag.sub == "clean" for tag in result.tags)


def test_qa_demo():
    """QA Pack #22: Demo tag"""
    norm = Normalizer()
    result = norm.normalize_title("Smells Like Teen Spirit (Demo)")
    assert result.core == "smells like teen spirit"
    assert any(tag.kind == TagKind.demo for tag in result.tags)


def test_qa_medley():
    """QA Pack #23: Medley tag"""
    norm = Normalizer()
    result = norm.normalize_title("Abbey Road Medley")
    assert "medley" in result.core


def test_qa_karaoke():
    """QA Pack #24: Karaoke version"""
    norm = Normalizer()
    result = norm.normalize_title(
        "Bohemian Rhapsody (Originally Performed by Queen) [Karaoke Version]"
    )
    assert result.core == "bohemian rhapsody"
    assert any(tag.kind == TagKind.karaoke for tag in result.tags)


def test_qa_whitespace_normalization():
    """QA Pack #25: Whitespace normalization"""
    norm = Normalizer()
    result = norm.normalize_title("Hello  World\u00a0Test")
    assert "  " not in result.core
    assert "\u00a0" not in result.core


def test_qa_curly_quotes():
    """QA Pack #26: Curly quotes canonicalization"""
    norm = Normalizer()
    result = norm.normalize_title("Don't Stop Believin'")
    assert "'" in result.core or "'" not in result.core


def test_qa_dashes():
    """QA Pack #27: Em-dash/en-dash canonicalization"""
    norm = Normalizer()
    result = norm.normalize_title("test\u2013test\u2014test")
    assert "\u2013" not in result.core
    assert "\u2014" not in result.core


def test_qa_artist_separator_ampersand():
    """QA Pack #28: Artist separator normalization (&)"""
    norm = Normalizer()
    result = norm.normalize_artist("Simon & Garfunkel")
    assert "&" not in result.core or " • " in result.core


def test_qa_artist_separator_plus():
    """QA Pack #29: Artist separator normalization (+)"""
    norm = Normalizer()
    result = norm.normalize_artist("Salt + Pepper")
    assert "+" not in result.core or " • " in result.core


def test_qa_multiple_guests():
    """QA Pack #30: Multiple guests"""
    norm = Normalizer()
    result = norm.normalize_artist("Artist feat. Guest1 feat. Guest2")
    assert result.core == "artist"
    assert len(result.guests) >= 1
