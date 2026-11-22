from collections.abc import Sequence
from itertools import pairwise

from fy_asr.asr import Word
from fy_asr.subtitles import MAX_LINE_CHARS, build_cues, to_srt


def spaced(texts: Sequence[str], step: float = 0.4) -> list[Word]:
    return [Word(t, i * step, i * step + step * 0.8) for i, t in enumerate(texts)]


def test_pause_starts_new_cue() -> None:
    words = [Word("hoi", 0.0, 0.3), Word("wrâld", 0.4, 0.8), Word("goeie", 3.0, 3.4)]
    cues = build_cues(words)
    assert [c.lines for c in cues] == [("hoi wrâld",), ("goeie",)]


def test_long_text_splits_into_two_balanced_lines_and_new_cues() -> None:
    cues = build_cues(spaced(["wurd"] * 30, step=0.1))
    assert len(cues) > 1
    assert all(len(c.lines) <= 2 for c in cues)
    assert all(len(line) <= MAX_LINE_CHARS for c in cues for line in c.lines)
    first = cues[0].lines
    assert len(first) == 2 and abs(len(first[0]) - len(first[1])) <= len("wurd ")


def test_short_cue_is_extended_but_never_overlaps_next() -> None:
    first = [
        Word(t, i * 0.05, i * 0.05 + 0.05)
        for i, t in enumerate(
            ["Dat", "is", "in", "hiel", "goede", "fraach,", "tink", "ik."]
        )
    ]
    second = [
        Word(t, 0.5 + i * 0.3, 0.7 + i * 0.3)
        for i, t in enumerate(
            [
                "Mar",
                "dêr",
                "moatte",
                "wy",
                "noch",
                "in",
                "kear",
                "mei",
                "de",
                "dokter",
                "oer",
                "prate.",
            ]
        )
    ]
    cues = build_cues(first + second)
    assert [len(c.lines) for c in cues] == [1, 2]
    assert cues[0].end == 0.5
    assert cues[1].end == second[-1].end
    assert all(a.end <= b.start for a, b in pairwise(cues))


def test_srt_format() -> None:
    srt = to_srt(build_cues([Word("hoi", 3661.5, 3662.75)]))
    assert srt == "1\n01:01:01,500 --> 01:01:02,750\nhoi\n"


def test_cues_end_at_a_sentence_when_the_text_needs_two_cues() -> None:
    text = (
        "Dat is goed. Mar wy moatte noch even sjen wat de dokter seit"
        " oer de útslach fan it bloedûndersyk."
    )
    cues = build_cues(spaced(text.split(), step=0.25))
    assert len(cues) == 2
    assert cues[0].lines == ("Dat is goed.",)


def test_line_break_prefers_a_comma_over_balance() -> None:
    text = "Ja, dat tink ik wol, mar ik wit it net seker fan dy saak."
    (cue,) = build_cues(spaced(text.split(), step=0.25))
    assert cue.lines[0].endswith(",")


def test_fillers_are_dropped() -> None:
    words = [
        Word("Jild", 0.0, 0.4),
        Word("uh,", 0.4, 0.8),
        Word("by", 0.8, 1.0),
        Word("my.", 1.0, 1.4),
    ]
    assert [c.lines for c in build_cues(words)] == [("Jild by my.",)]
