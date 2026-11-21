from itertools import pairwise

from fy_asr.asr import Word
from fy_asr.subtitles import MAX_LINE_CHARS, build_cues, to_srt


def spaced(texts: list[str], step: float = 0.4) -> list[Word]:
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
    words = [Word("ja", 0.0, 0.2), Word("nee", 1.2, 1.4), Word("dat", 4.0, 4.2)]
    cues = build_cues(words)
    assert cues[0].end == 1.0
    assert cues[1].start == 1.2 and cues[1].end == 2.2
    assert all(a.end <= b.start for a, b in pairwise(cues))


def test_srt_format() -> None:
    srt = to_srt(build_cues([Word("hoi", 3661.5, 3662.75)]))
    assert srt == "1\n01:01:01,500 --> 01:01:02,750\nhoi\n"
