import random
from itertools import pairwise

from fy_asr.corpora import Utterance
from fy_asr.train import WINDOW_SECONDS, gap, pack


def clip(i: int, duration: float) -> Utterance:
    return Utterance(f"{i}.wav", f"tekst {i}", "cv22", "s", duration)


def turn(recording: str, i: int, offset: float, duration: float) -> Utterance:
    return Utterance(
        f"{recording}_{i}.wav", f"turn {i}", "fris", "s", duration, recording, offset
    )


def length(window: list[Utterance]) -> float:
    return sum(u.duration for u in window) + sum(gap(a, b) for a, b in pairwise(window))


def test_pack_keeps_turns_in_order_and_repeats_dialogue() -> None:
    rng = random.Random(0)
    clips = [clip(i, rng.uniform(1.0, 12.0)) for i in range(100)]
    clips.append(clip(999, 31.0))
    turns = []
    for rec in ("a", "b"):
        offset = 0.0
        for i in range(40):
            duration = rng.uniform(1.0, 10.0)
            turns.append(turn(rec, i, offset, duration))
            offset += duration + rng.uniform(0.0, 3.0)
    windows = pack(clips + turns[::-1], random.Random(1), {"fris": 2})

    packed = [u.audio for w in windows for u in w]
    assert sorted(packed) == sorted(
        [c.audio for c in clips[:100]] + [t.audio for t in turns] * 2
    )
    for window in windows:
        assert length(window) <= WINDOW_SECONDS
        recordings = {u.recording for u in window}
        assert len(recordings) == 1
        if recordings != {""}:
            offsets = [u.offset for u in window]
            assert offsets == sorted(offsets)
