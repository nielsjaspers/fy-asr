import random

from fy_asr.corpora import Utterance
from fy_asr.train import GAP_SECONDS, WINDOW_SECONDS, pack


def utterance(i: int, duration: float) -> Utterance:
    return Utterance(f"{i}.wav", f"tekst {i}", "cv22", "s", duration)


def test_pack_fills_30s_windows_and_keeps_every_clip_that_fits() -> None:
    rng = random.Random(0)
    clips = [utterance(i, rng.uniform(1.0, 12.0)) for i in range(200)]
    clips.append(utterance(999, 31.0))
    windows = pack(clips, random.Random(1))
    packed = [u for window in windows for u in window]
    assert sorted(u.audio for u in packed) == sorted(c.audio for c in clips[:200])
    for window in windows:
        length = sum(u.duration for u in window) + GAP_SECONDS * (len(window) - 1)
        assert length <= WINDOW_SECONDS
    assert len(windows) < 200 / 3
