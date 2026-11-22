import random
from itertools import pairwise
from pathlib import Path

import numpy as np
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from fy_asr.audio import SAMPLE_RATE, write_wav
from fy_asr.corpora import Utterance
from fy_asr.train import WINDOW_SECONDS, Collator, gap, pack
from fy_asr.whisper import BASE_MODEL


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


def test_collator_scores_only_the_transcript_after_a_style_prompt(
    tmp_path: Path,
) -> None:
    write_wav(tmp_path / "a.wav", np.zeros(SAMPLE_RATE, np.float32))
    plain = Utterance(str(tmp_path / "a.wav"), "Goeie, hoe giet it?", "fris", "s", 1.0)
    styled = Utterance(str(tmp_path / "a.wav"), "goeie hoe giet it", "bhc", "s", 1.0)
    batch = Collator(BASE_MODEL)([[plain], [styled]])
    inputs, labels = batch["decoder_input_ids"], batch["labels"]

    plain_labels = labels[0][labels[0] != -100]
    shifted = shift_tokens_right(plain_labels[None], 50257, 50258)[0]
    assert inputs[0][: len(shifted)].tolist() == shifted.tolist()

    scored = labels[1] != -100
    assert inputs[1][0].item() == 50362
    first = int(scored.nonzero()[0])
    assert inputs[1][first].item() == 50258
    assert (
        labels[1][scored].tolist()[3:]
        == Collator(BASE_MODEL).tokenizer("goeie hoe giet it").input_ids[4:]
    )


def test_pack_fills_clips_without_a_recording_up_to_30s() -> None:
    windows = pack([clip(i, 9.0) for i in range(6)], random.Random(1))
    assert sorted(len(w) for w in windows) == [3, 3]
