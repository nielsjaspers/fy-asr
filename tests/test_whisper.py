from fy_asr.asr import Word
from fy_asr.whisper import snap_starts


def test_word_starts_in_silence_move_to_the_next_speech_onset() -> None:
    words = [Word("Oh,", 21.5, 26.2), Word("ik", 26.2, 26.4), Word("gap", 30.0, 30.5)]
    speech = [(20.0, 21.6), (25.9, 27.0), (29.0, 31.0)]
    assert snap_starts(words, speech) == [
        Word("Oh,", 25.9, 26.2),
        Word("ik", 26.2, 26.4),
        Word("gap", 30.0, 30.5),
    ]
