import subprocess
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

SAMPLE_RATE = 16_000
MAX_SEGMENT_SECONDS = 20.0


def load_audio(source: Path | bytes) -> np.ndarray:
    is_path = isinstance(source, Path)
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(source) if is_path else "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=None if is_path else source,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def speech_segments(audio: np.ndarray) -> list[tuple[int, int]]:
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        load_silero_vad(),
        sampling_rate=SAMPLE_RATE,
        max_speech_duration_s=MAX_SEGMENT_SECONDS,
        speech_pad_ms=200,
    )
    return [(ts["start"], ts["end"]) for ts in timestamps]
