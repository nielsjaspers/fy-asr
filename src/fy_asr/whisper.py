import bisect

import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from fy_asr.asr import Word
from fy_asr.audio import SAMPLE_RATE, merge_segments, speech_segments

BASE_MODEL = "openai/whisper-large-v3-turbo"
LANGUAGE = "nl"
WINDOW_SECONDS = 30.0
MAX_MERGE_GAP_SECONDS = 10.0


class WhisperTranscriber:
    def __init__(
        self, model_id: str, beams: int = 1, language: str | None = LANGUAGE
    ) -> None:
        self.beams = beams
        self.language = language
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.features = WhisperFeatureExtractor.from_pretrained(model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16
        ).eval()
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]

    @torch.inference_mode()
    def text(self, audio: np.ndarray) -> str:
        if len(audio) > WINDOW_SECONDS * SAMPLE_RATE:
            return " ".join(w.text for w in self.long_form(audio))
        inputs = self.features(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        ids = self.model.generate(
            inputs.to(self.device, torch.float16),
            language=self.language,
            task="transcribe",
            num_beams=self.beams,
        )
        assert isinstance(ids, torch.Tensor)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def words(self, audio: np.ndarray, offset: float = 0.0) -> list[Word]:
        inputs = self.features(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        out = self.model.generate(
            inputs.to(self.device, torch.float16),
            language=self.language,
            task="transcribe",
            num_beams=self.beams,
            return_token_timestamps=True,
        )
        assert isinstance(out, dict)
        ids = out["sequences"][0].tolist()
        ends = out["token_timestamps"][0].tolist()
        special = set(self.tokenizer.all_special_ids)
        tokens = [
            (token, end)
            for id_, token, end in zip(
                ids, self.tokenizer.convert_ids_to_tokens(ids), ends
            )
            if id_ not in special
        ]
        limit = len(audio) / SAMPLE_RATE
        words: list[Word] = []
        group: list[str] = []
        start = 0.0
        for i, (token, end) in enumerate(tokens):
            group.append(token)
            if i + 1 < len(tokens) and not tokens[i + 1][0].startswith("Ġ"):
                continue
            text = self.tokenizer.convert_tokens_to_string(group).strip()
            if text:
                words.append(
                    Word(text, offset + min(start, limit), offset + min(end, limit))
                )
            group, start = [], end
        return words

    def long_form(self, audio: np.ndarray) -> list[Word]:
        speech = speech_segments(audio)
        chunks = merge_segments(speech, WINDOW_SECONDS, MAX_MERGE_GAP_SECONDS)
        words = [
            word
            for start, end in chunks
            for word in self.words(audio[start:end], offset=start / SAMPLE_RATE)
        ]
        return snap_starts(
            words, [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in speech]
        )


def snap_starts(words: list[Word], speech: list[tuple[float, float]]) -> list[Word]:
    """Start each word no earlier than the last speech onset before its end.

    A word's start is the end of the token before it, so a word after a pause
    would otherwise begin during the silence.
    """
    onsets = [start for start, _ in speech]
    snapped: list[Word] = []
    for word in words:
        i = bisect.bisect_left(onsets, word.end) - 1
        if i >= 0 and onsets[i] > word.start:
            word = Word(word.text, onsets[i], word.end)
        snapped.append(word)
    return snapped
