from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from fy_asr.audio import SAMPLE_RATE, speech_segments

DEFAULT_MODEL = "greenw0lf/wav2vec2-large-xls-r-1b-frisian"


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float


class Transcriber:
    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.features = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.model = model.to(self.device)  # pyright: ignore[reportArgumentType]
        self.seconds_per_frame = self.model.config.inputs_to_logits_ratio / SAMPLE_RATE

    @torch.inference_mode()
    def words(self, audio: np.ndarray, offset: float = 0.0) -> list[Word]:
        inputs = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.device)).logits
        ids = logits.argmax(dim=-1)[0].cpu()
        decoded = self.tokenizer.decode(ids, output_word_offsets=True)
        return [
            Word(
                text=w["word"],
                start=offset + w["start_offset"] * self.seconds_per_frame,
                end=offset + w["end_offset"] * self.seconds_per_frame,
            )
            for w in decoded.word_offsets
        ]

    def text(self, audio: np.ndarray) -> str:
        return " ".join(w.text for w in self.words(audio))

    def long_form(self, audio: np.ndarray) -> list[Word]:
        return [
            word
            for start, end in speech_segments(audio)
            for word in self.words(audio[start:end], offset=start / SAMPLE_RATE)
        ]
