import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jiwer

from fy_asr.asr import Transcriber
from fy_asr.audio import load_audio
from fy_asr.corpora import Utterance


@dataclass(frozen=True)
class Score:
    wer: float
    cer: float
    items: int

    def __str__(self) -> str:
        return f"WER {self.wer:6.2%}  CER {self.cer:6.2%}  ({self.items} items)"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"['’‘`]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def score(references: Sequence[str], hypotheses: Sequence[str]) -> Score:
    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]
    return Score(jiwer.wer(refs, hyps), jiwer.cer(refs, hyps), len(refs))


def transcribe(transcriber: Transcriber, utterance: Utterance, long_form: bool) -> str:
    audio = load_audio(Path(utterance.audio))
    if long_form:
        return " ".join(w.text for w in transcriber.long_form(audio))
    return transcriber.text(audio)


def score_by_source(
    utterances: Sequence[Utterance], hypotheses: Sequence[str]
) -> dict[str, Score]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, utterance in enumerate(utterances):
        groups[utterance.source].append(i)
    scores = {
        source: score([utterances[i].text for i in idx], [hypotheses[i] for i in idx])
        for source, idx in sorted(groups.items())
    }
    return scores | {"all": score([u.text for u in utterances], hypotheses)}
