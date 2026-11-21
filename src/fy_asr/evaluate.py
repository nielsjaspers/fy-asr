import re
import unicodedata
from pathlib import Path

import jiwer
import pyarrow.parquet as pq

from fy_asr.asr import Transcriber
from fy_asr.audio import load_audio


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"['’‘`]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def evaluate(
    transcriber: Transcriber, parquet: Path, limit: int
) -> tuple[float, float, int]:
    rows = pq.read_table(parquet, columns=["audio", "text"]).slice(0, limit).to_pylist()
    references = [normalize(row["text"]) for row in rows]
    hypotheses = [
        normalize(transcriber.text(load_audio(row["audio"]["bytes"]))) for row in rows
    ]
    return (
        jiwer.wer(references, hypotheses),
        jiwer.cer(references, hypotheses),
        len(rows),
    )
