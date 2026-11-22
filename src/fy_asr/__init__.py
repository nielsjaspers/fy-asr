import argparse
from itertools import islice
from pathlib import Path

import torch
from transformers import AutoConfig

from fy_asr.asr import DEFAULT_MODEL, Transcriber
from fy_asr.audio import load_audio
from fy_asr.corpora import Utterance, prepare, read_manifest
from fy_asr.evaluate import score, score_by_source
from fy_asr.evaluate import transcribe as transcribe_item
from fy_asr.subtitles import build_cues, to_srt
from fy_asr.train import TrainConfig, train
from fy_asr.whisper import WhisperTranscriber

MPS_MEMORY_FRACTION = 0.75


def _transcriber(model_id: str, beams: int) -> Transcriber | WhisperTranscriber:
    if AutoConfig.from_pretrained(model_id).model_type == "whisper":
        return WhisperTranscriber(model_id, beams)
    return Transcriber(model_id)


def _first_per_source(
    utterances: list[Utterance], limit: int | None
) -> list[Utterance]:
    if limit is None:
        return utterances
    sources = dict.fromkeys(u.source for u in utterances)
    return [
        u
        for source in sources
        for u in islice((u for u in utterances if u.source == source), limit)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fy-asr")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="write SRT subtitles for an audio or video file"
    )
    transcribe.add_argument("input", type=Path)
    transcribe.add_argument("-o", "--output", type=Path)

    prepare_cmd = commands.add_parser(
        "prepare", help="download the corpora and write train/dev/test manifests"
    )
    prepare_cmd.add_argument("--data-dir", type=Path, default=Path("data"))

    eval_cmd = commands.add_parser("eval", help="WER/CER on a JSONL manifest")
    eval_cmd.add_argument("manifest", type=Path)
    eval_cmd.add_argument("-n", "--limit", type=int, help="max items per source")
    eval_cmd.add_argument(
        "--long-form", action="store_true", help="items are full recordings"
    )

    for command in (transcribe, eval_cmd):
        command.add_argument("--model", help="wav2vec2 CTC or Whisper checkpoint")
        command.add_argument(
            "--beams", type=int, default=1, help="beam size for Whisper decoding"
        )

    train_cmd = commands.add_parser(
        "train", help="LoRA fine-tune Whisper on a train manifest"
    )
    train_cmd.add_argument(
        "--train",
        type=Path,
        action="append",
        help="train manifest, repeatable (default data/prepared/train.jsonl)",
    )
    train_cmd.add_argument("--dev", type=Path, default=Path("data/prepared/dev.jsonl"))
    train_cmd.add_argument("--out", type=Path, required=True)
    train_cmd.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    train_cmd.add_argument("--lr", type=float, default=TrainConfig.lr)
    train_cmd.add_argument(
        "--base", default=TrainConfig.base, help="Whisper checkpoint"
    )
    train_cmd.add_argument(
        "--fris-repeat",
        type=int,
        default=dict(TrainConfig.repeat)["fris"],
        help="times FRIS is packed per epoch",
    )

    args = parser.parse_args()
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(MPS_MEMORY_FRACTION)

    if args.command == "prepare":
        for name, count in prepare(args.data_dir).items():
            print(f"{name}: {count}")
        return

    if args.command == "train":
        model = train(
            TrainConfig(
                tuple(args.train or [Path("data/prepared/train.jsonl")]),
                args.dev,
                args.out,
                base=args.base,
                epochs=args.epochs,
                repeat=(("fris", args.fris_repeat),),
                lr=args.lr,
            )
        )
        print(f"wrote {model}")
        return

    transcriber = _transcriber(args.model or DEFAULT_MODEL, args.beams)

    if args.command == "transcribe":
        words = transcriber.long_form(load_audio(args.input))
        output = args.output or args.input.with_suffix(".srt")
        output.write_text(to_srt(build_cues(words)), encoding="utf-8")
        print(f"wrote {output}")
        return

    utterances = _first_per_source(read_manifest(args.manifest), args.limit)
    hypotheses: list[str] = []
    for utterance in utterances:
        hypotheses.append(transcribe_item(transcriber, utterance, args.long_form))
        if args.long_form:
            print(
                f"{Path(utterance.audio).name}: {score([utterance.text], hypotheses[-1:])}"
            )
    for source, result in score_by_source(utterances, hypotheses).items():
        print(f"{source:>5}: {result}")
