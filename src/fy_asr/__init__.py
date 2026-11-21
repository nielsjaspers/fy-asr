import argparse
from pathlib import Path

from fy_asr.asr import DEFAULT_MODEL, Transcriber
from fy_asr.audio import load_audio
from fy_asr.evaluate import evaluate
from fy_asr.subtitles import build_cues, to_srt


def main() -> None:
    parser = argparse.ArgumentParser(prog="fy-asr")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="write SRT subtitles for an audio or video file"
    )
    transcribe.add_argument("input", type=Path)
    transcribe.add_argument("-o", "--output", type=Path)

    eval_cmd = commands.add_parser(
        "eval", help="WER/CER on a parquet file with audio and text columns"
    )
    eval_cmd.add_argument("parquet", type=Path)
    eval_cmd.add_argument("-n", "--limit", type=int, default=200)

    args = parser.parse_args()
    transcriber = Transcriber(args.model)

    match args.command:
        case "transcribe":
            words = transcriber.long_form(load_audio(args.input))
            output = args.output or args.input.with_suffix(".srt")
            output.write_text(to_srt(build_cues(words)), encoding="utf-8")
            print(f"wrote {output}")
        case "eval":
            wer, cer, clips = evaluate(transcriber, args.parquet, args.limit)
            print(f"WER {wer:.2%}  CER {cer:.2%}  on {clips} clips")
