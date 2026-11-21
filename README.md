# fy-asr

Automatic subtitles for West Frisian (Frysk) speech. Runs locally on Apple Silicon.

The current pipeline does not need any training:

1. ffmpeg decodes any audio or video file to 16 kHz mono.
2. Silero VAD finds speech segments (max 20 s).
3. [`greenw0lf/wav2vec2-large-xls-r-1b-frisian`](https://huggingface.co/greenw0lf/wav2vec2-large-xls-r-1b-frisian), a CTC model fine-tuned on Common Voice Frisian, transcribes each segment. CTC gives a timestamp for each word.
4. Words are grouped into subtitle cues (split at pauses, max 6 s, max two lines of 42 characters) and written as SRT.

Output is lowercase and has no punctuation, because the model vocabulary has neither.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg`).

```sh
uv sync
```

The first run downloads the model (about 3.6 GB) to the Hugging Face cache.

## Usage

Subtitle a file (writes `episode.srt` next to the input unless `-o` is given):

```sh
uv run fy-asr transcribe episode.mp4
```

Measure WER and CER on a parquet file with `audio` (bytes) and `text` columns, for example a shard of [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22):

```sh
uv run fy-asr eval test-00000-of-00002.parquet -n 200
```

Both commands take `--model` to use a different wav2vec2 CTC checkpoint.

## Results

| Data | WER | CER |
|---|---|---|
| Common Voice 22 test, first 200 clips (read speech) | 12.0% | 3.0% |
| `Juvoly/Frisian-dialogue`, one 58 min recording (spontaneous dialogue) | 57.0% | 27.9% |

Both after lowercasing and removing punctuation. A 58 min recording takes about 2 minutes on an M5 Pro.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright src tests
```
