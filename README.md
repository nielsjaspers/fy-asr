# fy-asr

Automatic subtitles for West Frisian (Frysk) speech. Runs locally on Apple Silicon.

Pipeline:

1. ffmpeg decodes any audio or video file to 16 kHz mono.
2. Silero VAD finds speech segments (max 20 s).
3. A speech model transcribes each segment and gives a timestamp for each word.
4. Words are grouped into subtitle cues (split at pauses, max 6 s, max two lines of 42 characters) and written as SRT.

Two speech models work in step 3:

- [`greenw0lf/wav2vec2-large-xls-r-1b-frisian`](https://huggingface.co/greenw0lf/wav2vec2-large-xls-r-1b-frisian) (default, no training needed). A CTC model fine-tuned on Common Voice Frisian. Output is lowercase without punctuation.
- [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) with a LoRA fine-tune from `fy-asr train` (see [Training](#training)). Output is cased and punctuated, and the error rate is much lower. Word timestamps come from Whisper's cross-attention.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg`).

```sh
uv sync
```

The first run downloads the model (about 3.6 GB for wav2vec2, 1.6 GB for Whisper) to the Hugging Face cache.

## Usage

Subtitle a file (writes `episode.srt` next to the input unless `-o` is given):

```sh
uv run fy-asr transcribe episode.mp4
uv run fy-asr transcribe episode.mp4 --model models/whisper-lora/model
```

## Data

```sh
uv run fy-asr prepare
```

This downloads two corpora at pinned revisions (about 8 GB) into `data/raw/` and writes JSONL manifests to `data/prepared/`:

- [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22): Common Voice 22 Frisian, read speech. Its train, dev and test splits share no speakers or sentences.
- [`Juvoly/Frisian-dialogue`](https://huggingface.co/datasets/Juvoly/Frisian-dialogue): spontaneous doctor-patient dialogue. The upstream train and test splits share every speaker, so `prepare` makes its own split. Recordings that share a caregiver or patient form one group. Whole groups go to test and dev until each has at least 2.5 h and 1.25 h; because groups are whole, the result is about 3.9 h and 2.0 h.

Caregivers and patients in dev and test never appear in train. Two kinds of speech can't be checked: the coordinator who announces each recording (one person, who is also a caregiver in train) and segments with an unknown speaker. Their clips are left out of the short dev and test sets. Long-form recordings still contain them, a few words per recording.

| Manifest | Contents |
|---|---|
| `train.jsonl`, `dev.jsonl`, `test.jsonl` | short utterances from both corpora (`audio`, `text`, `source`, `speaker`, `duration`) |
| `longform_dev.jsonl`, `longform_test.jsonl` | full held-out dialogue recordings, with the Frisian half of their transcript as reference |

Text is kept cased and punctuated. Evaluation lowercases it and removes punctuation.

## Evaluation

```sh
uv run fy-asr eval data/prepared/test.jsonl -n 500
uv run fy-asr eval data/prepared/longform_test.jsonl --long-form
```

`-n` limits the number of items per source. `--long-form` runs the full subtitle pipeline (VAD and segmentation) on each recording. Both commands, and `transcribe`, take `--model` for a different wav2vec2 CTC or Whisper checkpoint. The model type comes from the checkpoint config.

## Training

```sh
uv run fy-asr train --out models/whisper-lora
```

This trains LoRA adapters (rank 32 on all attention and feed-forward layers, 28M parameters) on `whisper-large-v3-turbo` with `train.jsonl`, and writes the merged model to `models/whisper-lora/model`.

- Whisper pads every input to 30 s. To use that compute, training shuffles the utterances and concatenates them into windows of up to 30 s (about 8,600 windows for one pass over 61 h).
- Whisper has no Frisian language token, so training and decoding use the Dutch prompt (`nl`).
- Batch size 1 with 8 accumulation steps, bf16 autocast, 2 epochs, learning rate 5e-4 with linear warmup and decay. The adapter with the lowest loss on 200 dev windows is kept.
- On an M5 Pro (48 GB) one run takes about 4.7 h and uses about 15 GB.

## Results

On the held-out sets from `prepare`. WER and CER after lowercasing and removing punctuation.

| Set | wav2vec2 (no training) | Whisper turbo + LoRA |
|---|---|---|
| CV22 test, 3,173 clips (read speech) | 14.7% / 4.0% | 10.2% / 3.1% |
| FRIS test, 1,770 clips (spontaneous dialogue, unseen speakers) | 59.8% / 31.0% | 21.7% / 12.7% |
| FRIS long-form test, 24 recordings, 4.6 h | 59.3% / 30.1% | 23.6% / 13.6% |

Without fine-tuning, Whisper turbo gets about 75% WER on a 100-clip sample of the test set.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright src tests
```
