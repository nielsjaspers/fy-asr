# fy-asr

Automatic subtitles for West Frisian (Frysk) speech. Runs locally on Apple Silicon.

Pipeline:

1. ffmpeg decodes any audio or video file to 16 kHz mono.
2. Silero VAD finds speech segments (max 20 s). For Whisper, adjacent segments are merged into chunks of up to 30 s when the pause between them is under 10 s, which matches how the model is trained.
3. A speech model transcribes each chunk and gives a timestamp for each word.
4. Words are cut into subtitle cues and written as SRT. A cue has at most two lines of 42 characters and lasts at most 6 s. Within those limits, cues end at a sentence end where possible, then at a pause or a comma, and only as a last resort in the middle of a phrase. Line breaks also prefer punctuation. Fillers such as "uh" are left out.

Two speech models work in step 3:

- [`greenw0lf/wav2vec2-large-xls-r-1b-frisian`](https://huggingface.co/greenw0lf/wav2vec2-large-xls-r-1b-frisian) (default, no training needed). A CTC model fine-tuned on Common Voice Frisian. Output is lowercase without punctuation.
- [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) with a LoRA fine-tune from `fy-asr train` (see [Training](#training)). Output is cased and punctuated, and the error rate is much lower. Word timestamps come from Whisper's cross-attention. Whisper gives the time a word ends; a word that follows a pause starts at the next speech onset that the VAD found.

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
uv run fy-asr transcribe episode.mp4 --model models/whisper-lora-v2/model --beams 5
```

`--beams 5` uses beam search: about 1 WER point better and about 3 times slower than the default greedy decoding.

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

### Optional: Boarnsterhim corpus

The [Boarnsterhim Corpus 2.0](https://taalmaterialen.ivdnt.org/download/boarnsterhim-corpus-2_0/) (Fryske Akademy, distributed by the Instituut voor de Nederlandse Taal) adds about 50 h of mostly spontaneous Frisian from bilingual speakers in their homes, recorded in 1982-1984 and again 35 years later. It requires an account and accepting a non-commercial licence. A model trained on it may only be used personally; publishing such a model requires written permission from the INT (licence articles 4.2 and 8).

Put `BoarnsterhimCorpus_2.0.zip` in `data/raw/boarnsterhim/` and run `prepare` again. It reads the zip directly and adds:

| Manifest | Contents |
|---|---|
| `train_bhc.jsonl` | Frisian read and spontaneous phrases, 49.9 h, 103 speakers |
| `test_bhc.jsonl` | 3.1 h of spontaneous speech (BHC2) from 9 speakers who are not in `train_bhc.jsonl` in either recording period |

`train.jsonl` never contains Boarnsterhim data. Transcripts are brought to the FRIS conventions: code-switch brackets and self-correction marks are removed, `eh`/`ehm` become `uh`, and phrases with unintelligible speech are skipped. Boarnsterhim text has no punctuation, so training puts a fixed style prompt before it (see [Training](#training)).

## Evaluation

```sh
uv run fy-asr eval data/prepared/test.jsonl -n 500
uv run fy-asr eval data/prepared/longform_test.jsonl --long-form
```

`-n` limits the number of items per source. `--long-form` runs the full subtitle pipeline (VAD and segmentation) on each recording. Both commands, and `transcribe`, take `--model` for a different wav2vec2 CTC or Whisper checkpoint. The model type comes from the checkpoint config.

## Training

Open data only (the model can be published):

```sh
uv run fy-asr train --out models/whisper-lora-v2
```

Continue that model with Boarnsterhim added (personal use only, see above):

```sh
uv run fy-asr train --train data/prepared/train.jsonl --train data/prepared/train_bhc.jsonl \
  --init models/whisper-lora-v2/adapter --epochs 1 --lr 2e-4 --fris-repeat 1 --out models/whisper-lora-v3
```

Training puts LoRA adapters (rank 32 on all attention and feed-forward layers, 28M parameters) on `whisper-large-v3-turbo` and writes the merged model to `<out>/model`.

- Whisper pads every input to 30 s, so training packs utterances into windows of up to 30 s. Utterances from the same recording (FRIS, Boarnsterhim) stay in recording order with their real pauses (0.1 to 1 s), so a window holds a stretch of real conversation; these windows end at a random length between 20 and 30 s. Common Voice clips are shuffled and fill 30 s windows. `--fris-repeat` packs FRIS that many times per epoch (default 2).
- Whisper has no Frisian language token, so training and decoding use the Dutch prompt (`nl`).
- Boarnsterhim windows get the text "sûnder ynterpunksje" in Whisper's previous-text slot. It is not scored, so the model learns that this prompt means unpunctuated text, and decoding without a prompt stays cased and punctuated.
- Batch size 1 with 8 accumulation steps, bf16 autocast, learning rate 5e-4 with linear warmup and decay, 2 epochs by default. `--init` continues an existing adapter, `--base` picks another Whisper checkpoint. The adapter with the lowest loss on 200 dev windows is kept.
- Each process may use at most 75% of the memory Metal recommends (about 28 GB on a 48 GB Mac). A larger batch fails with an out-of-memory error instead of filling swap. Batch 1 uses about 15 to 19 GB. Do not run two training jobs at the same time.
- On an M5 Pro, one 30 s window takes about 0.95 s. The open-data run (3,583 steps) takes about 7.5 h.

## Results

On the held-out test sets from `prepare`, with unseen speakers. WER and CER after lowercasing and removing punctuation. Greedy decoding unless stated.

| Test set | wav2vec2 (no training) | Whisper v2 (open data) | Whisper v3 (+ Boarnsterhim) |
|---|---|---|---|
| CV22, 3,173 clips (read speech) | 14.7% / 4.0% | 10.2% / 3.1% | 9.7% / 2.9% |
| FRIS, 1,770 clips (doctor-patient dialogue) | 59.8% / 31.0% | 20.8% / 12.8% | 21.2% / 13.0% |
| FRIS long-form, 24 recordings, 4.6 h | 59.3% / 30.1% | 20.2% / 11.6% | 20.0% / 11.5% |
| FRIS long-form, beam 5 | | 19.7% / 11.4% | 19.5% / 11.3% |
| Boarnsterhim, 9 speakers, 3.1 h (home interviews) | | 34.6% / 19.2% | 25.0% / 13.3% |

- v2 is trained on `train.jsonl` only and can be published. v3 continues v2 with Boarnsterhim and is for personal use only (see [Data](#optional-boarnsterhim-corpus)).
- On FRIS, v2 and v3 are within noise of each other (on the dev set v3 is 0.9 points better, on the test clips 0.4 points worse). Boarnsterhim mainly helps on speech like its own: home interviews with older speakers.
- Long-form gains came in steps. The first fine-tune with plain VAD segments got 23.6%. Merging segments into 30 s chunks brought that to 21.8% without retraining, the v2 training recipe to 20.2%, and beam search to 19.7%.
- Without fine-tuning, Whisper turbo gets about 75% WER on a 100-clip sample of the test set.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright src tests
```
