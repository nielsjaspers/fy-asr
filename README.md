# fy-asr

Speech recognition for West Frisian (Frysk), tuned for long recordings of spontaneous speech. The main result is [`whisper-large-v3-turbo-fy`](https://huggingface.co/nielsjaspers/whisper-large-v3-turbo-fy), a fine-tune of [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) on read and conversational Frisian. The repo contains the code to prepare the data, train and evaluate the model, and transcribe audio or video files to text or SRT subtitles. Everything runs locally on Apple Silicon.

| Model | Training data | Method | Can be published |
|---|---|---|---|
| `open-full` ([`whisper-large-v3-turbo-fy`](https://huggingface.co/nielsjaspers/whisper-large-v3-turbo-fy)) | Common Voice 22 + FRIS | all weights | yes; the recommended model |
| `open-lora` | Common Voice 22 + FRIS | LoRA adapters | yes |
| `private-lora` | `open-lora`, continued with Boarnsterhim | LoRA adapters | no (see [Boarnsterhim](#boarnsterhim-optional-restricted)) |

On 4.6 hours of held-out doctor-patient conversations, `open-full` reaches 19.3% WER (18.9% with beam search), against 58 to 62% for three public Frisian models ([results](#results)).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg`).

```sh
uv sync
uv run fy-asr transcribe recording.mp4 --beams 5
```

This writes `recording.srt` next to the input (`-o` sets another path). The default model is [`nielsjaspers/whisper-large-v3-turbo-fy`](https://huggingface.co/nielsjaspers/whisper-large-v3-turbo-fy), which is downloaded from Hugging Face on first use (1.6 GB). `--model` takes a model you trained yourself (such as `models/whisper-large-v3-turbo-fy/model`, see [Training](#training)) or any other Whisper or wav2vec2 CTC checkpoint. `--beams 5` lowers WER from 19.3% to 18.9% on the FRIS long-form test and is about 3 times slower.

## How it works

1. ffmpeg decodes the input to 16 kHz mono.
2. Silero VAD finds speech. Neighboring segments are merged into chunks of up to 30 s when the pause between them is under 10 s, which matches the 30 s windows the model is trained on.
3. Whisper transcribes each chunk. Output is cased and punctuated. Word times come from Whisper's cross-attention, and a word after a pause starts at the next speech onset the VAD found.
4. Words become subtitle cues of at most two lines of 42 characters and 6 s. Cues end at a sentence end where possible, then at a pause or comma. Fillers such as "uh" are left out of the subtitles.

## Data

```sh
uv run fy-asr prepare
```

This downloads two corpora at pinned revisions (about 8 GB) into `data/raw/` and writes JSONL manifests to `data/prepared/`. Text keeps its casing and punctuation; evaluation lowercases it and removes punctuation.

- [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22): Common Voice 22 Frisian, read sentences. Train, dev and test share no speakers or sentences.
- [`Juvoly/Frisian-dialogue`](https://huggingface.co/datasets/Juvoly/Frisian-dialogue) (FRIS): spontaneous doctor-patient conversations. The upstream splits share every speaker, so `prepare` regroups recordings by shared caregiver or patient and assigns whole groups to test (3.9 h) and dev (2.0 h). Their speakers never appear in train. The coordinator who announces each recording and unknown speakers can't be checked, so their clips are left out of the short dev and test sets.

| Manifest | Contents |
|---|---|
| `train.jsonl`, `dev.jsonl`, `test.jsonl` | short utterances from both corpora |
| `longform_dev.jsonl`, `longform_test.jsonl` | complete held-out FRIS recordings with their Frisian transcripts |

### Boarnsterhim (optional, restricted)

> **This corpus is optional and is not needed for the published model.** Its license allows personal, non-commercial use only. A model trained on it must not be published or shared without written permission from the Instituut voor de Nederlandse Taal (INT), license articles 4.2 and 8. `open-lora` and `open-full` contain no Boarnsterhim data.

The [Boarnsterhim Corpus 2.0](https://taalmaterialen.ivdnt.org/download/boarnsterhim-corpus-2_0/) (Fryske Akademy, distributed by the INT) holds Frisian and Dutch speech of bilingual speakers at home, recorded in 1982-1984 and again 35 years later. After accepting its license, put `BoarnsterhimCorpus_2.0.zip` in `data/raw/boarnsterhim/` and run `prepare` again. It adds `train_bhc.jsonl` (49.9 h of Frisian, 103 speakers) and `test_bhc.jsonl` (3.1 h of spontaneous speech from 9 other speakers). `train.jsonl` never contains Boarnsterhim data.

Results on the Boarnsterhim test set are reported with the full name of the corpus (see [Credits and license](#credits-and-license)), as article 4.3 of its license requires. The test set is never used to choose or tune an open model.

## Training

```sh
uv run fy-asr train --full --lr 1e-5 --out models/whisper-large-v3-turbo-fy   # open-full
uv run fy-asr train --out models/open-lora                                     # open-lora
uv run fy-asr train --train data/prepared/train.jsonl --train data/prepared/train_bhc.jsonl \
  --init models/open-lora/adapter --epochs 1 --lr 2e-4 --fris-repeat 1 --out models/private-lora-1
uv run fy-asr train --train data/prepared/train.jsonl --train data/prepared/train_bhc.jsonl \
  --init models/private-lora-1/adapter --epochs 1 --lr 1e-4 --warmup 50 --out models/private-lora
```

The trained model is written to `<out>/model`. A checkpoint is written every 250 steps, and a stopped run continues with the same command plus `--resume`.

- Utterances are packed into windows of up to 30 s, because Whisper pads every input to that length. FRIS and Boarnsterhim utterances stay in recording order with their real pauses, so a window holds a stretch of real conversation. FRIS is packed twice per epoch by default (`--fris-repeat`).
- Whisper has no Frisian language token, so training and decoding use the Dutch one (`nl`).
- Boarnsterhim transcripts have no punctuation. Their windows get an unscored style prompt ("sûnder ynterpunksje") in Whisper's previous-text slot, so normal decoding stays punctuated.
- Batch size 1 with 8 accumulation steps, bf16 autocast, linear warmup and decay, 2 epochs by default. The checkpoint with the lowest dev loss is kept. LoRA uses rank 32 on all attention and feed-forward layers; `--full` trains all 809M weights in fp32 with gradient checkpointing.
- Measured on an M5 Pro with 48 GB: `open-lora` uses about 16 GB and takes about 7.5 h, `open-full` about 19 GB and 9 h. Each process is capped at 75% of the memory Metal recommends, so a run that does not fit stops with an out-of-memory error instead of swapping. Run one training job at a time.

## Evaluation

```sh
uv run fy-asr eval data/prepared/test.jsonl
uv run fy-asr eval data/prepared/longform_test.jsonl --long-form --beams 5
uv run fy-asr eval data/prepared/test.jsonl --model bartelds/whisper-large-v3-turbo-fy-cv22 --language auto
```

`--long-form` runs the full transcription pipeline on each recording. `-n` limits the number of items per source. `--language auto` lets Whisper detect the language instead of forcing `nl`.

## Results

Word error rate on held-out test sets with unseen speakers, after lowercasing and removing punctuation. Greedy decoding unless stated. Bold marks the lowest WER on each test set. All choices for the open models were made on the dev sets.

| Test set | wav2vec2 baseline | Amooie et al. | Bartelds | `open-lora` | [`whisper-large-v3-turbo-fy`](https://huggingface.co/nielsjaspers/whisper-large-v3-turbo-fy) (`open-full`) | `private-lora` |
|---|---|---|---|---|---|---|
| Common Voice 17, 3,171 clips (read) | 14.8% | 13.4% | n/a | 10.3% | 10.0% | **9.4%** |
| Common Voice 22, 3,173 clips (read) | 14.7% | 13.3% | n/a | 10.2% | 9.9% | **9.3%** |
| FRIS, 1,770 clips (conversation) | 59.8% | 57.8% | 66.1% | 20.8% | **20.5%** | 20.6% |
| FRIS long-form, 24 recordings, 4.6 h | 59.3% | 62.4% | 58.2% | 20.2% | **19.3%** | 19.7% |
| FRIS long-form, beam 5 |  |  |  | 19.7% | 18.9% | **18.8%** |
| Boarnsterhim, 9 speakers, 3.1 h (home interviews) | 58.7% | 64.9% | 74.2% | 34.6% | 34.5% | **23.1%** |

The best publishable model is `whisper-large-v3-turbo-fy` (`open-full`). `private-lora` scores best on Common Voice and Boarnsterhim, but it is trained on Boarnsterhim and cannot be published (see [Boarnsterhim](#boarnsterhim-optional-restricted)).

- **wav2vec2 baseline:** [`greenw0lf/wav2vec2-large-xls-r-1b-frisian`](https://huggingface.co/greenw0lf/wav2vec2-large-xls-r-1b-frisian), XLS-R 1B fine-tuned on Common Voice Frisian.
- **Amooie et al.:** [`Reihaneh/wav2vec2_fy_nl_best_frisian_1`](https://huggingface.co/Reihaneh/wav2vec2_fy_nl_best_frisian_1), XLS-R 1B fine-tuned on Frisian and Dutch ([Amooie et al., 2025](https://arxiv.org/abs/2502.04883), which reports 13.1% on Common Voice 17). The model starts every transcript with a `[FY-NL]` language-ID tag, a label from its training setup rather than spoken words; the transcriber removes it. With the tag removed, its Common Voice 17 result matches the 13.1% the paper reports.
- **Bartelds:** [`bartelds/whisper-large-v3-turbo-fy-cv22`](https://huggingface.co/bartelds/whisper-large-v3-turbo-fy-cv22), the same Whisper base fine-tuned on Common Voice 22, decoded with its own language detection (`--language auto`). Its training data includes every sentence of the Common Voice test sets, so it is not scored there.
- The Common Voice 17 test set comes from the [`fsicoli/common_voice_17_0`](https://huggingface.co/datasets/fsicoli/common_voice_17_0) mirror and is not part of `prepare`. It shares 98% of its clips with the Common Voice 22 test set. None of its clips or speakers are in the training data of the models in this repo or of Amooie et al.; apart from 8 everyday phrases such as "dat wit ik net", neither are its sentences.
- Models trained only on read speech stay near 60% on conversation. The models in this repo, trained with FRIS conversations, reach about 20%. They also do better on Boarnsterhim, a corpus of spontaneous speech they never saw (34.5% against 58.7 to 74.2%). Training all weights instead of LoRA adapters helps most on long-form conversation (0.9 points); Boarnsterhim data helps most on speech like its own.

## Development

```sh
uv run pytest
uv run ruff check && uv run ruff format --check
uv run pyright src tests
```

## Credits and license

The code is released under the [MIT license](LICENSE).

- Base model: [Whisper](https://github.com/openai/whisper) (`whisper-large-v3-turbo`), OpenAI, MIT license.
- [Common Voice](https://commonvoice.mozilla.org/) Frisian, Mozilla Foundation, CC0, via [`LokaalHub/frisian-asr-cv22`](https://huggingface.co/datasets/LokaalHub/frisian-asr-cv22). The Common Voice terms forbid re-hosting the dataset and trying to identify speakers.
- [Project FRIS Frisian dialogue corpus](https://huggingface.co/datasets/Juvoly/Frisian-dialogue), Juvoly, CC BY-SA 4.0. Because of this license, the published models (`open-full`, `open-lora`) are released under CC BY-SA 4.0.
- Boarnsterhim Corpus (Version 2.0) (2025) [Data set]. Available at the Dutch Language Institute: http://hdl.handle.net/10032/tm-a3-c6. Fryske Akademy. Used only for `private-lora` and the Boarnsterhim test set, under its non-commercial license.
- [Silero VAD](https://github.com/snakers4/silero-vad), MIT license.
- Comparison models: [`greenw0lf/wav2vec2-large-xls-r-1b-frisian`](https://huggingface.co/greenw0lf/wav2vec2-large-xls-r-1b-frisian), [`Reihaneh/wav2vec2_fy_nl_best_frisian_1`](https://huggingface.co/Reihaneh/wav2vec2_fy_nl_best_frisian_1), [`bartelds/whisper-large-v3-turbo-fy-cv22`](https://huggingface.co/bartelds/whisper-large-v3-turbo-fy-cv22).

**AI assistance:** this project was developed with support from AI coding assistants: Claude Code (Anthropic) and Codex (OpenAI). They were used for writing code and documentation, literature and data research, and code review. All experiments were run, and all results measured, locally on the hardware described above. All code, documentation and results went through human review before publication.
