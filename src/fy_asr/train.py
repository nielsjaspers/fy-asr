import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)

from fy_asr.audio import SAMPLE_RATE, read_wav
from fy_asr.corpora import Utterance, read_manifest
from fy_asr.whisper import BASE_MODEL, LANGUAGE, WINDOW_SECONDS

GAP_SECONDS = 0.3
MAX_GAP_SECONDS = 1.0
MIN_WINDOW_SECONDS = 20.0
STYLE_PROMPTS = {"bhc": "sûnder ynterpunksje"}
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@dataclass(frozen=True)
class TrainConfig:
    train: tuple[Path, ...]
    dev: Path
    out: Path
    base: str = BASE_MODEL
    epochs: int = 2
    repeat: tuple[tuple[str, int], ...] = (("fris", 2),)
    accumulate: int = 8
    lr: float = 5e-4
    warmup_steps: int = 100
    eval_every: int = 250
    dev_windows: int = 200
    lora_rank: int = 32
    init: Path | None = None
    resume: bool = False
    seed: int = 0


def gap(a: Utterance, b: Utterance) -> float:
    if a.recording and a.recording == b.recording:
        return min(max(b.offset - a.offset - a.duration, 0.1), MAX_GAP_SECONDS)
    return GAP_SECONDS


def _fill(
    stream: list[Utterance], rng: random.Random, varied: bool
) -> list[list[Utterance]]:
    windows: list[list[Utterance]] = []
    current: list[Utterance] = []
    length = 0.0
    target = (
        rng.uniform(MIN_WINDOW_SECONDS, WINDOW_SECONDS) if varied else WINDOW_SECONDS
    )
    for u in stream:
        joined = length + gap(current[-1], u) + u.duration if current else u.duration
        if current and (joined > target or current[-1].recording != u.recording):
            windows.append(current)
            current = []
            if varied:
                target = rng.uniform(MIN_WINDOW_SECONDS, WINDOW_SECONDS)
            joined = u.duration
        length = joined
        current.append(u)
    if current:
        windows.append(current)
    return windows


def pack(
    utterances: list[Utterance],
    rng: random.Random,
    repeat: dict[str, int] | None = None,
) -> list[list[Utterance]]:
    """Pack utterances into windows of at most 30 s.

    Utterances from the same recording stay in recording order, so a window
    holds a stretch of real conversation, and end at a random length between
    20 and 30 s. Other utterances are shuffled and fill 30 s windows. `repeat`
    packs a source that many times per call.
    """
    fits = [u for u in utterances if u.duration <= WINDOW_SECONDS]
    windows: list[list[Utterance]] = []
    for source in sorted({u.source for u in fits}):
        rows = [u for u in fits if u.source == source]
        loose = [u for u in rows if not u.recording]
        recordings: dict[str, list[Utterance]] = {}
        for u in rows:
            if u.recording:
                recordings.setdefault(u.recording, []).append(u)
        for _ in range((repeat or {}).get(source, 1)):
            rng.shuffle(loose)
            windows += _fill(loose, rng, varied=False)
            names = sorted(recordings)
            rng.shuffle(names)
            stream = [
                u for n in names for u in sorted(recordings[n], key=lambda u: u.offset)
            ]
            windows += _fill(stream, rng, varied=True)
    rng.shuffle(windows)
    return windows


class Collator:
    def __init__(self, base: str) -> None:
        self.features = WhisperFeatureExtractor.from_pretrained(base)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            base, language=LANGUAGE, task="transcribe"
        )

    def __call__(self, batch: list[list[Utterance]]) -> dict[str, torch.Tensor]:
        audios = [
            np.concatenate(
                [
                    part
                    for i, u in enumerate(window)
                    for part in (
                        [np.zeros(int(gap(window[i - 1], u) * SAMPLE_RATE), np.float32)]
                        if i
                        else []
                    )
                    + [read_wav(Path(u.audio))]
                ]
            )[: int(WINDOW_SECONDS * SAMPLE_RATE)]
            for window in batch
        ]
        features = self.features(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        sequences = [self._sequence(window) for window in batch]
        longest = max(len(seq) for seq, _ in sequences) - 1
        inputs = torch.full((len(batch), longest), self.tokenizer.eos_token_id)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for row, (seq, prompt) in enumerate(sequences):
            inputs[row, : len(seq) - 1] = torch.tensor(seq[:-1])
            labels[row, prompt : len(seq) - 1] = torch.tensor(seq[prompt + 1 :])
        return {
            "input_features": features,
            "decoder_input_ids": inputs,
            "labels": labels,
        }

    def _sequence(self, window: list[Utterance]) -> tuple[list[int], int]:
        """Decoder tokens for a window and how many of them are an unscored prompt."""
        target = self.tokenizer(" ".join(u.text for u in window)).input_ids
        style = STYLE_PROMPTS.get(window[0].source)
        if style is None:
            return target, 0
        prompt = [self.tokenizer.convert_tokens_to_ids("<|startofprev|>")]
        prompt += self.tokenizer(" " + style, add_special_tokens=False).input_ids
        return prompt + target, len(prompt)


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model(base: str, rank: int, init: Path | None) -> PeftModel:
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.config.use_cache = False
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    if init:
        return PeftModel.from_pretrained(model, init, is_trainable=True)
    peft = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=LORA_TARGETS
        ),
    )
    assert isinstance(peft, PeftModel)
    return peft


@torch.no_grad()
def _dev_loss(model: PeftModel, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        with torch.autocast(device, dtype=torch.bfloat16):
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
        total += loss.item()
        count += 1
    model.train()
    return total / count


def train(config: TrainConfig) -> Path:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    device = _device()
    train_set = [u for path in config.train for u in read_manifest(path)]
    repeat = dict(config.repeat)
    dev_windows = pack(read_manifest(config.dev), random.Random(config.seed))
    dev_loader = DataLoader(
        dev_windows[: config.dev_windows],  # pyright: ignore[reportArgumentType]
        collate_fn=Collator(config.base),
        num_workers=2,
    )
    epochs = [pack(train_set, rng, repeat) for _ in range(config.epochs)]
    epochs = [w[: len(w) - len(w) % config.accumulate] for w in epochs]
    total_steps = sum(len(w) for w in epochs) // config.accumulate
    warmup = max(1, min(config.warmup_steps, total_steps // 10))

    checkpoint = latest_checkpoint(config.out) if config.resume else None
    init = checkpoint / "adapter" if checkpoint else config.init
    model = _model(config.base, config.lora_rank, init).to(device)  # pyright: ignore[reportArgumentType]
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(
            (s + 1) / warmup,
            max(0.0, (total_steps - s) / (total_steps - warmup)),
        ),
    )

    adapter = config.out / "adapter"
    best = math.inf
    step = start_epoch = start_window = 0
    if checkpoint:
        state = torch.load(checkpoint / "state.pt", weights_only=False)
        if state["total_steps"] != total_steps:
            raise ValueError(
                f"{checkpoint} belongs to a run with other data or settings"
            )
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step, best = state["step"], state["best"]
        start_epoch, start_window = state["epoch"], state["window"]
        print(f"resuming from {checkpoint} at step {step}", flush=True)
    resumed_at = step
    started = time.time()
    losses: list[float] = []
    model.train()
    for epoch in range(start_epoch, len(epochs)):
        skip = start_window if epoch == start_epoch else 0
        loader = DataLoader(
            epochs[epoch][skip:],  # pyright: ignore[reportArgumentType]
            collate_fn=Collator(config.base),
            num_workers=3,
        )
        for i, batch in enumerate(loader, start=skip + 1):
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            (loss / config.accumulate).backward()
            losses.append(loss.item())
            if i % config.accumulate:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            step += 1
            if step % 25 == 0:
                done = step - resumed_at
                hours = (time.time() - started) / done * (total_steps - step) / 3600
                print(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {sum(losses) / len(losses):.3f} "
                    f"lr {schedule.get_last_lr()[0]:.2e} eta {hours:.1f} h",
                    flush=True,
                )
                losses.clear()
            if step % config.eval_every == 0 or step == total_steps:
                dev = _dev_loss(model, dev_loader, device)
                print(f"step {step} dev loss {dev:.3f}", flush=True)
                if dev < best:
                    best = dev
                    model.save_pretrained(str(adapter))
                save_checkpoint(
                    config.out,
                    model,
                    {
                        "total_steps": total_steps,
                        "step": step,
                        "epoch": epoch,
                        "window": i,
                        "best": best,
                        "optimizer": optimizer.state_dict(),
                        "schedule": schedule.state_dict(),
                    },
                )
    return merge(config.base, adapter, config.out / "model")


def latest_checkpoint(out: Path) -> Path | None:
    """The newest checkpoint whose state file was fully written."""
    done = sorted(
        d for d in (out / "checkpoints").glob("*") if (d / "state.pt").exists()
    )
    return done[-1] if done else None


def save_checkpoint(out: Path, model: PeftModel, state: dict) -> None:
    """Write a checkpoint for --resume, then delete older ones.

    The state file is written last, so a checkpoint without it is incomplete.
    """
    target = out / "checkpoints" / f"{state['step']:07d}"
    model.save_pretrained(str(target / "adapter"))
    torch.save(state, target / "state.pt")
    for old in (out / "checkpoints").glob("*"):
        if old != target:
            shutil.rmtree(old)


def merge(base: str, adapter: Path, out: Path) -> Path:
    model = WhisperForConditionalGeneration.from_pretrained(base)
    merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    merged.generation_config.forced_decoder_ids = None
    merged.save_pretrained(out)
    WhisperFeatureExtractor.from_pretrained(base).save_pretrained(out)
    WhisperTokenizer.from_pretrained(base).save_pretrained(out)
    return out
