import math
import random
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
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


@dataclass(frozen=True)
class TrainConfig:
    train: Path
    dev: Path
    out: Path
    epochs: int = 2
    accumulate: int = 8
    lr: float = 5e-4
    warmup_steps: int = 100
    eval_every: int = 250
    dev_windows: int = 200
    lora_rank: int = 32
    seed: int = 0


def pack(utterances: list[Utterance], rng: random.Random) -> list[list[Utterance]]:
    """Shuffle utterances and concatenate them into windows of at most 30 s."""
    order = [u for u in utterances if u.duration <= WINDOW_SECONDS]
    rng.shuffle(order)
    windows: list[list[Utterance]] = []
    current: list[Utterance] = []
    length = 0.0
    for u in order:
        if current and length + GAP_SECONDS + u.duration > WINDOW_SECONDS:
            windows.append(current)
            current, length = [], 0.0
        length += (GAP_SECONDS if current else 0.0) + u.duration
        current.append(u)
    if current:
        windows.append(current)
    return windows


class Collator:
    def __init__(self) -> None:
        self.features = WhisperFeatureExtractor.from_pretrained(BASE_MODEL)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            BASE_MODEL, language=LANGUAGE, task="transcribe"
        )
        self.gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype=np.float32)

    def __call__(self, batch: list[list[Utterance]]) -> dict[str, torch.Tensor]:
        audios = [
            np.concatenate(
                [
                    part
                    for i, u in enumerate(window)
                    for part in ([self.gap] if i else []) + [read_wav(Path(u.audio))]
                ]
            )[: int(WINDOW_SECONDS * SAMPLE_RATE)]
            for window in batch
        ]
        features = self.features(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features
        ids = [
            self.tokenizer(" ".join(u.text for u in window)).input_ids[1:]
            for window in batch
        ]
        longest = max(len(x) for x in ids)
        labels = torch.full((len(ids), longest), -100, dtype=torch.long)
        for row, x in zip(labels, ids):
            row[: len(x)] = torch.tensor(x)
        return {"input_features": features, "labels": labels}


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model(rank: int) -> PeftModel:
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.use_cache = False
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
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
    train_set = read_manifest(config.train)
    dev_windows = pack(read_manifest(config.dev), random.Random(config.seed))
    dev_loader = DataLoader(
        dev_windows[: config.dev_windows],  # pyright: ignore[reportArgumentType]
        collate_fn=Collator(),
        num_workers=2,
    )
    epochs = [pack(train_set, rng) for _ in range(config.epochs)]
    epochs = [w[: len(w) - len(w) % config.accumulate] for w in epochs]
    total_steps = sum(len(w) for w in epochs) // config.accumulate
    warmup = min(config.warmup_steps, max(1, total_steps // 10))

    model = _model(config.lora_rank).to(device)  # pyright: ignore[reportArgumentType]
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
    step = 0
    started = time.time()
    losses: list[float] = []
    model.train()
    for epoch, windows in enumerate(epochs):
        loader = DataLoader(
            windows,  # pyright: ignore[reportArgumentType]
            collate_fn=Collator(),
            num_workers=3,
        )
        for i, batch in enumerate(loader, start=1):
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
                hours = (time.time() - started) / step * (total_steps - step) / 3600
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
    return merge(adapter, config.out / "model")


def merge(adapter: Path, out: Path) -> Path:
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.generation_config.forced_decoder_ids = None
    merged.save_pretrained(out)
    WhisperFeatureExtractor.from_pretrained(BASE_MODEL).save_pretrained(out)
    WhisperTokenizer.from_pretrained(BASE_MODEL).save_pretrained(out)
    return out
