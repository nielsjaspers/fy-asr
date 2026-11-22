import hashlib
import json
import re
import subprocess
import tarfile
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

Source = Literal["cv22", "fris"]
Split = Literal["train", "dev", "test"]

CV22 = ("LokaalHub/frisian-asr-cv22", "54a731d22d602f17499abc6455a9cf387cebb49a")
FRIS = ("Juvoly/Frisian-dialogue", "35769611c51bddbfc877e549a971666a9ac085cd")
FRIS_HELD_OUT_HOURS: dict[Split, float] = {"test": 2.5, "dev": 1.25}
FRIS_SPEAKER_ROLES = {"Caregiver", "Patient"}


@dataclass(frozen=True)
class Utterance:
    audio: str
    text: str
    source: Source
    speaker: str
    duration: float
    recording: str = ""
    offset: float = 0.0


def read_manifest(path: Path) -> list[Utterance]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        Utterance(**{**row, "audio": str(path.parent / row["audio"])}) for row in rows
    ]


def _write_manifest(path: Path, rows: Iterable[Utterance]) -> int:
    lines = [json.dumps(asdict(row), ensure_ascii=False) for row in rows]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(lines)


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean(text: str) -> str:
    return " ".join(text.split())


def _cv22(raw: Path, out: Path) -> dict[Split, list[Utterance]]:
    splits: dict[Split, list[Utterance]] = {"train": [], "dev": [], "test": []}
    for split, rows in splits.items():
        audio_dir = out / "audio" / "cv22" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        for shard in sorted((raw / "data").glob(f"{split}-*.parquet")):
            table = pq.read_table(
                shard, columns=["audio", "text", "client_id", "duration"]
            )
            for row in table.to_pylist():
                target = audio_dir / Path(row["audio"]["path"]).name
                if not target.exists():
                    target.write_bytes(row["audio"]["bytes"])
                rows.append(
                    Utterance(
                        audio=str(target.relative_to(out)),
                        text=_clean(row["text"]),
                        source="cv22",
                        speaker=row["client_id"],
                        duration=row["duration"],
                    )
                )
    return splits


def _known_speaker(segment: dict) -> bool:
    return (
        segment["speaker_role"] in FRIS_SPEAKER_ROLES
        and str(segment["speaker_id"]) != "unk"
    )


def _speaker_groups(segments: list[dict]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def root(node: str) -> str:
        while parent.setdefault(node, node) != node:
            node = parent[node]
        return node

    for seg in segments:
        if _known_speaker(seg):
            parent[root(f"rec:{seg['src_audio_filepath']}")] = root(
                f"spk:{seg['speaker_id']}"
            )
    recordings = {seg["src_audio_filepath"] for seg in segments}
    return {rec: root(f"rec:{rec}") for rec in recordings}


def _assign_held_out(
    groups: dict[str, str], hours: dict[str, float]
) -> dict[str, Split]:
    by_hash = sorted(
        set(groups.values()), key=lambda g: hashlib.sha1(g.encode()).hexdigest()
    )
    assignment: dict[str, Split] = {}
    for split, target in FRIS_HELD_OUT_HOURS.items():
        taken = 0.0
        for group in by_hash:
            if group in assignment or taken >= target:
                continue
            assignment[group] = split
            taken += hours[group]
    return {rec: assignment.get(group, "train") for rec, group in groups.items()}


_TRANSLATION_HEADING = re.compile(r"^\s*oersetting\s*/\s*vertaling\s*$", re.IGNORECASE)
_HEADER = re.compile(r"^\s*(transkripsje\b|[SPC]\s*=)", re.IGNORECASE)
_TURN_PREFIX = re.compile(r"^\s*\d{1,2}[:.]\d\d\s*[A-Z]?\s*:?\s*")


def frisian_transcript(docx: Path) -> str:
    """The Frisian half of a FRIS `T&O` transcript; the Dutch translation follows a heading."""
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode()
    lines = [
        line
        for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.DOTALL)
        for line in "".join(
            "\n" if m[1] is None else m[1]
            for m in re.finditer(r"<w:t[^>]*>([^<]*)</w:t>|<w:(?:br|cr)\b[^>]*/>", p)
        ).split("\n")
    ]
    end = next(i for i, line in enumerate(lines) if _TRANSLATION_HEADING.match(line))
    turns = [
        _TURN_PREFIX.sub("", line)
        for line in lines[:end]
        if line.strip() and not _HEADER.match(line)
    ]
    return _clean(" ".join(turns))


def _audio_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return float(result.stdout)


def _fris(
    raw: Path, out: Path
) -> tuple[dict[Split, list[Utterance]], dict[Split, list[Utterance]]]:
    segments = [
        json.loads(line)
        for name in ("manifest_train.json", "manifest_test.json")
        for line in (raw / name).read_text(encoding="utf-8").splitlines()
    ]
    for seg in segments:
        seg["src_audio_filepath"] = _nfc(seg["src_audio_filepath"])
    groups = _speaker_groups(segments)
    hours: dict[str, float] = defaultdict(float)
    for seg in segments:
        hours[groups[seg["src_audio_filepath"]]] += seg["duration"] / 3600
    split_of = _assign_held_out(groups, hours)

    audio_dir = out / "audio" / "fris"
    if not audio_dir.exists():
        with tarfile.open(raw / "audio_aligned.tar.gz") as tar:
            tar.extractall(audio_dir.parent / "fris_tmp", filter="data")
        (audio_dir.parent / "fris_tmp" / "audio_aligned").rename(audio_dir)
        (audio_dir.parent / "fris_tmp").rmdir()
    wavs = {_nfc(p.name): p for p in audio_dir.glob("*.wav")}

    segment_rows: dict[Split, list[Utterance]] = {"train": [], "dev": [], "test": []}
    for seg in segments:
        text = _clean(seg["text"])
        split = split_of[seg["src_audio_filepath"]]
        if not text or (split != "train" and not _known_speaker(seg)):
            continue
        segment_rows[split].append(
            Utterance(
                audio=str(
                    wavs[_nfc(Path(seg["audio_filepath"]).name)].relative_to(out)
                ),
                text=text,
                source="fris",
                speaker=str(seg["speaker_id"]),
                duration=seg["duration"],
                recording=seg["src_audio_filepath"],
                offset=seg["src_offset"],
            )
        )

    long_dir = out / "longform"
    long_dir.mkdir(exist_ok=True)
    held_out = {rec for rec, split in split_of.items() if split != "train"}
    with tarfile.open(raw / "fris_raw.tar.gz") as tar:
        members = {_nfc(Path(m.name).name): m for m in tar.getmembers() if m.isfile()}
        for rec in held_out:
            for name in (rec, f"T&O {Path(rec).stem}.docx"):
                target = long_dir / name
                if not target.exists():
                    source = tar.extractfile(members[name])
                    assert source is not None
                    target.write_bytes(source.read())

    long_rows: dict[Split, list[Utterance]] = {"train": [], "dev": [], "test": []}
    for rec in sorted(held_out):
        audio = long_dir / rec
        long_rows[split_of[rec]].append(
            Utterance(
                audio=str(audio.relative_to(out)),
                text=frisian_transcript(long_dir / f"T&O {audio.stem}.docx"),
                source="fris",
                speaker=groups[rec],
                duration=_audio_seconds(audio),
            )
        )
    return segment_rows, long_rows


def prepare(data_dir: Path) -> dict[str, int]:
    raw, out = data_dir / "raw", data_dir / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    cv22_raw = Path(
        snapshot_download(
            CV22[0], repo_type="dataset", revision=CV22[1], local_dir=raw / "cv22"
        )
    )
    fris_raw = Path(
        snapshot_download(
            FRIS[0],
            repo_type="dataset",
            revision=FRIS[1],
            local_dir=raw / "fris",
            allow_patterns=["*.json", "*.tar.gz"],
        )
    )
    cv22 = _cv22(cv22_raw, out)
    fris, longform = _fris(fris_raw, out)
    counts = {
        split: _write_manifest(out / f"{split}.jsonl", [*cv22[split], *fris[split]])
        for split in cv22
    }
    for split in ("dev", "test"):
        counts[f"longform_{split}"] = _write_manifest(
            out / f"longform_{split}.jsonl", longform[split]
        )
    return counts
