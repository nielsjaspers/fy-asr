from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fy_asr.asr import Word

MAX_LINE_CHARS = 42
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 1.0
PAUSE_SECONDS = 0.8


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    lines: tuple[str, ...]


def _layout(words: Sequence[str]) -> tuple[str, ...] | None:
    text = " ".join(words)
    if len(text) <= MAX_LINE_CHARS:
        return (text,)
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    fitting = [s for s in splits if all(len(line) <= MAX_LINE_CHARS for line in s)]
    return min(fitting, key=lambda s: abs(len(s[0]) - len(s[1]))) if fitting else None


def _group(words: Iterable[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    for word in words:
        current = groups[-1] if groups else None
        if (
            current is None
            or word.start - current[-1].end > PAUSE_SECONDS
            or word.end - current[0].start > MAX_CUE_SECONDS
            or _layout([w.text for w in [*current, word]]) is None
        ):
            groups.append([word])
        else:
            current.append(word)
    return groups


def build_cues(words: Iterable[Word]) -> list[Cue]:
    groups = _group(words)
    cues: list[Cue] = []
    for i, group in enumerate(groups):
        start = group[0].start
        next_start = groups[i + 1][0].start if i + 1 < len(groups) else float("inf")
        end = min(max(group[-1].end, start + MIN_CUE_SECONDS), next_start)
        texts = [w.text for w in group]
        cues.append(Cue(start, end, _layout(texts) or (" ".join(texts),)))
    return cues


def _timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(cues: Sequence[Cue]) -> str:
    return "\n".join(
        f"{i}\n{_timestamp(c.start)} --> {_timestamp(c.end)}\n"
        + "\n".join(c.lines)
        + "\n"
        for i, c in enumerate(cues, start=1)
    )
