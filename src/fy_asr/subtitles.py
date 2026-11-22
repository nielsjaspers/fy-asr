from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fy_asr.asr import Word

MAX_LINE_CHARS = 42
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 1.0
PAUSE_SECONDS = 0.8
MAX_PAUSE_IN_CUE = 2.0


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    lines: tuple[str, ...]


SENTENCE_END = (".", "?", "!")
CLAUSE_END = (",", ";", ":")
CUE_COST = 2.0
PAUSE_COST = 0.5
SHORT_CUE_CHARS = 12
SHORT_CUE_COST = 1.5
FILLERS = {"uh", "eh", "ehm", "uhm", "hm"}
MAX_CUE_WORDS = 24


def _end_cost(word: str) -> float:
    if word.endswith(SENTENCE_END):
        return 0.0
    if word.endswith(CLAUSE_END):
        return 1.0
    return 4.0


def _layout(words: Sequence[str]) -> tuple[str, ...] | None:
    text = " ".join(words)
    if len(text) <= MAX_LINE_CHARS:
        return (text,)
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    fitting = [s for s in splits if all(len(line) <= MAX_LINE_CHARS for line in s)]
    if not fitting:
        return None
    return min(
        fitting,
        key=lambda s: abs(len(s[0]) - len(s[1])) + 10 * _end_cost(s[0]),
    )


def _group(words: Sequence[Word]) -> list[list[Word]]:
    """Cut words into cues at the lowest total cost.

    A cue of two or more words must fit two lines and MAX_CUE_SECONDS and may
    not span a silence longer than MAX_PAUSE_IN_CUE; a single word always
    forms a cue. Cutting after a sentence end is free, at a
    pause or after a clause cheap, and anywhere else expensive. Every cue costs
    CUE_COST, so short sentences share a cue, and a short cue that does not end
    a sentence costs SHORT_CUE_COST extra.
    """
    best = [0.0] + [float("inf")] * len(words)
    cut = [0] * (len(words) + 1)
    for end in range(1, len(words) + 1):
        for start in range(end - 1, max(-1, end - 1 - MAX_CUE_WORDS), -1):
            group = words[start:end]
            if start < end - 1 and (
                words[start + 1].start - words[start].end > MAX_PAUSE_IN_CUE
                or group[-1].end - group[0].start > MAX_CUE_SECONDS
                or _layout([w.text for w in group]) is None
            ):
                break
            closing = 0.0
            if end < len(words):
                closing = _end_cost(group[-1].text)
                if words[end].start - group[-1].end > PAUSE_SECONDS:
                    closing = min(closing, PAUSE_COST)
            short = len(" ".join(w.text for w in group)) <= SHORT_CUE_CHARS
            penalty = SHORT_CUE_COST if short and closing else 0.0
            cost = best[start] + CUE_COST + closing + penalty
            if cost < best[end]:
                best[end], cut[end] = cost, start
    groups: list[list[Word]] = []
    end = len(words)
    while end:
        groups.append(list(words[cut[end] : end]))
        end = cut[end]
    return groups[::-1]


def build_cues(words: Iterable[Word]) -> list[Cue]:
    """Subtitle cues for words; fillers such as "uh" are left out."""
    groups = _group([w for w in words if w.text.lower().strip(",.?!;:") not in FILLERS])
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
