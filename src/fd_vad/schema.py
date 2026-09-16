"""Data schema for FD-VAD (PRD §2). One atomic sample = one synthesized
utterance with a semantic-completeness class and a stop-speaking timestamp.

The JSONL manifest is the contract between the data pipeline (M1) and the
training/eval code (M2-M4). Validation is strict and fail-loud by design
(the user hates silent failures): `validate_utterance` raises on any drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator, Literal

SemClass = Literal["complete", "incomplete"]

# Label token conventions (paper §3).
CONTINUE_SPEAKING = 0   # <|C-S|>  keep listening
STOP_SPEAKING = 1       # <|S-S|>  user finished

# Timeout thresholds (paper §4.1): short for complete, long for incomplete.
TIMEOUT_COMPLETE_MS = 400    # T_s
TIMEOUT_INCOMPLETE_MS = 1000  # T_l

SCHEMA_VERSION = "1.0"


@dataclass
class WordTiming:
    word: str
    start_s: float
    end_s: float


@dataclass
class Utterance:
    """One synthesized utterance = one training/eval sample.

    stop_speaking_s is the ground-truth boundary (seconds from audio start) after
    which the user has finished; for `complete` it is the end of speech, for
    `incomplete` it is also end-of-audio but the label semantics differ (a truly
    incomplete query should keep predicting Continue right up to the cutoff).
    """
    id: str
    sem_class: SemClass
    text: str                       # the (possibly truncated) spoken text
    source_text: str                # the original complete text it derived from
    audio_path: str                 # relative to the dataset root
    sample_rate: int
    duration_s: float
    stop_speaking_s: float          # boundary timestamp (fbank-frame-derivable)
    speaker: str                    # voice id / speaker-prompt id
    words: list[WordTiming] = field(default_factory=list)
    timeout_ms: int = 0             # T_s or T_l depending on sem_class
    source: str = ""                # provenance: dataset/corpus name
    extra: dict = field(default_factory=dict)  # arbitrary metadata (midfiller, synthetic, ...)
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(d: dict) -> "Utterance":
        words = [WordTiming(**w) for w in d.get("words", [])]
        d = {**d, "words": words}
        # tolerate unknown extra keys by dropping them (forward-compat), but warn-loud
        allowed = set(Utterance.__dataclass_fields__.keys())
        extra = set(d.keys()) - allowed
        if extra:
            raise ValueError(f"Utterance {d.get('id')!r} has unknown fields: {sorted(extra)}")
        return Utterance(**{k: v for k, v in d.items() if k in allowed})


def expected_timeout_ms(sem_class: SemClass) -> int:
    return TIMEOUT_COMPLETE_MS if sem_class == "complete" else TIMEOUT_INCOMPLETE_MS


def validate_utterance(u: Utterance, dataset_root: Path | None = None,
                       check_audio: bool = False, require_text: bool = False) -> list[str]:
    """Return a list of problems (empty == valid). Never silently passes bad data.

    require_text: smart-turn (ASR-free) has no transcript, so text may be empty by
    design; set True only for corpora where a transcript is expected.
    """
    errs: list[str] = []
    if not u.id:
        errs.append("empty id")
    if u.sem_class not in ("complete", "incomplete"):
        errs.append(f"bad sem_class {u.sem_class!r}")
    if require_text and not u.text.strip():
        errs.append("empty text")
    if u.sample_rate <= 0:
        errs.append(f"bad sample_rate {u.sample_rate}")
    if u.duration_s <= 0:
        errs.append(f"non-positive duration {u.duration_s}")
    if not (0 <= u.stop_speaking_s <= u.duration_s + 1e-6):
        errs.append(f"stop_speaking_s {u.stop_speaking_s} outside [0, duration {u.duration_s}]")
    if u.timeout_ms and u.timeout_ms != expected_timeout_ms(u.sem_class):
        errs.append(f"timeout_ms {u.timeout_ms} != expected {expected_timeout_ms(u.sem_class)} for {u.sem_class}")
    # word timings monotonic & within duration
    prev = 0.0
    for i, w in enumerate(u.words):
        if w.start_s < prev - 1e-3:
            errs.append(f"word[{i}] {w.word!r} start {w.start_s} < prev end {prev}")
        if w.end_s < w.start_s:
            errs.append(f"word[{i}] {w.word!r} end < start")
        if w.end_s > u.duration_s + 1e-3:
            errs.append(f"word[{i}] {w.word!r} end {w.end_s} > duration {u.duration_s}")
        prev = w.end_s
    if check_audio and dataset_root is not None:
        p = dataset_root / u.audio_path
        if not p.exists():
            errs.append(f"audio file missing: {p}")
    return errs


def write_manifest(path: str | Path, utts: Iterable[Utterance]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for u in utts:
            f.write(json.dumps(u.to_json()) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> Iterator[Utterance]:
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Utterance.from_json(json.loads(line))
            except Exception as e:
                raise ValueError(f"manifest {path} line {line_no}: {e}") from e


def validate_manifest(path: str | Path, dataset_root: Path | None = None,
                      check_audio: bool = False) -> dict:
    """Validate an entire manifest; returns a summary dict and raises on hard errors."""
    n, n_complete, n_incomplete, problems = 0, 0, 0, []
    ids: set[str] = set()
    for u in read_manifest(path):
        n += 1
        if u.id in ids:
            problems.append(f"duplicate id {u.id}")
        ids.add(u.id)
        n_complete += u.sem_class == "complete"
        n_incomplete += u.sem_class == "incomplete"
        errs = validate_utterance(u, dataset_root, check_audio)
        problems.extend(f"{u.id}: {e}" for e in errs)
    summary = {
        "n": n, "n_complete": n_complete, "n_incomplete": n_incomplete,
        "n_problems": len(problems), "problems": problems[:50],
    }
    if problems:
        raise ValueError(f"manifest {path} has {len(problems)} problems; first: {problems[0]}")
    return summary
