"""Clip-level → chunk-level label derivation (docs/labeling.md, S2 scheme).

The one place that encodes: COMPLETE clips get a Stop boundary at the speech offset;
INCOMPLETE clips are Continue everywhere (their end is not a valid endpoint).
"""
from __future__ import annotations

from .schema import Utterance, CONTINUE_SPEAKING, STOP_SPEAKING
from .windowing import (
    WindowConfig, num_chunks, chunk_label, seconds_to_fbank_frames,
)


def clip_chunk_labels(u: Utterance, cfg: WindowConfig) -> list[int]:
    """Per-chunk labels (0/1) for every chunk c=1..C of the utterance."""
    n = seconds_to_fbank_frames(u.duration_s)
    C = num_chunks(n, cfg)
    if u.sem_class == "incomplete":
        return [CONTINUE_SPEAKING] * C                    # never a valid endpoint
    stop = seconds_to_fbank_frames(u.stop_speaking_s)
    return [chunk_label(c, stop, n, cfg) for c in range(1, C + 1)]


def label_sanity(u: Utterance, cfg: WindowConfig) -> list[str]:
    """Validation gates from docs/labeling.md. Empty list == passes."""
    errs: list[str] = []
    labels = clip_chunk_labels(u, cfg)
    if not labels:
        errs.append("no chunks (zero-length clip)")
        return errs
    n_stop = sum(labels)
    if u.sem_class == "complete":
        if n_stop < 1:
            errs.append("complete clip has 0 Stop chunks (boundary past clip end?)")
        if n_stop == len(labels):
            errs.append("complete clip is ALL Stop (boundary at/near 0?)")
        # Stop chunks must be a contiguous tail
        first_stop = next((i for i, v in enumerate(labels) if v == STOP_SPEAKING), None)
        if first_stop is not None and any(v == CONTINUE_SPEAKING for v in labels[first_stop:]):
            errs.append("Stop chunks not contiguous at tail")
        if not (0.0 < u.stop_speaking_s <= u.duration_s + 1e-6):
            errs.append(f"stop_speaking_s {u.stop_speaking_s} outside (0, duration]")
    else:  # incomplete
        if n_stop != 0:
            errs.append(f"incomplete clip has {n_stop} Stop chunks (must be 0)")
    return errs
