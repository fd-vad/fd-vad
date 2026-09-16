"""Voice activity detection to locate the speech offset (end-of-speech).

Used only to place the Stop boundary inside COMPLETE clips (docs/labeling.md).
Default is a relative-threshold energy VAD with hysteresis — cheap, dependency-free,
and adequate on smart-turn's mostly-clean (often synthetic) audio. Structured so a
neural VAD (Silero) can drop in behind the same interface later.
"""
from __future__ import annotations

import numpy as np


def energy_vad_offset(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
    rel_thresh_db: float = -40.0,
    min_speech_ms: float = 100.0,
    hangover_ms: float = 80.0,
) -> tuple[float, float]:
    """Return (onset_s, offset_s): first and last speech instants.

    Speech frames = those whose RMS is within `rel_thresh_db` of the clip's peak-frame
    RMS. `hangover_ms` merges short gaps so a single pause inside a word doesn't split
    the region; `min_speech_ms` guards against blips. If no speech is found, returns
    (0.0, duration) as a safe fallback (treat whole clip as speech).
    """
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float64)
    dur = len(y) / sr
    n = max(1, int(sr * frame_ms / 1000.0))
    nf = len(y) // n
    if nf == 0:
        return 0.0, dur
    frames = y[: nf * n].reshape(nf, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    peak = db.max()
    speech = db > (peak + rel_thresh_db)

    # hangover: fill gaps shorter than hangover_ms between speech frames
    hang = int(round(hangover_ms / frame_ms))
    if hang > 0:
        idx = np.where(speech)[0]
        if len(idx):
            for a, b in zip(idx[:-1], idx[1:]):
                if 1 < (b - a) <= hang + 1:
                    speech[a:b] = True

    idx = np.where(speech)[0]
    if len(idx) == 0:
        return 0.0, dur
    onset = idx[0] * frame_ms / 1000.0
    offset = (idx[-1] + 1) * frame_ms / 1000.0
    # enforce a minimum speech span; if too short, fall back to whole clip
    if (offset - onset) * 1000.0 < min_speech_ms:
        return 0.0, dur
    return float(onset), float(min(offset, dur))
