"""smart-turn-data-v3.1 → FD-VAD Utterance conversion (docs/labeling.md).

Each parquet row: {audio:{bytes(flac),path}, id, language, endpoint_bool, midfiller,
endfiller, synthetic, dataset}. endpoint_bool True→complete(Stop), False→incomplete(Continue).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from ..schema import Utterance, expected_timeout_ms
from ..windowing import WindowConfig
from .vad import energy_vad_offset

TARGET_SR = 16000


@dataclass
class STConfig:
    language: str = "eng"
    min_trailing_stop_chunks: int = 1     # pad COMPLETE clips to guarantee ≥ this many post-endpoint chunks
    rel_thresh_db: float = -40.0
    max_pad_s: float = 1.0                # never pad more than this (safety)


def _decode(audio_field: dict) -> tuple[np.ndarray, int]:
    y, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def row_to_utterance(
    row: dict,
    win: WindowConfig,
    st: STConfig,
) -> tuple[Utterance, np.ndarray, int]:
    """Convert one parquet row → (Utterance, audio_float32_mono, sr).

    Caller writes the audio to `utt.audio_path`. Raises on malformed audio (fail-loud).
    """
    y, sr = _decode(row["audio"])
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    if len(y) < sr * 0.1:
        raise ValueError(f"clip {row['id']} too short: {len(y)/sr:.3f}s")

    endpoint = bool(row["endpoint_bool"])
    sem_class = "complete" if endpoint else "incomplete"
    stride_s = win.stride_frames / 100.0

    if endpoint:
        _, offset_s = energy_vad_offset(y, sr, rel_thresh_db=st.rel_thresh_db)
        stop_s = float(offset_s)
        # guarantee >= min_trailing_stop_chunks full chunks after the boundary
        need_end = stop_s + st.min_trailing_stop_chunks * stride_s
        cur_dur = len(y) / sr
        if cur_dur < need_end:
            pad = min(need_end - cur_dur, st.max_pad_s)
            y = np.concatenate([y, np.zeros(int(pad * sr), dtype=y.dtype)])
    else:
        stop_s = len(y) / sr  # unused for labeling (incomplete → all Continue)

    dur = len(y) / sr
    stop_s = min(stop_s, dur)
    utt = Utterance(
        id=f"st_{st.language}_{row['id']}",
        sem_class=sem_class,
        text="",                       # ASR-free: no transcript in smart-turn
        source_text="",
        audio_path="",                 # set by caller
        sample_rate=sr,
        duration_s=round(dur, 4),
        stop_speaking_s=round(stop_s, 4),
        speaker=str(row.get("dataset", "unknown")),
        timeout_ms=expected_timeout_ms(sem_class),
        source=f"smart-turn-v3.1/{row.get('dataset','')}",
        extra={
            "endpoint_bool": endpoint,
            "midfiller": row.get("midfiller"),
            "endfiller": row.get("endfiller"),
            "synthetic": row.get("synthetic"),
            "language": row.get("language"),
        },
    )
    return utt, y, sr
