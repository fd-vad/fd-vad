"""Sliding-window math for FD-VAD (paper §2.2, PRD §1).

This is the spec-critical core of the whole system. All frame conventions and
formulas below follow the paper (arXiv 2509.20410) and fd_vad_replication_prd.md
EXACTLY. Every function is pure and unit-tested in tests/test_windowing.py.

Frame-rate conventions
----------------------
- Fbank features are computed at **100 Hz** (10 ms / frame). This is the base unit
  in which W_f and S_f are expressed.
- The Zipformer encoder consumes fbank and emits features at **25 Hz** (a 4x temporal
  downsample). So an encoder output frame spans 40 ms.
- The modality adapter further downsamples the 25 Hz encoder features by a factor `k`
  (concatenate k consecutive frames) → effective LLM audio-token rate 25/k Hz.

Paper defaults (Table, §4.1)
----------------------------
- W_f = 256 fbank frames = 2560 ms  (window size)
- S_f = 32  fbank frames = 320 ms   (chunk stride)

Sliding window (paper §2.2, 1-indexed in the paper):
    l_c = max(1, min(c * S_f, N) - W_f)
    r_c = min(c * S_f, N)
    F_c = [f_{l_c}, ..., f_{r_c}]
where N is the total number of fbank frames and c is the 1-based chunk index.

We implement 0-indexed, half-open [start, end) slices in code (Pythonic), and expose
the paper's 1-indexed inclusive form in the docstrings/tests for verification.
"""
from __future__ import annotations

from dataclasses import dataclass

FBANK_HZ = 100          # fbank frame rate (Hz)
ENCODER_HZ = 25         # Zipformer output frame rate (Hz)
MS_PER_FBANK_FRAME = 1000 // FBANK_HZ  # 10 ms


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window hyperparameters, in fbank (100 Hz) frame units unless noted."""
    window_frames: int = 256   # W_f  (2560 ms at 100 Hz)
    stride_frames: int = 32    # S_f  (320 ms  at 100 Hz)

    def __post_init__(self) -> None:
        if self.window_frames <= 0 or self.stride_frames <= 0:
            raise ValueError("window_frames and stride_frames must be positive")
        if self.stride_frames > self.window_frames:
            raise ValueError(
                f"stride_frames ({self.stride_frames}) > window_frames "
                f"({self.window_frames}); windows would skip audio"
            )

    @property
    def window_ms(self) -> float:
        return self.window_frames * MS_PER_FBANK_FRAME

    @property
    def stride_ms(self) -> float:
        return self.stride_frames * MS_PER_FBANK_FRAME


def num_chunks(n_frames: int, cfg: WindowConfig) -> int:
    """Number of chunks c=1..C that tile an utterance of `n_frames` fbank frames.

    The last chunk's right edge r_c reaches N exactly when c*S_f >= N. We define
    C = ceil(N / S_f), so the final window always ends at frame N (via the min()).
    An utterance with 0 frames yields 0 chunks.
    """
    if n_frames < 0:
        raise ValueError("n_frames must be >= 0")
    if n_frames == 0:
        return 0
    return -(-n_frames // cfg.stride_frames)  # ceil division


def window_bounds(c: int, n_frames: int, cfg: WindowConfig) -> tuple[int, int]:
    """Return (start, end) as a 0-indexed half-open [start, end) fbank-frame slice
    for 1-based chunk index `c`, matching the paper's l_c/r_c (1-indexed inclusive).

    Paper (1-indexed inclusive): l_c = max(1, min(c*S_f, N) - W_f), r_c = min(c*S_f, N).
    The window is [l_c, r_c] inclusive, i.e. length r_c - l_c + 1 ... but note the paper's
    l_c uses `min(c*S_f,N) - W_f` which is the frame *before* the W_f-long span, so the
    inclusive span [l_c+1 .. r_c] has length W_f once past the ramp-up. We reproduce this
    precisely as a 0-indexed half-open slice: start = max(0, r - W_f), end = r,
    where r = min(c*S_f, N). Length = min(r, W_f) frames (ramps up then saturates at W_f).
    """
    if c < 1:
        raise ValueError(f"chunk index c must be >= 1, got {c}")
    if n_frames <= 0:
        raise ValueError("n_frames must be > 0 to extract a window")
    r = min(c * cfg.stride_frames, n_frames)   # r_c  (0-indexed half-open end)
    start = max(0, r - cfg.window_frames)       # l_c  (0-indexed start)
    return start, r


def all_windows(n_frames: int, cfg: WindowConfig) -> list[tuple[int, int]]:
    """[(start,end), ...] half-open slices for every chunk c=1..C of the utterance."""
    return [window_bounds(c, n_frames, cfg) for c in range(1, num_chunks(n_frames, cfg) + 1)]


def seconds_to_fbank_frames(t_seconds: float) -> int:
    """Convert a timestamp in seconds to a 0-indexed fbank frame count (floor)."""
    if t_seconds < 0:
        raise ValueError("timestamp must be >= 0")
    return int(t_seconds * FBANK_HZ)


def chunk_label(c: int, stop_frame: int, n_frames: int, cfg: WindowConfig) -> int:
    """Ground-truth per-chunk label for chunk `c` (paper §2.3 / PRD §2 labeling).

    A chunk classifies the user's state at the *right edge* r_c of its window:
      - y_c = 1 (Stop Speaking / <|S-S|>)  if the utterance has ended by r_c
              (r_c >= stop_frame), i.e. the user has finished speaking.
      - y_c = 0 (Continue Speaking / <|C-S|>) otherwise (still mid-utterance).

    `stop_frame` is the stop-speaking boundary in fbank frames (from timestamp
    annotation, §2 stage 3). During TRAINING, supervision is applied only to the
    last chunk of each sampled window (Eq. 2); this function gives the label for
    ANY chunk, which the eval harness (chunk-level, Table 2) also uses.
    """
    _, r = window_bounds(c, n_frames, cfg)
    return 1 if r >= stop_frame else 0


def boundary_chunk(stop_frame: int, n_frames: int, cfg: WindowConfig) -> int:
    """The 1-based index of the first chunk whose label flips to 1 (Stop).

    Useful for §6.2 boundary-focused resampling (distance-to-boundary metadata):
    the transition chunk c* is the first c with r_c >= stop_frame. Returns the
    total chunk count + 0 semantics: if the utterance never reaches stop_frame
    within its chunks, returns num_chunks (last chunk)."""
    C = num_chunks(n_frames, cfg)
    for c in range(1, C + 1):
        if chunk_label(c, stop_frame, n_frames, cfg) == 1:
            return c
    return C
