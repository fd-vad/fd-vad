"""Windowed training dataset (paper §2.2). Each example = one sliding window ending
at chunk c → predict that chunk's label. Window audio = y[l_c*160 : r_c*160]
(160 samples per 100Hz fbank frame at 16kHz).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from ..schema import read_manifest
from ..windowing import (
    WindowConfig, num_chunks, window_bounds, seconds_to_fbank_frames, boundary_chunk,
)
from ..labeling import clip_chunk_labels

SAMPLES_PER_FRAME = 160   # 16kHz / 100Hz fbank


class WindowDataset(Dataset):
    def __init__(self, manifest_path, audio_root, win: WindowConfig,
                 balance: bool = False, max_examples: int | None = None,
                 continue_keep_prob: float = 1.0, seed: int = 1234):
        self.audio_root = Path(audio_root)
        self.win = win
        self._cache: dict[str, np.ndarray] = {}
        rng = random.Random(seed)

        self.utts = list(read_manifest(manifest_path))
        # examples: (utt_idx, chunk_c, label, dist_to_boundary). dist is chunk-distance
        # from the Stop transition (§6.2); for incomplete clips (no boundary) dist = -1 (far).
        self.examples: list[tuple[int, int, int, int]] = []
        for ui, u in enumerate(self.utts):
            n = seconds_to_fbank_frames(u.duration_s)
            C = num_chunks(n, win)
            labels = clip_chunk_labels(u, win)
            if u.sem_class == "complete":
                b = boundary_chunk(seconds_to_fbank_frames(u.stop_speaking_s), n, win)
            else:
                b = None
            for c in range(1, C + 1):
                lab = labels[c - 1]
                if lab == 0 and continue_keep_prob < 1.0 and rng.random() > continue_keep_prob:
                    continue
                dist = abs(c - b) if b is not None else -1
                self.examples.append((ui, c, lab, dist))

        if balance:
            pos = [e for e in self.examples if e[2] == 1]
            neg = [e for e in self.examples if e[2] == 0]
            rng.shuffle(neg)
            neg = neg[: len(pos)] if pos else neg
            self.examples = pos + neg
            rng.shuffle(self.examples)
        if max_examples:
            rng.shuffle(self.examples)
            self.examples = self.examples[:max_examples]

    def _load(self, u) -> np.ndarray:
        if u.audio_path not in self._cache:
            y, sr = sf.read(self.audio_root / u.audio_path, dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            self._cache[u.audio_path] = y
        return self._cache[u.audio_path]

    def label_counts(self):
        pos = sum(1 for e in self.examples if e[2] == 1)
        return {"stop": pos, "continue": len(self.examples) - pos, "total": len(self.examples)}

    def boundary_weights(self, near: int = 2, factor: float = 4.0) -> list[float]:
        """§6.2: weight for each example — `factor`x for windows within `near` chunks of
        the Stop boundary (hard negatives/positives), 1.0 otherwise. Use with
        torch.utils.data.WeightedRandomSampler."""
        w = []
        for (_, _, lab, dist) in self.examples:
            near_boundary = (dist >= 0 and dist <= near)
            w.append(factor if near_boundary else 1.0)
        return w

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ui, c, lab, _dist = self.examples[i]
        u = self.utts[ui]
        y = self._load(u)
        n = seconds_to_fbank_frames(u.duration_s)
        s, e = window_bounds(c, n, self.win)
        w = y[s * SAMPLES_PER_FRAME: e * SAMPLES_PER_FRAME]
        if len(w) < SAMPLES_PER_FRAME:                    # guard tiny slice
            w = y[max(0, len(y) - SAMPLES_PER_FRAME):]
        return torch.from_numpy(np.ascontiguousarray(w)), lab


def collate(batch):
    wavs, labels = zip(*batch)
    lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)
    T = int(lens.max())
    out = torch.zeros(len(wavs), T, dtype=torch.float32)
    for i, w in enumerate(wavs):
        out[i, : len(w)] = w
    return out, lens, torch.tensor(labels, dtype=torch.long)
