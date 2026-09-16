"""Modality adapter (paper §2.1): downsample encoder features by concatenating every
k consecutive frames, then a 2-layer MLP with ReLU projecting into the LLM embedding
space. A_P = W2( ReLU( W1 · A_I + b1 ) ) + b2, where A_I concatenates k frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ModalityAdapter(nn.Module):
    def __init__(self, in_dim: int, k: int, hidden_dim: int, out_dim: int):
        super().__init__()
        if k < 1:
            raise ValueError("downsample k must be >= 1")
        self.k = k
        self.in_dim = in_dim
        self.lin1 = nn.Linear(in_dim * k, hidden_dim)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, feats: torch.Tensor, feat_lens: torch.Tensor):
        """feats [B,S,D], feat_lens [B] -> (adapted [B,S//k,out], out_lens [B])."""
        B, S, D = feats.shape
        assert D == self.in_dim, f"encoder dim {D} != adapter in_dim {self.in_dim}"
        S2 = (S // self.k) * self.k
        if S2 == 0:
            # window shorter than k encoder frames: keep a single group by left-padding
            pad = self.k - S
            feats = torch.cat([feats, feats.new_zeros(B, pad, D)], dim=1)
            S2 = self.k
        x = feats[:, :S2, :].reshape(B, S2 // self.k, D * self.k)
        x = self.lin2(self.act(self.lin1(x)))
        out_lens = (feat_lens // self.k).clamp(min=1)
        return x, out_lens
