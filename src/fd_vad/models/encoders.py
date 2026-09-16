"""Frozen speech encoders behind one interface (PRD §1, D3).

The encoder is always frozen (paper §2.1). Torch-native SSL encoders (wav2vec2, WavLM)
are efficient on the 2560ms sliding windows and correct for causal/streaming training
(each window encoded independently). A Zipformer-ONNX encoder can drop in behind the
same interface later as a fidelity upgrade.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpeechEncoder(nn.Module):
    """encode(wav[B,T], wav_lens[B]) -> (feats[B,S,D], feat_lens[B]). Frozen."""
    output_dim: int
    output_hz: float

    def encode(self, wav, wav_lens):
        raise NotImplementedError


class _HFSSLEncoder(SpeechEncoder):
    """Wraps any HF SSL model exposing last_hidden_state, attention_mask, and
    _get_feat_extract_output_lengths (wav2vec2, WavLM, HuBERT share this API)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # frozen encoder: gradient checkpointing is pointless and (under no_grad+autocast)
        # caused a severe slowdown/stall. Disable explicitly.
        self.model.config.gradient_checkpointing = False
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass
        self.output_dim = self.model.config.hidden_size
        self.output_hz = 16000.0 / 320.0            # ~50 Hz (conv total stride 320)
        self._do_normalize = True

    @torch.no_grad()
    def encode(self, wav: torch.Tensor, wav_lens: torch.Tensor):
        mask = torch.arange(wav.shape[1], device=wav.device)[None, :] < wav_lens[:, None]
        if self._do_normalize:                       # per-utterance zero-mean/unit-var
            m = (wav * mask).sum(1) / wav_lens.clamp(min=1)
            v = ((wav - m[:, None]) ** 2 * mask).sum(1) / wav_lens.clamp(min=1)
            wav = ((wav - m[:, None]) / (v[:, None].sqrt() + 1e-7)) * mask
        out = self.model(wav, attention_mask=mask.long()).last_hidden_state
        feat_lens = self.model._get_feat_extract_output_lengths(wav_lens).long()
        return out, feat_lens


class Wav2Vec2Encoder(_HFSSLEncoder):
    def __init__(self, name: str = "facebook/wav2vec2-base"):
        from transformers import Wav2Vec2Model
        super().__init__(Wav2Vec2Model.from_pretrained(name))


class WavLMEncoder(_HFSSLEncoder):
    def __init__(self, name: str = "microsoft/wavlm-base-plus"):
        from transformers import WavLMModel
        super().__init__(WavLMModel.from_pretrained(name))


def build_encoder(kind: str, checkpoint: str = "") -> SpeechEncoder:
    kind = kind.lower()
    if kind in ("wav2vec2", "w2v2"):
        return Wav2Vec2Encoder(checkpoint or "facebook/wav2vec2-base")
    if kind == "wavlm":
        return WavLMEncoder(checkpoint or "microsoft/wavlm-base-plus")
    if kind == "zipformer":
        raise NotImplementedError(
            "Zipformer-ONNX encoder not yet wired; use kind=wav2vec2/wavlm (see STATE.md D3).")
    raise ValueError(f"unknown encoder kind: {kind}")
