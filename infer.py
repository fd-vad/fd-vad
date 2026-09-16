#!/usr/bin/env python
"""End-to-end FD-VAD end-of-turn (EOT) inference on a single audio file.

Downloads the frozen base models (WavLM-base-plus, Qwen2.5-0.5B-Instruct) automatically
on first run, applies this repo's trained adapter + LoRA (fd_vad.pt), and slides a 2.56 s
window over the audio at a 320 ms stride to produce a per-chunk P(Stop) stream. It commits
an EOT the first time P(Stop) exceeds `--tau` for `--k` consecutive chunks.

Usage:
    pip install -r requirements.txt
    python infer.py path/to/audio.wav              # 16 kHz mono recommended (auto-resampled)
    python infer.py audio.wav --tau 0.9 --k 1      # more conservative endpoint
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))          # fd_vad package lives under src/
from fd_vad.config import load_config
from fd_vad.models.fd_vad import FDVAD

SR = 16_000
WIN = 256 * 160          # 2.56 s window
STRIDE = 32 * 160        # 320 ms stride
CHUNK_S = STRIDE / SR    # 0.32 s


def load_audio(path: str) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav


@torch.no_grad()
def p_stop_stream(model, wav: np.ndarray, device: str, batch: int = 128) -> np.ndarray:
    """Return P(Stop) per 320 ms chunk (causal: each window ends at the chunk time)."""
    x = torch.from_numpy(wav).float()
    n_chunks = max(1, int(np.ceil(len(x) / STRIDE)))
    padded = torch.cat([x.new_zeros(WIN), x])                       # left-pad one window
    ends = torch.clamp(torch.arange(1, n_chunks + 1) * STRIDE, max=len(x)) + WIN
    ar = torch.arange(WIN)
    probs = np.zeros(n_chunks, dtype=np.float32)
    for b0 in range(0, n_chunks, batch):
        b1 = min(b0 + batch, n_chunks)
        idx = ends[b0:b1].view(-1, 1) + ar.view(1, -1) - WIN       # slice [end-WIN : end]
        inp = padded[idx].to(device)                                # [B, WIN]
        lens = torch.full((b1 - b0,), WIN, dtype=torch.long)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(inp, lens, labels=None)
        else:
            out = model(inp, lens, labels=None)
        probs[b0:b1] = out.probs2[:, 1].float().cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--ckpt", default=str(HERE / "fd_vad.pt"))
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--tau", type=float, default=0.5, help="P(Stop) threshold to commit")
    ap.add_argument("--k", type=int, default=1, help="consecutive chunks above tau before commit")
    ap.add_argument("--show-stream", action="store_true", help="print per-chunk P(Stop)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config(args.config); cfg.encoder.kind = "wavlm"

    # Prefer the bundled base weights (fully offline, cwd-independent). Falls back to
    # downloading microsoft/wavlm-base-plus + Qwen/Qwen2.5-0.5B-Instruct if base/ is absent.
    base_wavlm, base_qwen = HERE / "base/wavlm-base-plus", HERE / "base/qwen2.5-0.5b"
    if base_wavlm.exists() and base_qwen.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        cfg.encoder.checkpoint = str(base_wavlm)
        cfg.llm.model = str(base_qwen)

    if not Path(args.ckpt).exists():
        sys.exit(
            f"[fd-vad] checkpoint not found: {args.ckpt}\n"
            "  Download the model weights from https://huggingface.co/puneetUMD/fd-vad\n"
            "  e.g.:  huggingface-cli download puneetUMD/fd-vad fd_vad.pt --local-dir .\n"
            "  (optionally also download the base/ folder from there for fully-offline use;\n"
            "   otherwise WavLM + Qwen2.5-0.5B download automatically on first run.)"
        )

    torch.manual_seed(1234)
    model = FDVAD(cfg).to(device).eval()
    model.load_trainable_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])

    wav = load_audio(args.audio)
    probs = p_stop_stream(model, wav, device)

    if args.show_stream:
        for c, p in enumerate(probs):
            print(f"  t={c*CHUNK_S:5.2f}s  P(Stop)={p:.3f}")

    # commit: first chunk with tau exceeded for k consecutive chunks
    run = 0; commit = None
    for c, p in enumerate(probs):
        run = run + 1 if p >= args.tau else 0
        if run >= args.k:
            commit = (c + 1) * CHUNK_S            # chunk end time
            break
    if commit is not None:
        print(f"\nEOT: user turn COMPLETE — endpoint at {commit:.2f}s "
              f"(tau={args.tau}, k={args.k}).")
    else:
        print(f"\nEOT: no endpoint — turn appears INCOMPLETE across {len(probs)} chunks "
              f"(max P(Stop)={probs.max():.3f}, tau={args.tau}).")


if __name__ == "__main__":
    main()
