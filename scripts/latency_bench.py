#!/usr/bin/env python
"""M4 §4.3 latency benchmark: measure per-window model processing time t_model on GPU,
then report first-packet latency T_stride + t_model vs the full-utterance-wait baseline."""
from __future__ import annotations
import argparse, sys, time, json
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.config import load_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--encoder", default="wavlm")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--speech-s", type=float, default=3.0, help="assumed user utterance length for the arithmetic")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt).parent
    cfg = load_config(ckpt_dir / "config.yaml"); cfg.encoder.kind = args.encoder
    win = WindowConfig(cfg.window.window_frames, cfg.window.stride_frames)
    dev = "cuda"
    model = FDVAD(cfg).to(dev)
    model.load_trainable_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()

    # one full 2560ms window, batch 1 (streaming inference is per-window, batch 1)
    T = int(16000 * win.window_ms / 1000.0)
    wav = torch.zeros(1, T, device=dev)
    lens = torch.tensor([T], device=dev)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(wav, lens, labels=None)
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.iters):
            t0 = time.time()
            model(wav, lens, labels=None)
            torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    t_model = sum(ts) / len(ts)
    p50, p90 = ts[len(ts) // 2], ts[int(len(ts) * 0.9)]

    T_stride = win.stride_ms
    speech_ms = args.speech_s * 1000.0
    fd_vad_fp = T_stride + t_model                  # first-packet latency
    cascade_fp = speech_ms + 284                    # paper's cascaded baseline (t_cascade=284ms)
    naive_wait = speech_ms + t_model                # naive full-utterance-wait with our model

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "t_model_ms": {"mean": round(t_model, 2), "p50": round(p50, 2), "p90": round(p90, 2)},
        "T_stride_ms": T_stride,
        "assumed_speech_ms": speech_ms,
        "first_packet_latency_ms": {
            "fd_vad (ours, streaming) = T_stride + t_model": round(fd_vad_fp, 1),
            "naive full-utterance-wait (ours) = speech + t_model": round(naive_wait, 1),
            "cascaded ASR baseline (paper) = speech + 284": round(cascade_fp, 1),
        },
        "latency_reduction_vs_naive_wait": f"{100*(1 - fd_vad_fp/naive_wait):.1f}%",
    }
    print(json.dumps(report, indent=2))
    (ckpt_dir / "latency.json").write_text(json.dumps(report, indent=2))
    print(f"[wrote] {ckpt_dir/'latency.json'}")


if __name__ == "__main__":
    main()
