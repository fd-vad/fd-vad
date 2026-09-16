#!/usr/bin/env python
"""M2+M3 smoke test (PRD §5 acceptance):
  M2: a forward pass on one batch yields {0,1} logits without shape errors.
  M3: loss decreases over a short overfit run on a tiny balanced set.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.config import load_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD
from fd_vad.data.dataset import WindowDataset, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--manifest", default="/mnt/localssd/svad/data/smartturn_en/pilot/manifest.jsonl")
    ap.add_argument("--audio-root", default="/mnt/localssd/svad/data/smartturn_en/pilot")
    ap.add_argument("--encoder", default="wav2vec2")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-examples", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.encoder.kind = args.encoder
    win = WindowConfig(cfg.window.window_frames, cfg.window.stride_frames)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[build] encoder={cfg.encoder.kind} llm={cfg.llm.model} k={cfg.adapter.downsample_k}")
    model = FDVAD(cfg).to(dev)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[params] trainable={n_train/1e6:.2f}M total={n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}% trainable)")

    ds = WindowDataset(args.manifest, args.audio_root, win, balance=True,
                       max_examples=args.max_examples)
    print(f"[data] examples={len(ds)} counts={ds.label_counts()}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate, drop_last=True)

    # ---- M2: single forward (inference) ----
    wav, lens, labels = next(iter(dl))
    model.eval()
    with torch.no_grad():
        out = model(wav, lens, labels=None)
    assert out.logits2.shape == (wav.shape[0], 2), out.logits2.shape
    print(f"[M2 ✓] forward ok. logits2={tuple(out.logits2.shape)} "
          f"probs[0]={out.probs2[0].tolist()}")

    # ---- M3: overfit loop, loss should drop ----
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    model.train()
    losses, accs = [], []
    it = iter(dl); t0 = time.time()
    for step in range(args.steps):
        try:
            wav, lens, labels = next(it)
        except StopIteration:
            it = iter(dl); wav, lens, labels = next(it)
        out = model(wav, lens, labels=labels)
        opt.zero_grad(); out.loss.backward(); opt.step()
        pred = out.logits2.argmax(-1).cpu()
        acc = (pred == labels).float().mean().item()
        losses.append(out.loss.item()); accs.append(acc)
        if step % 20 == 0 or step == args.steps - 1:
            print(f"  step {step:3d} loss {out.loss.item():.4f} acc {acc:.2f}")
    dt = time.time() - t0
    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    print(f"[M3] first10 loss {first:.4f} -> last10 loss {last:.4f} "
          f"| last10 acc {sum(accs[-10:])/10:.2f} | {dt:.1f}s ({args.steps/dt:.1f} it/s)")
    assert last < first * 0.8, f"loss did not drop enough: {first:.3f}->{last:.3f}"
    print("[M3 ✓] loss decreased on overfit set. Model stack is wired correctly.")


if __name__ == "__main__":
    main()
