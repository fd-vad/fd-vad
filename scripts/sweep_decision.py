#!/usr/bin/env python
"""§6.1 sweep: run inference once, then sweep (threshold, K) debounce policies and
report sentence-level accuracy — an accuracy-vs-latency tradeoff table (PRD §6.1)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.config import load_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD
from fd_vad.eval import infer_manifest
from fd_vad.decision import clip_records_to_sequences, sentence_accuracy, debounce_sequence


def mean_stop_latency_chunks(clips, threshold, K):
    """Avg extra chunks waited before committing Stop on COMPLETE clips (latency proxy)."""
    lat = []
    for d in clips.values():
        if d["sem"] != "complete":
            continue
        preds = debounce_sequence(d["pstop"], threshold, K)
        gt_first = next((i for i, v in enumerate(d["gt"]) if v == 1), None)
        pr_first = next((i for i, v in enumerate(preds) if v == 1), None)
        if gt_first is not None and pr_first is not None:
            lat.append(pr_first - gt_first)
    return round(sum(lat) / len(lat), 3) if lat else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/localssd/svad/checkpoints/run1/latest.pt")
    ap.add_argument("--test", default="/mnt/localssd/svad/data/smartturn_en/test")
    ap.add_argument("--encoder", default="wav2vec2")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default="results/decision_sweep.json")
    args = ap.parse_args()

    if args.cache and Path(args.cache).exists():
        records = json.load(open(args.cache))
        print(f"[cache] loaded {len(records)} records from {args.cache}")
    else:
        ckpt_dir = Path(args.ckpt).parent
        cfg = load_config(ckpt_dir / "config.yaml"); cfg.encoder.kind = args.encoder
        win = WindowConfig(cfg.window.window_frames, cfg.window.stride_frames)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = FDVAD(cfg).to(dev)
        model.load_trainable_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
        print(f"[infer] {args.ckpt}")
        records = infer_manifest(model, f"{args.test}/manifest.jsonl", args.test, win, 32, dev)
        if args.cache:
            json.dump(records, open(args.cache, "w"))

    clips = clip_records_to_sequences(records)
    print(f"[data] {len(clips)} clips")

    base = sentence_accuracy(clips, 0.5, 1)
    print(f"\nBASELINE (thr=0.5,K=1): all={base['all']['accuracy']} "
          f"complete={base['complete']['accuracy']} incomplete={base['incomplete']['accuracy']}")

    grid = []
    print("\n=== §6.1 sweep: sentence accuracy (complete / incomplete / all) | stop-latency chunks ===")
    print(f"{'thr':>5} {'K':>2} | {'complete':>9} {'incompl':>9} {'all':>6} | {'lat':>5}")
    for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
        for K in (1, 2, 3, 4):
            s = sentence_accuracy(clips, thr, K)
            lat = mean_stop_latency_chunks(clips, thr, K)
            grid.append({"threshold": thr, "K": K, "sentence": s, "stop_latency_chunks": lat})
            print(f"{thr:>5} {K:>2} | {s['complete']['accuracy']:>9} {s['incomplete']['accuracy']:>9} "
                  f"{s['all']['accuracy']:>6} | {str(lat):>5}")

    best = max(grid, key=lambda g: g["sentence"]["all"]["accuracy"])
    print(f"\nBEST all-acc: thr={best['threshold']} K={best['K']} -> "
          f"all={best['sentence']['all']['accuracy']} "
          f"(complete={best['sentence']['complete']['accuracy']} incomplete={best['sentence']['incomplete']['accuracy']}) "
          f"vs baseline all={base['all']['accuracy']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"baseline": base, "grid": grid, "best": best}, open(args.out, "w"), indent=2)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
