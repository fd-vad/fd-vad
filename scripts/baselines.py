#!/usr/bin/env python
"""Baseline comparison (paper Table 3 style) on our English smart-turn TEST set.

Per-clip binary decision: complete (turn ended) vs incomplete. Reports per-class accuracy
(= recall per class), matching the paper's sentence-level protocol, for:
  - smart-turn-v3.1 (pipecat, Whisper-tiny+linear, ONNX) — the paper's own baseline, in-domain here
  - energy-VAD / trailing-silence threshold — acoustic-only lower bound (no semantics)
  - FD-VAD (ours): per-clip END decision (last-chunk prediction) — apples-to-apples single decision
Also prints our STRICT all-chunk sentence accuracy for reference.
"""
from __future__ import annotations
import argparse, json, sys, io
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.data.vad import energy_vad_offset

TEST = "/mnt/localssd/svad/data/smartturn_en/test"


def load_test():
    utts = []
    for line in open(f"{TEST}/manifest.jsonl"):
        d = json.loads(line)
        utts.append(d)
    return utts


def per_class_acc(preds, gts):
    """preds/gts: 1=complete, 0=incomplete. Returns dict of per-class recall + overall."""
    import collections
    c = collections.Counter()
    for p, g in zip(preds, gts):
        c[(g, p)] += 1
    n_c = sum(1 for g in gts if g == 1); n_i = sum(1 for g in gts if g == 0)
    comp_acc = c[(1, 1)] / n_c if n_c else 0
    inc_acc = c[(0, 0)] / n_i if n_i else 0
    overall = (c[(1, 1)] + c[(0, 0)]) / len(gts)
    return {"complete": round(comp_acc, 4), "incomplete": round(inc_acc, 4),
            "overall": round(overall, 4), "n_complete": n_c, "n_incomplete": n_i}


PREDS = {}  # model -> list of (id, gt, pred), filled by each runner for bootstrap CIs


def run_smart_turn(utts):
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    # verified against smart-turn's own inference.py: chunk_length=8, do_normalize=True,
    # last-8s of audio, and the ONNX output is ALREADY a probability (no sigmoid).
    onnx = hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.1-gpu.onnx")
    sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    fe = WhisperFeatureExtractor(chunk_length=8)
    N = 8 * 16000
    preds, gts = [], []
    for u in utts:
        y, sr = sf.read(f"{TEST}/{u['audio_path']}", dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        y = y[-N:] if len(y) > N else y                          # last 8s (endpoint at end)
        feat = fe(y, sampling_rate=16000, return_tensors="np", padding="max_length",
                  max_length=N, truncation=True, do_normalize=True).input_features.astype(np.float32)
        p = sess.run(None, {"input_features": feat})[0][0].item()   # already a probability
        preds.append(1 if p > 0.5 else 0)
        gts.append(1 if u["sem_class"] == "complete" else 0)
    PREDS["smart_turn_v3.1"] = [(u["id"], g, p) for u, g, p in zip(utts, gts, preds)]
    return per_class_acc(preds, gts)


def run_energy_vad(utts, sweep=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8)):
    """Trailing-silence threshold: predict complete if trailing silence >= T (acoustic-only)."""
    trail, gts = [], []
    for u in utts:
        y, sr = sf.read(f"{TEST}/{u['audio_path']}", dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        _, off = energy_vad_offset(y, sr)
        trail.append(len(y) / sr - off)
        gts.append(1 if u["sem_class"] == "complete" else 0)
    best = None; best_preds = None
    for T in sweep:
        preds = [1 if t >= T else 0 for t in trail]
        m = per_class_acc(preds, gts); m["threshold"] = T
        if best is None or m["overall"] > best["overall"]:
            best = m; best_preds = preds
    PREDS["energy_vad"] = [(u["id"], g, p) for u, g, p in zip(utts, gts, best_preds)]
    return best


def run_ours(utts, records_path):
    """FD-VAD per-clip END decision = last-chunk prediction; also strict all-chunk sent-acc."""
    recs = json.load(open(records_path))
    from collections import defaultdict
    clips = defaultdict(list)
    for r in recs: clips[r["id"]].append(r)
    end_preds, gts, strict_ok = [], [], []
    sem = {u["id"]: u["sem_class"] for u in utts}
    ids_order = []
    for cid, rs in clips.items():
        rs.sort(key=lambda x: x["c"])
        end_preds.append(rs[-1]["pred"])                          # decision at end of utterance
        gts.append(1 if sem[cid] == "complete" else 0)
        strict_ok.append(all(x["pred"] == x["gt"] for x in rs))
        ids_order.append(cid)
    PREDS["fd_vad_ours"] = list(zip(ids_order, gts, end_preds))
    m = per_class_acc(end_preds, gts)
    # strict per-class
    import collections
    sc = collections.Counter()
    for ok, g in zip(strict_ok, gts): sc[(g, ok)] += 1
    n_c = sum(g for g in gts); n_i = len(gts) - n_c
    m["strict_complete"] = round(sc[(1, True)] / n_c, 4)
    m["strict_incomplete"] = round(sc[(0, True)] / n_i, 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="/mnt/localssd/svad/checkpoints/run_full_wavlm_ddp/test_records.json")
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()
    utts = load_test()
    print(f"[test] {len(utts)} clips")
    out = {}
    print("running energy-VAD baseline..."); out["energy_vad"] = run_energy_vad(utts); print(" ", out["energy_vad"])
    print("running FD-VAD (ours)...");   out["fd_vad_ours"] = run_ours(utts, args.records); print(" ", out["fd_vad_ours"])
    print("running smart-turn-v3.1 (may take a few min)..."); out["smart_turn_v3.1"] = run_smart_turn(utts); print(" ", out["smart_turn_v3.1"])
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    json.dump(out, open(args.out, "w"), indent=2)
    json.dump({k: [[i, g, p] for i, g, p in v] for k, v in PREDS.items()},
              open(args.out.replace(".json", "_preds.json"), "w"))
    print("\n=== Table 3 style (per-class accuracy) ===")
    print(f"{'model':<26}{'complete':>10}{'incomplete':>12}{'overall':>9}")
    for k in ("energy_vad", "smart_turn_v3.1", "fd_vad_ours"):
        m = out[k]; print(f"{k:<26}{m['complete']:>10}{m['incomplete']:>12}{m['overall']:>9}")
    o = out["fd_vad_ours"]
    print(f"{'fd-vad (strict all-chunk)':<26}{o['strict_complete']:>10}{o['strict_incomplete']:>12}")
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
