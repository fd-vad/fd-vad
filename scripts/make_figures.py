#!/usr/bin/env python
"""Generate paper figures into tech_report/figures/:
  fig1: §6.1 accuracy-vs-latency tradeoff (confidence-debounce sweep on final model)
  fig2: length-stratified strict complete-sentence accuracy (why the strict metric is length-bound)
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "tech_report" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
CK = "/mnt/localssd/svad/checkpoints/run_full_wavlm_ddp"
TEST = "/mnt/localssd/svad/data/smartturn_en/test"


def fig1_acc_vs_latency():
    sweep = ROOT / "results" / "run_full_decision_sweep.json"
    if not sweep.exists():
        print("no sweep json; skip fig1"); return
    grid = json.load(open(sweep))["grid"]
    g = [x for x in grid if x["K"] == 1 and x.get("stop_latency_chunks") is not None]
    g.sort(key=lambda x: x["stop_latency_chunks"])
    lat = [x["stop_latency_chunks"] * 320 for x in g]   # chunks -> ms
    comp = [x["sentence"]["complete"]["accuracy"] for x in g]
    inc = [x["sentence"]["incomplete"]["accuracy"] for x in g]
    ov = [x["sentence"]["all"]["accuracy"] for x in g]
    plt.figure(figsize=(6, 4))
    plt.plot(lat, comp, "o-", label="complete", color="#2a6f97")
    plt.plot(lat, inc, "s-", label="incomplete", color="#e26d5c")
    plt.plot(lat, ov, "^--", label="overall", color="#555")
    plt.xlabel("added stop latency on complete clips (ms)")
    plt.ylabel("sentence accuracy")
    plt.title("§6.1 Confidence-debounce: accuracy vs latency (K=1)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "fig1_accuracy_vs_latency.png", dpi=150); plt.close()
    print("wrote fig1")


def fig2_length_stratified():
    recs = json.load(open(f"{CK}/test_records.json"))
    dur = {json.loads(l)["id"]: json.loads(l)["duration_s"] for l in open(f"{TEST}/manifest.jsonl")}
    clips = defaultdict(lambda: {"sem": None, "rows": []})
    for r in recs:
        clips[r["id"]]["sem"] = r["sem_class"]; clips[r["id"]]["rows"].append(r)
    bins = [(0, 4), (4, 7), (7, 11), (11, 99)]
    labels = ["0-4s", "4-7s", "7-11s", "11-16s"]
    comp_acc, inc_acc = [], []
    for lo, hi in bins:
        cc = [c for cid, c in clips.items() if lo <= dur.get(cid, 0) < hi and c["sem"] == "complete"]
        ii = [c for cid, c in clips.items() if lo <= dur.get(cid, 0) < hi and c["sem"] == "incomplete"]
        def acc(cs): return np.mean([all(x["pred"] == x["gt"] for x in c["rows"]) for c in cs]) if cs else 0
        comp_acc.append(acc(cc)); inc_acc.append(acc(ii))
    x = np.arange(len(labels)); w = 0.38
    plt.figure(figsize=(6, 4))
    plt.bar(x - w/2, comp_acc, w, label="complete", color="#2a6f97")
    plt.bar(x + w/2, inc_acc, w, label="incomplete", color="#e26d5c")
    plt.axhline(0.888, ls=":", color="#2a6f97", alpha=0.7, label="paper complete (ref)")
    plt.xticks(x, labels); plt.ylim(0, 1.05)
    plt.xlabel("clip length"); plt.ylabel("strict all-chunk sentence accuracy")
    plt.title("Strict sentence accuracy vs clip length")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(FIG / "fig2_length_stratified.png", dpi=150); plt.close()
    print("wrote fig2")


if __name__ == "__main__":
    fig1_acc_vs_latency()
    fig2_length_stratified()
    print(f"figures in {FIG}")
