#!/usr/bin/env python
"""Rigor add-ons: bootstrap 95% CIs on per-class/overall accuracy for every model, and
per-sub-domain (dataset source) accuracy for our model (a cross-domain robustness signal,
since we lack a true OOD set)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

TEST = "/mnt/localssd/svad/data/smartturn_en/test"
RES = Path(__file__).resolve().parents[1] / "results"


def acc_by_class(gts, preds, idx):
    g = np.array(gts)[idx]; p = np.array(preds)[idx]
    c = (g == 1); i = (g == 0)
    comp = ((p[c] == 1).mean()) if c.any() else 0
    inc = ((p[i] == 0).mean()) if i.any() else 0
    ov = (p == g).mean()
    return comp, inc, ov


def boot(gts, preds, n=2000, seed=0):
    rng = np.random.default_rng(seed); N = len(gts)
    comps, incs, ovs = [], [], []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        c, i, o = acc_by_class(gts, preds, idx)
        comps.append(c); incs.append(i); ovs.append(o)
    def ci(a): return (round(float(np.mean(a)), 3), round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3))
    return {"complete": ci(comps), "incomplete": ci(incs), "overall": ci(ovs)}


def main():
    models = {}
    bp = json.load(open(RES / "baselines_preds.json"))
    for k, rows in bp.items():
        models[k] = {r[0]: (r[1], r[2]) for r in rows}   # id -> (gt,pred)
    cp = RES / "baseline_cascade_preds.json"
    if cp.exists():
        d = json.load(open(cp))
        # cascade preds were saved in manifest order; drop failed GPT calls via ok mask
        utt_ids = [json.loads(l)["id"] for l in open(f"{TEST}/manifest.jsonl")][:len(d["preds"])]
        oks = d.get("ok", [True] * len(d["preds"]))
        models["cascade_whisper_gpt4omini"] = {i: (g, p) for i, g, p, ok in
                                               zip(utt_ids, d["gts"], d["preds"], oks) if ok}

    # bootstrap each model over ITS OWN clips (cascade may have fewer after dropping failures)
    common = set.intersection(*[set(m.keys()) for m in models.values() if m])
    print(f"[bootstrap] {len(common)} common test clips across {len(models)} models\n")
    print(f"{'model':<28}{'complete (95% CI)':>24}{'incomplete (95% CI)':>26}{'overall (95% CI)':>24}")
    ids = sorted(common)
    ci_out = {}
    for name, m in models.items():
        gts = [m[i][0] for i in ids]; preds = [m[i][1] for i in ids]
        c = boot(gts, preds); ci_out[name] = c
        f = lambda t: f"{t[0]:.3f} [{t[1]:.3f},{t[2]:.3f}]"
        print(f"{name:<28}{f(c['complete']):>24}{f(c['incomplete']):>26}{f(c['overall']):>24}")

    # per-sub-domain accuracy for ours
    sem = {}; src = {}
    for l in open(f"{TEST}/manifest.jsonl"):
        u = json.loads(l); sem[u["id"]] = u["sem_class"]
        src[u["id"]] = (u.get("extra") or {}).get("dataset", u.get("speaker", "?"))
    ours = models.get("fd_vad_ours", {})
    from collections import defaultdict
    bysrc = defaultdict(lambda: [0, 0])
    for i, (g, p) in ours.items():
        bysrc[src.get(i, "?")][0] += int(p == g); bysrc[src.get(i, "?")][1] += 1
    print("\n[per sub-domain] FD-VAD (ours) overall endpoint accuracy by dataset source:")
    for s, (ok, n) in sorted(bysrc.items(), key=lambda x: -x[1][1]):
        if n >= 10: print(f"  {s:<22} n={n:<4} acc={ok/n:.3f}")

    json.dump(ci_out, open(RES / "eval_confidence_intervals.json", "w"), indent=2)
    print(f"\n[wrote] {RES/'eval_confidence_intervals.json'}")


if __name__ == "__main__":
    main()
