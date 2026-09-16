#!/usr/bin/env python
"""Score FD-VAD on TurnBench dev and build an apples-to-apples dev leaderboard.

- Merges our sharded probs → probs-{eot,int}.json.
- Sweeps our per-frame probs (turnbench.sweep) → operating point = highest recall at
  fp_rate ≤ budget (the same rule baselines used to pick their committed dev point).
- Scores every baseline's committed predictions-dev.json (turnbench.score) on the same
  dev gold.
- Prints a dev leaderboard ranked by EOT recall (qualifiers fp≤budget first) and writes
  a JSON summary.

Runs in the svad env (turnbench installed --no-deps). Local dataset dir avoids the
remote HfFileSystem path.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from turnbench.data import resolve_dataset
from turnbench.sweep import load_probs, operating_point, sweep
from turnbench.submission import load_submission
from turnbench.score import score_submission

BASELINES = Path("/mnt/localssd/svad/turnbench/baselines")


def merge_shards(out_dir: str, task: str) -> Path:
    shards = sorted(glob.glob(f"{out_dir}/probs-{task}.shard*of*.json"))
    outp = Path(out_dir) / f"probs-{task}.json"
    if not shards:
        return outp  # already whole
    merged = None
    probs: list = []
    for s in shards:
        d = json.loads(Path(s).read_text())
        merged = merged or {k: d[k] for k in ("schema_version", "task", "frame_rate_hz")}
        probs += d["probs"]
    merged["probs"] = probs
    outp.write_text(json.dumps(merged))
    return outp


def norm_sweeprow(op):
    if op is None:
        return None
    return {"recall": op.recall, "fp_rate": op.fp_rate,
            "lat_p50": op.lat_p50, "theta": op.theta}


def norm_taskscore(ts):
    lat = ts.latency()
    return {"recall": ts.recall, "fp_rate": ts.fp_rate, "lat_p50": lat.p50}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="dir with our (sharded) probs")
    ap.add_argument("--dataset", required=True, help="local dev parquet dir (…/dev/data)")
    ap.add_argument("--fp-budget", type=float, default=0.1)
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args()

    ds = resolve_dataset(source=args.dataset)
    results: dict[str, dict] = {}

    # ---- ours: sweep probs → operating point ----
    for task in ("eot", "int"):
        merged = merge_shards(args.out_dir, task)
        if not merged.exists():
            continue
        probs = load_probs(merged)
        rows = sweep(probs, ds)
        op = operating_point(rows, fp_budget=args.fp_budget)
        results.setdefault("FD-VAD (ours)", {})[task] = norm_sweeprow(op)

    # ---- baselines: score committed dev predictions ----
    for d in sorted(BASELINES.glob("*/")):
        f = d / "predictions-dev.json"
        if not f.exists():
            continue
        sc = score_submission(load_submission(f), ds)
        results[d.name] = {"eot": norm_taskscore(sc.task_eot),
                           "int": norm_taskscore(sc.task_int)}

    # ---- print leaderboard (rank by EOT recall; qualifiers fp≤budget first) ----
    def sort_key(item):
        eot = item[1].get("eot")
        if not eot:
            return (2, 0.0)
        q = 0 if (eot["fp_rate"] == eot["fp_rate"] and eot["fp_rate"] <= args.fp_budget) else 1
        return (q, -eot["recall"])

    def fmt(x):
        if not x or x.get("recall") != x.get("recall"):
            return f"{'—':>7} {'—':>6} {'—':>7}"
        return f"{x['recall']:7.3f} {x['fp_rate']:6.3f} {x['lat_p50']:6.0f}ms"

    print(f"\n=== TurnBench DEV leaderboard (fp budget {args.fp_budget}) ===")
    print(f"{'model':<30}  {'EOT rec':>7} {'fp':>6} {'lat50':>8}   {'INT rec':>7} {'fp':>6} {'lat50':>8}   {'qual':>4}")
    for name, r in sorted(results.items(), key=sort_key):
        eot, intt = r.get("eot"), r.get("int")
        q = ""
        if eot and eot["recall"] == eot["recall"]:
            q = "✓" if eot["fp_rate"] <= args.fp_budget else "over"
        star = "  <== OURS" if name.startswith("FD-VAD") else ""
        print(f"{name:<30}  {fmt(eot)}   {fmt(intt)}   {q:>4}{star}")

    out = args.summary_out or f"{args.out_dir}/dev_leaderboard.json"
    Path(out).write_text(json.dumps(results, indent=2, default=lambda o: None))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
