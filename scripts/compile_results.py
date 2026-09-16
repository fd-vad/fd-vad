#!/usr/bin/env python
"""Compile all run test-metrics into a comparison table (results/RESULTS.md)."""
from __future__ import annotations
import json, sys
from pathlib import Path

# (label, path-to-test_metrics.json, note)
RUNS = [
    ("run1 wav2vec2-base 3K keep0.15", "checkpoints/run1/test_metrics.json", "first baseline"),
    ("run2 wav2vec2-large 8K r32 2ep", "checkpoints/run2/test_metrics.json", "bigger encoder — regressed"),
    ("run3 wav2vec2-base 8K natural", "checkpoints/run3/test_metrics.json", "balance fix"),
    ("run_wavlm WavLM 8K natural", "checkpoints/run_wavlm/test_metrics.json", "encoder A/B winner"),
    ("run_full_wavlm_ddp WavLM 63K", "checkpoints/run_full_wavlm_ddp/test_metrics.json", "FINAL full-data model"),
]
BASE = Path("/mnt/localssd/svad")


def load(p):
    fp = BASE / p
    if not fp.exists():
        return None
    return json.load(open(fp))


def main():
    rows = []
    for label, p, note in RUNS:
        d = load(p)
        if not d:
            rows.append((label, note, None)); continue
        cl, sl = d["chunk_level"], d["sentence_level"]
        rows.append((label, note, {
            "chunkAcc": cl["all"]["accuracy"], "chunkF1": cl["all"]["f1"],
            "compF1": cl["complete"]["f1"], "fsr": cl["incomplete"].get("false_stop_rate", "-"),
            "sentC": sl["complete"]["accuracy"], "sentI": sl["incomplete"]["accuracy"],
        }))

    lines = ["# FD-VAD replication — results (English smart-turn, shared 1000-clip test set)\n",
             "| Run | chunkAcc | chunkF1 | complete-F1 | incomp FSR | sent complete | sent incomplete | note |",
             "|---|---|---|---|---|---|---|---|"]
    for label, note, m in rows:
        if m is None:
            lines.append(f"| {label} | — | — | — | — | — | — | {note} (pending) |")
        else:
            lines.append(f"| {label} | {m['chunkAcc']} | {m['chunkF1']} | {m['compF1']} | {m['fsr']} | "
                         f"{m['sentC']} | {m['sentI']} | {note} |")
    lines += [
        "\n**Reference points (from paper / baseline):**",
        "- FD-VAD paper (own synthetic set, Zipformer, 400K samples): chunk-acc ~0.986; sentence complete ~0.888 / incomplete ~0.892.",
        "- smart-turn-v3 baseline (paper Table 3): sentence complete 0.80 / incomplete 0.56.",
        "- Note: our clips (smart-turn, 5-16s) are longer than the paper's (~3s), so strict all-chunk sentence accuracy is harder.",
    ]
    out = Path(__file__).resolve().parents[1] / "results" / "RESULTS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
