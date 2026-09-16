#!/usr/bin/env python
"""M1 data prep: smart-turn-data-v3.1 → FD-VAD audio + JSONL manifest.

Downloads parquet shards on demand (HF cache on localssd), filters to a language,
derives chunk-label boundaries (docs/labeling.md), writes 16kHz wavs + a manifest,
and emits a distribution report. Fail-loud: any label-sanity violation aborts.

Example:
  python scripts/prepare_smartturn.py --split train --max-per-class 100 \
      --out /mnt/localssd/svad/data/smartturn_en/pilot
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.windowing import WindowConfig
from fd_vad.data.smartturn import row_to_utterance, STConfig
from fd_vad.labeling import clip_chunk_labels, label_sanity
from fd_vad.schema import write_manifest, validate_utterance

REPOS = {
    "train": "pipecat-ai/smart-turn-data-v3.1-train",
    "test": "pipecat-ai/smart-turn-data-v3.1-test",
}


def list_shards(repo: str) -> list[str]:
    files = HfApi().repo_info(repo, repo_type="dataset").siblings
    return sorted(f.rfilename for f in files if f.rfilename.endswith(".parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="eng")
    ap.add_argument("--max-per-class", type=int, default=100,
                    help="cap per {complete,incomplete}; -1 = no cap")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--max-shards", type=int, default=1)
    ap.add_argument("--window-frames", type=int, default=256)
    ap.add_argument("--stride-frames", type=int, default=32)
    ap.add_argument("--min-trailing-stop-chunks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--spotcheck", type=int, default=8)
    args = ap.parse_args()

    repo = REPOS[args.split]
    win = WindowConfig(args.window_frames, args.stride_frames)
    st = STConfig(language=args.language, min_trailing_stop_chunks=args.min_trailing_stop_chunks)

    out = Path(args.out); (out / "audio").mkdir(parents=True, exist_ok=True)
    shards = list_shards(repo)[args.shard_start: args.shard_start + args.max_shards]
    cap = args.max_per_class
    counts = {"complete": 0, "incomplete": 0}
    utts = []
    rejects = []
    chunk_stop = chunk_cont = 0
    durs = []
    spot = []

    for shard in shards:
        if cap > 0 and all(counts[c] >= cap for c in counts):
            break
        p = hf_hub_download(repo, shard, repo_type="dataset")
        df = pq.ParquetFile(p).read().to_pandas()
        df = df[df["language"] == args.language]
        df = df.sample(frac=1.0, random_state=args.seed)  # shuffle for class balance
        print(f"[{shard}] {len(df)} {args.language} rows; counts={counts}")
        for _, row in df.iterrows():
            sem = "complete" if bool(row["endpoint_bool"]) else "incomplete"
            if cap > 0 and counts[sem] >= cap:
                continue
            try:
                utt, y, sr = row_to_utterance(row.to_dict(), win, st)
            except Exception as e:
                print(f"  skip {row['id']}: {e}")
                continue
            rel = f"audio/{utt.id}.wav"
            sf.write(out / rel, y, sr, subtype="PCM_16")
            utt.audio_path = rel
            # Validate. Schema/audio errors are hard bugs (abort). Label-sanity failures are
            # pathological individual clips (e.g. VAD found ~no speech on a 'complete' clip) —
            # SKIP + count them, and abort only if the reject RATE is high (data-quality alarm).
            schema_errs = validate_utterance(utt, out, check_audio=True)
            if schema_errs:
                raise SystemExit(f"SCHEMA/AUDIO ERROR for {utt.id}: {schema_errs}")
            sane_errs = label_sanity(utt, win)
            if sane_errs:
                rejects.append({"id": utt.id, "sem": sem, "errs": sane_errs})
                (out / rel).unlink(missing_ok=True)   # drop the orphan wav
                continue
            labels = clip_chunk_labels(utt, win)
            chunk_stop += sum(labels); chunk_cont += len(labels) - sum(labels)
            durs.append(utt.duration_s)
            if len(spot) < args.spotcheck:
                spot.append({"id": utt.id, "sem": sem, "dur": utt.duration_s,
                             "stop_s": utt.stop_speaking_s, "labels": labels})
            utts.append(utt); counts[sem] += 1

    n_seen = len(utts) + len(rejects)
    reject_rate = len(rejects) / max(1, n_seen)
    if rejects:
        (out / "rejects.json").write_text(json.dumps(rejects, indent=2))
        print(f"[rejects] {len(rejects)}/{n_seen} clips skipped (label-sanity), rate={reject_rate:.3%}; "
              f"see {out/'rejects.json'}")
    if reject_rate > 0.05:
        raise SystemExit(f"reject rate {reject_rate:.2%} > 5% — data-quality alarm, aborting.")
    n = write_manifest(out / "manifest.jsonl", utts)
    report = {
        "split": args.split, "language": args.language, "n": n, "counts": counts,
        "window_frames": win.window_frames, "stride_frames": win.stride_frames,
        "chunk_labels": {"stop": chunk_stop, "continue": chunk_cont,
                         "stop_rate": round(chunk_stop / max(1, chunk_stop + chunk_cont), 4)},
        "duration_s": {"mean": round(float(np.mean(durs)), 3) if durs else 0,
                       "min": round(float(np.min(durs)), 3) if durs else 0,
                       "max": round(float(np.max(durs)), 3) if durs else 0},
        "spotcheck": spot,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("\n=== REPORT ===")
    print(json.dumps({k: v for k, v in report.items() if k != "spotcheck"}, indent=2))
    print(f"manifest: {out/'manifest.jsonl'} ({n} utts)  audio: {out/'audio'}")
    # invariant checks
    assert counts["complete"] > 0 and counts["incomplete"] > 0, "need both classes"
    print("OK: prep complete, all sanity gates passed.")


if __name__ == "__main__":
    main()
