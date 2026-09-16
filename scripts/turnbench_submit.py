#!/usr/bin/env python
"""Build a TurnBench predictions.json at a FROZEN operating point.

Applies the exact deployed decision rule to raw per-frame P(Stop): Silero-VAD gate
(min_sil/settle) then a fixed threshold theta committed via TurnBench's own rising-edge
rule (`commit_events`). The operating point (min_sil, theta) is chosen ONCE on dev and
applied UNCHANGED to test — the benchmark's causal-submission requirement. Interruption
is not a task FD-VAD addresses, so INT event lists are empty.

Same code path for dev and test, so a file that validates/scores on dev validates on test.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tb_patch  # noqa: F401  cheap datasets fingerprint (large test table, see module)

import numpy as np
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

from turnbench.data import resolve_dataset, conversation, conversation_ids
from turnbench.sweep import commit_events, frame_count
from turnbench.submission import (
    SCHEMA_VERSION, Submission, ConversationPrediction, SpeakerEvents,
    validate_coverage, validate_event_times,
)

SR = 16_000


def speech_flags(wav16, vad, n_frames, hop):
    ts = get_speech_timestamps(wav16, vad, sampling_rate=SR)
    flags = np.zeros(n_frames, dtype=bool)
    for seg in ts:
        lo, hi = seg["start"] // hop, min((seg["end"] - 1) // hop, n_frames - 1)
        if lo <= hi:
            flags[lo:hi + 1] = True
    return flags


def gate(raw, flags, min_sil, settle):
    g = np.zeros_like(raw)
    active, sil = False, 0
    for i in range(len(raw)):
        if flags[i]:
            active, sil = True, 0
        else:
            sil += 1
            if active and min_sil <= sil <= settle:
                g[i] = raw[i]
            if sil > settle:
                active = False
    return g


def merged_raw(raw_arg):
    p = Path(raw_arg)
    if p.is_dir():
        shards = sorted(glob.glob(str(p / "probs-eot.shard*of*.json"))) or [str(p / "probs-eot.json")]
    elif "*" in raw_arg:
        shards = sorted(glob.glob(raw_arg))
    else:
        shards = [raw_arg]
    by = {}
    for s in shards:
        for e in json.loads(Path(s).read_text())["probs"]:
            by[e["conversation_id"]] = e
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw probs-eot.json OR dir with shards")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--min-sil-ms", type=int, default=1000)
    ap.add_argument("--settle-ms", type=int, default=2500)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--refractory-s", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hop = int(round(SR / args.fps))
    mnf = int(round(args.min_sil_ms / 1000 * args.fps))
    settle = int(round(args.settle_ms / 1000 * args.fps))
    raw_by = merged_raw(args.raw)
    ds = resolve_dataset(source=args.dataset)
    ids = conversation_ids(ds)
    vad = load_silero_vad(onnx=True)

    preds = []
    for k, cid in enumerate(ids):
        conv = conversation(ds, cid)
        n = frame_count(conv.duration_s, args.fps)
        entry = raw_by[cid]
        spk_ev = {}
        for spk in (1, 2):
            raw = np.asarray(entry[f"speaker_{spk}"]["prob"], dtype=np.float32)[:n]
            w, sr = conv.audio(spk)
            w = torch.from_numpy(np.ascontiguousarray(w)).float()
            if sr != SR:
                w = torchaudio.functional.resample(w, sr, SR)
            fl = speech_flags(w, vad, len(raw), hop)
            g = gate(raw, fl, mnf, settle)
            eot = commit_events(g.tolist(), args.fps, args.theta, refractory_s=args.refractory_s)
            # clamp strictly < duration for the validator's `> duration` check
            eot = [min(t, conv.duration_s) for t in eot]
            spk_ev[spk] = SpeakerEvents(eot=eot, interruption=[])
        pred = ConversationPrediction(conversation_id=cid, speaker_1=spk_ev[1], speaker_2=spk_ev[2])
        validate_event_times(pred, conv.duration_s)
        preds.append(pred)
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(ids)}] {cid}", flush=True)

    sub = Submission(schema_version=SCHEMA_VERSION, predictions=preds)
    validate_coverage(sub, ids)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(sub.model_dump_json(indent=2))
    n_eot = sum(len(p.speaker_1.eot) + len(p.speaker_2.eot) for p in preds)
    print(f"[submit] wrote {args.out} | {len(preds)} convs, {n_eot} EOT events "
          f"(min_sil={args.min_sil_ms}ms theta={args.theta})")


if __name__ == "__main__":
    main()
