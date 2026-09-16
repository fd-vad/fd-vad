#!/usr/bin/env python
"""Robustness variant of turnbench_submit.py: STREAMING (causal) Silero VAD gate.

Identical to turnbench_submit.py in every respect — same raw FD-VAD probs, same
gate settling (min_sil/settle), same fixed theta, same TurnBench `commit_events`
rising-edge commit — EXCEPT the speech flags feeding the gate come from Silero's
streaming `VADIterator` (512-sample chunks, stateful RNN, no lookahead) instead of
the offline whole-clip `get_speech_timestamps`. Every frame's speech flag then
depends only on audio up to that frame's end, so every committed event time is
strictly causal. This exists to show the leaderboard result does not rest on the
offline VAD's lookahead.

Outputs go to a SEPARATE directory (results/turnbench/submission_streaming_vad/)
so they can never be confused with the official offline-VAD submission files.

Supports conversation sharding (--shard-idx/--shard-num) for parallel workers;
merge the per-shard prediction files with --merge.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tb_patch  # noqa: F401  cheap datasets fingerprint (large test table)

import numpy as np
import torch
import torchaudio
from silero_vad import load_silero_vad, VADIterator

from turnbench.data import resolve_dataset, conversation, conversation_ids
from turnbench.sweep import commit_events, frame_count
from turnbench.submission import (
    SCHEMA_VERSION, Submission, ConversationPrediction, SpeakerEvents,
    validate_coverage, validate_event_times,
)

SR = 16_000
VAD_WIN = 512  # Silero's required chunk size @ 16kHz (32 ms)


def streaming_speech_flags(wav16, model, n_frames, hop):
    """Causal per-frame speech flags via streaming Silero VADIterator.

    Feeds consecutive 512-sample chunks through the stateful model in order, so a
    speech start/end at sample s is emitted using only audio ≤ s (plus Silero's
    fixed internal min_silence_duration to confirm an end — far smaller than the
    gate's own 1000 ms settling). Frame i is speech iff any of its samples fall in
    a detected speech span."""
    vad = VADIterator(model, sampling_rate=SR)
    vad.reset_states()
    N = wav16.shape[0]
    speech = np.zeros(N, dtype=bool)
    cur = None
    for s0 in range(0, N - VAD_WIN + 1, VAD_WIN):
        out = vad(wav16[s0:s0 + VAD_WIN], return_seconds=False)
        if out:
            if "start" in out:
                cur = int(out["start"])
            if "end" in out and cur is not None:
                speech[max(0, cur):min(N, int(out["end"]))] = True
                cur = None
    if cur is not None:
        speech[max(0, cur):] = True
    flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        a, b = i * hop, min((i + 1) * hop, N)
        if a < b and speech[a:b].any():
            flags[i] = True
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


def do_merge(shard_glob, dataset, out):
    ds = resolve_dataset(source=dataset, skip_audio=True)
    ids = conversation_ids(ds)
    by = {}
    for s in sorted(glob.glob(shard_glob)):
        sub = Submission.model_validate_json(Path(s).read_text())
        for p in sub.predictions:
            by[p.conversation_id] = p
    preds = [by[cid] for cid in ids]
    sub = Submission(schema_version=SCHEMA_VERSION, predictions=preds)
    validate_coverage(sub, ids)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(sub.model_dump_json(indent=2))
    n_eot = sum(len(p.speaker_1.eot) + len(p.speaker_2.eot) for p in preds)
    print(f"[merge] wrote {out} | {len(preds)} convs, {n_eot} EOT events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="raw probs-eot.json OR dir with shards")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--min-sil-ms", type=int, default=1000)
    ap.add_argument("--settle-ms", type=int, default=2500)
    ap.add_argument("--theta", type=float, default=0.0036)
    ap.add_argument("--refractory-s", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--merge", metavar="SHARD_GLOB", help="merge per-shard files into --out and exit")
    args = ap.parse_args()

    if args.merge:
        do_merge(args.merge, args.dataset, args.out)
        return

    hop = int(round(SR / args.fps))
    mnf = int(round(args.min_sil_ms / 1000 * args.fps))
    settle = int(round(args.settle_ms / 1000 * args.fps))
    raw_by = merged_raw(args.raw)
    ds = resolve_dataset(source=args.dataset)
    ids = conversation_ids(ds)
    if args.shard_num > 1:
        ids = ids[args.shard_idx::args.shard_num]
    model = load_silero_vad(onnx=False)

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
            fl = streaming_speech_flags(w, model, len(raw), hop)
            g = gate(raw, fl, mnf, settle)
            eot = commit_events(g.tolist(), args.fps, args.theta, refractory_s=args.refractory_s)
            eot = [min(t, conv.duration_s) for t in eot]
            spk_ev[spk] = SpeakerEvents(eot=eot, interruption=[])
        pred = ConversationPrediction(conversation_id=cid, speaker_1=spk_ev[1], speaker_2=spk_ev[2])
        validate_event_times(pred, conv.duration_s)
        preds.append(pred)
        print(f"  [{k+1}/{len(ids)}] {cid}", flush=True)

    sub = Submission(schema_version=SCHEMA_VERSION, predictions=preds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(sub.model_dump_json(indent=2))
    n_eot = sum(len(p.speaker_1.eot) + len(p.speaker_2.eot) for p in preds)
    print(f"[submit-streaming] wrote {args.out} | shard {args.shard_idx}/{args.shard_num}: "
          f"{len(preds)} convs, {n_eot} EOT events (streaming VAD, theta={args.theta})")


if __name__ == "__main__":
    main()
