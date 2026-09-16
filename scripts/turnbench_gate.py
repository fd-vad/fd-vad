#!/usr/bin/env python
"""VAD-gated re-scoring of FD-VAD's TurnBench probs (Phase 1 improvement).

The competitive endpointer baselines (e.g. smart_turn_v3) don't query their model every
frame — they run a VAD + settling pipeline: only emit an endpoint score after speech has
been followed by a minimum silence, and hold otherwise. That suppresses false fires during
active speech and short mid-turn pauses. We replicate that as a POST-PROCESS over the raw
per-frame P(Stop) we already computed: a Silero-VAD pass gives per-frame speech flags, and a
settling gate zeroes P(Stop) except in the [min_sil, settle] silence window after a speech
segment. The threshold is still chosen by TurnBench's own dev sweep.

min_sil is the key fp knob: requiring more trailing silence before a fire is allowed
suppresses short mid-turn pauses (at the cost of latency). We grid it on dev.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

from turnbench.data import resolve_dataset, conversation, conversation_ids
from turnbench.sweep import ProbsFile, ConversationProbs, SpeakerProbs, sweep, operating_point

SR = 16_000


def speech_flags(wav16: torch.Tensor, vad, n_frames: int, hop: int) -> np.ndarray:
    ts = get_speech_timestamps(wav16, vad, sampling_rate=SR)
    flags = np.zeros(n_frames, dtype=bool)
    for seg in ts:
        lo = seg["start"] // hop
        hi = min((seg["end"] - 1) // hop, n_frames - 1)
        if lo <= hi:
            flags[lo:hi + 1] = True
    return flags


def gate(raw: np.ndarray, flags: np.ndarray, min_sil: int, settle: int) -> np.ndarray:
    """Zero P(Stop) except during the [min_sil, settle]-frame silence window after speech."""
    g = np.zeros_like(raw)
    active = False
    sil = 0
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="merged raw probs-eot.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--min-sil-ms", nargs="+", type=int, default=[0, 240, 400, 640, 1000])
    ap.add_argument("--settle-ms", type=int, default=2500)
    ap.add_argument("--out", required=True, help="where to write the best gated probs-eot.json")
    args = ap.parse_args()

    hop = int(round(SR / args.fps))
    raw = json.loads(Path(args.raw).read_text())
    raw_by_id = {e["conversation_id"]: e for e in raw["probs"]}
    ds = resolve_dataset(source=args.dataset)
    ids = conversation_ids(ds)
    vad = load_silero_vad(onnx=True)

    # per-channel: (raw_prob, speech_flags), computed once
    print("[gate] computing Silero speech flags ...", flush=True)
    cache: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    t0 = time.time()
    for k, cid in enumerate(ids):
        conv = conversation(ds, cid)
        entry = raw_by_id[cid]
        cache[cid] = {}
        for spk in (1, 2):
            rp = np.asarray(entry[f"speaker_{spk}"]["prob"], dtype=np.float32)
            w, sr = conv.audio(spk)
            w = torch.from_numpy(np.ascontiguousarray(w)).float()
            if sr != SR:
                w = torchaudio.functional.resample(w, sr, SR)
            fl = speech_flags(w, vad, len(rp), hop)
            cache[cid][spk] = (rp, fl)
        print(f"  [{k+1}/{len(ids)}] conv {cid} | {time.time()-t0:.0f}s", flush=True)

    settle = int(round(args.settle_ms / 1000 * args.fps))
    print(f"\n[gate] sweeping configs (settle={args.settle_ms}ms={settle}f)", flush=True)
    print(f"{'min_sil_ms':>10}  {'rec@fp.10':>9} {'fp':>6}  {'rec@fp.15':>9} {'fp':>6}")
    best = None
    for ms in args.min_sil_ms:
        mnf = int(round(ms / 1000 * args.fps))
        entries = []
        for cid in ids:
            g1 = gate(*cache[cid][1], mnf, settle)
            g2 = gate(*cache[cid][2], mnf, settle)
            entries.append(ConversationProbs(
                conversation_id=cid,
                speaker_1=SpeakerProbs(prob=g1.tolist()),
                speaker_2=SpeakerProbs(prob=g2.tolist())))
        pf = ProbsFile(schema_version=1, task="eot", frame_rate_hz=args.fps, probs=entries)
        rows = sweep(pf, ds)
        op1 = operating_point(rows, fp_budget=0.10)
        op15 = operating_point(rows, fp_budget=0.15)
        r1 = f"{op1.recall:.3f}" if op1 else "  —  "
        f1 = f"{op1.fp_rate:.3f}" if op1 else "  —  "
        r15 = f"{op15.recall:.3f}" if op15 else "  —  "
        f15 = f"{op15.fp_rate:.3f}" if op15 else "  —  "
        print(f"{ms:>10}  {r1:>9} {f1:>6}  {r15:>9} {f15:>6}", flush=True)
        score = (op1.recall if op1 else 0.0)
        if best is None or score > best[0]:
            best = (score, ms, pf, op1, op15)

    _, best_ms, best_pf, bop1, bop15 = best
    Path(args.out).write_text(best_pf.model_dump_json())
    print(f"\n[gate] best min_sil={best_ms}ms -> rec@.10={bop1.recall:.3f} rec@.15={bop15.recall:.3f}")
    print(f"[gate] wrote {args.out}")


if __name__ == "__main__":
    main()
