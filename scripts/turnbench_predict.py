#!/usr/bin/env python
"""Run FD-VAD over TurnBench conversations → per-frame P(Stop) probs (TurnBench ProbsFile).

FD-VAD is a streaming per-chunk endpoint detector: given a 2.56s audio window it emits
P(Stop) = P(turn complete). We slide that window over each speaker channel at `fps`
frames/s (window END = frame end time, causal: no audio past the timestamp is used) and
record P(Stop) per frame. The result is a TurnBench EOT probabilities file; the benchmark's
own sweep/commit/score machinery then picks the operating point and scores recall/fp/latency.

INT (interruption) is not a task FD-VAD is built for; we optionally emit probs-int = 1 - P(Stop)
(a "speaker is actively talking" proxy, same trick as the smart_turn_v3 baseline) for completeness.

Runs in the `svad` training env (torch 2.6 + fd_vad). Scoring runs in the same env via
turnbench.sweep (installed --no-deps). Uses the dataset's committed durations for the
canonical frame grid, so the emitted file validates against the scorer exactly.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path("/home/colligo/semanticVAD/src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tb_patch  # noqa: F401  cheap datasets fingerprint (large test table, see module)
from fd_vad.config import load_config
from fd_vad.models.fd_vad import FDVAD

from turnbench.data import resolve_dataset, conversation, conversation_ids
from turnbench.sweep import (
    ConversationProbs, ProbsFile, SpeakerProbs, frame_count,
)

SR = 16_000
WIN_SAMPLES = 256 * 160          # 2.56s window (256 fbank frames @ 100Hz, 160 samples/frame @ 16kHz)


def load_model(ckpt: str, config: str, device: str) -> FDVAD:
    cfg = load_config(config)
    cfg.encoder.kind = "wavlm"
    torch.manual_seed(1234)                       # match training seed (seeded init reproducibility)
    model = FDVAD(cfg).to(device)
    ck = torch.load(ckpt, map_location="cpu")
    model.load_trainable_state_dict(ck["model"])
    model.eval()
    print(f"[fd_vad] loaded {ckpt} (step {ck.get('step')})", flush=True)
    return model


@torch.no_grad()
def score_channel(model: FDVAD, wav16: torch.Tensor, n_frames: int, fps: float,
                  device: str, batch: int = 192) -> np.ndarray:
    """wav16: 1-D float32 tensor on `device` @ 16kHz. Returns P(Stop) per frame, len == n_frames.
    Frame i (0-based) covers [i/fps,(i+1)/fps); its window ENDS at (i+1)/fps (causal).

    Left-pad by one window so every frame's 2.56s window is a fixed-length slice
    padded[e : e+WIN] (ramp-up frames at the very start read the zero pad — negligible,
    only the first 2.56s of each channel). bf16 autocast matches training numerics."""
    N = wav16.shape[0]
    WIN = WIN_SAMPLES
    padded = torch.cat([wav16.new_zeros(WIN), wav16])
    ends = torch.clamp(
        torch.round((torch.arange(1, n_frames + 1, device=device) / fps) * SR).long(), 1, N)
    ar = torch.arange(WIN, device=device)
    probs = np.zeros(n_frames, dtype=np.float32)
    for b0 in range(0, n_frames, batch):
        b1 = min(b0 + batch, n_frames)
        idx = ends[b0:b1].view(-1, 1) + ar.view(1, -1)     # padded[e : e+WIN] == orig [e-WIN:e]
        inp = padded[idx]                                   # [B, WIN]
        lens = torch.full((b1 - b0,), WIN, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(inp, lens, labels=None)
        probs[b0:b1] = out.probs2[:, 1].float().cpu().numpy()
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, help="local dir of parquet shards (…/dev/data)")
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emit-int", action="store_true", help="also write probs-int.json (=1-P(Stop))")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N conversations")
    ap.add_argument("--shard-idx", type=int, default=0, help="this shard's index (multi-GPU)")
    ap.add_argument("--shard-num", type=int, default=1, help="total shards (multi-GPU)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.config, device)
    ds = resolve_dataset(source=args.dataset)
    ids = conversation_ids(ds)
    if args.limit:
        ids = ids[: args.limit]
    if args.shard_num > 1:
        ids = ids[args.shard_idx :: args.shard_num]
    print(f"[fd_vad] shard {args.shard_idx}/{args.shard_num}: {len(ids)} conversations @ fps={args.fps}", flush=True)

    resampler_cache: dict[int, torchaudio.transforms.Resample] = {}
    eot_entries, int_entries = [], []
    t0 = time.time()
    for k, cid in enumerate(ids):
        conv = conversation(ds, cid)
        n_frames = frame_count(conv.duration_s, args.fps)
        chan_probs = {}
        for spk in (1, 2):
            wav, sr = conv.audio(spk)
            w = torch.from_numpy(np.ascontiguousarray(wav)).float()
            if sr != SR:
                if sr not in resampler_cache:
                    resampler_cache[sr] = torchaudio.transforms.Resample(sr, SR)
                w = resampler_cache[sr](w)
            w = w.to(device)
            chan_probs[spk] = score_channel(model, w, n_frames, args.fps, device, args.batch)
            del w
        eot_entries.append(ConversationProbs(
            conversation_id=cid,
            speaker_1=SpeakerProbs(prob=chan_probs[1].tolist()),
            speaker_2=SpeakerProbs(prob=chan_probs[2].tolist()),
        ))
        if args.emit_int:
            int_entries.append(ConversationProbs(
                conversation_id=cid,
                speaker_1=SpeakerProbs(prob=(1.0 - chan_probs[1]).tolist()),
                speaker_2=SpeakerProbs(prob=(1.0 - chan_probs[2]).tolist()),
            ))
        el = time.time() - t0
        print(f"  [{k+1}/{len(ids)}] conv {cid} dur={conv.duration_s:.0f}s frames={n_frames} "
              f"| {el:.0f}s elapsed", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f".shard{args.shard_idx}of{args.shard_num}" if args.shard_num > 1 else ""
    eot = ProbsFile(schema_version=1, task="eot", frame_rate_hz=args.fps, probs=eot_entries)
    (out_dir / f"probs-eot{sfx}.json").write_text(eot.model_dump_json())
    print(f"[fd_vad] wrote {out_dir/('probs-eot'+sfx+'.json')}", flush=True)
    if args.emit_int:
        it = ProbsFile(schema_version=1, task="int", frame_rate_hz=args.fps, probs=int_entries)
        (out_dir / f"probs-int{sfx}.json").write_text(it.model_dump_json())
        print(f"[fd_vad] wrote {out_dir/('probs-int'+sfx+'.json')}", flush=True)


if __name__ == "__main__":
    main()
