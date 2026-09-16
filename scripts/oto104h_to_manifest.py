#!/usr/bin/env python
"""Convert otoSpeech-full-duplex-turn-104h (TurnBench's in-domain train set) into FD-VAD
manifest clips, for in-domain fine-tuning.

Each conversation dir has per-speaker audio (`speaker_{1,2}_audio.wav`) and SRT annotations
(`speaker_{1,2}_annotation_a.srt`) whose cues are tagged `[Normal Turn]`, `[Overlap]`,
`[... Backchannel]`, etc. We reconstruct each speaker's turns (merge consecutive floor-holding
cues separated by < MERGE_GAP) and emit FD-VAD clips in the smart-turn style:
  - COMPLETE: audio ending shortly after a turn end, `stop_speaking_s` = the turn's speech offset.
  - INCOMPLETE: the same turn truncated mid-way, no stop (keep predicting Continue).
Clips are written as 16 kHz mono wavs; a JSONL manifest per split references them. Conversations
are split train/val by id so val is speaker/'conv-disjoint from train.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.schema import Utterance, write_manifest, expected_timeout_ms

SR = 16_000
FLOOR_LABELS = ("normal turn", "overlap")   # cues that hold/take the floor (turn-forming)
MERGE_GAP = 1.0        # s: same-speaker floor cues within this gap = one turn
MAX_CLIP = 8.0         # s: cap clip length (keep last MAX_CLIP ending at the boundary)
PRE = 0.5              # s: context before a turn start
TRAIL_MIN, TRAIL_MAX = 0.3, 1.0   # s: trailing silence appended after a complete turn

_TS = re.compile(r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)")
_LABEL = re.compile(r"\[([^\]]+)\]")


def _sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path: Path):
    """-> list of (start_s, end_s, label_lower, text)."""
    segs = []
    block = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        if line.strip() == "":
            if block:
                tsline = next((l for l in block if _TS.search(l)), None)
                if tsline:
                    g = _TS.search(tsline).groups()
                    st, en = _sec(*g[:4]), _sec(*g[4:])
                    txt = " ".join(l for l in block if not _TS.search(l) and not l.strip().isdigit())
                    lm = _LABEL.search(txt)
                    label = (lm.group(1).strip().lower() if lm else "")
                    text = _LABEL.sub("", txt, count=1).strip()
                    if en > st:
                        segs.append((st, en, label, text))
                block = []
        else:
            block.append(line)
    segs.sort(key=lambda x: x[0])
    return segs


def build_turns(segs):
    """Merge consecutive floor-holding cues within MERGE_GAP into turns -> [(start,end,text)]."""
    floor = [(s, e, t) for (s, e, lab, t) in segs if any(f in lab for f in FLOOR_LABELS)]
    turns = []
    for s, e, t in floor:
        if turns and s - turns[-1][1] < MERGE_GAP:
            ps, pe, pt = turns[-1]
            turns[-1] = (ps, e, (pt + " " + t).strip())
        else:
            turns.append((s, e, t))
    return turns


def next_activity_after(segs, t):
    nxt = [s for (s, e, lab, txt) in segs if s > t]
    return min(nxt) if nxt else t + 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="oto104h download dir")
    ap.add_argument("--out", required=True, help="manifest output root")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)
    conv_dirs = sorted([d for d in root.iterdir() if d.is_dir() and (d / "metadata.json").exists()],
                       key=lambda d: int(d.name) if d.name.isdigit() else d.name)
    if args.limit:
        conv_dirs = conv_dirs[: args.limit]
    val_ids = set(rng.sample([d.name for d in conv_dirs], max(1, int(len(conv_dirs) * args.val_frac))))

    out = Path(args.out)
    rows = {"train": [], "val": []}
    audio_dirs = {sp: (out / sp / "audio") for sp in ("train", "val")}
    for d in audio_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    n_clips = {"complete": 0, "incomplete": 0}
    for ci, cd in enumerate(conv_dirs):
        split = "val" if cd.name in val_ids else "train"
        for spk in (1, 2):
            srt = cd / f"speaker_{spk}_annotation_a.srt"
            wavp = cd / f"speaker_{spk}_audio.wav"
            if not srt.exists() or not wavp.exists():
                continue
            segs = parse_srt(srt)
            turns = build_turns(segs)
            if not turns:
                continue
            wav, sr = sf.read(wavp, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
            dur_total = len(wav) / SR
            for ti, (ts, te, text) in enumerate(turns):
                te = min(te, dur_total)
                if te <= ts:
                    continue
                # --- complete clip ---
                trail = min(max(next_activity_after(segs, te) - te, TRAIL_MIN), TRAIL_MAX)
                ce = min(te + trail, dur_total)
                cs = max(0.0, ts - PRE)
                if ce - cs > MAX_CLIP:
                    cs = ce - MAX_CLIP
                if ce - cs < 0.5:
                    continue
                clip = wav[int(cs * SR):int(ce * SR)]
                cid = f"{cd.name}_s{spk}_t{ti}_c"
                rel = f"audio/{cid}.wav"
                sf.write(audio_dirs[split] / f"{cid}.wav", clip, SR)
                rows[split].append(Utterance(
                    id=cid, sem_class="complete", text=text[:200], source_text=text[:200],
                    audio_path=rel, sample_rate=SR, duration_s=len(clip) / SR,
                    stop_speaking_s=min(te - cs, len(clip) / SR), speaker=f"{cd.name}_{spk}",
                    timeout_ms=expected_timeout_ms("complete"), source="oto104h"))
                n_clips["complete"] += 1
                # --- incomplete clip (truncate the turn before its end) ---
                if te - ts > 1.2:
                    cut = rng.uniform(ts + 0.6, te - 0.4)
                    ics = max(0.0, ts - PRE)
                    if cut - ics > MAX_CLIP:
                        ics = cut - MAX_CLIP
                    iclip = wav[int(ics * SR):int(cut * SR)]
                    if len(iclip) / SR >= 0.5:
                        iid = f"{cd.name}_s{spk}_t{ti}_i"
                        irel = f"audio/{iid}.wav"
                        sf.write(audio_dirs[split] / f"{iid}.wav", iclip, SR)
                        rows[split].append(Utterance(
                            id=iid, sem_class="incomplete", text=text[:200], source_text=text[:200],
                            audio_path=irel, sample_rate=SR, duration_s=len(iclip) / SR,
                            stop_speaking_s=len(iclip) / SR, speaker=f"{cd.name}_{spk}",
                            timeout_ms=expected_timeout_ms("incomplete"), source="oto104h"))
                        n_clips["incomplete"] += 1
        if (ci + 1) % 20 == 0:
            print(f"  [{ci+1}/{len(conv_dirs)}] clips={n_clips}", flush=True)

    for split in ("train", "val"):
        n = write_manifest(out / split / "manifest.jsonl", rows[split])
        print(f"[{split}] {n} clips -> {out/split/'manifest.jsonl'}")
    print(f"[done] totals: {n_clips} | val convs={len(val_ids)}")


if __name__ == "__main__":
    main()
