#!/usr/bin/env python
"""M4 eval: load a trained checkpoint, report chunk-level (Table 2) + sentence-level
(Table 3) metrics on a test manifest."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.config import load_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD
from fd_vad.eval import full_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="defaults to <ckpt_dir>/config.yaml")
    ap.add_argument("--encoder", default="wav2vec2")
    ap.add_argument("--test", default="/mnt/localssd/svad/data/smartturn_en/test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1234, help="match training seed so a frozen/untrained adapter reproduces")
    ap.add_argument("--allow-missing", action="store_true", help="allow ckpt to omit frozen params (kept at seeded init)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ckpt_dir = Path(args.ckpt).parent
    cfg_path = args.config or (ckpt_dir / "config.yaml")
    cfg = load_config(cfg_path); cfg.encoder.kind = args.encoder
    win = WindowConfig(cfg.window.window_frames, cfg.window.stride_frames)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = FDVAD(cfg).to(dev)
    ck = torch.load(args.ckpt, map_location="cpu")
    model.load_trainable_state_dict(ck["model"], allow_missing=args.allow_missing)
    print(f"[eval] loaded {args.ckpt} (step {ck.get('step')})")

    m = full_eval(model, f"{args.test}/manifest.jsonl", args.test, win, batch_size=args.batch, device=dev)
    print(json.dumps(m, indent=2))
    out = args.out or (ckpt_dir / "test_metrics.json")
    Path(out).write_text(json.dumps(m, indent=2))

    cl, sl = m["chunk_level"], m["sentence_level"]
    print("\n=== SUMMARY (vs paper Table 2/3; baseline smart-turn-v3 sent-acc ~0.80 complete / 0.56 incomplete) ===")
    print(f"chunk-level Acc (all): {cl['all']['accuracy']}  F1 (all): {cl['all']['f1']}")
    print(f"complete subset : chunk-F1={cl['complete']['f1']} recall={cl['complete']['recall']} prec={cl['complete']['precision']}")
    print(f"incomplete subset: false_stop_rate={cl['incomplete']['false_stop_rate']} specificity={cl['incomplete']['specificity']}  (F1 N/A: no Stop GT)")
    print(f"sentence-level acc: complete={sl['complete']['accuracy']} incomplete={sl['incomplete']['accuracy']}")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
