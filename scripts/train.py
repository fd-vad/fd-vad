#!/usr/bin/env python
"""M3 training (paper §2.3, Eq. 2): last-chunk CE, cosine LR + warmup, checkpointing.

Crash-resilient: saves {trainable weights, optimizer, scheduler, step} to localssd
every --save-every steps and on exit; --resume continues from latest.pt. Encoder + LLM
base are frozen, so checkpoints are small (LoRA + adapter only).
"""
from __future__ import annotations
import argparse, json, os, sys, time, math, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fd_vad.config import load_config, dump_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD
from fd_vad.data.dataset import WindowDataset, collate
from fd_vad.eval import full_eval


def save_ckpt(path, model, opt, sched, step, extra):
    tmp = str(path) + ".tmp"
    torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched else None, "step": step,
                "rng": torch.get_rng_state(), **extra}, tmp)
    os.replace(tmp, path)  # atomic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--encoder", default="wav2vec2")
    ap.add_argument("--encoder-ckpt", default="")
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--train", default="/mnt/localssd/svad/data/smartturn_en/train")
    ap.add_argument("--val", default="/mnt/localssd/svad/data/smartturn_en/val")
    ap.add_argument("--out", default="/mnt/localssd/svad/checkpoints/run1")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--continue-keep-prob", type=float, default=0.15)
    ap.add_argument("--boundary-oversample", type=float, default=0.0,
                    help="§6.2: >0 enables WeightedRandomSampler with this factor for windows within 2 chunks of the Stop boundary")
    ap.add_argument("--window-frames", type=int, default=0, help="override W_f (100Hz frames)")
    ap.add_argument("--stride-frames", type=int, default=0, help="override S_f (M5: 16 = 160ms chunk)")
    ap.add_argument("--freeze-adapter", action="store_true", help="M5: non-joint proxy — train LoRA only")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default="", help="fine-tune: load model weights from this ckpt, fresh optimizer/schedule")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    # ---- DDP setup (torchrun sets LOCAL_RANK); single-GPU when unset ----
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp = local_rank >= 0
    if ddp:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
        world = dist.get_world_size(); rank = dist.get_rank(); is_main = rank == 0
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        world = 1; rank = 0; is_main = True

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "metrics.jsonl", "a") if is_main else None
    def log(msg):
        if is_main: print(msg, flush=True)

    cfg = load_config(args.config); cfg.encoder.kind = args.encoder
    if args.encoder_ckpt: cfg.encoder.checkpoint = args.encoder_ckpt
    if args.lora_rank: cfg.llm.lora_rank = args.lora_rank
    if args.lora_alpha: cfg.llm.lora_alpha = args.lora_alpha
    if args.window_frames: cfg.window.window_frames = args.window_frames
    if args.stride_frames: cfg.window.stride_frames = args.stride_frames
    dump_config(cfg, out / "config.yaml")
    win = WindowConfig(cfg.window.window_frames, cfg.window.stride_frames)

    model = FDVAD(cfg).to(dev)
    if args.init_from:
        ick = torch.load(args.init_from, map_location="cpu")
        model.load_trainable_state_dict(ick["model"])
        log(f"[init-from] loaded pretrained weights from {args.init_from} (step {ick.get('step')}); fresh optimizer/schedule")
    if args.freeze_adapter:
        for p in model.adapter.parameters():
            p.requires_grad_(False)
        log("[M5] adapter FROZEN (non-joint training proxy — LoRA only)")
    log(f"[params] trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.2f}M | ddp={ddp} world={world}")
    net = model
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    ds = WindowDataset(f"{args.train}/manifest.jsonl", args.train, win,
                       continue_keep_prob=args.continue_keep_prob, seed=args.seed)
    log(f"[data] train examples={len(ds)} counts={ds.label_counts()}")
    dist_sampler = None
    if ddp:
        from torch.utils.data import DistributedSampler
        dist_sampler = DistributedSampler(ds, shuffle=True, seed=args.seed, drop_last=True)
        dl = DataLoader(ds, batch_size=args.batch, sampler=dist_sampler, collate_fn=collate,
                        drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
    elif args.boundary_oversample > 0:
        from torch.utils.data import WeightedRandomSampler
        w = ds.boundary_weights(near=2, factor=args.boundary_oversample)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True)
        dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, collate_fn=collate,
                        drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)
        log(f"[data] §6.2 boundary oversample x{args.boundary_oversample} within 2 chunks of boundary")
    else:
        dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                        drop_last=True, num_workers=2, pin_memory=True, persistent_workers=True)

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    log(f"[sched] total_optim_steps={total_steps} warmup={int(args.warmup_ratio*total_steps)} (per-rank; world={world})")

    start_step = 0
    latest = out / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu")
        model.load_trainable_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ck.get("sched"): sched.load_state_dict(ck["sched"])
        start_step = ck["step"]; torch.set_rng_state(ck["rng"])
        log(f"[resume] from step {start_step}")

    net.train(); opt.zero_grad()
    gstep = start_step; micro = 0; t0 = time.time(); run_loss = []
    for epoch in range(args.epochs):
        if dist_sampler is not None:
            dist_sampler.set_epoch(epoch)
        for wav, lens, labels in dl:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=("cuda" in str(dev))):
                out_ = net(wav, lens, labels=labels)
                loss = out_.loss / args.grad_accum
            loss.backward(); run_loss.append(out_.loss.item()); micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                if gstep % 20 == 0 and is_main:
                    rl = sum(run_loss[-40:]) / len(run_loss[-40:])
                    rate = (gstep - start_step) / (time.time() - t0)
                    log(f"  step {gstep}/{total_steps} loss {rl:.4f} lr {sched.get_last_lr()[0]:.2e} {rate:.2f} it/s")
                    logf.write(json.dumps({"step": gstep, "loss": rl, "lr": sched.get_last_lr()[0]}) + "\n"); logf.flush()
                if gstep % args.save_every == 0 and is_main:
                    save_ckpt(latest, model, opt, sched, gstep, {"config": args.config})
                # periodic eval is skipped under DDP (would desync ranks); use large --eval-every
                if gstep % args.eval_every == 0 and is_main and not ddp:
                    m = full_eval(model, f"{args.val}/manifest.jsonl", args.val, win, batch_size=args.batch, device=dev)
                    cl = m["chunk_level"]["all"]; sl = m["sentence_level"]
                    log(f"  [val@{gstep}] chunk F1={cl['f1']} acc={cl['accuracy']} "
                        f"| sent complete={sl['complete']['accuracy']} incomplete={sl['incomplete']['accuracy']}")
                    logf.write(json.dumps({"step": gstep, "val": m}) + "\n"); logf.flush()
                    net.train()

    if ddp:
        import torch.distributed as dist
        dist.barrier()
    if is_main:
        save_ckpt(latest, model, opt, sched, gstep, {"config": args.config, "final": True})
        m = full_eval(model, f"{args.val}/manifest.jsonl", args.val, win, batch_size=args.batch, device=dev)
        (out / "final_val_metrics.json").write_text(json.dumps(m, indent=2))
        log("=== FINAL VAL ==="); log(json.dumps(m, indent=2))
        log(f"[done] ckpt: {latest}")
    if ddp:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
