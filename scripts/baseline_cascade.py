#!/usr/bin/env python
"""Cascaded ASR->LLM baseline (paper's Paraformer+TEN analog): the 'semantic but
NON-streaming, ASR-dependent' class that FD-VAD claims to beat on latency.

Whisper (transformers) transcribes each test clip; GPT-4o-mini (via the Adobe LLM proxy)
judges complete vs incomplete from the transcript. Reports per-class accuracy on the SAME
test set as scripts/baselines.py, so all baselines are directly comparable.
"""
from __future__ import annotations
import argparse, json, sys, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import requests

TEST = "/mnt/localssd/svad/data/smartturn_en/test"
SECRETS = "/home/colligo/secrets.txt"


def get_proxy():
    url = [l for l in open(SECRETS) if "API URl:" in l][0].split("API URl:")[1].strip().rstrip("/")
    key = [l for l in open(SECRETS) if l.startswith("Key:")][0].split("Key:")[1].strip()
    return url, key


def transcribe_all(utts, device="cuda"):
    import torch
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base.en",
                   device=0 if device == "cuda" else -1, torch_dtype=torch.float16)
    texts = []
    for i, u in enumerate(utts):
        y, sr = sf.read(f"{TEST}/{u['audio_path']}", dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        out = asr({"array": y, "sampling_rate": sr})
        texts.append(out["text"].strip())
        if (i + 1) % 200 == 0: print(f"  asr {i+1}/{len(utts)}", flush=True)
    return texts


JUDGE_SYS = ("You judge whether a speaker has FINISHED their conversational turn. "
             "Given a (possibly cut-off) transcript of what they said, decide if it is a "
             "COMPLETE thought/turn (they are done, it's the assistant's turn to respond) or "
             "INCOMPLETE (they were interrupted mid-thought and would keep talking). "
             "Answer with exactly one word: complete or incomplete.")


def judge_one(url, key, text):
    """Returns (pred, ok). Handles 429 rate-limits with exponential backoff (the proxy
    rate-limits per user, so we must retry 429s, not treat them as failures)."""
    text = text if text else "(silence)"
    for attempt in range(8):
        try:
            r = requests.post(f"{url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "gpt-4o-mini", "temperature": 0, "max_tokens": 3,
                      "messages": [{"role": "system", "content": JUDGE_SYS},
                                   {"role": "user", "content": f'Transcript: "{text}"'}]},
                timeout=40)
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 20)); continue
            if r.status_code != 200:
                time.sleep(1); continue
            ans = r.json()["choices"][0]["message"]["content"].strip().lower()
            return (1 if ans.startswith("complete") else 0), True
        except Exception:
            time.sleep(1.5)
    return 0, False


def per_class(preds, gts):
    n_c = sum(g for g in gts); n_i = len(gts) - n_c
    cc = sum(1 for p, g in zip(preds, gts) if g == 1 and p == 1)
    ii = sum(1 for p, g in zip(preds, gts) if g == 0 and p == 0)
    return {"complete": round(cc / n_c, 4), "incomplete": round(ii / n_i, 4),
            "overall": round((cc + ii) / len(gts), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0=all test clips")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="results/baseline_cascade.json")
    args = ap.parse_args()
    utts = [json.loads(l) for l in open(f"{TEST}/manifest.jsonl")]
    if args.limit: utts = utts[:args.limit]
    gts = [1 if u["sem_class"] == "complete" else 0 for u in utts]
    # cache transcripts so we never redo ASR when re-judging (rate-limit reruns are cheap)
    tcache = Path("results/cascade_transcripts.json")
    cached = json.load(open(tcache)) if tcache.exists() else {}
    need = [u for u in utts if u["id"] not in cached]
    if need:
        print(f"[cascade] transcribing {len(need)} clips with Whisper-base.en...")
        newt = transcribe_all(need)
        cached.update({u["id"]: t for u, t in zip(need, newt)})
        tcache.parent.mkdir(exist_ok=True, parents=True); json.dump(cached, open(tcache, "w"))
    else:
        print(f"[cascade] using cached transcripts for all {len(utts)} clips")
    texts = [cached[u["id"]] for u in utts]
    url, key = get_proxy()
    print(f"[cascade] judging with gpt-4o-mini (workers={args.workers}, 429-backoff)...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda t: judge_one(url, key, t), texts))
    preds = [p for p, ok in res]; fails = sum(1 for _, ok in res if not ok)
    print(f"[cascade] GPT failures: {fails}/{len(res)}")
    # exclude failed calls from scoring (don't count a failed call as a wrong prediction)
    keep = [i for i, (_, ok) in enumerate(res) if ok]
    preds_k = [preds[i] for i in keep]; gts_k = [gts[i] for i in keep]
    m = per_class(preds_k, gts_k)
    m["n"] = len(keep); m["n_failed"] = fails
    oks = [ok for _, ok in res]
    json.dump({"cascade_whisper_gpt4omini": m,
               "samples": [{"gt": g, "pred": p, "ok": ok, "text": t[:100]}
                           for g, (p, ok), t in zip(gts, res, texts)][:30]},
              open(args.out, "w"), indent=2)
    # per-clip preds for bootstrap CI (+ ok mask so failed calls are excluded)
    json.dump({"preds": preds, "gts": gts, "ok": oks, "texts": texts},
              open(args.out.replace(".json", "_preds.json"), "w"))
    print("=== cascade (Whisper->GPT-4o-mini) per-class accuracy ===")
    print(m)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
