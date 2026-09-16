"""Evaluation harness (M4, PRD §4): chunk-level (Table 2) + sentence-level (Table 3).

Chunk-level: per-class confusion matrix + Recall/Precision/F1/Accuracy, computed
separately for the `complete` and `incomplete` subsets. Sentence-level: a clip is
correct iff ALL its chunk predictions match GT (paper's strict criterion).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .windowing import WindowConfig
from .data.dataset import WindowDataset, collate


@torch.no_grad()
def infer_manifest(model, manifest_path, audio_root, win: WindowConfig,
                   batch_size: int = 32, device: str = "cuda") -> list[dict]:
    """Return per-(clip,chunk) records: {id, sem_class, c, gt, pred, p_stop}."""
    ds = WindowDataset(manifest_path, audio_root, win, balance=False, continue_keep_prob=1.0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    records, pos = [], 0
    for wav, lens, labels in dl:
        out = model(wav, lens, labels=None)
        pred = out.logits2.argmax(-1).cpu().tolist()
        pstop = out.probs2[:, 1].cpu().tolist()
        for j in range(len(pred)):
            ui, c, gt, dist = ds.examples[pos + j]
            u = ds.utts[ui]
            records.append({"id": u.id, "sem_class": u.sem_class, "c": c, "dist": dist,
                            "gt": gt, "pred": pred[j], "p_stop": pstop[j]})
        pos += len(pred)
    return records


def _prf(tp, fp, fn, tn):
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    # specificity / false-Stop rate are the meaningful metrics for the incomplete
    # subset (which has no Stop-positive chunks under the S2 scheme, so recall/F1
    # are undefined there). false_stop_rate = fraction of Continue chunks wrongly Stopped.
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fsr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"recall": round(rec, 4), "precision": round(prec, 4),
            "f1": round(f1, 4), "accuracy": round(acc, 4),
            "specificity": round(spec, 4), "false_stop_rate": round(fsr, 4)}


def chunk_level_metrics(records: list[dict]) -> dict:
    """Per sem_class: confusion (for Stop=positive class) + Recall/Precision/F1/Accuracy."""
    out = {}
    for sem in ("complete", "incomplete", "all"):
        rs = records if sem == "all" else [r for r in records if r["sem_class"] == sem]
        tp = sum(r["gt"] == 1 and r["pred"] == 1 for r in rs)
        fp = sum(r["gt"] == 0 and r["pred"] == 1 for r in rs)
        fn = sum(r["gt"] == 1 and r["pred"] == 0 for r in rs)
        tn = sum(r["gt"] == 0 and r["pred"] == 0 for r in rs)
        out[sem] = {"n_chunks": len(rs),
                    "confusion": {"tp_stop": tp, "fp": fp, "fn": fn, "tn_continue": tn},
                    **_prf(tp, fp, fn, tn)}
    return out


def sentence_level_metrics(records: list[dict]) -> dict:
    """Strict: a clip is correct iff ALL chunk preds match GT. Accuracy per sem_class."""
    by_clip: dict[str, dict] = {}
    for r in records:
        d = by_clip.setdefault(r["id"], {"sem": r["sem_class"], "ok": True, "n": 0})
        d["n"] += 1
        if r["pred"] != r["gt"]:
            d["ok"] = False
    out = {}
    for sem in ("complete", "incomplete", "all"):
        cs = [d for d in by_clip.values() if sem == "all" or d["sem"] == sem]
        n = len(cs); correct = sum(d["ok"] for d in cs)
        out[sem] = {"n_clips": n, "correct": correct,
                    "accuracy": round(correct / n, 4) if n else 0.0}
    return out


def near_boundary_metrics(records: list[dict], near: int = 2) -> dict:
    """§6.2: chunk-level Stop metrics restricted to windows within `near` chunks of the
    Stop boundary (complete clips only; incomplete have dist=-1). This is the metric
    boundary-focused resampling should move."""
    rs = [r for r in records if 0 <= r.get("dist", -1) <= near]
    tp = sum(r["gt"] == 1 and r["pred"] == 1 for r in rs)
    fp = sum(r["gt"] == 0 and r["pred"] == 1 for r in rs)
    fn = sum(r["gt"] == 1 and r["pred"] == 0 for r in rs)
    tn = sum(r["gt"] == 0 and r["pred"] == 0 for r in rs)
    return {"near": near, "n_chunks": len(rs), **_prf(tp, fp, fn, tn)}


def tolerance_sentence_metrics(records: list[dict], tol_chunks: int = 1) -> dict:
    """Real-world sentence metric: a COMPLETE clip is correct if its first predicted Stop
    is within ±tol_chunks of the first GT Stop (endpoint detected within tol*320ms), with
    no Stop fired more than tol chunks early. INCOMPLETE clips correct if no Stop fired.
    Less brittle than strict all-chunk matching, which scales badly with clip length."""
    from collections import defaultdict
    clips = defaultdict(lambda: {"sem": None, "rows": []})
    for r in records:
        clips[r["id"]]["sem"] = r["sem_class"]; clips[r["id"]]["rows"].append(r)
    def first_stop(seq):
        return next((i for i, v in enumerate(seq) if v == 1), None)
    out = {"complete": [0, 0], "incomplete": [0, 0]}
    for d in clips.values():
        d["rows"].sort(key=lambda x: x["c"])
        gt = [x["gt"] for x in d["rows"]]; pr = [x["pred"] for x in d["rows"]]
        if d["sem"] == "incomplete":
            ok = (1 not in pr)
        else:
            gi, pi = first_stop(gt), first_stop(pr)
            ok = (gi is not None and pi is not None and abs(pi - gi) <= tol_chunks)
        out[d["sem"]][0] += int(ok); out[d["sem"]][1] += 1
    return {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4) if v[1] else 0.0}
            for k, v in out.items() if v[1]}


def full_eval(model, manifest_path, audio_root, win, batch_size=32, device="cuda") -> dict:
    recs = infer_manifest(model, manifest_path, audio_root, win, batch_size, device)
    return {"chunk_level": chunk_level_metrics(recs),
            "sentence_level": sentence_level_metrics(recs),
            "sentence_tolerance_1chunk": tolerance_sentence_metrics(recs, tol_chunks=1),
            "near_boundary": near_boundary_metrics(recs, near=2),
            "n_records": len(recs)}
