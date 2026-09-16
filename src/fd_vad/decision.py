"""§6.1 Confidence-gated dynamic decision (inference-time only, no retraining).

Replaces the hard per-chunk 0/1 with hysteresis/debounce: commit to Stop only after
the Stop confidence exceeds a threshold for K consecutive chunks, and once committed
stay Stopped (a streaming endpoint is terminal). This suppresses isolated false-Stop
chunks — the dominant sentence-level failure mode (PRD §6.1).
"""
from __future__ import annotations


def debounce_sequence(pstop: list[float], threshold: float = 0.5, K: int = 1) -> list[int]:
    """Per-chunk Stop/Continue after debounce.

    threshold: min p_stop to count a chunk as a Stop vote.
    K: number of consecutive Stop votes required before committing Stop.
    K=1, threshold=0.5 reproduces the raw argmax decision.
    """
    preds, run, committed = [], 0, False
    for p in pstop:
        if committed:
            preds.append(1)
            continue
        if p >= threshold:
            run += 1
            if run >= K:
                committed = True
                preds.append(1)
            else:
                preds.append(0)
        else:
            run = 0
            preds.append(0)
    return preds


def clip_records_to_sequences(records: list[dict]) -> dict[str, dict]:
    """Group flat (clip,chunk) records into per-clip ordered sequences."""
    clips: dict[str, dict] = {}
    for r in records:
        d = clips.setdefault(r["id"], {"sem": r["sem_class"], "rows": []})
        d["rows"].append(r)
    for d in clips.values():
        d["rows"].sort(key=lambda x: x["c"])
        d["gt"] = [x["gt"] for x in d["rows"]]
        d["pstop"] = [x["p_stop"] for x in d["rows"]]
    return clips


def sentence_accuracy(clips: dict[str, dict], threshold: float, K: int) -> dict:
    """Strict sentence-level accuracy (all chunks match GT) under a debounce policy."""
    out = {"complete": [0, 0], "incomplete": [0, 0], "all": [0, 0]}
    for d in clips.values():
        preds = debounce_sequence(d["pstop"], threshold, K)
        ok = preds == d["gt"]
        for key in (d["sem"], "all"):
            out[key][0] += int(ok)
            out[key][1] += 1
    return {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4) if v[1] else 0.0}
            for k, v in out.items()}
