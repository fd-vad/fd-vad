# Labeling design — smart-turn → FD-VAD chunk labels (M1)

This is the contract that turns a `smart-turn-data-v3.1` clip into per-320ms-chunk
`Continue(0)/Stop(1)` targets. Getting this right is the single highest-leverage
correctness decision in the project, so it is documented, tested, and spot-checked
before any scale-up.

## Source facts (measured 2026-09-06)
- `smart-turn-data-v3.1`: 16kHz FLAC, `endpoint_bool` label per clip, English ~24%.
- Clips are **cut at the decision point**:
  - `endpoint_bool=True` (COMPLETE): speaker finished a complete thought. Measured
    trailing silence after speech-end: mean 0.43s, p50 0.27s, p90 0.93s.
  - `endpoint_bool=False` (INCOMPLETE): speaker truncated mid-thought. Trailing
    silence: mean 0.26s, p50 0.04s (cut ~right at speech).
- The two trailing-silence distributions **overlap** → the model cannot cheat via
  silence duration; it must judge semantic completeness. This is the point of the task.

## Label scheme (S2 — "clip cut at decision point")
Per clip with `n_frames` fbank frames (100Hz) and stride/window from config:

- **COMPLETE** (`endpoint_bool=True`): `stop_frame = VAD speech-offset` (fbank frame).
  chunk `c` label = `1 (Stop)` iff `r_c >= stop_frame`, else `0 (Continue)`
  (via `windowing.chunk_label`). Yields Continue during speech, Stop in the trailing
  post-endpoint region (≥1 Stop chunk; we optionally pad to guarantee one — see below).
- **INCOMPLETE** (`endpoint_bool=False`): `stop_frame = +∞` (set to `n_frames+1`).
  → every chunk label = `0 (Continue)`. The truncation is NOT a valid endpoint.

`sem_class`: complete→"complete" (timeout `T_s=400ms`), incomplete→"incomplete"
(timeout `T_l=1000ms`). Timeout is metadata for the decision layer (§6.1), not a chunk label.

### Why this differs from a naive "Stop at every acoustic offset"
An energy VAD alone would fire Stop at the end of *every* clip (both classes). The whole
value of FD-VAD is using semantics to fire Stop ONLY at complete endpoints. So the
label MUST come from `endpoint_bool`, with VAD used only to *locate* the boundary within
complete clips. This is the correct reading of smart-turn's design.

### Relation to paper Table 2
FD-VAD's own (unreleased) synthetic test set shows ~1 Stop chunk per sample in BOTH
the complete and incomplete subsets, implying their incomplete samples still contain a
final acoustic endpoint (an "S1" scheme). We cannot reproduce their exact data; smart-turn's
clips are constructed differently (S2). We therefore report metrics on smart-turn's own
test split (internally consistent) and note this as a known data-construction difference,
not a bug. If we later add synthetic data (IndexTTS + seed-tts, §D2), we can build S1-style
samples to match the paper more closely.

## Training construction (Eq. 2: supervise last chunk only)
Each clip yields training examples = windows ending at chunk `c` for `c=1..C`, each
supervised on its terminal chunk with `chunk_label(c)`. Class balance is naturally skewed
(Stop only at complete-clip trailing chunks) — mirrors Table 2's ~8% Stop rate. Sampling
weights (and §6.2 boundary oversampling) are applied at the sampler, not by dropping data.

## Config knobs (configs)
- `vad.kind`: energy (default) | silero. `vad.rel_thresh_db`: -40 (relative to peak).
- `label.min_trailing_stop_chunks`: 1 — pad COMPLETE clips with trailing silence so at
  least this many full post-endpoint chunks exist (guarantees a clean Stop target even
  when measured trailing silence < one stride). INCOMPLETE clips are never padded.
- `data.language`: "eng" for the English replication (D9).

## Validation gates (run before scale-up)
1. Label sanity: COMPLETE clips have ≥1 Stop chunk and ≥1 Continue chunk; INCOMPLETE
   clips have 0 Stop chunks. (Assert in prep; fail-loud.)
2. Boundary sanity: for COMPLETE, `0 < stop_frame <= n_frames`; Stop chunks are contiguous
   at the tail.
3. Distribution report: per-class counts, chunk-label counts, Stop-rate, duration stats.
4. Spot-check: dump ~10 clips (audio + derived boundary + labels) for manual listen.
