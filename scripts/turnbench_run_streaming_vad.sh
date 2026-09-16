#!/usr/bin/env bash
# Causal-VAD robustness run: re-gate the existing raw FD-VAD probs with STREAMING
# Silero (turnbench_submit_streaming.py) for BOTH dev and test, into a SEPARATE
# directory so nothing mixes with the official offline-VAD submission.
# Same theta/min_sil/settle/commit as the official run — only the VAD is causal.
set -euo pipefail

REPO=/home/colligo/semanticVAD
OUTDIR=$REPO/results/turnbench/submission_streaming_vad
DEV_DS=/mnt/localssd/svad/turnbench_data/dev/data
TEST_DS=/mnt/localssd/svad/turnbench_data/test/data
DEV_RAW=/mnt/localssd/svad/turnbench_out/dev
TEST_RAW=/mnt/localssd/svad/turnbench_out/test
TMP=/mnt/localssd/svad/turnbench_out/streaming_vad
NSHARD=8
mkdir -p "$OUTDIR" "$TMP"

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/localssd/svad/envs/svad
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""

run_split () {  # name raw_dir dataset
  local name=$1 raw=$2 ds=$3
  echo "[streaming] $name: $NSHARD parallel streaming-VAD shards"
  for i in $(seq 0 $((NSHARD-1))); do
    python -u scripts/turnbench_submit_streaming.py \
      --raw "$raw" --dataset "$ds" \
      --out "$TMP/predictions-$name.shard${i}of${NSHARD}.json" \
      --shard-idx "$i" --shard-num "$NSHARD" \
      >"$TMP/$name.shard${i}.log" 2>&1 &
  done
  wait
  echo "[streaming] $name: merging shards"
  python -u scripts/turnbench_submit_streaming.py --merge "$TMP/predictions-$name.shard*of${NSHARD}.json" \
    --dataset "$ds" --out "$OUTDIR/predictions-$name.json"
}

run_split dev  "$DEV_RAW"  "$DEV_DS"
run_split test "$TEST_RAW" "$TEST_DS"

echo "[streaming] scoring DEV (gold available):"
python -m turnbench.score "$OUTDIR/predictions-dev.json" --dataset "$DEV_DS" 2>&1 | tail -6
echo "[streaming] DONE -> $OUTDIR/predictions-{dev,test}.json (test recall is server-scored)"
