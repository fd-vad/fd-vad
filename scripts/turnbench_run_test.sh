#!/usr/bin/env bash
# Turnkey TurnBench TEST-split prediction at the FROZEN operating point.
#
# Mirrors the dev pipeline exactly (see results/turnbench/RESULTS.md):
#   1. 2-GPU sharded raw P(Stop) via scripts/turnbench_predict.py
#   2. VAD-gate + fixed-threshold commit via scripts/turnbench_submit.py
# The operating point (min_sil=1000ms, settle=2500ms, theta=0.0036) was chosen ONCE on
# dev and is applied UNCHANGED here — TurnBench's causal-submission rule.
#
# PREREQ: test audio must be downloaded first (gated HF repo mundo-ai/turn-benchmark-test):
#   HF_TOKEN=$(grep -iA3 "hf access token" /home/colligo/secrets.txt | grep -oE 'hf_[A-Za-z0-9]+' | head -1)
#   HUGGING_FACE_HUB_TOKEN=$HF_TOKEN python -c "from huggingface_hub import snapshot_download; \
#     snapshot_download('mundo-ai/turn-benchmark-test', repo_type='dataset', \
#     allow_patterns=['data/*.parquet'], local_dir='/mnt/localssd/svad/turnbench_data/test')"
set -euo pipefail

REPO=/home/colligo/semanticVAD
CKPT=/mnt/localssd/svad/checkpoints/run_full_wavlm_ddp/latest.pt   # == fd_vad.pt
CONFIG=$REPO/configs/base.yaml
DATASET=/mnt/localssd/svad/turnbench_data/test/data
OUT=/mnt/localssd/svad/turnbench_out/test
mkdir -p "$OUT"

source /opt/conda/etc/profile.d/conda.sh
conda activate /mnt/localssd/svad/envs/svad

if ! ls "$DATASET"/*.parquet >/dev/null 2>&1; then
  echo "ERROR: no test parquet at $DATASET — download the gated test audio first (see header)." >&2
  exit 1
fi

echo "[test] raw probs: 2-GPU sharded predict"
CUDA_VISIBLE_DEVICES=0 python -u scripts/turnbench_predict.py \
  --ckpt "$CKPT" --config "$CONFIG" --dataset "$DATASET" \
  --out-dir "$OUT" --shard-idx 0 --shard-num 2 >"$OUT/shard0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u scripts/turnbench_predict.py \
  --ckpt "$CKPT" --config "$CONFIG" --dataset "$DATASET" \
  --out-dir "$OUT" --shard-idx 1 --shard-num 2 >"$OUT/shard1.log" 2>&1 &
wait
echo "[test] raw shards done"

echo "[test] gate + commit at frozen op point (min_sil=1000ms theta=0.0036)"
python -u scripts/turnbench_submit.py \
  --raw "$OUT" --dataset "$DATASET" \
  --min-sil-ms 1000 --settle-ms 2500 --theta 0.0036 \
  --out "$REPO/results/turnbench/submission/predictions-test.json"

echo "[test] DONE -> results/turnbench/submission/predictions-test.json"
