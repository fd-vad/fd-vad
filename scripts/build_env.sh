#!/bin/bash
set -euo pipefail
export CONDA_PKGS_DIRS=/mnt/localssd/svad/.conda_pkgs
export PIP_CACHE_DIR=/mnt/localssd/svad/.pip_cache
export TMPDIR=/mnt/localssd/svad/tmp
ENV=/mnt/localssd/svad/envs/svad
echo "[env] creating conda env at $ENV"
/opt/conda/bin/conda create -y -p $ENV python=3.11 >/dev/null
PY=$ENV/bin/python
$PY -m pip install --upgrade pip >/dev/null
echo "[env] installing torch (cu124)"
$PY -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
echo "[env] installing core ML/audio stack"
$PY -m pip install \
  transformers==4.46.3 "peft==0.14.0" accelerate datasets huggingface_hub \
  soundfile librosa scipy numpy pandas pyyaml tqdm jiwer einops \
  pypdf sentencepiece
echo "[env] verify"
$PY - <<'PYEOF'
import torch, transformers, peft, soundfile, librosa
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
print("transformers", transformers.__version__, "peft", peft.__version__)
PYEOF
echo "[env] DONE"
