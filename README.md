# FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech

FD-VAD is a compact, **ASR-free, streaming** semantic endpoint detector for full-duplex
voice agents. It feeds short sliding windows of audio directly to a small LLM and emits a per-chunk
`Continue` / `Stop` decision, using the LLM's semantic understanding to place the turn boundary — so it
distinguishes a mid-thought pause from a genuine end-of-turn, which acoustic VAD cannot.

**Architecture:** frozen WavLM-base-plus encoder → 2-layer modality adapter → Qwen2.5-0.5B-Instruct
(LoRA, 10.3M trainable params) → `{0,1}` token per 320 ms chunk. Trained with a sliding-window,
last-chunk cross-entropy objective.

## Model weights

This folder is **code only** — it does not ship the weights. Get them from the Hugging Face repo:

**➡ https://huggingface.co/puneetUMD/fd-vad**

```bash
# the trained FD-VAD adapter/LoRA checkpoint (required):
huggingface-cli download puneetUMD/fd-vad fd_vad.pt --local-dir .
# optional: the frozen base encoder/LLM, for fully-offline use (skip to auto-download instead):
huggingface-cli download puneetUMD/fd-vad --include "base/*" --local-dir .
```

Place `fd_vad.pt` in this folder. If you also download `base/`, inference runs fully offline; otherwise
`microsoft/wavlm-base-plus` and `Qwen/Qwen2.5-0.5B-Instruct` download automatically from the Hub on first run.

## Quickstart — end-of-turn inference

```bash
pip install -r requirements.txt
python infer.py path/to/audio.wav                 # prints the committed EOT time, or "incomplete"
python infer.py audio.wav --tau 0.9 --k 1         # more conservative endpoint
python infer.py audio.wav --show-stream           # per-320ms P(Stop) trace
```

To train / evaluate / reproduce from scratch, see **Reproduce** below (uses `scripts/` + `src/fd_vad`).

## Install
```bash
conda create -y -p ./env python=3.11 && conda activate ./env
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # installs the fd_vad package
pip install -r requirements-lock.txt   # pinned versions used for the paper
```

## Quickstart
```python
import torch
from fd_vad.config import load_config
from fd_vad.windowing import WindowConfig
from fd_vad.models.fd_vad import FDVAD

cfg = load_config("configs/base.yaml"); cfg.encoder.kind = "wavlm"
model = FDVAD(cfg).cuda()
model.load_trainable_state_dict(torch.load("checkpoints/fd_vad.pt")["model"])
# per-window inference -> out.probs2[:,1] = P(Stop)
```

## Reproduce
```bash
python scripts/prepare_smartturn.py --split train --max-shards 75 --max-per-class -1 --out data/train
python scripts/train.py --encoder wavlm --train data/train --val data/val --epochs 1   # 2-GPU: torchrun --nproc_per_node=2
python scripts/eval.py --ckpt <latest.pt> --encoder wavlm --test data/test             
python scripts/baselines.py            # baseline comparison (energy-VAD, smart-turn, ours)
python scripts/baseline_cascade.py     # cascaded ASR->LLM baseline
python scripts/latency_bench.py --ckpt <latest.pt> --encoder wavlm                       
```

## Repository layout
```
src/fd_vad/            # package: windowing, schema, config, data, models, eval, decision
scripts/               # prepare / train / eval / baselines / latency / figures
configs/base.yaml      # config (window, adapter k, LoRA, prompts)
tests/                 # unit tests (windowing, labeling)
```

## Data & checkpoints
- **Data:** derived from the public `pipecat-ai/smart-turn-data-v3.1` corpus (English), converted to
  per-chunk labels by `scripts/prepare_smartturn.py` (see [`docs/labeling.md`](docs/labeling.md)).
- **Checkpoints:** small (LoRA + adapter only, ~120 MB); the encoder and LLM base are loaded from HF.

## License
MIT (see [`LICENSE`](LICENSE)). Uses WavLM (MIT), Qwen2.5 (Apache-2.0), and the
smart-turn-v3.1 dataset (BSD-2-Clause) — please honor their respective licenses.

## Citation
```bibtex
@inproceedings{fdvad,
  title     = {FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech},
  author    = {Puneet Mathur, Dinesh Manocha},
  booktitle = {Submitted},
  year      = {2026}
}
```
