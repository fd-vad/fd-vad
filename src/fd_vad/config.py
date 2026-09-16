"""Config system (PRD §5 M0). YAML-backed, dataclass-validated, versioned.

Kept deliberately lightweight (no Hydra dependency) but structured so prompt
wording, downsample factor k, LoRA params etc. live in config, not code
(PRD §1 flags these as tunables that affect results).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml


@dataclass
class WindowCfg:
    window_frames: int = 256
    stride_frames: int = 32


@dataclass
class EncoderCfg:
    kind: str = "zipformer"          # zipformer | whisper | wav2vec2
    checkpoint: str = ""             # HF id or local path
    output_hz: int = 25
    frozen: bool = True


@dataclass
class AdapterCfg:
    downsample_k: int = 4            # concatenate k consecutive 25Hz frames (PRD §1: sweep)
    hidden_dim: int = 2048


@dataclass
class LLMCfg:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    system_prompt: str = "You are a voice activity detector."
    user_prompt: str = "Has the user finished speaking? Answer 0 (continue) or 1 (stop)."


@dataclass
class TrainCfg:
    epochs: int = 1
    batch_size: int = 64
    lr: float = 5e-5
    warmup_ratio: float = 0.03
    lr_schedule: str = "cosine"
    grad_accum: int = 1
    seed: int = 1234
    supervise_last_chunk_only: bool = True   # Eq. 2


@dataclass
class DataCfg:
    root: str = "/mnt/localssd/svad/data"
    sample_rate: int = 24000
    timeout_complete_ms: int = 400
    timeout_incomplete_ms: int = 1000


@dataclass
class Config:
    name: str = "base"
    window: WindowCfg = field(default_factory=WindowCfg)
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    adapter: AdapterCfg = field(default_factory=AdapterCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)

    def validate(self) -> None:
        if self.window.stride_frames > self.window.window_frames:
            raise ValueError("stride_frames > window_frames")
        if self.adapter.downsample_k < 1:
            raise ValueError("downsample_k must be >= 1")
        if self.llm.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")


def _merge(dc, d: dict):
    """Recursively overlay dict `d` onto dataclass instance `dc`."""
    for k, v in (d or {}).items():
        if not hasattr(dc, k):
            raise ValueError(f"unknown config key: {k}")
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = _merge(Config(), raw)
    cfg.validate()
    return cfg


def dump_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
