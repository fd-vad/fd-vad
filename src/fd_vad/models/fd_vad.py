"""FDVAD (paper §2): frozen encoder → modality adapter → Qwen2.5-0.5B (LoRA).

Per window: inputs_embeds = [ prompt_embeds ; adapted_audio_embeds ] and the LLM
predicts a single label token in {"0","1"} followed by <eos>. Training supervises only
that terminal target (Eq. 1-2) with cross-entropy.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoders import build_encoder
from .adapter import ModalityAdapter

LABEL_STR = {0: "0", 1: "1"}


@dataclass
class FDVADOutput:
    loss: torch.Tensor | None
    logits2: torch.Tensor          # [B,2] logits over {0,1} at the decision position
    probs2: torch.Tensor           # softmax of logits2


class FDVAD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg
        self.encoder = build_encoder(cfg.encoder.kind, cfg.encoder.checkpoint)

        self.tok = AutoTokenizer.from_pretrained(cfg.llm.model)
        llm = AutoModelForCausalLM.from_pretrained(cfg.llm.model, torch_dtype=torch.float32)
        lora = LoraConfig(
            r=cfg.llm.lora_rank, lora_alpha=cfg.llm.lora_alpha,
            lora_dropout=getattr(cfg.llm, "lora_dropout", 0.05),
            target_modules=list(cfg.llm.lora_targets), task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(llm, lora)
        self.embed = self.llm.get_input_embeddings()
        llm_dim = self.llm.config.hidden_size

        self.adapter = ModalityAdapter(
            self.encoder.output_dim, cfg.adapter.downsample_k, cfg.adapter.hidden_dim, llm_dim)

        # fixed prompt (versioned in config, PRD §1)
        msgs = [{"role": "system", "content": cfg.llm.system_prompt},
                {"role": "user", "content": cfg.llm.user_prompt}]
        prompt_ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        self.register_buffer("prompt_ids", torch.tensor(prompt_ids, dtype=torch.long), persistent=False)

        self.label_token_ids = [self.tok.encode(LABEL_STR[i], add_special_tokens=False)[0] for i in (0, 1)]
        assert len(self.label_token_ids) == 2, "label tokens must be single-token"
        self.eos_id = self.tok.eos_token_id

    # ---- device helpers ----
    @property
    def device(self):
        return next(self.llm.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """Params needed to reproduce the model: LoRA + the FULL adapter (always saved,
        even if frozen, so a frozen-adapter/non-joint checkpoint is self-contained)."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        names |= {n for n, _ in self.adapter.named_parameters(prefix="adapter")}
        return {n: v.detach().cpu() for n, v in self.state_dict().items() if n in names}

    def load_trainable_state_dict(self, sd: dict, allow_missing: bool = False):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"unexpected keys in checkpoint: {list(unexpected)[:5]}")
        loaded = set(sd.keys())
        train_names = {n for n, p in self.named_parameters() if p.requires_grad}
        not_loaded = train_names - loaded
        if not_loaded and not allow_missing:
            raise ValueError(f"checkpoint missing {len(not_loaded)} trainable params, e.g. {list(not_loaded)[:3]}")
        if not_loaded:
            print(f"[warn] {len(not_loaded)} params not in checkpoint (kept seeded init): {list(not_loaded)[:3]}")

    def _audio_embeds(self, wav, wav_lens):
        feats, flens = self.encoder.encode(wav, wav_lens)
        return self.adapter(feats.to(self.embed.weight.dtype), flens)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor,
                labels: torch.Tensor | None = None) -> FDVADOutput:
        """wav [B,T] float, wav_lens [B], labels [B] in {0,1} or None (inference)."""
        B = wav.shape[0]
        dev = self.device
        wav, wav_lens = wav.to(dev), wav_lens.to(dev)
        audio, audio_lens = self._audio_embeds(wav, wav_lens)     # [B,S,dim], [B]

        prompt_emb = self.embed(self.prompt_ids.to(dev))          # [Lp, dim]
        Lp = prompt_emb.shape[0]
        train = labels is not None
        tgt_len = 2 if train else 0                                # [label, eos]

        seqs, lab_rows, dec_pos = [], [], []
        max_len = 0
        for i in range(B):
            a = audio[i, : int(audio_lens[i])]                    # [Ai, dim]
            parts = [prompt_emb, a]
            if train:
                lid = self.label_token_ids[int(labels[i])]
                tgt_ids = torch.tensor([lid, self.eos_id], device=dev)
                parts.append(self.embed(tgt_ids))
            seq = torch.cat(parts, dim=0)                         # [Li, dim]
            seqs.append(seq)
            dec_pos.append(Lp + a.shape[0] - 1)                   # position that predicts the label
            if train:
                row = torch.full((seq.shape[0],), -100, dtype=torch.long, device=dev)
                row[Lp + a.shape[0]] = self.label_token_ids[int(labels[i])]
                row[Lp + a.shape[0] + 1] = self.eos_id
                lab_rows.append(row)
            max_len = max(max_len, seq.shape[0])

        dim = prompt_emb.shape[1]
        inp = torch.zeros(B, max_len, dim, device=dev, dtype=prompt_emb.dtype)
        attn = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        lab = torch.full((B, max_len), -100, dtype=torch.long, device=dev) if train else None
        for i, seq in enumerate(seqs):
            L = seq.shape[0]
            inp[i, :L] = seq
            attn[i, :L] = 1
            if train:
                lab[i, :L] = lab_rows[i]

        out = self.llm(inputs_embeds=inp, attention_mask=attn, labels=lab)
        loss = out.loss if train else None

        # decision logits at each sample's decision position, restricted to {0,1}
        idx = torch.tensor(dec_pos, device=dev)
        dec_logits = out.logits[torch.arange(B, device=dev), idx]      # [B, vocab]
        logits2 = dec_logits[:, self.label_token_ids]                  # [B, 2]
        probs2 = torch.softmax(logits2, dim=-1)
        return FDVADOutput(loss=loss, logits2=logits2.detach(), probs2=probs2.detach())
