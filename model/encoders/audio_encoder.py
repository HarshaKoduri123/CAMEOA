from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Expects:
        audio: Tensor[B, T]  raw waveform or already padded waveform
    Returns:
        Tensor[B, d_model]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)

        self.backbone = Wav2Vec2Model.from_pretrained(cfg.audio_model_name)
        hidden = self.backbone.config.hidden_size

        if bool(cfg.freeze_audio):
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(float(cfg.audio_dropout)),
            nn.Linear(hidden, self.d_model),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [B, T]
        """
        if audio is None:
            raise ValueError("AudioEncoder received audio=None")

        attn_mask = (audio != 0).long() if audio.dim() == 2 else None
        out = self.backbone(input_values=audio, attention_mask=attn_mask)
        x = out.last_hidden_state.mean(dim=1)  # [B, hidden]
        x = self.proj(x)
        return x