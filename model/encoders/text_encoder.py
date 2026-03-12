from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    """
    Supports either:
      1) input_ids + attention_mask
      2) a dict with keys {"input_ids", "attention_mask"}

    Returns:
        Tensor[B, d_model]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)

        self.backbone = AutoModel.from_pretrained(cfg.text_model_name)
        hidden = self.backbone.config.hidden_size

        if bool(cfg.freeze_text):
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(float(cfg.text_dropout)),
            nn.Linear(hidden, self.d_model),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            x = out.pooler_output
        else:
            x = out.last_hidden_state[:, 0]  # CLS token

        x = self.proj(x)
        return x