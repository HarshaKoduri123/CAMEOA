from __future__ import annotations

import torch
import torch.nn as nn


class VideoEncoder(nn.Module):
    """
    Current implementation assumes precomputed video embeddings are used.

    Expects:
        video_embed: Tensor[B, video_embed_dim]
    Returns:
        Tensor[B, d_model]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = int(cfg.video_embed_dim)
        d_model = int(cfg.d_model)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(float(cfg.video_dropout)),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.video_dropout)),
            nn.Linear(d_model, d_model),
        )

    def forward(self, video_embed: torch.Tensor) -> torch.Tensor:
        if video_embed is None:
            raise ValueError("VideoEncoder received video_embed=None")
        return self.net(video_embed)