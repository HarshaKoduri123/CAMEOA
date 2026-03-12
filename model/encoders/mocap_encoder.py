from __future__ import annotations

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        return x + self.pe[:, : x.size(1)]


class MoCapEncoder(nn.Module):
    """
    Expects:
        mocap: Tensor[B, T, F]
        mocap_len: Tensor[B]
    Returns:
        Tensor[B, d_model]
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        feat_dim = int(cfg.mocap_feat_dim)
        d_model = int(cfg.d_model)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.mocap_dropout)),
        )
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=int(cfg.mocap_max_len) + 8)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.mocap_nhead),
            dim_feedforward=4 * d_model,
            dropout=float(cfg.mocap_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.mocap_layers))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, mocap: torch.Tensor, mocap_len: torch.Tensor | None = None) -> torch.Tensor:
        if mocap is None:
            raise ValueError("MoCapEncoder received mocap=None")

        x = self.input_proj(mocap)
        x = self.pos(x)

        key_padding_mask = None
        if mocap_len is not None:
            B, T, _ = mocap.shape
            idx = torch.arange(T, device=mocap.device).unsqueeze(0).expand(B, T)
            key_padding_mask = idx >= mocap_len.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if mocap_len is not None:
            mask = (~key_padding_mask).float().unsqueeze(-1)  # [B,T,1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            x = x.mean(dim=1)

        x = self.out_proj(x)
        return x