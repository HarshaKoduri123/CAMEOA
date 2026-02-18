import torch
import torch.nn as nn

class MocapEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.GELU())

    def forward(self, mocap: torch.Tensor, mocap_len: torch.Tensor):
        B, T, F = mocap.shape  # F = 576
        x = self.in_proj(mocap)

        mask = torch.arange(T, device=mocap.device)[None, :] >= mocap_len[:, None]
        x = self.tr(x, src_key_padding_mask=mask)

        x = x.masked_fill(mask[:, :, None], 0.0)
        denom = mocap_len.clamp(min=1).float().unsqueeze(-1)
        pooled = x.sum(dim=1) / denom
        return self.out(pooled)

