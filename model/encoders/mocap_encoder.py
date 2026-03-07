from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class SqueezeExcite(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.fc(x)


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.net(x)



class JointGroupDecomposition(nn.Module):

    def __init__(self, feat_dim: int, group_dim: int, n_groups: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_groups  = n_groups
        self.group_dim = group_dim
        per_group_in   = feat_dim // n_groups

        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(per_group_in + (feat_dim - per_group_in * n_groups if i == n_groups - 1 else 0),
                          group_dim),
                nn.LayerNorm(group_dim),
                nn.GELU(),
                nn.Linear(group_dim, group_dim),
                nn.LayerNorm(group_dim),
                nn.GELU(),
            )
            for i in range(n_groups)
        ])

        self.feat_dim   = feat_dim
        self.per_group  = feat_dim // n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        parts = []
        for i, proj in enumerate(self.group_projs):
            start = i * self.per_group
            end   = (i + 1) * self.per_group if i < self.n_groups - 1 else self.feat_dim
            parts.append(proj(x[:, :, start:end])) 
        return torch.cat(parts, dim=-1)             


class SpatialBodyGraphAttention(nn.Module):

    def __init__(self, group_dim: int, n_groups: int = 4, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_groups = n_groups
        self.group_dim = group_dim

        self.body_pos = nn.Parameter(torch.randn(1, n_groups, group_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=group_dim, nhead=n_heads,
            dim_feedforward=group_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.graph_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj   = nn.Linear(group_dim * n_groups, group_dim * n_groups)
        self.norm       = nn.LayerNorm(group_dim * n_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D = x.shape
        gd = self.group_dim

        nodes = x.reshape(B * T, self.n_groups, gd)
        nodes = nodes + self.body_pos.expand(B * T, -1, -1)

        fused = self.graph_attn(nodes)       
        fused = fused.reshape(B, T, D)       

        return self.norm(x + self.out_proj(fused))


class MotionDynamicsExtractor(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.vel_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )
        self.acc_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        vel = torch.zeros_like(x)
        vel[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]

        acc = torch.zeros_like(vel)
        acc[:, 1:, :] = vel[:, 1:, :] - vel[:, :-1, :]

        v_feat = self.vel_proj(vel)  
        a_feat = self.acc_proj(acc)  

        return self.merge(torch.cat([x, v_feat, a_feat], dim=-1)) 


class TemporalTransformerEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 max_len: int = 400, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:

        T = x.size(1)
        x = x + self.pos_emb[:, :T, :]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm_out(x)



class MultiScaleTemporalPoolMoCap(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.slow_attn = nn.Sequential(nn.Linear(d_model, 1))
        self.fast_attn = nn.Sequential(nn.Linear(d_model, 1))

        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

        mask_float = valid_mask.float()
        s_scores = self.slow_attn(x).squeeze(-1)               
        s_scores = s_scores.masked_fill(valid_mask == 0, -1e9)
        s_weights = torch.softmax(s_scores, dim=1).unsqueeze(-1)
        slow_pool = (s_weights * x).sum(dim=1)                 

        x_fast  = x[:, ::4, :]
        m_fast  = valid_mask[:, ::4]
        f_scores = self.fast_attn(x_fast).squeeze(-1)
        f_scores = f_scores.masked_fill(m_fast == 0, -1e9)
        f_weights = torch.softmax(f_scores, dim=1).unsqueeze(-1)
        fast_pool = (f_weights * x_fast).sum(dim=1)            

        return self.fuse(torch.cat([slow_pool, fast_pool], dim=-1))  

class MocapEncoder(nn.Module):

    def __init__(
        self,
        feat_dim:   int,
        d_model:    int,
        nhead:      int   = 8,
        num_layers: int   = 4,
        n_groups:   int   = 4,
        max_len:    int   = 400,
        dropout:    float = 0.1,
    ):
        super().__init__()

        group_dim  = d_model // n_groups  
        total_group = group_dim * n_groups 

        self.group_decomp = JointGroupDecomposition(
            feat_dim, group_dim, n_groups=n_groups, dropout=dropout
        )

        self.spatial_graph = SpatialBodyGraphAttention(
            group_dim, n_groups=n_groups, n_heads=max(1, n_groups // 2), dropout=dropout
        )

        self.in_proj = nn.Sequential(
            nn.Linear(total_group, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.dynamics = MotionDynamicsExtractor(d_model, dropout=dropout)
        self.temporal = TemporalTransformerEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            max_len=max_len, dropout=dropout,
        )

        self.ms_pool = MultiScaleTemporalPoolMoCap(d_model, dropout=dropout)
        self.adapter = nn.Sequential(
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=2, dropout=dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, mocap: torch.Tensor, mocap_len: torch.Tensor) -> torch.Tensor:

        B, T, _ = mocap.shape

        pad_mask   = torch.arange(T, device=mocap.device)[None, :] >= mocap_len[:, None]
        valid_mask = ~pad_mask   # True = valid
        x = self.group_decomp(mocap)      
        x = self.spatial_graph(x)          
        x = self.in_proj(x)                
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.dynamics(x)            
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.temporal(x, pad_mask)      

        out = self.ms_pool(x, valid_mask)  

        return self.adapter(out)          
