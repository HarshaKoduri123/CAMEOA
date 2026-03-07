from __future__ import annotations
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

class TemporalContrastExtractor(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.diff1_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU()
        )
        self.diff2_proj = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU()
        )

        self.stat_conv  = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim),
            nn.GELU(),
        )

        self.merge = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, D = x.shape

        diff1 = torch.zeros_like(x)
        diff1[:, 1:] = x[:, 1:] - x[:, :-1]

        diff2 = torch.zeros_like(diff1)
        diff2[:, 1:] = diff1[:, 1:] - diff1[:, :-1]


        stats = self.stat_conv(x.transpose(1, 2)).transpose(1, 2) 
        d1 = self.diff1_proj(diff1)
        d2 = self.diff2_proj(diff2)

        return self.merge(torch.cat([x, d1, d2, stats], dim=-1))



class EmotionFrameSelector(nn.Module):

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_peak   = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.q_valley = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.q_global = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.attn_peak   = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_valley = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_global = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)


        self.fuse = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B = x.size(0)
        qp = self.q_peak.expand(B, -1, -1)
        qv = self.q_valley.expand(B, -1, -1)
        qg = self.q_global.expand(B, -1, -1)

        peak,   _ = self.attn_peak(qp,   x, x)   # (B, 1, D)
        valley, _ = self.attn_valley(qv, x, x)
        global_,_ = self.attn_global(qg, x, x)

        return self.fuse(torch.cat([
            peak.squeeze(1), valley.squeeze(1), global_.squeeze(1)
        ], dim=-1))




class CrossFrameRelationTransformer(nn.Module):

    def __init__(self, dim: int, n_heads: int = 4, num_layers: int = 3,
                 max_frames: int = 64, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_frames, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        T = x.size(1)
        x = x + self.pos_emb[:, :T, :]
        return self.norm(self.transformer(x))


class VideoEncoder(nn.Module):


    def __init__(
        self,
        backbone:       str   = "r3d_18",
        d_model:        int   = 256,
        freeze:         bool  = True,
        dropout:        float = 0.1,
        embed_dim:      int   = 512,
        temporal_embed: bool  = False,
    ):
        super().__init__()
        self.temporal_embed = temporal_embed
        self._use_backbone = False
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            weights       = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
            self.backbone.fc = nn.Identity()
            backbone_dim  = 512
            self._use_backbone = True
            if freeze:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        except Exception:
            backbone_dim = embed_dim

        working_dim = min(d_model * 2, 512)   
        self.in_proj = nn.Sequential(
            nn.Linear(embed_dim if not self._use_backbone else backbone_dim, working_dim),
            nn.LayerNorm(working_dim),
            nn.GELU(),
        )

        self.temporal_contrast = TemporalContrastExtractor(working_dim, dropout=dropout)

        self.cross_frame = CrossFrameRelationTransformer(
            dim=working_dim, n_heads=max(1, working_dim // 64),
            num_layers=3, dropout=dropout,
        )

        self.frame_selector = EmotionFrameSelector(
            working_dim, n_heads=max(1, working_dim // 64), dropout=dropout
        )

        self.pre_proj = nn.Sequential(
            nn.Linear(working_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.adapter = nn.Sequential(
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=2, dropout=dropout),
            nn.LayerNorm(d_model),
        )

    def _embed_to_temporal(self, embed: torch.Tensor) -> torch.Tensor:

        if embed.dim() == 2:
            return embed.unsqueeze(1)
        return embed                 
    def forward(self, video_input: torch.Tensor) -> torch.Tensor:

        if self._use_backbone and video_input.dim() == 5:
            x = video_input.permute(0, 2, 1, 3, 4).contiguous() 
            x = self.backbone(x)                                  
            x = x.unsqueeze(1)                                    

        else:
            if video_input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                video_input = video_input.float()
            x = self._embed_to_temporal(video_input)              


        x = self.in_proj(x)                   
        x = self.temporal_contrast(x)         
        x = self.cross_frame(x)               
        x = self.frame_selector(x)            


        out = self.pre_proj(x)    
        out = self.adapter(out)   
        return out