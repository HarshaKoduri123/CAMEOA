from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class SqueezeExcite(nn.Module):

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MultiScaleTemporalPool(nn.Module):

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        # Learnable query for each scale
        self.q_fine   = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.q_coarse = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.q_global = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

        self.attn_fine   = nn.MultiheadAttention(hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_coarse = nn.MultiheadAttention(hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_global = nn.MultiheadAttention(hidden, num_heads=4, dropout=dropout, batch_first=True)

        # Dilated 1D conv for fine-scale local context
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden), 
            nn.Conv1d(hidden, hidden, kernel_size=1),                             
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 
        B, T, H = x.shape
        x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)  
        x_norm = self.norm(x + x_conv)
        stride = max(1, T // 20)
        x_coarse = x_norm[:, ::stride, :]  

        q_f = self.q_fine.expand(B, -1, -1)
        q_c = self.q_coarse.expand(B, -1, -1)
        q_g = self.q_global.expand(B, -1, -1)

        fine,   _ = self.attn_fine(q_f,   x_norm,   x_norm)    
        coarse, _ = self.attn_coarse(q_c, x_coarse, x_coarse)  
        global_, _ = self.attn_global(q_g, x,       x)     

        return fine.squeeze(1), coarse.squeeze(1), global_.squeeze(1)


class ProsodicExtractor(nn.Module):

    def __init__(self, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.energy_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=k, stride=k // 2, padding=k // 4),
                nn.GELU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.GELU(),
            )
            for k in [160, 320, 640] 
        ])

        self.merge = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool_attn = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = wav.unsqueeze(1)  
        parts = []
        min_len = None
        for conv in self.energy_convs:
            out = conv(x) 
            if min_len is None or out.size(2) < min_len:
                min_len = out.size(2)
            parts.append(out)

        parts = [p[:, :, :min_len] for p in parts]
        merged = self.merge(torch.cat(parts, dim=1))  
        merged = merged.transpose(1, 2)             
        weights = torch.softmax(self.pool_attn(merged), dim=1) 
        pooled  = (weights * merged).sum(dim=1)                 
        return self.proj(self.dropout(pooled))


class CrossScaleFusion(nn.Module):

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 4, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj = nn.Linear(hidden * 4, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        fine:     torch.Tensor,   
        coarse:   torch.Tensor, 
        global_:  torch.Tensor,   
        prosodic: torch.Tensor,  
    ) -> torch.Tensor:
        tokens = torch.stack([fine, coarse, global_, prosodic], dim=1)  
        tokens = tokens + self.pos_emb
        fused  = self.transformer(tokens)                              
        out    = self.out_proj(fused.reshape(fused.size(0), -1))        
        return self.norm(out)

class AudioEncoder(nn.Module):

    def __init__(
        self,
        model_name: str,
        d_model:    int,
        freeze:     bool  = True,
        unfreeze_top_n_layers: int = 0,   
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size  
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_top_n_layers > 0:
            n_layers = len(self.backbone.encoder.layers)
            for i in range(n_layers - unfreeze_top_n_layers, n_layers):
                for p in self.backbone.encoder.layers[i].parameters():
                    p.requires_grad = True
            for p in self.backbone.encoder.layer_norm.parameters():
                p.requires_grad = True

        self.ms_pool = MultiScaleTemporalPool(hidden, dropout=dropout)
        self.prosodic = ProsodicExtractor(hidden, dropout=dropout)
        self.fusion = CrossScaleFusion(hidden, dropout=dropout)


        self.pre_proj = nn.Sequential(
            nn.Linear(hidden, d_model),
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

    def forward(self, wav: torch.Tensor) -> torch.Tensor:

        with torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        ):
            hidden = self.backbone(wav).last_hidden_state  
        fine, coarse, global_ = self.ms_pool(hidden)      
        prosodic = self.prosodic(wav)                      
        fused = self.fusion(fine, coarse, global_, prosodic) 
        out = self.pre_proj(fused)  
        out = self.adapter(out)      
        return out