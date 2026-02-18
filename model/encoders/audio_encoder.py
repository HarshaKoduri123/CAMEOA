import torch
import torch.nn as nn
from transformers import Wav2Vec2Model



class AudioEncoder(nn.Module):
    """
    wav2vec2 -> mean pool -> adapter to d_model
    """
    def __init__(self, model_name: str, d_model: int, freeze: bool = True):
        super().__init__()
        
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size

        self.adapter = nn.Sequential(
            nn.Linear(self.hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, T) float
        returns hA: (B, d_model)
        """
        x = self.backbone(wav).last_hidden_state  # (B, L, hidden)
        x = x.mean(dim=1)  # (B, hidden)
        return self.adapter(x)
