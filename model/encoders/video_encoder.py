import torch
import torch.nn as nn

import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights


class VideoEncoder(nn.Module):
    """
    torchvision r3d_18 pretrained -> global pool -> adapter to d_model
    Input video tensor: (B, T, 3, H, W)
    """
    def __init__(self, backbone: str, d_model: int, freeze: bool = True):
        super().__init__()

        weights = R3D_18_Weights.DEFAULT
        self.backbone = r3d_18(weights=weights)
        # remove classifier
        self.backbone.fc = nn.Identity()

        self.hidden = 512  # r3d_18 output dim

        self.adapter = nn.Sequential(
            nn.Linear(self.hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: (B, T, 3, H, W)
        torchvision expects (B, 3, T, H, W)
        returns hV: (B, d_model)
        """
        x = video.permute(0, 2, 1, 3, 4).contiguous()  # (B,3,T,H,W)
        feat = self.backbone(x)  # (B, 512)
        return self.adapter(feat)
