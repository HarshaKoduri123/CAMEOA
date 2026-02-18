# loss.py
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def info_nce_loss_masked(
    anchor: torch.Tensor,          # (B, D)
    positive: torch.Tensor,        # (B, D)
    temperature: float,
    row_mask: torch.Tensor | None = None,  # (B,) 1=use row, 0=ignore row
) -> torch.Tensor:
    """
    Full-batch InfoNCE:
      logits = anchor @ positive.T / temp   (B,B)
      positives are diagonal.

    If row_mask is provided:
      - loss is averaged only over valid rows
      - BUT negatives remain full batch (critical for stability).
    """
    anchor = l2_normalize(anchor)
    positive = l2_normalize(positive)

    logits = (anchor @ positive.t()) / temperature  # (B,B)
    labels = torch.arange(anchor.size(0), device=anchor.device)

    per_row = F.cross_entropy(logits, labels, reduction="none")  # (B,)

    if row_mask is None:
        return per_row.mean()

    row_mask = row_mask.float()
    denom = row_mask.sum().clamp(min=1.0)
    return (per_row * row_mask).sum() / denom


def orthogonality_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    """
    shared/private: (B, D). Penalize cosine similarity magnitude (ShaSpec-style).
    """
    shared = l2_normalize(shared)
    private = l2_normalize(private)
    cos = (shared * private).sum(dim=-1).abs()  # (B,)
    return cos.mean()


def consistency_loss(
    z_masked: torch.Tensor,   # (B,D)
    z_full: torch.Tensor,     # (B,D)
    used_mask: torch.Tensor,  # (B,4)
    full_mask: torch.Tensor,  # (B,4)
) -> torch.Tensor:
    """
    Apply consistency only when dropout actually removed something.
    """
    drop_any = (used_mask.sum(dim=1) < full_mask.sum(dim=1))  # (B,)
    if drop_any.any():
        return F.mse_loss(z_masked[drop_any], z_full[drop_any])
    return torch.zeros((), device=z_masked.device)


def sigma_reg_loss(
    sigmas: list[torch.Tensor],
    mode: str = "log2",         # "log2" or "mean"
    target: float = 1.0,
) -> torch.Tensor:
    """
    Prevent sigma heads from collapsing to extremes.
    sigmas: list of (B,) tensors, each sigma >= 0.

    mode="log2": mean( (log(sigma))^2 )
    mode="mean": mean( (sigma - target)^2 )
    """
    if len(sigmas) == 0:
        return torch.zeros(())

    device = sigmas[0].device
    eps = 1e-6
    s = torch.stack([x.clamp(min=eps) for x in sigmas], dim=1)  # (B,M)

    if mode == "log2":
        return (torch.log(s) ** 2).mean()
    elif mode == "mean":
        return ((s - target) ** 2).mean()
    else:
        raise ValueError(f"Unknown sigma_reg_loss mode: {mode}")
