# loss.py  (UPDATED v2)
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


# ─────────────────────────────────────────────
#  Contrastive
# ─────────────────────────────────────────────

def info_nce_loss_masked(
    anchor: torch.Tensor,                  
    positive: torch.Tensor,               
    temperature: float,
    row_mask: torch.Tensor | None = None, 
) -> torch.Tensor:

    anchor   = l2_normalize(anchor)
    positive = l2_normalize(positive)

    logits = (anchor @ positive.t()) / temperature  
    labels = torch.arange(anchor.size(0), device=anchor.device)

    per_row_a2p = F.cross_entropy(logits,   labels, reduction="none")  

    per_row_p2a = F.cross_entropy(logits.t(), labels, reduction="none")
    per_row = 0.5 * (per_row_a2p + per_row_p2a)

    if row_mask is None:
        return per_row.mean()

    row_mask = row_mask.float()
    denom    = row_mask.sum().clamp(min=1.0)
    return (per_row * row_mask).sum() / denom


def cross_modal_contrastive_loss(
    reps:      dict[str, dict[str, torch.Tensor]],  
    used_mask: torch.Tensor,                        
    mod_idx:   dict[str, int],
    temperature: float,
    proj_key:  str = "proj",                        
) -> torch.Tensor:

    keys     = [k for k in ["A", "T", "V", "M"] if k in reps]
    loss     = torch.zeros((), device=used_mask.device)
    count    = 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            mi = used_mask[:, mod_idx[ki]]
            mj = used_mask[:, mod_idx[kj]]
            pair_mask = mi * mj
            if pair_mask.sum() < 2:
                continue

            zi = reps[ki][proj_key]   
            zj = reps[kj][proj_key]

            loss  += info_nce_loss_masked(zi, zj, temperature, row_mask=pair_mask)
            count += 1

    return loss / max(1, count)


def hub_contrastive_loss(
    z_proj:    torch.Tensor,                        
    reps:      dict[str, dict[str, torch.Tensor]],
    full_mask: torch.Tensor,                       
    mod_idx:   dict[str, int],
    temperature: float,
    proj_key:  str = "proj",
) -> torch.Tensor:

    loss  = torch.zeros((), device=z_proj.device)
    count = 0

    for k in ["A", "T", "V", "M"]:
        if k not in reps:
            continue
        mk = full_mask[:, mod_idx[k]]
        if mk.sum() < 2:
            continue

        loss  += info_nce_loss_masked(z_proj, reps[k][proj_key], temperature, row_mask=mk)
        count += 1

    return loss / max(1, count)


# ─────────────────────────────────────────────
#  Orthogonality
# ─────────────────────────────────────────────

def orthogonality_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:

    shared  = l2_normalize(shared)  
    private = l2_normalize(private)   

    if shared.size(-1) == private.size(-1):
       
        return (shared * private).sum(dim=-1).abs().mean()
    else:
   
        cos = (shared * private[:, :shared.size(-1)]).sum(dim=-1).abs() \
            if shared.size(-1) <= private.size(-1) else \
              (shared[:, :private.size(-1)] * private).sum(dim=-1).abs()
        return cos.mean()


# ─────────────────────────────────────────────
#  Consistency 
# ─────────────────────────────────────────────

def consistency_loss(
    z_masked:  torch.Tensor,  
    z_full:    torch.Tensor,  
    used_mask: torch.Tensor, 
    full_mask: torch.Tensor,  
) -> torch.Tensor:

    drop_any = used_mask.sum(dim=1) < full_mask.sum(dim=1)   
    if drop_any.any():
        return F.mse_loss(z_masked[drop_any], z_full[drop_any].detach())
    return torch.zeros((), device=z_masked.device)


# ─────────────────────────────────────────────
#  ShaSpec auxiliary (DAO / DCO)
# ─────────────────────────────────────────────

def dao_uniform_loss(
    logits:   torch.Tensor,               
    row_mask: torch.Tensor | None = None, 
) -> torch.Tensor:
    logp   = F.log_softmax(logits, dim=-1)
    target = torch.full_like(logp, 0.25)
    per_row = -(target * logp).sum(dim=-1)  

    if row_mask is None:
        return per_row.mean()
    row_mask = row_mask.float()
    return (per_row * row_mask).sum() / row_mask.sum().clamp(min=1.0)


def dco_loss(
    logits:    torch.Tensor,              
    target_idx: int,
    row_mask:  torch.Tensor | None = None,  
) -> torch.Tensor:

    B      = logits.size(0)
    target = torch.full((B,), int(target_idx), device=logits.device, dtype=torch.long)
    per_row = F.cross_entropy(logits, target, reduction="none")

    if row_mask is None:
        return per_row.mean()
    row_mask = row_mask.float()
    return (per_row * row_mask).sum() / row_mask.sum().clamp(min=1.0)


# ─────────────────────────────────────────────
#  Distributional / prior losses
# ─────────────────────────────────────────────

def kl_to_standard_normal(
    mu:        torch.Tensor,              
    log_sigma: torch.Tensor,              
    row_mask:  torch.Tensor | None = None,
) -> torch.Tensor:

    logvar  = 2.0 * log_sigma
    var     = torch.exp(logvar)
    per_row = 0.5 * (mu.pow(2) + var - logvar - 1.0).sum(dim=-1)

    if row_mask is None:
        return per_row.mean()
    row_mask = row_mask.float()
    return (per_row * row_mask).sum() / row_mask.sum().clamp(min=1.0)


def sigma_reg_loss(
    log_sigma_list: list[torch.Tensor],
    mode: str = "log2",
    target_var: float = 1.0,
) -> torch.Tensor:

    if not log_sigma_list:
        return torch.zeros(())

    ls = torch.cat(log_sigma_list, dim=0)  
    if mode == "log2":
        return (ls ** 2).mean()
    elif mode == "meanvar":
        var = torch.exp(2.0 * ls)
        return ((var - target_var) ** 2).mean()
    else:
        raise ValueError(f"Unknown sigma_reg_loss mode: {mode}")


# ─────────────────────────────────────────────
#  Task loss (CE with smoothing + class weights)
# ─────────────────────────────────────────────

def task_cross_entropy(
    logits:         torch.Tensor,          
    targets:        torch.Tensor,          
    class_weights:  torch.Tensor | None,  
    label_smoothing: float = 0.0,
) -> torch.Tensor:

    return F.cross_entropy(
        logits,
        targets,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )