from __future__ import annotations

import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.encoders.audio_encoder import AudioEncoder
from model.encoders.text_encoder import TextEncoder
from model.encoders.video_encoder import VideoEncoder
from model.encoders.mocap_encoder import MocapEncoder

from loss import (
    cross_modal_contrastive_loss,
    hub_contrastive_loss,
    orthogonality_loss,
    consistency_loss,
    dao_uniform_loss,
    dco_loss,
    kl_to_standard_normal,
    sigma_reg_loss,
    task_cross_entropy,
)


MOD_KEYS = ["A", "T", "V", "M"]
MOD_IDX  = {k: i for i, k in enumerate(MOD_KEYS)}


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def _grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class ShaSpecEAUModel(nn.Module):


    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.enc_audio = AudioEncoder(
            model_name=cfg.audio_model_name,
            d_model=cfg.d_model,
            freeze=cfg.freeze_audio,
            unfreeze_top_n_layers=0,  
            dropout=float(getattr(cfg, "audio_dropout", 0.1)),
        )
        self.enc_text = TextEncoder(
            model_name=cfg.text_model_name,
            d_model=cfg.d_model,
            max_len=cfg.max_text_len,
            freeze=cfg.freeze_text,
            unfreeze_top_n_layers=0,
            dropout=float(getattr(cfg, "text_dropout", 0.1)),
        )
        self.enc_video = VideoEncoder(
            backbone=cfg.video_backbone,
            d_model=cfg.d_model,
            freeze=cfg.freeze_video,
            embed_dim=int(getattr(cfg, "video_embed_dim", 512)),
            temporal_embed=bool(getattr(cfg, "video_temporal_embed", False)),
            dropout=float(getattr(cfg, "video_dropout", 0.1)),
        )
        self.enc_mocap = MocapEncoder(
            feat_dim=cfg.mocap_feat_dim,
            d_model=cfg.d_model,
            nhead=cfg.mocap_nhead,
            num_layers=cfg.mocap_layers,
            n_groups=int(getattr(cfg, "mocap_n_groups", 4)),
            max_len=int(getattr(cfg, "mocap_max_len", 400)),
            dropout=float(getattr(cfg, "mocap_dropout", 0.1)),
        )


        self.video_embed_proj = None   


        def mk_mu():
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_shared),
                nn.LayerNorm(cfg.d_shared),
                nn.GELU(),
            )

        def mk_logsig():

            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_shared),
                nn.LayerNorm(cfg.d_shared),
            )

        def mk_priv():
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_private),
                nn.LayerNorm(cfg.d_private),
                nn.GELU(),
            )

        self.muA, self.lsA, self.pA = mk_mu(), mk_logsig(), mk_priv()
        self.muT, self.lsT, self.pT = mk_mu(), mk_logsig(), mk_priv()
        self.muV, self.lsV, self.pV = mk_mu(), mk_logsig(), mk_priv()
        self.muM, self.lsM, self.pM = mk_mu(), mk_logsig(), mk_priv()

        d_proj = int(getattr(cfg, "d_proj", 128))
        self.con_proj = nn.Sequential(
            nn.Linear(cfg.d_shared, cfg.d_shared),
            nn.ReLU(),
            nn.Linear(cfg.d_shared, d_proj),
        )

        self.dao_head = nn.Linear(cfg.d_shared, 4)
        self.dco_head = nn.Linear(cfg.d_private, 4)

        n_speakers = int(getattr(cfg, "n_speakers", 10))
        self.speaker_adv_head = nn.Sequential(
            nn.Linear(cfg.d_shared, cfg.d_shared // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(cfg.d_shared // 2, n_speakers),
        )
        self._adv_alpha = 0.0   
        self.post_fuse_ln = nn.LayerNorm(cfg.d_shared)


        self.num_classes = int(getattr(cfg, "num_classes", 4))
        self.classifier  = nn.Sequential(
            nn.Linear(cfg.d_shared, cfg.d_shared),
            nn.GELU(),
            nn.Dropout(float(getattr(cfg, "enc_dropout", 0.1))),
            nn.Linear(cfg.d_shared, self.num_classes),
        )


        self._class_weights: Optional[torch.Tensor] = None
        if getattr(cfg, "use_class_weights", False) and cfg.class_weights is not None:
            self._class_weights = torch.tensor(
                list(cfg.class_weights), dtype=torch.float32
            )


    def _clamp_log_sigma(self, ls: torch.Tensor) -> torch.Tensor:
        mn = float(getattr(self.cfg, "log_sigma_min", -6.0))
        mx = float(getattr(self.cfg, "log_sigma_max",  2.0))
        return ls.clamp(mn, mx)


    def sample_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """Returns (B, 4) float mask for [A, T, V, M]."""
        cfg = self.cfg
        probs = [
            ("keep_all",   cfg.p_keep_all),
            ("drop_video", cfg.p_drop_video),
            ("drop_text",  cfg.p_drop_text),
            ("drop_audio", cfg.p_drop_audio),
            ("drop_mocap", cfg.p_drop_mocap),
            ("drop_two",   cfg.p_drop_two),
        ]
        total  = sum(p for _, p in probs)
        probs  = [(k, p / total) for k, p in probs]
        masks  = []

        for _ in range(B):
            r, acc, mode = random.random(), 0.0, "keep_all"
            for k, p in probs:
                acc += p
                if r <= acc:
                    mode = k
                    break

            m = [1, 1, 1, 1]
            if mode == "drop_video":
                m[2] = 0
            elif mode == "drop_text":
                m[1] = 0
            elif mode == "drop_audio":
                m[0] = 0
            elif mode == "drop_mocap":
                m[3] = 0
            elif mode == "drop_two":
                for di in random.sample([0, 1, 2, 3], 2):
                    m[di] = 0
                if sum(m) == 0:
                    m[random.randint(0, 3)] = 1

            masks.append(m)

        return torch.tensor(masks, dtype=torch.float32, device=device)


    def encode(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """Returns {A,T,V,M} ⊆ keys, each (B, d_model)."""
        h: Dict[str, torch.Tensor] = {}

        if batch.get("audio") is not None:
            h["A"] = self.enc_audio(batch["audio"].to(device, non_blocking=True))

        if batch.get("text") is not None:
            h["T"] = self.enc_text(batch["text"], device=device)

        # VideoEncoder DEEP v2 handles both (B,D) and (B,T,D) natively
        if batch.get("video_embed") is not None:
            ve = batch["video_embed"].to(device, non_blocking=True)
            if ve.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                ve = ve.float()
            h["V"] = self.enc_video(ve)
        elif batch.get("video") is not None:
            h["V"] = self.enc_video(batch["video"].to(device, non_blocking=True))

        if batch.get("mocap") is not None:
            h["M"] = self.enc_mocap(
                batch["mocap"].to(device, non_blocking=True),
                batch["mocap_len"].to(device, non_blocking=True),
            )

        return h



    def shared_private(
        self, h: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Returns dict[modality] → {mu, ls, p, proj}
          mu   : (B, d_shared)   — mean of shared Gaussian
          ls   : (B, d_shared)   — log σ of shared Gaussian
          p    : (B, d_private)  — private representation
          proj : (B, d_proj)     — projected mu (for contrastive only)
        """
        out: Dict[str, Dict[str, torch.Tensor]] = {}

        heads = {
            "A": (self.muA, self.lsA, self.pA),
            "T": (self.muT, self.lsT, self.pT),
            "V": (self.muV, self.lsV, self.pV),
            "M": (self.muM, self.lsM, self.pM),
        }

        for k, (mu_head, ls_head, p_head) in heads.items():
            if k not in h:
                continue
            mu   = mu_head(h[k])
            ls   = self._clamp_log_sigma(ls_head(h[k]))
            priv = p_head(h[k])
            proj = self.con_proj(mu)             
            out[k] = {"mu": mu, "ls": ls, "p": priv, "proj": proj}

        return out


    def fuse(
        self,
        reps: Dict[str, Dict[str, torch.Tensor]],
        mask: torch.Tensor,           # (B, 4)  [A,T,V,M]
    ) -> torch.Tensor:

        B      = next(iter(reps.values()))["mu"].size(0)
        device = next(iter(reps.values()))["mu"].device

        mu_list      = []
        var_sc_list  = []
        present_list = []

        for i, k in enumerate(MOD_KEYS):
            if k in reps:
                mu       = reps[k]["mu"]                       
                var_sc   = torch.exp(2.0 * reps[k]["ls"]).mean(dim=-1) 
                mu_list.append(mu)
                var_sc_list.append(var_sc)
                present_list.append(mask[:, i])
            else:
                mu_list.append(torch.zeros(B, self.cfg.d_shared, device=device))
                var_sc_list.append(torch.full((B,), 1e6, device=device))
                present_list.append(torch.zeros(B, device=device))

        MU  = torch.stack(mu_list,      dim=1) 
        VAR = torch.stack(var_sc_list,  dim=1)  
        PM  = torch.stack(present_list, dim=1)  

        inv   = PM * (1.0 / (VAR + 1e-8))
        denom = inv.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha = inv / denom                  

        z = (alpha.unsqueeze(-1) * MU).sum(dim=1)  
        return self.post_fuse_ln(z)


    @torch.no_grad()
    def get_z(
        self,
        batch: Dict,
        device: torch.device,
        apply_mask: bool = False,
    ) -> torch.Tensor:
        h    = self.encode(batch, device=device)
        reps = self.shared_private(h)

        hasA = batch["has_audio"].to(device, non_blocking=True).float()
        hasT = batch["has_text"].to(device, non_blocking=True).float()
        hasV = batch["has_video"].to(device, non_blocking=True).float()
        hasM = batch["has_mocap"].to(device, non_blocking=True).float()
        full_mask = torch.stack([hasA, hasT, hasV, hasM], dim=1)

        if apply_mask:
            drop_mask = self.sample_mask(full_mask.size(0), device)
            used_mask = full_mask * drop_mask
        else:
            used_mask = full_mask

        return self.fuse(reps, used_mask)

    def set_adv_alpha(self, alpha: float):
        """Call from train.py each epoch to ramp up gradient reversal."""
        self._adv_alpha = float(alpha)



    def compute_losses(
        self,
        batch:      Dict,
        device:     torch.device,
        train:      bool = True,
        apply_mask  = None,
    ) -> Dict[str, torch.Tensor]:

        h    = self.encode(batch, device=device)
        reps = self.shared_private(h)          
        B = next(iter(reps.values()))["mu"].size(0)

        # availability flags
        hasA = batch["has_audio"].to(device, non_blocking=True).float()
        hasT = batch["has_text"].to(device, non_blocking=True).float()
        hasV = batch["has_video"].to(device, non_blocking=True).float()
        hasM = batch["has_mocap"].to(device, non_blocking=True).float()
        full_mask = torch.stack([hasA, hasT, hasV, hasM], dim=1)  
        if apply_mask is None:
            apply_mask = train

        if apply_mask:
            drop_mask = self.sample_mask(B, device)
            used_mask = full_mask * drop_mask
        else:
            used_mask = full_mask

        z_masked = self.fuse(reps, used_mask)
        z_full   = self.fuse(reps, full_mask) if apply_mask else z_masked
        z_proj   = self.con_proj(z_full)      

        loss_con_cross = cross_modal_contrastive_loss(
            reps, used_mask, MOD_IDX,
            temperature=self.cfg.temperature,
            proj_key="proj",
        )
        loss_con_hub = hub_contrastive_loss(
            z_proj, reps, full_mask, MOD_IDX,
            temperature=self.cfg.temperature,
            proj_key="proj",
        )

        lambda_con_cross = float(getattr(self.cfg, "lambda_con_cross", 1.0))
        lambda_con_hub   = float(getattr(self.cfg, "lambda_con_hub",   0.5))
        loss_con = lambda_con_cross * loss_con_cross + lambda_con_hub * loss_con_hub

        loss_cons = consistency_loss(z_masked, z_full, used_mask, full_mask)

        loss_orth = torch.stack([
            orthogonality_loss(reps[k]["mu"], reps[k]["p"])
            for k in reps
        ]).mean()


        dao_parts, dco_parts = [], []
        for k in reps:
            mk = used_mask[:, MOD_IDX[k]]
            if mk.sum() < 1:
                continue
            dao_parts.append(dao_uniform_loss(self.dao_head(reps[k]["mu"]),  row_mask=mk))
            dco_parts.append(dco_loss(self.dco_head(reps[k]["p"]), MOD_IDX[k], row_mask=mk))

        loss_dao = torch.stack(dao_parts).mean() if dao_parts else torch.zeros((), device=device)
        loss_dco = torch.stack(dco_parts).mean() if dco_parts else torch.zeros((), device=device)

        kl_parts = []
        for k in reps:
            mk = used_mask[:, MOD_IDX[k]]
            if mk.sum() < 1:
                continue
            kl_parts.append(kl_to_standard_normal(reps[k]["mu"], reps[k]["ls"], row_mask=mk))
        loss_prior = torch.stack(kl_parts).mean() if kl_parts else torch.zeros((), device=device)

        lambda_sigma = float(getattr(self.cfg, "lambda_sigma_reg", 0.0))
        sigma_mode   = getattr(self.cfg, "sigma_reg_mode", "log2")
        loss_sigma   = sigma_reg_loss([reps[k]["ls"] for k in reps], mode=sigma_mode)

        loss_task = torch.zeros((), device=device)
        lambda_task = float(getattr(self.cfg, "lambda_task", 0.0))
        if lambda_task > 0:
            y_raw = [self.cfg.label_map.get(str(lab), -1) for lab in batch["label"]]
            y     = torch.tensor(y_raw, device=device, dtype=torch.long)
            keep  = (y >= 0) & (y < self.num_classes)

            if keep.sum() > 0:
                cw = (
                    self._class_weights.to(device)
                    if self._class_weights is not None
                    else None
                )
                logits    = self.classifier(z_full[keep])
                loss_task = task_cross_entropy(
                    logits, y[keep], cw,
                    label_smoothing=float(getattr(self.cfg, "label_smoothing", 0.0)),
                )

        loss_speaker = torch.zeros((), device=device)
        lambda_spk   = float(getattr(self.cfg, "lambda_speaker_adv", 0.0))
        if lambda_spk > 0 and self._adv_alpha > 0 and "speaker_id" in batch:

            alpha  = self._adv_alpha

            z_rev  = z_full + (z_full.detach() * (-1 - 1) + z_full * (1 + 1)) * 0

            z_rev  = _grad_reverse(z_full, alpha)
            spk_ids = batch["speaker_id"].to(device, non_blocking=True).long()
            spk_logits = self.speaker_adv_head(z_rev)
            loss_speaker = F.cross_entropy(spk_logits, spk_ids)

        lambda_cons  = float(getattr(self.cfg, "lambda_cons",  0.5))
        lambda_orth  = float(getattr(self.cfg, "lambda_orth",  0.05))
        lambda_dao   = float(getattr(self.cfg, "lambda_dao",   0.01))
        lambda_dco   = float(getattr(self.cfg, "lambda_dco",   0.01))
        lambda_prior = float(getattr(self.cfg, "lambda_prior", 0.0))

        total = (
            loss_con
            + lambda_cons  * loss_cons
            + lambda_orth  * loss_orth
            + lambda_task  * loss_task
            + lambda_dao   * loss_dao
            + lambda_dco   * loss_dco
            + lambda_prior * loss_prior
            + lambda_sigma * loss_sigma
            + lambda_spk   * loss_speaker
        )

        return {
            "total":     total,
            "con":       loss_con.detach(),
            "con_cross": loss_con_cross.detach(),
            "con_hub":   loss_con_hub.detach(),
            "cons":      loss_cons.detach(),
            "orth":      loss_orth.detach(),
            "task":      loss_task.detach(),
            "dao":       loss_dao.detach(),
            "dco":       loss_dco.detach(),
            "prior":     loss_prior.detach(),
            "sigma":     loss_sigma.detach(),
            "speaker":   loss_speaker.detach(),
        }