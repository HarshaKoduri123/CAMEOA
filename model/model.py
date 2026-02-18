# model/model.py  (UPDATED ShaSpecEAUModel: masked full-batch InfoNCE + selective consistency + sigma reg + get_z)

import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoders.audio_encoder import AudioEncoder
from model.encoders.text_encoder import TextEncoder
from model.encoders.video_encoder import VideoEncoder
from model.encoders.mocap_encoder import MocapEncoder
from loss import (
    info_nce_loss_masked,
    orthogonality_loss,
    consistency_loss,
    sigma_reg_loss,
)


class ShaSpecEAUModel(nn.Module):
    """
    Clean A+T+V+M:
    - Encoders -> h^m (utterance-level)
    - ShaSpec: h^m -> shared s^m and private p^m
    - EAU: h^m -> sigma^m
    - Uncertainty-weighted fusion over available modalities -> z
    - Losses: hub InfoNCE (full batch negs) + consistency (only when dropped) + orth + optional classifier + sigma reg
    - Video : batch["video_embed"] exists (B, video_embed_dim).

    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # encoders
        self.enc_audio = AudioEncoder(cfg.audio_model_name, cfg.d_model, freeze=cfg.freeze_audio)
        self.enc_text = TextEncoder(cfg.text_model_name, cfg.d_model, cfg.max_text_len, freeze=cfg.freeze_text)

        # Optional raw-video encoder (only used if batch["video"] is provided)
        self.enc_video = VideoEncoder(cfg.video_backbone, cfg.d_model, freeze=cfg.freeze_video)

        self.enc_mocap = MocapEncoder(
            cfg.mocap_feat_dim,
            cfg.d_model,
            nhead=cfg.mocap_nhead,
            num_layers=cfg.mocap_layers
        )

        # If using precomputed embeddings (Option A): project (Dv -> d_model)
        video_embed_dim = int(getattr(cfg, "video_embed_dim", 512))
        self.video_embed_proj = nn.Sequential(
            nn.Linear(video_embed_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU()
        )

        # ShaSpec projections
        def mk_shared():
            return nn.Sequential(nn.Linear(cfg.d_model, cfg.d_shared), nn.LayerNorm(cfg.d_shared), nn.GELU())

        def mk_priv():
            return nn.Sequential(nn.Linear(cfg.d_model, cfg.d_private), nn.LayerNorm(cfg.d_private), nn.GELU())

        self.sA, self.pA = mk_shared(), mk_priv()
        self.sT, self.pT = mk_shared(), mk_priv()
        self.sV, self.pV = mk_shared(), mk_priv()
        self.sM, self.pM = mk_shared(), mk_priv()

        # Uncertainty heads (EAU-style): scalar sigma per modality
        def mk_sigma():
            return nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
                nn.Softplus(),
            )

        self.uA, self.uT, self.uV, self.uM = mk_sigma(), mk_sigma(), mk_sigma(), mk_sigma()

        # Optional classifier head
        self.num_classes = 4
        self.classifier = nn.Linear(cfg.d_shared, self.num_classes)

        # Optional post-fusion norm (helps retrieval geometry)
        self.post_fuse_ln = nn.LayerNorm(cfg.d_shared)

    # ------------------------
    # masking schedule
    # ------------------------
    def sample_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """
        Returns mask (B,4) for [A,T,V,M] where 1=keep, 0=drop
        """
        cfg = self.cfg
        probs = [
            ("keep_all", cfg.p_keep_all),
            ("drop_video", cfg.p_drop_video),
            ("drop_text", cfg.p_drop_text),
            ("drop_audio", cfg.p_drop_audio),
            ("drop_mocap", cfg.p_drop_mocap),
            ("drop_two", cfg.p_drop_two),
        ]
        total = sum(p for _, p in probs)
        probs = [(k, p / total) for k, p in probs]

        masks = []
        for _ in range(B):
            r = random.random()
            acc = 0.0
            mode = "keep_all"
            for k, p in probs:
                acc += p
                if r <= acc:
                    mode = k
                    break

            m = [1, 1, 1, 1]  # A,T,V,M
            if mode == "drop_video":
                m[2] = 0
            elif mode == "drop_text":
                m[1] = 0
            elif mode == "drop_audio":
                m[0] = 0
            elif mode == "drop_mocap":
                m[3] = 0
            elif mode == "drop_two":
                drop_idx = random.sample([0, 1, 2, 3], 2)
                for di in drop_idx:
                    m[di] = 0
                if sum(m) == 0:
                    m[random.randint(0, 3)] = 1

            masks.append(m)

        return torch.tensor(masks, dtype=torch.float32, device=device)

    # ------------------------
    # encode
    # ------------------------
    def encode(self, batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute h^m for modalities that exist in batch.
        Output: dict with keys subset of {A,T,V,M}, each (B, d_model)
        """
        h: Dict[str, torch.Tensor] = {}

        if batch.get("audio") is not None:
            h["A"] = self.enc_audio(batch["audio"].to(device, non_blocking=True))

        if batch.get("text") is not None:
            h["T"] = self.enc_text(batch["text"], device=device)

        if batch.get("video_embed") is not None:
            ve = batch["video_embed"].to(device, non_blocking=True)
            if ve.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                ve = ve.float()
            h["V"] = self.video_embed_proj(ve)
        elif batch.get("video") is not None:
            h["V"] = self.enc_video(batch["video"].to(device, non_blocking=True))

        if batch.get("mocap") is not None:
            h["M"] = self.enc_mocap(
                batch["mocap"].to(device, non_blocking=True),
                batch["mocap_len"].to(device, non_blocking=True),
            )

        return h

    def shared_private(self, h: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        out = {}
        if "A" in h:
            out["A"] = {"s": self.sA(h["A"]), "p": self.pA(h["A"]), "h": h["A"], "sig": self.uA(h["A"]).squeeze(-1)}
        if "T" in h:
            out["T"] = {"s": self.sT(h["T"]), "p": self.pT(h["T"]), "h": h["T"], "sig": self.uT(h["T"]).squeeze(-1)}
        if "V" in h:
            out["V"] = {"s": self.sV(h["V"]), "p": self.pV(h["V"]), "h": h["V"], "sig": self.uV(h["V"]).squeeze(-1)}
        if "M" in h:
            out["M"] = {"s": self.sM(h["M"]), "p": self.pM(h["M"]), "h": h["M"], "sig": self.uM(h["M"]).squeeze(-1)}
        return out

    def fuse(self, reps: Dict[str, Dict[str, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """
        reps: dict with keys in {"A","T","V","M"}
        mask: (B,4) corresponding to [A,T,V,M]
        returns z: (B, d_shared)
        """
        keys = ["A", "T", "V", "M"]
        B = next(iter(reps.values()))["s"].size(0)
        device = next(iter(reps.values()))["s"].device

        s_list = []
        sig_list = []
        present_mask = []
        for i, k in enumerate(keys):
            if k in reps:
                s_list.append(reps[k]["s"])
                sig_list.append(reps[k]["sig"].clamp(min=1e-6))
                present_mask.append(mask[:, i])
            else:
                s_list.append(torch.zeros((B, self.cfg.d_shared), device=device))
                sig_list.append(torch.ones((B,), device=device) * 1e6)
                present_mask.append(torch.zeros((B,), device=device))

        S = torch.stack(s_list, dim=1)         # (B,4,D)
        SIG = torch.stack(sig_list, dim=1)     # (B,4)
        PM = torch.stack(present_mask, dim=1)  # (B,4)

        inv = PM * (1.0 / (SIG ** 2))
        denom = inv.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha = inv / denom
        z = (alpha.unsqueeze(-1) * S).sum(dim=1)

        # ✅ helpful for stable contrastive / retrieval
        z = self.post_fuse_ln(z)
        return z

    # ------------------------
    # helper: get z for retrieval/eval
    # ------------------------
    @torch.no_grad()
    def get_z(self, batch: Dict, device: torch.device, apply_mask: bool):
        h = self.encode(batch, device=device)
        reps = self.shared_private(h)

        hasA = batch["has_audio"].to(device, non_blocking=True).float()
        hasT = batch["has_text"].to(device, non_blocking=True).float()
        hasV = batch["has_video"].to(device, non_blocking=True).float()
        hasM = batch["has_mocap"].to(device, non_blocking=True).float()
        full_mask = torch.stack([hasA, hasT, hasV, hasM], dim=1)

        if apply_mask:
            B = full_mask.size(0)
            drop_mask = self.sample_mask(B, device=device)
            used_mask = full_mask * drop_mask
        else:
            used_mask = full_mask

        return self.fuse(reps, used_mask)

    # ------------------------
    # losses
    # ------------------------
    def compute_losses(
        self,
        batch: Dict,
        device: torch.device,
        train: bool = True,
        apply_mask=None,   # None => defaults to train behavior
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(batch, device=device)
        reps = self.shared_private(h)

        B = next(iter(reps.values()))["s"].size(0)

        # availability flags from dataloader
        hasA = batch["has_audio"].to(device, non_blocking=True).float()
        hasT = batch["has_text"].to(device, non_blocking=True).float()
        hasV = batch["has_video"].to(device, non_blocking=True).float()
        hasM = batch["has_mocap"].to(device, non_blocking=True).float()
        full_mask = torch.stack([hasA, hasT, hasV, hasM], dim=1)

        if apply_mask is None:
            apply_mask = train

        if apply_mask:
            drop_mask = self.sample_mask(B, device=device)
            used_mask = full_mask * drop_mask
        else:
            used_mask = full_mask

        z_masked = self.fuse(reps, used_mask)

        if apply_mask:
            z_full = self.fuse(reps, full_mask)
        else:
            z_full = z_masked

        loss_con = torch.zeros((), device=device)
        count = 0
        mod_idx = {"A": 0, "T": 1, "V": 2, "M": 3}

        for k in ["A", "T", "V", "M"]:
            if k not in reps:
                continue
            mk = used_mask[:, mod_idx[k]]  # (B,)
            if mk.sum() < 2:
                continue

            loss_con = loss_con + info_nce_loss_masked(
                z_masked, reps[k]["s"], self.cfg.temperature, row_mask=mk
            )
            count += 1

        if count > 0:
            loss_con = loss_con / count

        loss_cons = consistency_loss(z_masked, z_full, used_mask, full_mask)
        loss_orth = torch.zeros((), device=device)
        orth_count = 0
        for k in reps.keys():
            loss_orth = loss_orth + orthogonality_loss(reps[k]["s"], reps[k]["p"])
            orth_count += 1
        if orth_count > 0:
            loss_orth = loss_orth / orth_count

        sigmas = [reps[k]["sig"] for k in reps.keys()]
        loss_sigma = sigma_reg_loss(sigmas, mode=getattr(self.cfg, "sigma_reg_mode", "log2"))

        loss_task = torch.zeros((), device=device)
        if self.cfg.lambda_task > 0:
            y = []
            for lab in batch["label"]:
                y.append(self.cfg.label_map.get(str(lab), -1))
            y = torch.tensor(y, device=device, dtype=torch.long)

            keep = (y >= 0) & (y <= 3)
            if keep.sum() > 0:
                logits = self.classifier(z_full[keep])
                loss_task = F.cross_entropy(logits, y[keep])

        lambda_sigma = float(getattr(self.cfg, "lambda_sigma", 0.0))

        total = (
            loss_con
            + self.cfg.lambda_cons * loss_cons
            + self.cfg.lambda_orth * loss_orth
            + self.cfg.lambda_task * loss_task
            + lambda_sigma * loss_sigma
        )

        return {
            "total": total,
            "con": loss_con.detach(),
            "cons": loss_cons.detach(),
            "orth": loss_orth.detach(),
            "task": loss_task.detach(),
            "sigma": loss_sigma.detach(),
        }
