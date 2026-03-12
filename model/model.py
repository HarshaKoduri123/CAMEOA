from __future__ import annotations

import math
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2Model, AutoModel


# =========================================================
# Encoder modules
# =========================================================

class AudioEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(cfg.audio_model_name)
        hidden = self.backbone.config.hidden_size

        if bool(cfg.freeze_audio):
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(float(cfg.audio_dropout)),
            nn.Linear(hidden, int(cfg.d_model)),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        attn_mask = (audio != 0).long() if audio.dim() == 2 else None
        out = self.backbone(input_values=audio, attention_mask=attn_mask)
        x = out.last_hidden_state.mean(dim=1)
        return self.proj(x)


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(cfg.text_model_name)
        hidden = self.backbone.config.hidden_size

        if bool(cfg.freeze_text):
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(float(cfg.text_dropout)),
            nn.Linear(hidden, int(cfg.d_model)),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            x = out.pooler_output
        else:
            x = out.last_hidden_state[:, 0]
        return self.proj(x)


class VideoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = int(cfg.video_embed_dim)
        d_model = int(cfg.d_model)

        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(float(cfg.video_dropout)),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.video_dropout)),
            nn.Linear(d_model, d_model),
        )

    def forward(self, video_embed: torch.Tensor) -> torch.Tensor:
        return self.net(video_embed)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MoCapEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        feat_dim = int(cfg.mocap_feat_dim)
        d_model = int(cfg.d_model)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(float(cfg.mocap_dropout)),
        )
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=int(cfg.mocap_max_len) + 8)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(cfg.mocap_nhead),
            dim_feedforward=4 * d_model,
            dropout=float(cfg.mocap_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg.mocap_layers))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, mocap: torch.Tensor, mocap_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(mocap)
        x = self.pos(x)

        key_padding_mask = None
        if mocap_len is not None:
            B, T, _ = mocap.shape
            idx = torch.arange(T, device=mocap.device).unsqueeze(0).expand(B, T)
            key_padding_mask = idx >= mocap_len.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if mocap_len is not None:
            mask = (~key_padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            x = x.mean(dim=1)

        return self.out_proj(x)


# =========================================================
# CAMEO building blocks
# =========================================================

class GaussianProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.mu = nn.Linear(d_in, d_out)
        self.log_sigma = nn.Linear(d_in, d_out)

    def forward(self, h: torch.Tensor):
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        return mu, log_sigma


class FeedForwardRefinement(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.mlp(self.norm(z))


# =========================================================
# Main CAMEO model
# =========================================================

class CAMEOModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.modalities = ["audio", "video", "text", "mocap"]
        self.mod_idx = {"audio": 0, "video": 1, "text": 2, "mocap": 3}

        d_model = int(cfg.d_model)
        d_shared = int(cfg.d_shared)

        # Encoders
        self.enc_audio = AudioEncoder(cfg)
        self.enc_video = VideoEncoder(cfg)
        self.enc_text = TextEncoder(cfg)
        self.enc_mocap = MoCapEncoder(cfg)

        # Gaussian heads
        self.gauss = nn.ModuleDict({
            "audio": GaussianProjection(d_model, d_shared),
            "video": GaussianProjection(d_model, d_shared),
            "text": GaussianProjection(d_model, d_shared),
            "mocap": GaussianProjection(d_model, d_shared),
        })

        # Refiner + classifier
        self.refiner = FeedForwardRefinement(d_shared, dropout=float(cfg.classifier_dropout))
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_shared),
            nn.Dropout(float(cfg.classifier_dropout)),
            nn.Linear(d_shared, int(cfg.num_classes)),
        )

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def _get_text_inputs(self, batch: Dict, device: torch.device):
        if "text_input_ids" in batch:
            input_ids = batch["text_input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("text_attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            return input_ids, attention_mask

        if "input_ids" in batch:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            return input_ids, attention_mask

        return None, None

    def _labels_to_tensor(self, labels, device):
        if torch.is_tensor(labels):
            return labels.to(device)

        out = []
        for l in labels:
            if isinstance(l, str):
                out.append(self.cfg.label_map[l])
            else:
                out.append(int(l))
        return torch.tensor(out, dtype=torch.long, device=device)

    def _full_presence_mask(self, batch: Dict, device: torch.device) -> torch.Tensor:
        has_audio = batch["has_audio"].float().to(device)
        has_video = batch["has_video"].float().to(device)
        has_text = batch["has_text"].float().to(device)
        has_mocap = batch["has_mocap"].float().to(device)
        return torch.stack([has_audio, has_video, has_text, has_mocap], dim=1)

    def _sample_modality_mask(self, batch: Dict, device: torch.device) -> torch.Tensor:
        mask = self._full_presence_mask(batch, device).clone()
        B = mask.size(0)

        probs = [
            float(self.cfg.p_keep_all),
            float(self.cfg.p_drop_video),
            float(self.cfg.p_drop_text),
            float(self.cfg.p_drop_audio),
            float(self.cfg.p_drop_mocap),
            float(self.cfg.p_drop_two),
        ]
        s = sum(probs)
        probs = [p / s for p in probs]

        choices = torch.multinomial(torch.tensor(probs, device=device), num_samples=B, replacement=True)

        for i in range(B):
            c = int(choices[i].item())

            if c == 0:
                pass
            elif c == 1:
                mask[i, self.mod_idx["video"]] = 0.0
            elif c == 2:
                mask[i, self.mod_idx["text"]] = 0.0
            elif c == 3:
                mask[i, self.mod_idx["audio"]] = 0.0
            elif c == 4:
                mask[i, self.mod_idx["mocap"]] = 0.0
            elif c == 5:
                drop_ids = random.sample([0, 1, 2, 3], 2)
                mask[i, drop_ids[0]] = 0.0
                mask[i, drop_ids[1]] = 0.0

            if mask[i].sum() < 1:
                alive = torch.where(self._full_presence_mask(batch, device)[i] > 0)[0]
                if len(alive) > 0:
                    mask[i, alive[0]] = 1.0

        return mask

    # -----------------------------------------------------
    # Core forward pieces
    # -----------------------------------------------------
    def encode(self, batch: Dict, device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
        h = {}

        # Audio
        h["audio"] = None
        if batch.get("audio") is not None:
            h["audio"] = self.enc_audio(batch["audio"].to(device, non_blocking=True))

        # Video
        h["video"] = None
        if batch.get("video_embed") is not None:
            h["video"] = self.enc_video(batch["video_embed"].to(device, non_blocking=True))

        # Text
        h["text"] = None
        input_ids, attention_mask = self._get_text_inputs(batch, device)
        if input_ids is not None:
            h["text"] = self.enc_text(input_ids=input_ids, attention_mask=attention_mask)

        # MoCap
        h["mocap"] = None
        if batch.get("mocap") is not None:
            mocap = batch["mocap"].to(device, non_blocking=True)
            mocap_len = batch.get("mocap_len", None)
            if mocap_len is not None:
                mocap_len = mocap_len.to(device, non_blocking=True)
            h["mocap"] = self.enc_mocap(mocap, mocap_len)

        return h

    def project_distributions(self, h: Dict[str, Optional[torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        reps = {}
        for m in self.modalities:
            if h[m] is None:
                reps[m] = None
                continue

            mu, log_sigma = self.gauss[m](h[m])
            log_sigma = torch.clamp(
                log_sigma,
                min=float(self.cfg.log_sigma_min),
                max=float(self.cfg.log_sigma_max),
            )
            sigma = torch.exp(log_sigma)
            reps[m] = {
                "mu": mu,
                "log_sigma": log_sigma,
                "sigma": sigma,
            }
        return reps

    def fuse(self, reps: Dict[str, Dict[str, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        mus = []
        precisions = []

        for m in self.modalities:
            rep = reps[m]
            if rep is None:
                continue

            idx = self.mod_idx[m]
            present = mask[:, idx].unsqueeze(-1)

            mu = rep["mu"]
            sigma = rep["sigma"]
            precision = 1.0 / (sigma ** 2 + 1e-8)

            mu = mu * present
            precision = precision * present

            mus.append(mu)
            precisions.append(precision)

        precision_sum = torch.stack(precisions, dim=0).sum(dim=0).clamp_min(1e-8)
        weighted_sum = sum([p * mu for p, mu in zip(precisions, mus)])
        z = weighted_sum / precision_sum
        return z

    def forward(self, batch: Dict, device: torch.device, apply_mask: bool = False):
        h = self.encode(batch, device)

        reps = self.project_distributions(h)

        # Debug Gaussian projection
        for m, rep in reps.items():
            if rep is None:
                continue

            mu = rep["mu"]
            sigma = rep["sigma"]
            log_sigma = rep["log_sigma"]


        if apply_mask:
            mask = self._sample_modality_mask(batch, device)
        else:
            mask = self._full_presence_mask(batch, device)

        z = self.fuse(reps, mask)
        z_refined = self.refiner(z)
        logits = self.classifier(z_refined)

        return {
            "h": h,
            "reps": reps,
            "mask": mask,
            "z": z,
            "z_refined": z_refined,
            "logits": logits,
        }

    # -----------------------------------------------------
    # Losses
    # -----------------------------------------------------
    def alignment_loss(self, z: torch.Tensor, reps: Dict[str, Dict[str, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """
        Contrastive-style alignment between fused representation z and each available modality mean mu_m.
        """
        tau = float(self.cfg.temperature)
        z = F.normalize(z, dim=-1)

        losses = []
        for m in self.modalities:
            rep = reps[m]
            if rep is None:
                continue

            idx = self.mod_idx[m]
            present = mask[:, idx] > 0
            if present.sum() < 2:
                continue

            z_m = z[present]
            mu_m = F.normalize(rep["mu"][present], dim=-1)

            logits = z_m @ mu_m.T / tau
            targets = torch.arange(logits.size(0), device=logits.device)
            losses.append(F.cross_entropy(logits, targets))

        if len(losses) == 0:
            return z.new_tensor(0.0)

        return torch.stack(losses).mean()

    def kl_loss(self, reps: Dict[str, Dict[str, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """
        KL( N(mu, sigma^2) || N(0, I) )
        """
        losses = []
        for m in self.modalities:
            rep = reps[m]
            if rep is None:
                continue

            idx = self.mod_idx[m]
            present = mask[:, idx].unsqueeze(-1)

            mu = rep["mu"]
            log_sigma = rep["log_sigma"]
            sigma2 = torch.exp(2.0 * log_sigma)

            kl = 0.5 * (mu.pow(2) + sigma2 - 1.0 - 2.0 * log_sigma)
            kl = kl * present
            denom = present.sum().clamp_min(1.0)
            losses.append(kl.sum() / denom)

        if len(losses) == 0:
            return mask.new_tensor(0.0, dtype=torch.float32)

        return torch.stack(losses).mean()

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if bool(self.cfg.use_class_weights) and self.cfg.class_weights is not None:
            weight = torch.tensor(self.cfg.class_weights, dtype=torch.float32, device=logits.device)
        else:
            weight = None

        return F.cross_entropy(
            logits,
            labels,
            weight=weight,
            label_smoothing=float(self.cfg.label_smoothing),
        )

    def compute_losses(self, batch: Dict, device: torch.device, train: bool = True, apply_mask: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(batch, device=device, apply_mask=apply_mask)
        labels = self._labels_to_tensor(batch["label"], device)

        loss_align = self.alignment_loss(out["z"], out["reps"], out["mask"])
        loss_kl = self.kl_loss(out["reps"], out["mask"])
        loss_cls = self.classification_loss(out["logits"], labels)

        total = (
            float(self.cfg.lambda_align) * loss_align
            + float(self.cfg.lambda_kl) * loss_kl
            + float(self.cfg.lambda_cls) * loss_cls
        )

        return {
            "total": total,
            "align": loss_align.detach(),
            "kl": loss_kl.detach(),
            "cls": loss_cls.detach(),
        }