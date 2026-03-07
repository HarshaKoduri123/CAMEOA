# model/encoders/text_encoder.py  (DEEP v2)
#
# Architecture:
#   RoBERTa backbone (optionally partial-unfreeze)
#   → Hierarchical Token Attention  (word-level → phrase-level → sentence-level)
#   → Emotion Lexicon Gate          (amplify emotionally salient tokens)
#   → Span Feature Extractor        (contiguous emotion spans via conv)
#   → Deep Adapter MLP (3 layers + residual)
#   → d_model output
#
# Motivation:
#   CLS token misses span-level and local emotional cues.
#   "I'm absolutely NOT happy" — CLS may average out the negation.
#   Hierarchical attention + span convolutions capture these patterns.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers (same as audio_encoder, kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(1, dim // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, dim // reduction), dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# ─────────────────────────────────────────────────────────────────────────────
#  Hierarchical Attention Pooling
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalAttentionPool(nn.Module):
    """
    3-level pooling over transformer token sequence:
      Level 1 (word):    scored attention over all tokens
      Level 2 (phrase):  local window attention (±3 tokens)
      Level 3 (sentence): CLS + SEP anchored summary
    All three are fused via a small MLP.
    """
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        # Word-level: scalar attention score per token
        self.word_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        # Phrase-level: local conv window
        self.phrase_conv = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3, groups=hidden),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.phrase_attn = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        # Sentence-level: CLS + attended summary with SEP
        self.sent_proj = nn.Linear(hidden * 2, hidden)

        # Fusion of 3 levels
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(
        self,
        token_emb:      torch.Tensor,     # (B, L, H)
        attention_mask: torch.Tensor,     # (B, L)  1=real, 0=pad
    ) -> torch.Tensor:
        B, L, H = token_emb.shape
        mask_float = attention_mask.float()
        mask_bool  = attention_mask.bool()

        # ── Level 1: word attention ──────────────────────────────────────────
        scores = self.word_attn(token_emb).squeeze(-1)        # (B, L)
        scores = scores.masked_fill(~mask_bool, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        word_pool = (weights * token_emb).sum(dim=1)           # (B, H)

        # ── Level 2: phrase (local conv) attention ───────────────────────────
        x_t = token_emb.transpose(1, 2)   # (B, H, L)
        phrase_feat = self.phrase_conv(x_t).transpose(1, 2)   # (B, L, H)
        p_scores = self.phrase_attn(phrase_feat).squeeze(-1)
        p_scores = p_scores.masked_fill(~mask_bool, -1e9)
        p_weights = torch.softmax(p_scores, dim=1).unsqueeze(-1)
        phrase_pool = (p_weights * phrase_feat).sum(dim=1)     # (B, H)

        # ── Level 3: CLS + [SEP-like summary] ───────────────────────────────
        cls_emb  = token_emb[:, 0, :]                          # (B, H)
        # Last real token (before pad) as sentence-end anchor
        lens     = mask_float.sum(dim=1).long().clamp(min=1) - 1
        last_emb = token_emb[torch.arange(B), lens]            # (B, H)
        sent_pool = self.sent_proj(torch.cat([cls_emb, last_emb], dim=-1))  # (B, H)

        # ── Fuse ─────────────────────────────────────────────────────────────
        return self.fuse(torch.cat([word_pool, phrase_pool, sent_pool], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion Lexicon Gate
# ─────────────────────────────────────────────────────────────────────────────

class EmotionLexiconGate(nn.Module):
    """
    Learns to amplify tokens that carry emotional content.
    Implemented as a token-level gating network — no hard-coded lexicon needed,
    the gate learns from supervision which tokens matter.
    Also uses a negation-aware span detector to handle "not happy" etc.
    """
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        # Token-level gate: predict salience ∈ (0,1) per token
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        # Bigram context: each token sees its neighbours before gating
        self.context_conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        token_emb:      torch.Tensor,   # (B, L, H)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Returns gated token embeddings (B, L, H)."""
        # Add bigram context
        ctx   = self.context_conv(token_emb.transpose(1, 2)).transpose(1, 2)
        x     = self.norm(token_emb + ctx)
        gate  = self.gate(x)                             # (B, L, 1) ∈ (0,1)
        # Keep padding as zero
        mask  = attention_mask.unsqueeze(-1).float()
        return x * (1.0 + gate) * mask                  # amplify, not replace


# ─────────────────────────────────────────────────────────────────────────────
#  Span Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class SpanFeatureExtractor(nn.Module):
    """
    Multi-kernel 1D convolutions to capture emotion spans of different lengths:
      · bigram   (k=2): "very sad", "not good"
      · trigram  (k=3): "I feel bad"
      · 4-gram   (k=4): "I am so angry"
      · 5-gram   (k=5): longer phrases
    Each scale → attention pool → concatenate → project
    """
    def __init__(self, hidden: int, n_filters: int = 128, dropout: float = 0.1):
        super().__init__()
        self.kernels = [2, 3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, n_filters, kernel_size=k, padding=k // 2),
                nn.GELU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=1),
                nn.GELU(),
            )
            for k in self.kernels
        ])
        self.proj = nn.Sequential(
            nn.Linear(n_filters * len(self.kernels), hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_emb:      torch.Tensor,   # (B, L, H)
        attention_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        x   = token_emb.transpose(1, 2)   # (B, H, L)
        mask = attention_mask.float().unsqueeze(1)  # (B, 1, L)

        pooled = []
        for conv in self.convs:
            feat = conv(x)                                          # (B, n_filters, L')
            feat_len = feat.size(2)
            m = F.interpolate(mask, size=feat_len, mode="nearest")  # (B, 1, L')
            # Zero out padding, set to -inf so max-pool ignores it
            feat = feat * m + (1.0 - m) * (-1e9)
            pooled.append(feat.max(dim=2).values)                   # (B, n_filters)

        return self.proj(torch.cat(pooled, dim=-1))  # (B, H)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Deep text encoder for speech emotion recognition.

    RoBERTa → EmotionLexiconGate (token amplification)
            → HierarchicalAttentionPool (word/phrase/sentence)
            + SpanFeatureExtractor (bigram..5gram conv)
            → Fusion MLP
            → Deep Adapter (3x ResidualMLP + SE)
            → d_model
    """

    def __init__(
        self,
        model_name: str,
        d_model:    int,
        max_len:    int,
        freeze:     bool  = True,
        unfreeze_top_n_layers: int = 0,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone  = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size   # 768 for roberta-base
        self.max_len = max_len

        # Freeze all
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Optionally unfreeze top N transformer layers
        if unfreeze_top_n_layers > 0:
            layers = self.backbone.encoder.layer
            n = len(layers)
            for i in range(n - unfreeze_top_n_layers, n):
                for p in layers[i].parameters():
                    p.requires_grad = True
            for p in self.backbone.pooler.parameters():
                p.requires_grad = True

        # Emotion lexicon gate
        self.lex_gate = EmotionLexiconGate(hidden, dropout=dropout)

        # Hierarchical attention pool
        self.hier_pool = HierarchicalAttentionPool(hidden, dropout=dropout)

        # Span feature extractor
        self.span_feat = SpanFeatureExtractor(hidden, n_filters=128, dropout=dropout)

        # Fuse hier_pool + span_feat
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Project hidden → d_model
        self.pre_proj = nn.Sequential(
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Deep adapter
        self.adapter = nn.Sequential(
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=4, dropout=dropout),
            SqueezeExcite(d_model),
            ResidualMLP(d_model, expansion=2, dropout=dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        texts: list[str] of length B
        returns: (B, d_model)
        """
        texts = [(t if isinstance(t, str) else "") for t in texts]
        toks  = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        grad_ctx = torch.set_grad_enabled(
            any(p.requires_grad for p in self.backbone.parameters())
        )
        with grad_ctx:
            out = self.backbone(**toks)

        token_emb = out.last_hidden_state          # (B, L, 768)
        attn_mask = toks["attention_mask"]         # (B, L)

        # Emotion lexicon gating
        gated = self.lex_gate(token_emb, attn_mask)  # (B, L, 768)

        # Hierarchical attention pool
        hier = self.hier_pool(gated, attn_mask)       # (B, 768)

        # Span features
        span = self.span_feat(gated, attn_mask)       # (B, 768)

        # Fuse
        fused = self.fuse(torch.cat([hier, span], dim=-1))  # (B, 768)

        # Project + deep adapter
        out = self.pre_proj(fused)   # (B, d_model)
        out = self.adapter(out)      # (B, d_model)
        return out