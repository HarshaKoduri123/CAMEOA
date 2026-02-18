import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel



class TextEncoder(nn.Module):
    """
    BERT/RoBERTa -> CLS pool -> adapter to d_model
    """
    def __init__(self, model_name: str, d_model: int, max_len: int, freeze: bool = True):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden = self.backbone.config.hidden_size
        self.max_len = max_len

        self.adapter = nn.Sequential(
            nn.Linear(self.hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """
        texts: list[str] length B
        returns hT: (B, d_model)
        """
        # handle missing/None texts by empty string
        texts = [(t if isinstance(t, str) else "") for t in texts]
        toks = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(device)

        out = self.backbone(**toks).last_hidden_state  # (B, L, hidden)
        cls = out[:, 0, :]  # (B, hidden)
        return self.adapter(cls)
