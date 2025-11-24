"""
MIL + Transformer aggregator for study-level classification.
Produces study logits and attention maps (top-k slice indices).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1️⃣ Attention-based MIL Aggregator
# ============================================================
class AttentionMIL(nn.Module):
    """
    Lightweight Attention-based Multiple Instance Learning (MIL) model.
    Input: x [B, N, D] — batch of studies with N slice embeddings
    Output: logits [B, C], attn_weights [B, N]
    """

    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, num_classes: int = 2):
        super().__init__()
        hidden_dim = hidden_dim or max(embed_dim // 2, 64)

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D]
        B, N, D = x.shape
        scores = self.attention(x.view(B * N, D)).view(B, N)  # [B, N]
        attn = F.softmax(scores, dim=1)                       # normalize attention
        attn_exp = attn.unsqueeze(-1)                         # [B, N, 1]
        study_emb = torch.sum(attn_exp * x, dim=1)            # weighted sum
        logits = self.classifier(study_emb)                   # [B, num_classes]
        return logits, attn


# ============================================================
# 2️⃣ Positional Encoding for Transformer
# ============================================================
class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding.
    Adds positional information to slice embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        x = x + self.pe[:, :N, :D]
        return x


# ============================================================
# 3️⃣ Transformer Aggregator
# ============================================================
class TransformerAggregator(nn.Module):
    """
    Small Transformer-based aggregator using a CLS token to summarize slice embeddings.
    Returns study-level logits and per-slice attention-like weights (cosine similarity).
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int = 2,
        num_heads: int = 4,
        depth: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        max_slices: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_enc = PositionalEncoding(embed_dim, max_len=max_slices + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, D]
        B, N, D = x.shape

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)          # [B, N+1, D]
        x = self.pos_enc(x)

        # Transformer expects [S, B, D]
        x_t = x.transpose(0, 1)
        out = self.transformer(x_t)                   # [N+1, B, D]
        out = out.transpose(0, 1)                     # [B, N+1, D]

        # Extract CLS output
        cls_out = out[:, 0, :]                        # [B, D]
        cls_out = self.norm(cls_out)
        logits = self.classifier(cls_out)             # [B, num_classes]

        # Compute attention-like weights for interpretability
        slice_embs = out[:, 1:, :]                    # [B, N, D]
        cls_norm = cls_out.unsqueeze(1)               # [B, 1, D]
        attn_scores = F.cosine_similarity(cls_norm, slice_embs, dim=-1)  # [B, N]
        attn = F.softmax(attn_scores, dim=1)

        return logits, attn


# ============================================================
# 4️⃣ Factory Function
# ============================================================
def build_aggregator(cfg: dict):
    """
    Factory method for selecting MIL or Transformer aggregator.
    Example cfg:
      {
        'type': 'transformer',
        'embed_dim': 512,
        'num_classes': 2,
        'num_heads': 4,
        'depth': 2,
        'mlp_ratio': 2.0,
        'dropout': 0.1,
        'max_slices': 256
      }
    """
    t = cfg.get("type", "mil").lower()

    if t == "mil":
        return AttentionMIL(
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg.get("hidden_dim"),
            num_classes=cfg.get("num_classes", 2),
        )

    elif t == "transformer":
        return TransformerAggregator(
            embed_dim=cfg["embed_dim"],
            num_classes=cfg.get("num_classes", 2),
            num_heads=cfg.get("num_heads", 4),
            depth=cfg.get("depth", 2),
            mlp_ratio=cfg.get("mlp_ratio", 2.0),
            dropout=cfg.get("dropout", 0.1),
            max_slices=cfg.get("max_slices", 256),
        )

    else:
        raise ValueError(f"Unknown aggregator type: {t}")
