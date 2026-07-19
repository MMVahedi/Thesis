"""
Match3 classifier model: embedding -> stack of (attention + feed-forward)
encoder layers -> per-token binary classification head.

This reproduces the "33" experiment config from strassen-attention-neurips25's
`cmds/empirical/match3/hyperparams.json` (M=37, hidden_dim=128, 1 layer,
2 heads, dropout=0.4, no LayerNorm), restricted to comparing exactly two
attention mechanisms: "standard" and "strassen" (see `models/attentions/`).
"""

from typing import Optional

import torch
import torch.nn as nn

from models.attentions.standard import StandardAttention
from models.attentions.strassen import StrassenAttention
from models.embeddings.match3 import Match3Embedding

ATTENTION_CLASSES = {
    "standard": StandardAttention,
    "strassen": StrassenAttention,
}


def _init_attention_weights(attention: nn.Module) -> None:
    for name, param in attention.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class Match3EncoderLayer(nn.Module):
    """
    One encoder block: self-attention (standard or Strassen) + a 3-layer ReLU
    feed-forward network, each wrapped in a plain residual connection.

    `use_norm=False` (the paper's setting for this task) skips LayerNorm
    entirely, matching `BackboneTransformerLayer` in the reference repo.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_type: str,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        use_attention_dropout: bool = True,
        ffn_depth: int = 3,
        use_norm: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        if attention_type not in ATTENTION_CLASSES:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        self.attention = ATTENTION_CLASSES[attention_type](
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dtype=dtype,
            device=device,
            use_dropout=use_attention_dropout,
        )
        _init_attention_weights(self.attention)

        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                nn.ReLU(),
            )
            for _ in range(ffn_depth)
        ])

        self.use_norm = use_norm
        if use_norm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_state: torch.Tensor, batch_mask: Optional[dict] = None) -> torch.Tensor:
        x = self.norm1(hidden_state) if self.use_norm else hidden_state
        attention_output, _ = self.attention(hidden_state=x, batch_mask=batch_mask)
        hidden_state = hidden_state + self.dropout1(attention_output)

        x = self.norm2(hidden_state) if self.use_norm else hidden_state
        for layer in self.ffn:
            x = layer(x)
        hidden_state = hidden_state + self.dropout2(x)

        return hidden_state


class TokenClassifier(nn.Module):
    """Per-token binary classification head: Linear -> ReLU -> Linear -> 1 logit."""

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(hidden_dim, 1, dtype=dtype, device=device)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.activation(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class Match3Model(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        embedding_norm_scalar: float = 1.0,
        use_norm: bool = False,
        use_attention_dropout: bool = True,
        share_layers: bool = False,
        ffn_depth: int = 3,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = f"match3_{attention_type}_h{hidden_dim}_l{num_layers}_heads{num_heads}"

        self.embedding = Match3Embedding(
            hidden_dim=hidden_dim,
            embedding_norm_scalar=embedding_norm_scalar,
            dtype=dtype,
            device=device,
        )

        def make_layer() -> Match3EncoderLayer:
            return Match3EncoderLayer(
                hidden_dim=hidden_dim,
                attention_type=attention_type,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_attention_dropout=use_attention_dropout,
                ffn_depth=ffn_depth,
                use_norm=use_norm,
                dtype=dtype,
                device=device,
            )

        if share_layers:
            shared_layer = make_layer()
            self.layers = nn.ModuleList([shared_layer] * num_layers)
        else:
            self.layers = nn.ModuleList([make_layer() for _ in range(num_layers)])

        self.classifier = TokenClassifier(
            hidden_dim=hidden_dim, dropout_rate=dropout_rate, dtype=dtype, device=device
        )

    def forward(self, batch: dict):
        hidden_state = self.embedding(batch)
        for layer in self.layers:
            hidden_state = layer(hidden_state, batch_mask=batch["batch_mask"])

        logits = self.classifier(hidden_state)
        y_hat = torch.sigmoid(logits)
        return y_hat, hidden_state
