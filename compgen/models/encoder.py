"""
Task-agnostic transformer encoder layer: a registered attention architecture
plus a ReLU feed-forward stack, each wrapped in a plain residual connection.

This module composes only shared pieces (the attention registry in
`compgen/models/attentions/`) and must not import task-specific code — task
models under `compgen/models/tasks/` assemble this layer with their own
task-specific embedding.
"""

from typing import Optional

import torch
import torch.nn as nn

from compgen.models.attentions import ATTENTION_CLASSES


def _init_attention_weights(attention: nn.Module) -> None:
    for name, param in attention.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class EncoderLayer(nn.Module):
    """
    One encoder block: self-attention (any architecture registered in
    `compgen.models.attentions.ATTENTION_CLASSES`) + a `ffn_depth`-layer ReLU
    feed-forward network, each wrapped in a plain residual connection.

    `use_norm=False` skips LayerNorm entirely (the strassen-attention paper's
    setting for Match3), matching `BackboneTransformerLayer` in the reference
    repo.
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