"""
Task-agnostic prediction heads for the model layer.

These heads compose hidden states produced by any task's encoder stack and
must not import task-specific code — task models under
`compgen/models/tasks/` pick the head that matches their output type.
"""

import torch
import torch.nn as nn


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


class RegressionHead(nn.Module):
    """Per-token scalar regression head: LayerNorm -> Dropout -> Linear(hidden, 1).

    Mirrors the reference fuzzy-logic transformer's final normalization followed
    by a scalar output projection. Task-agnostic: it maps any
    `(batch, length, hidden)` hidden state to `(batch, length, 1)`.
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-6,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(hidden_dim, 1, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.dropout(self.norm(x)))