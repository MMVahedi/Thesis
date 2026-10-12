"""
Task-agnostic feed-forward blocks for the encoder layer.

Two variants, constructed directly by the caller:

- `ReluStackFFN`: the historical ``ffn_depth`` copies of
  ``Dropout -> Linear -> ReLU`` over the hidden width; unchanged behavior.
- `GeluFFN`: a two-projection ``Linear(ffn_dim) -> GELU(approximate="tanh")
  -> Dropout -> Linear(hidden)`` block, matching the reference fuzzy-logic
  transformer.

There is no name-keyed factory: a caller (task model) instantiates the variant
it needs and passes the constructed module into
`compgen.models.encoder.EncoderLayer`, which consults no feed-forward registry.

A block owns no initialization: it keeps PyTorch's default parameter
initialization, and the composing task model applies its single policy across
the whole model.
"""

from typing import Optional

import torch
import torch.nn as nn


class BaseFeedForward(nn.Module):
    """Shared construction context for feed-forward blocks."""

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dtype = dtype
        self.device = device


class ReluStackFFN(BaseFeedForward):
    """``ffn_depth`` blocks of ``Dropout -> Linear -> ReLU`` at the hidden width."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_depth: int = 3,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__(hidden_dim, dropout_rate, dtype, device)
        self.ffn_depth = ffn_depth
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                nn.ReLU(),
            )
            for _ in range(ffn_depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class GeluFFN(BaseFeedForward):
    """Two-projection ``Linear(ffn_dim) -> GELU(tanh) -> Dropout -> Linear(hidden)``."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__(hidden_dim, dropout_rate, dtype, device)
        self.ffn_dim = hidden_dim if ffn_dim is None else ffn_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, self.ffn_dim, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_rate),
            nn.Linear(self.ffn_dim, hidden_dim, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
