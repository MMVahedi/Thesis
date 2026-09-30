"""
Fuzzy-logic task embedding.

Projects a fuzzy-logic input batch ``x`` of shape
``(batch, length, num_variables + 1)`` to ``(batch, length, hidden_dim)`` with a
single linear projection followed by dropout, matching the reference
transformer's input embedding plus input dropout.
"""

import torch
import torch.nn as nn


class FuzzyLogicEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.input_proj = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype).to(self.device)
        return self.dropout(self.input_proj(x))
