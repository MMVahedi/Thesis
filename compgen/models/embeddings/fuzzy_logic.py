"""
Fuzzy-logic task embedding.

Projects a fuzzy-logic input batch ``x`` of shape
``(batch, length, num_variables + 1)`` to ``(batch, length, hidden_dim)`` with a
single linear projection and adds a fixed absolute positional encoding. Position
enters once here, at the input, rather than as an attention bias inside the
encoder stack.
"""

import torch
import torch.nn as nn

from compgen.models.position import SinusoidalPositionalEncoding


class FuzzyLogicEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.input_proj = nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device)
        self.position = SinusoidalPositionalEncoding(
            hidden_dim, dtype=dtype, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.dtype).to(self.device)
        hidden = self.input_proj(x)
        hidden = hidden + self.position(
            hidden.shape[1], device=hidden.device, dtype=hidden.dtype
        )
        return hidden
