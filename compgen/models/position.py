"""
Reusable fixed absolute positional encoding.

Produces a deterministic, parameter-free positional encoding using the standard
sinusoidal formulation from "Attention Is All You Need" (Vaswani et al., 2017).
For position ``p`` and channel ``c`` (``i = c // 2``)::

    PE[p, c] = sin(p / 10000^(2i / hidden_dim))   if c is even
    PE[p, c] = cos(p / 10000^(2i / hidden_dim))   if c is odd

so each pair of channels shares one frequency, running from ``10000^0`` to
``10000^((hidden_dim - 2) / hidden_dim)``.

This is a task-agnostic block: it knows nothing about any task model. Consumers
add its output to a batch of token embeddings of shape ``(batch, length,
hidden_dim)``; the ``(length, hidden_dim)`` result broadcasts over the batch.
"""

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal absolute positional encoding (no learned parameters).

    ``forward(length)`` returns a ``(length, hidden_dim)`` tensor. The encoding
    has no parameters and no length bound: it is computed on demand, so it can
    be evaluated for any sequence length and adds nothing to a checkpoint.
    """

    def __init__(
        self,
        hidden_dim: int,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        length: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        channels = torch.arange(self.hidden_dim, device=device, dtype=dtype)
        # Each channel pair (2i, 2i + 1) shares the divisor 10000^(2i / hidden).
        divisor = torch.pow(
            10000.0,
            2.0 * (channels // 2) / self.hidden_dim,
        )
        angle = position / divisor
        return torch.where(
            (channels % 2 == 0).unsqueeze(0),
            torch.sin(angle),
            torch.cos(angle),
        )
