"""
Reusable relative-position bias block.

Produces an additive attention bias indexed by head and query/key position,
following the T5-style unidirectional bucketing used by the reference
fuzzy-logic transformer: nearby distances get their own bucket, farther
distances are spread logarithmically up to `max_distance`, and the bias is
looked up from a learned per-head embedding. The result broadcasts over the
batch dimension of attention scores `(B, H, N, N)`.

This is a task-agnostic block: it knows nothing about any task model, and
`EncoderLayer` can own one to feed attention through its `score_bias` argument.
"""

import math

import torch
import torch.nn as nn


def relative_position_buckets(
    relative_position: torch.Tensor, num_buckets: int, max_distance: int
) -> torch.Tensor:
    """Map integer relative positions to bucket indices (unidirectional, T5-style)."""
    n = torch.maximum(-relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (
        torch.log(n.float().clamp_min(1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
    return torch.where(is_small, n, large)


class RelativePositionBias(nn.Module):
    """Learned per-head additive attention bias from relative positions.

    `forward(length)` returns a `(1, num_heads, length, length)` tensor that
    can be added directly to attention scores.
    """

    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 16,
        max_distance: int = 16,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(
            num_buckets, num_heads, dtype=dtype, device=device
        )

    def forward(self, length: int, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = self.embedding.weight.device
        context = torch.arange(length, device=device)[:, None]
        memory = torch.arange(length, device=device)[None, :]
        buckets = relative_position_buckets(
            memory - context, self.num_buckets, self.max_distance
        )
        return self.embedding(buckets).permute(2, 0, 1).unsqueeze(0)
