"""
Shared configuration and mechanics for attention architectures.

Every attention architecture in this package builds on `BaseAttention`, which
owns the pieces they all need: the head count, the resolved query/key and value
head dimensions, the score scaler, dtype/device, and reconciliation of the
aggregated value width back to `hidden_dim`. A concrete architecture implements
only its score function and value aggregation.

Attention is a pure hidden-state transform: it applies no padding masking, no
dropout, and no projection bias, and takes only the hidden state as input.

Dimension resolution (`qk_head_dim` / `v_head_dim`):

- neither supplied -> both become ``hidden_dim // num_heads`` (the historical
  single-head-dimension behavior; requires ``hidden_dim % num_heads == 0``);
- exactly one supplied -> both take that value;
- both supplied -> used independently.

Output-width reconciliation: the aggregated value path has width
``num_heads * v_head_dim``. When that differs from `hidden_dim`, the base adds
an output projection back to `hidden_dim`. In the default equal-dimension case
the widths match, so **no output projection is added** and the previous
parameterization is preserved.

Initialization: attention owns no initialization. A constructed architecture
keeps PyTorch's default parameter initialization; the composing task model
applies its single initialization policy across the whole model after
construction, and the encoder layer never initializes a passed attention module.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class BaseAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        qk_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dtype = dtype

        # Resolve per-branch head dimensions (see module docstring).
        if qk_head_dim is None and v_head_dim is None:
            assert (
                hidden_dim % num_heads == 0
            ), "hidden_dim must be divisible by num_heads when no head dimension is given."
            qk_head_dim = v_head_dim = hidden_dim // num_heads
        elif qk_head_dim is None:
            qk_head_dim = v_head_dim
        elif v_head_dim is None:
            v_head_dim = qk_head_dim

        assert qk_head_dim > 0 and v_head_dim > 0, "head dimensions must be positive."

        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_dim = num_heads * qk_head_dim
        self.v_dim = num_heads * v_head_dim
        self.scaler = 1 / math.sqrt(qk_head_dim)
        self.device = device

        # Reconcile the value width back to hidden_dim only when it differs.
        # Projections are always bias-free.
        self.out_proj = None
        if self.v_dim != hidden_dim:
            self.out_proj = nn.Linear(
                self.v_dim, hidden_dim, bias=False, dtype=dtype, device=device
            )

    # ------------------------------------------------------------------ #
    # Head-splitting / merge helpers
    # ------------------------------------------------------------------ #
    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """(B, N, H*D) -> (B, H, N, D)."""
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.num_heads, head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, N, D) -> (B, N, H*D)."""
        batch, heads, length, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, length, heads * head_dim)

    def _apply_out_proj(self, x: torch.Tensor) -> torch.Tensor:
        """Reconcile a value-width tensor back to hidden_dim (identity by default)."""
        return self.out_proj(x) if self.out_proj is not None else x
