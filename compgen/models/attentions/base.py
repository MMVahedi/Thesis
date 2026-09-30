"""
Shared configuration and mechanics for attention architectures.

Every attention architecture in this package builds on `BaseAttention`, which
owns the pieces they all need: the head count, the resolved query/key and value
head dimensions, the score scaler, dropout, dtype/device, padding-mask
construction, and reconciliation of the aggregated value width back to
`hidden_dim`. A concrete architecture implements only its score function and
value aggregation.

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
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        bias: bool = False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dtype = dtype
        self.bias = bias

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
        self.mask_padding_value = mask_padding_value
        self.device = device

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        # Reconcile the value width back to hidden_dim only when it differs.
        self.out_proj = None
        if self.v_dim != hidden_dim:
            self.out_proj = nn.Linear(
                self.v_dim, hidden_dim, bias=bias, dtype=dtype, device=device
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

    def _drop_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return self.dropout(weights) if self.use_dropout else weights

    # ------------------------------------------------------------------ #
    # Mask helpers
    # ------------------------------------------------------------------ #
    def _flatten_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.flatten(1).to(self.device)

    def construct_mask(self, attention_mask: torch.Tensor, layout: str) -> torch.Tensor:
        """
        Expand a `(B, N)` padding mask (True = padding) to a score-shaped
        boolean mask.

        Layouts:
        - ``"b1nn"``  -> (B, 1, N, N)          (broadcastable over heads)
        - ``"bnnn1"`` -> (B, N, N, N, 1)       (pair-level triangular scores)
        - ``"b1nn2"`` -> (B, 1, N, N*N)        (third-order scores over N*N keys)
        """
        mask = self._flatten_padding_mask(attention_mask)
        if layout == "b1nn":
            return (mask.unsqueeze(1) + mask.unsqueeze(2)).unsqueeze(1)
        if layout == "bnnn1":
            pair = mask.unsqueeze(2).unsqueeze(3)
            pair = pair + mask.unsqueeze(1).unsqueeze(3)
            pair = pair + mask.unsqueeze(1).unsqueeze(2)
            return pair.unsqueeze(4)
        if layout == "b1nn2":
            pair = mask.unsqueeze(2) + mask.unsqueeze(1)
            pair = pair.unsqueeze(3) + mask.unsqueeze(1).unsqueeze(2)
            return pair.unsqueeze(1).flatten(-2)
        raise ValueError(f"Unknown mask layout: {layout}")
