"""
Linear attention mixer.

The sibling of `SoftmaxAttention`: the same projections, scaling, head split,
and output reconciliation, but the attention weights are the scaled scores
directly — the final softmax normalization is removed. Configuration and shared
mechanics live in `BaseAttention`.
"""

import torch
import torch.nn as nn
from opt_einsum import contract

from compgen.models.attentions.base import BaseAttention


class LinearAttention(BaseAttention):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 1,
        qk_head_dim: int = None,
        v_head_dim: int = None,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dtype=dtype,
            device=device,
        )

        # Projections are bias-free; see `SoftmaxAttention` for the row-vector
        # interpretation of `nn.Linear` over the last axis.
        self.query = nn.Linear(
            hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device
        )
        self.key = nn.Linear(
            hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device
        )
        self.value = nn.Linear(
            hidden_dim, self.v_dim, bias=False, dtype=dtype, device=device
        )

    def forward(self, hidden_state):
        q = self._split_heads(self.query(hidden_state), self.qk_head_dim)
        k = self._split_heads(self.key(hidden_state), self.qk_head_dim)
        v = self._split_heads(self.value(hidden_state), self.v_head_dim)

        scores = contract("bhid,bhjd->bhij", q, k) * self.scaler

        # No softmax: the scaled scores are the attention weights.
        att_weights = scores

        att = contract("bhij,bhjd->bhid", att_weights, v)
        att = self._merge_heads(att)
        att = self._apply_out_proj(att)

        return att, att_weights
