"""
Softmax attention mixer.

Scaled dot-product attention with independent query/key and value head
dimensions (see `BaseAttention`). Scores are normalized with softmax over
keys, which applies per-row stabilization internally, then used to aggregate
the values. Configuration and shared mechanics live in `BaseAttention`.
"""

import torch
import torch.nn as nn
from opt_einsum import contract

from compgen.models.attentions.base import BaseAttention


class SoftmaxAttention(BaseAttention):
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

        # nn.Linear operates on the last axis, applying one weight matrix W to
        # every token independently: (B, N, hidden_dim) -> (B, N, *_dim), i.e.
        # y_i = W x_i for each row-vector x_i. This is the textbook Wq X
        # (PyTorch just stores W transposed as (out, in)). The output stacks
        # num_heads contiguous per-head blocks of *_head_dim, which
        # _split_heads re-exposes as the head axis. Projections are bias-free.
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

        # No explicit max-subtraction here: torch.softmax already normalizes
        # with the numerically stable (per-row max-subtracted) formulation.
        att_weights = scores.softmax(dim=-1)

        att = contract("bhij,bhjd->bhid", att_weights, v)
        att = self._merge_heads(att)
        att = self._apply_out_proj(att)

        return att, att_weights
