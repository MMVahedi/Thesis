# NEED REVIEW
"""
Triangular attention mixer (retained high-order variant).

A pair-level attention over triples of positions: it operates on sequence
*pairs* `(B, N, N, C)` and returns pair representations `(B, N, N, hidden_dim)`,
so unlike the other architectures it is **not sequence-composable** through
`EncoderLayer` (which feeds `(B, N, C)`). Refactored onto `BaseAttention`: the
two key projections use the query/key width and the two value projections use
the value width. Contractions and normalization are unchanged from the original
implementation. Attention applies no masking or dropout.
"""

import torch
import torch.nn as nn
from opt_einsum import contract

from compgen.models.attentions.base import BaseAttention


class TriangularAttention(BaseAttention):
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

        self.left_k = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)
        self.right_k = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)
        self.left_v = nn.Linear(hidden_dim, self.v_dim, bias=False, dtype=dtype, device=device)
        self.right_v = nn.Linear(hidden_dim, self.v_dim, bias=False, dtype=dtype, device=device)
        self.query = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_state):
        batch, length, _, _ = hidden_state.shape
        heads = self.num_heads

        left_k = self.left_k(hidden_state).view(batch, length, length, heads, self.qk_head_dim)
        right_k = self.right_k(hidden_state).view_as(left_k)
        left_v = self.left_v(hidden_state).view(batch, length, length, heads, self.v_head_dim)
        right_v = self.right_v(hidden_state).view_as(left_v)
        query = self.query(hidden_state).view_as(left_k)

        scores = contract("bxahd,bayhd->bxayh", left_k, right_k) * self.scaler

        val = contract("bxahd,bayhd->bxayhd", left_v, right_v)

        scores = scores - scores.max(dim=2, keepdim=True).values
        att_weights = scores.softmax(dim=2)

        att = contract("bxayh,bxayhd->bxyhd", att_weights, val)
        att = att.reshape(batch, length, length, self.v_dim)
        att = self._apply_out_proj(att)

        return att, att_weights
