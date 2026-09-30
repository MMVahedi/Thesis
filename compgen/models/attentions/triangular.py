"""
Triangular attention mixer (retained high-order variant).

A pair-level attention over triples of positions: it operates on sequence
*pairs* `(B, N, N, C)` and returns pair representations `(B, N, N, hidden_dim)`,
so unlike the other architectures it is **not sequence-composable** through
`EncoderLayer` (which feeds `(B, N, C)`). Refactored onto `BaseAttention`: the
two key projections use the query/key width and the two value projections use
the value width. Contractions and normalization are unchanged from the original
implementation.

Divergence from the original: padding-mask expansion now flattens the mask and
builds a well-defined `(B, N, N, N, 1)` layout, whereas the original chained
`unsqueeze` calls only made sense for a 3-D mask and were never exercised (this
variant has no sequence-composable mask path in the task models).
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
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        bias: bool = False,
        mask_padding_value: float = -1e4,
        device: str = "cpu",
        use_dropout: bool = True,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            bias=bias,
            mask_padding_value=mask_padding_value,
            device=device,
            use_dropout=use_dropout,
        )

        self.left_k = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)
        self.right_k = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)
        self.left_v = nn.Linear(hidden_dim, self.v_dim, bias=bias, dtype=dtype, device=device)
        self.right_v = nn.Linear(hidden_dim, self.v_dim, bias=bias, dtype=dtype, device=device)
        self.query = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, hidden_state, batch_mask=None, score_bias=None):
        # `score_bias` is accepted for interface uniformity with `EncoderLayer`;
        # this pair-level variant does not use an additive bias.
        batch, length, _, _ = hidden_state.shape
        heads = self.num_heads

        left_k = self.left_k(hidden_state).view(batch, length, length, heads, self.qk_head_dim)
        right_k = self.right_k(hidden_state).view_as(left_k)
        left_v = self.left_v(hidden_state).view(batch, length, length, heads, self.v_head_dim)
        right_v = self.right_v(hidden_state).view_as(left_v)
        query = self.query(hidden_state).view_as(left_k)

        scores = contract("bxahd,bayhd->bxayh", left_k, right_k) * self.scaler

        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]
        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask, "bnnn1")
            scores = scores.masked_fill(expanded_mask, self.mask_padding_value)

        val = contract("bxahd,bayhd->bxayhd", left_v, right_v)

        scores = scores - scores.max(dim=2, keepdim=True).values
        att_weights = scores.softmax(dim=2)
        att_weights = self._drop_weights(att_weights)

        att = contract("bxayh,bxayhd->bxyhd", att_weights, val)
        att = att.reshape(batch, length, length, self.v_dim)
        att = self._apply_out_proj(att)

        return att, att_weights
