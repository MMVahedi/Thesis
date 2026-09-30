"""
Linear attention mixer.

The sibling of `SoftmaxAttention`: the same projections, scaling, head split,
dropout, masking, and output reconciliation, but the attention weights are the
scaled scores directly — the final softmax normalization is removed. Because an
unnormalized sum would still add a large additive padding value, padding is
masked by zeroing instead of by adding `mask_padding_value`. This masking
difference is the one deliberate divergence from `SoftmaxAttention`. An optional
additive `score_bias` (for example a relative-position bias) is added to the
scaled scores before masking.
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

        self.query = nn.Linear(
            hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device
        )
        self.key = nn.Linear(
            hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device
        )
        self.value = nn.Linear(
            hidden_dim, self.v_dim, bias=bias, dtype=dtype, device=device
        )

    def forward(self, hidden_state, batch_mask=None, score_bias=None):
        batch, length, _ = hidden_state.shape

        q = self._split_heads(self.query(hidden_state), self.qk_head_dim)
        k = self._split_heads(self.key(hidden_state), self.qk_head_dim)
        v = self._split_heads(self.value(hidden_state), self.v_head_dim)

        scores = contract("bhid,bhjd->bhij", q, k) * self.scaler
        if score_bias is not None:
            # Additive bias (e.g. a relative-position bias) broadcast over batch.
            scores = scores + score_bias

        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]
        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask, "b1nn")
            # Zeroing keeps masked positions out of the unnormalized sum.
            scores = scores.masked_fill(expanded_mask, 0.0)

        # No softmax: the (masked) scaled scores are the attention weights.
        att_weights = self._drop_weights(scores)

        att = contract("bhij,bhjd->bhid", att_weights, v)
        att = self._merge_heads(att)
        att = self._apply_out_proj(att)

        return att, att_weights
