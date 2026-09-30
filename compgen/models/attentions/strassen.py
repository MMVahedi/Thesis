"""
Strassen attention mixer (retained high-order variant).

Three bilinear score matrices combined trilinearly, with an exponential (not
softmax) normalization. Refactored onto `BaseAttention`: score-producing
matrices use the query/key width and the two value factors use the value width.
The contraction equations and normalization are unchanged from the original
implementation; the only structural change is that the single fused projection
became one projection per matrix.
"""

import torch
import torch.nn as nn
from opt_einsum import contract

from compgen.models.attentions.base import BaseAttention


class StrassenAttention(BaseAttention):
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

        self.w_a = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)
        self.w_b = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)
        self.w_c = nn.Linear(hidden_dim, self.qk_dim, bias=bias, dtype=dtype, device=device)
        self.w_v1 = nn.Linear(hidden_dim, self.v_dim, bias=bias, dtype=dtype, device=device)
        self.w_v2 = nn.Linear(hidden_dim, self.v_dim, bias=bias, dtype=dtype, device=device)

    def forward(self, hidden_state, batch_mask=None, score_bias=None):
        # `score_bias` is accepted for interface uniformity with `EncoderLayer`;
        # this high-order variant does not use an additive bias.
        a = self._split_heads(self.w_a(hidden_state), self.qk_head_dim)
        b = self._split_heads(self.w_b(hidden_state), self.qk_head_dim)
        c = self._split_heads(self.w_c(hidden_state), self.qk_head_dim)
        v1 = self._split_heads(self.w_v1(hidden_state), self.v_head_dim)
        v2 = self._split_heads(self.w_v2(hidden_state), self.v_head_dim)

        X = contract("bhid,bhjd->bhij", a, b) * self.scaler
        Y = contract("bhjd,bhkd->bhjk", b, c) * self.scaler
        Z = contract("bhkd,bhid->bhki", c, a) * self.scaler

        attention_mask = None
        if batch_mask is not None:
            attention_mask = batch_mask["attention_mask"]
        if attention_mask is not None:
            expanded_mask = self.construct_mask(attention_mask, "b1nn")
            X = X.masked_fill(expanded_mask, self.mask_padding_value)
            Y = Y.masked_fill(expanded_mask, self.mask_padding_value)
            Z = Z.masked_fill(expanded_mask, self.mask_padding_value)

        X = X - torch.max(X, dim=-1, keepdim=True).values
        Y = Y - torch.max(
            torch.max(Y, dim=-1, keepdim=True).values, dim=-2, keepdim=True
        ).values
        Z = Z - torch.max(Z, dim=-2, keepdim=True).values

        X = X.exp()
        Y = Y.exp()
        Z = Z.exp()

        X = self._drop_weights(X)
        Y = self._drop_weights(Y)
        Z = self._drop_weights(Z)

        V = contract("bhjd,bhkd->bhjkd", v1, v2)

        up = contract("bhikd,bhki->bhid", contract("bhij,bhjk,bhjkd->bhikd", X, Y, V), Z)
        down = contract("bhik,bhki->bhi", contract("bhij,bhjk->bhik", X, Y), Z)
        down = down + 1e-9

        att = up / down.unsqueeze(-1)
        att = self._merge_heads(att)
        att = self._apply_out_proj(att)

        return att, None
