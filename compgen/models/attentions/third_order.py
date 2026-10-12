# NEED REVIEW
"""
Third-order attention mixer (retained high-order variant).

A score tensor over triples of tokens, normalized with softmax over the N*N
key pairs. Refactored onto `BaseAttention`: the score-producing projections
(`q_i`, `k_j`, `k_k`) use the query/key width and the value factors (`v_j`,
`v_k`) use the value width. Contractions and normalization are unchanged from
the original implementation. Attention applies no masking or dropout.
"""

import torch
import torch.nn as nn
from opt_einsum import contract

from compgen.models.attentions.base import BaseAttention


class ThirdOrderAttention(BaseAttention):
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

        self.w_qi = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)
        self.w_kj = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)
        self.w_kk = nn.Linear(hidden_dim, self.qk_dim, bias=False, dtype=dtype, device=device)
        self.w_vj = nn.Linear(hidden_dim, self.v_dim, bias=False, dtype=dtype, device=device)
        self.w_vk = nn.Linear(hidden_dim, self.v_dim, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_state):
        batch, length, _ = hidden_state.shape
        nn_len = length ** 2

        qi = self._split_heads(self.w_qi(hidden_state), self.qk_head_dim)
        kj = self._split_heads(self.w_kj(hidden_state), self.qk_head_dim)
        kk = self._split_heads(self.w_kk(hidden_state), self.qk_head_dim)
        vj = self._split_heads(self.w_vj(hidden_state), self.v_head_dim)
        vk = self._split_heads(self.w_vk(hidden_state), self.v_head_dim)

        Kjk = contract("bhjd,bhkd->bhjkd", kj, kk)

        scores = contract("bhid,bhjkd->bhijk", qi, Kjk)
        scores = scores.reshape(batch, self.num_heads, length, nn_len)
        scores = scores * self.scaler

        scores = scores - scores.max(dim=-1, keepdim=True).values
        att_weights = torch.softmax(scores, -1)

        att_weights = att_weights.reshape(batch, self.num_heads, length, length, length)

        Vjk = contract("bhjd,bhkd->bhjkd", vj, vk)

        att = contract("bhijk,bhjkd->bhid", att_weights, Vjk)
        att = self._merge_heads(att)
        att = self._apply_out_proj(att)

        return att, None
