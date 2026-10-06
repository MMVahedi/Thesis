"""
Fuzzy-logic regression model: task-specific input embedding -> stack of shared
encoder layers (any sequence-composable architecture in
`compgen.models.attentions`) -> shared regression head.

The model composes the reference fuzzy-logic transformer skeleton (pre-norm
layers, a GELU feed-forward block, multi-head attention from the shared
registry, and the reference initialization). Two deliberate divergences from
the reference: position is supplied by a fixed absolute positional encoding
added to the input embedding rather than by a learned bias inside attention,
and attention dropout uses the package's standard per-element dropout rather
than the reference's single mask broadcast across batch and heads.

This file is thin glue: the embedding (including the fixed positional encoding)
lives in `compgen/models/embeddings/fuzzy_logic.py`, the encoder layer in
`compgen/models/encoder.py`, and the head in `compgen/models/heads.py`.
"""

import torch
import torch.nn as nn

from compgen.models.embeddings.fuzzy_logic import FuzzyLogicEmbedding
from compgen.models.encoder import EncoderLayer
from compgen.models.heads import RegressionHead
from compgen.models.initialization import init_reference_weights


class FuzzyLogicModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        attention_type: str = "softmax",
        num_layers: int = 2,
        num_heads: int = 8,
        qk_head_dim: int = 2,
        v_head_dim: int = 2,
        ffn_type: str = "gelu",
        ffn_dim: int = 256,
        dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.1,
        use_norm: bool = True,
        attention_bias: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = f"fuzzy_logic_{attention_type}_h{hidden_dim}_l{num_layers}_heads{num_heads}"

        self.embedding = FuzzyLogicEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            dtype=dtype,
            device=device,
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_dim=hidden_dim,
                attention_type=attention_type,
                num_heads=num_heads,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                attention_bias=attention_bias,
                ffn_type=ffn_type,
                ffn_dim=ffn_dim,
                use_norm=use_norm,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ])

        self.head = RegressionHead(hidden_dim, eps=eps, dtype=dtype, device=device)
        init_reference_weights(self)

    @staticmethod
    def _to_input(x):
        """Accept a raw tensor or any batch object exposing `.x`."""
        return x.x if hasattr(x, "x") else x

    def forward(self, x, return_attention=False):
        hidden_state = self.embedding(self._to_input(x))
        attentions = []
        for layer in self.layers:
            hidden_state, attention = layer(
                hidden_state, batch_mask=None, return_attention=True
            )
            attentions.append(attention)

        prediction = self.head(hidden_state)
        return (prediction, attentions) if return_attention else prediction

    @staticmethod
    def response_token_codes(attentions):
        """Response-token self-attention code across heads: `(batch, heads)`."""
        return attentions[-1][:, :, -1, -1]
