"""
Fuzzy-logic regression model: task-specific input embedding -> stack of shared
encoder layers (any sequence-composable architecture in
`compgen.models.attentions`) -> shared regression head.

The model composes the reference fuzzy-logic transformer skeleton (pre-norm
layers, a GELU feed-forward block, multi-head attention from any attention
class, and the reference initialization). The model is handed
caller-supplied, already-constructed attention and feed-forward modules — the
same modules are shared across every encoder layer — and hands both to the
shared encoder layer, which performs no selection and applies no dropout. Two
deliberate divergences from the reference: position is supplied by a fixed
absolute positional encoding
added to the input embedding rather than by a learned bias inside attention, and
attention applies no dropout at all (attention is a pure hidden-state
transform).

Initialization is owned here, at the model level: the model accepts a
`WeightInitializer` policy class (default `ReferenceInitializer`) and its single
`_init_weights` method applies that one uniform policy to the whole assembled
model — including the caller-supplied sub-modules — so the sub-level modules
perform no initialization of their own.

This file is thin glue: the embedding (including the fixed positional encoding)
lives in `compgen/models/embeddings/fuzzy_logic.py`, the encoder layer in
`compgen/models/encoder.py`, the feed-forward variants in
`compgen/models/ffn.py`, and the head in `compgen/models/heads.py`.
"""

from typing import Type

import torch
import torch.nn as nn

from compgen.models.embeddings.fuzzy_logic import FuzzyLogicEmbedding
from compgen.models.encoder import EncoderLayer
from compgen.models.heads import RegressionHead
from compgen.models.initialization import ReferenceInitializer, WeightInitializer


class FuzzyLogicModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        attention: nn.Module,
        ffn: nn.Module,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout_rate: float = 0.0,
        use_norm: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        initializer: Type[WeightInitializer] = ReferenceInitializer,
    ):
        super().__init__()
        if not isinstance(attention, nn.Module):
            raise TypeError(
                "attention must be a constructed nn.Module, not an architecture "
                "name or a factory"
            )
        if not isinstance(ffn, nn.Module):
            raise TypeError(
                "ffn must be a constructed nn.Module, not a feed-forward variant "
                "name or a factory"
            )
        self.name = f"fuzzy_logic_h{hidden_dim}_l{num_layers}_heads{num_heads}"

        self.embedding = FuzzyLogicEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device,
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_dim=hidden_dim,
                attention=attention,
                ffn=ffn,
                use_norm=use_norm,
                eps=eps,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ])

        self.head = RegressionHead(hidden_dim, eps=eps, dtype=dtype, device=device)
        self.initializer = initializer()
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the model's single initialization policy to the whole model."""
        self.initializer(self)

    @staticmethod
    def _to_input(x):
        """Accept a raw tensor or any batch object exposing `.x`."""
        return x.x if hasattr(x, "x") else x

    def forward(self, x, return_attention=False):
        hidden_state = self.embedding(self._to_input(x))
        attentions = []
        for layer in self.layers:
            hidden_state, attention = layer(
                hidden_state, return_attention=True
            )
            attentions.append(attention)

        prediction = self.head(hidden_state)
        return (prediction, attentions) if return_attention else prediction

    @staticmethod
    def response_token_codes(attentions):
        """Response-token self-attention code across heads: `(batch, heads)`."""
        return attentions[-1][:, :, -1, -1]
