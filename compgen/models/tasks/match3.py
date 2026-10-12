# NEED REVIEW
"""
Match3 classifier model: task-specific embedding -> stack of shared encoder
layers (each with caller-supplied attention and feed-forward modules) ->
shared per-token binary classification head.

This reproduces the "33" experiment config from strassen-attention-neurips25's
`cmds/empirical/match3/hyperparams.json` (M=37, hidden_dim=128, 1 layer,
2 heads, dropout=0.4, no LayerNorm). Any architecture is supported — the model
is handed caller-supplied, already-constructed attention and feed-forward
modules, which it shares across every encoder layer, and hands both to each
shared encoder layer; which architectures a study compares is decided by the
calling notebook/experiment config (see `compgen/models/factory.py`). Dropout
lives in the feed-forward block and the classifier; the encoder applies none.
Initialization is owned here, at the model level: the model accepts a
`WeightInitializer` policy class (default `XavierInitializer`) and its single
`_init_weights` method applies that one uniform policy to the whole assembled
model, so the sub-level modules perform no initialization of their own.

This file is intentionally thin glue: the encoder layer and the head live in
`compgen/models/encoder.py` and `compgen/models/heads.py`, the feed-forward
variants in `compgen/models/ffn.py`, and the task-specific embedding lives in
`compgen/models/embeddings/match3.py`.
"""

from typing import Type

import torch
import torch.nn as nn

from compgen.models.embeddings.match3 import Match3Embedding
from compgen.models.encoder import EncoderLayer
from compgen.models.heads import TokenClassifier
from compgen.models.initialization import WeightInitializer, XavierInitializer


class Match3Model(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention: nn.Module,
        ffn: nn.Module,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        embedding_norm_scalar: float = 1.0,
        use_norm: bool = False,
        share_layers: bool = False,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        initializer: Type[WeightInitializer] = XavierInitializer,
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
        self.name = f"match3_h{hidden_dim}_l{num_layers}_heads{num_heads}"

        self.embedding = Match3Embedding(
            hidden_dim=hidden_dim,
            embedding_norm_scalar=embedding_norm_scalar,
            dtype=dtype,
            device=device,
        )

        def make_layer() -> EncoderLayer:
            return EncoderLayer(
                hidden_dim=hidden_dim,
                attention=attention,
                ffn=ffn,
                use_norm=use_norm,
                dtype=dtype,
                device=device,
            )

        if share_layers:
            shared_layer = make_layer()
            self.layers = nn.ModuleList([shared_layer] * num_layers)
        else:
            self.layers = nn.ModuleList([make_layer() for _ in range(num_layers)])

        self.classifier = TokenClassifier(
            hidden_dim=hidden_dim, dropout_rate=dropout_rate, dtype=dtype, device=device
        )
        self.initializer = initializer()
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply the model's single initialization policy to the whole model."""
        self.initializer(self)

    def forward(self, batch: dict, return_attention: bool = False):
        hidden_state = self.embedding(batch)
        attentions = []
        for layer in self.layers:
            if return_attention:
                hidden_state, attention = layer(
                    hidden_state, return_attention=True
                )
                attentions.append(attention)
            else:
                hidden_state = layer(hidden_state)

        y_hat = self.classifier(hidden_state)
        if return_attention:
            return y_hat, hidden_state, attentions
        return y_hat, hidden_state