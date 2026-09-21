"""
Match3 classifier model: task-specific embedding -> stack of shared encoder
layers (any attention architecture registered in `compgen.models.attentions`)
-> shared per-token binary classification head.

This reproduces the "33" experiment config from strassen-attention-neurips25's
`cmds/empirical/match3/hyperparams.json` (M=37, hidden_dim=128, 1 layer,
2 heads, dropout=0.4, no LayerNorm). All architectures present in the shared
registry are supported; which architectures a study compares is decided by
the calling notebook/experiment config.

This file is intentionally thin glue: the encoder layer and the head live in
`compgen/models/encoder.py` and `compgen/models/heads.py`, and the
task-specific embedding lives in `compgen/models/embeddings/match3.py`.
"""

import torch
import torch.nn as nn

from compgen.models.embeddings.match3 import Match3Embedding
from compgen.models.encoder import EncoderLayer
from compgen.models.heads import TokenClassifier


class Match3Model(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attention_type: str,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
        embedding_norm_scalar: float = 1.0,
        use_norm: bool = False,
        use_attention_dropout: bool = True,
        share_layers: bool = False,
        ffn_depth: int = 3,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.name = f"match3_{attention_type}_h{hidden_dim}_l{num_layers}_heads{num_heads}"

        self.embedding = Match3Embedding(
            hidden_dim=hidden_dim,
            embedding_norm_scalar=embedding_norm_scalar,
            dtype=dtype,
            device=device,
        )

        def make_layer() -> EncoderLayer:
            return EncoderLayer(
                hidden_dim=hidden_dim,
                attention_type=attention_type,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                use_attention_dropout=use_attention_dropout,
                ffn_depth=ffn_depth,
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

    def forward(self, batch: dict):
        hidden_state = self.embedding(batch)
        for layer in self.layers:
            hidden_state = layer(hidden_state, batch_mask=batch["batch_mask"])

        logits = self.classifier(hidden_state)
        y_hat = torch.sigmoid(logits)
        return y_hat, hidden_state