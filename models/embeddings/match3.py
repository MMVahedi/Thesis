"""
Match3 "theoretical" embedding, from "Strassen Attention, Split VC Dimension,
and Compositionality in Transformers".

Maps each token's (position, value) pair to a hand-crafted feature vector
`[position, value, value**2, 1]` (each divided by `embedding_norm_scalar**2`
to keep magnitudes controlled for a given modulus M), optionally projected to
`hidden_dim`. This specific feature set is what makes the paper's theoretical
constructions possible: a token's own value, squared value, and position are
exactly the ingredients needed to express the triple-sum-mod-M predicate (see
`dataset/generators/match3.py`) as a bilinear/trilinear attention score.
"""

import torch
import torch.nn as nn


class Match3Embedding(nn.Module):
    THEORETICAL_DIM = 4  # [position, value, value**2, 1]

    def __init__(
        self,
        hidden_dim: int,
        embedding_norm_scalar: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_norm_scalar = embedding_norm_scalar
        self.dtype = dtype
        self.device = device

        # Only project if hidden_dim differs from the raw 4-d feature vector,
        # so the "theoretical" constructions (which reason about the raw
        # features directly) still apply when hidden_dim == 4.
        self.in_proj = None
        if hidden_dim != self.THEORETICAL_DIM:
            self.in_proj = nn.Linear(self.THEORETICAL_DIM, hidden_dim, dtype=dtype, device=device)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with "seq" (batch, seq_len, 1) token values and
                "position" (batch, seq_len, 1) integer positions.

        Returns:
            (batch, seq_len, hidden_dim) embeddings.
        """
        seq = batch["seq"].to(self.dtype).to(self.device)
        position = batch["position"].to(self.dtype).to(self.device)
        norm_factor = self.embedding_norm_scalar ** 2

        embd = torch.cat(
            [
                position / norm_factor,
                seq / norm_factor,
                (seq ** 2) / norm_factor,
                torch.ones_like(seq),
            ],
            dim=-1,
        )
        if self.in_proj is not None:
            embd = self.in_proj(embd)
        return embd
