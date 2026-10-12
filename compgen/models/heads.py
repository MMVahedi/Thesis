"""
Task-agnostic prediction heads for the model layer.

These heads compose hidden states produced by any task's encoder stack and
must not import task-specific code — task models under
`compgen/models/tasks/` pick the head that matches their output type.
"""

import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    """Per-token binary classification head: Linear -> ReLU -> Linear -> sigmoid -> 1 probability.

    Binary, not multiclass: only two outcomes exist ("match" vs "no match"), so a
    single scalar encodes both and the head emits ONE probability per token. A
    K-class head would instead emit K logits + softmax + argmax; here the single
    logit is passed through `sigmoid` (the two-class special case of softmax,
    since softmax([z, 0]) == sigmoid(z)) and thresholded at 0.5.

    The head owns the full binary readout and returns probabilities bounded within
    `[0, 1]` (mathematically interior to `(0, 1)`; float64 may round to the
    endpoints for large logits), NOT raw logits: the sigmoid lives here and must
    not be reapplied by a task model. Because no logits are exposed, training
    consumes the output with a probability-input binary cross-entropy loss
    (`nn.BCELoss`), not `BCEWithLogitsLoss`. See `compgen/models/tasks/match3.py`.

    The head owns no initialization: it keeps PyTorch's default parameter
    initialization, and the composing task model applies its single policy
    across the whole model.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.activation = nn.ReLU()
        # Collapse the hidden vector to a single logit per token: (B, L, H) -> (B, L, 1).
        self.out_proj = nn.Linear(hidden_dim, 1, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, length, hidden_dim) -> (batch, length, 1) probabilities bounded within [0, 1]."""
        x = self.dropout(x)
        x = self.activation(self.dense(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out_proj(x))


class RegressionHead(nn.Module):
    """Per-token scalar regression head: LayerNorm -> Dropout -> Linear(hidden, 1).

    Mirrors the reference fuzzy-logic transformer's final normalization followed
    by a scalar output projection. Task-agnostic: it maps any
    `(batch, length, hidden)` hidden state to `(batch, length, 1)`.

    Regression, not classification: the single output is a continuous score, so
    there is no sigmoid/softmax and no logits/probability convention — unlike the
    classification head its raw projection is the prediction. The head owns no
    initialization: its LayerNorm and projection keep PyTorch's default
    initialization, and the composing task model applies its single policy.
    """

    def __init__(
        self,
        hidden_dim: int,
        eps: float = 1e-6,
        dropout_rate: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        # Normalize last, as in the reference transformer, before the scalar readout.
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout_rate)
        # Collapse the hidden vector to a single scalar per token.
        self.out_proj = nn.Linear(hidden_dim, 1, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, length, hidden_dim) -> (batch, length, 1) scalar predictions."""
        return self.out_proj(self.dropout(self.norm(x)))