"""
Task-agnostic transformer encoder layer: caller-constructed attention and
feed-forward modules, each wrapped in a plain residual connection.

This module composes only shared pieces and the two modules it is handed; it
does not import task-specific code, the attention registry, or the
feed-forward module family — task models under `compgen/models/tasks/` resolve
a registered architecture and a feed-forward variant, construct both modules,
and assemble this layer with their own task-specific embedding (which is where
positional information is added).

The layer applies no dropout and performs no initialization: the modules it is
handed are used exactly as constructed, and its own normalization modules keep
PyTorch's defaults. Any dropout lives inside the sub-modules, and the composing
task model applies its single initialization policy across the whole model. The
feed-forward variants (a residual ReLU stack and a two-projection GELU block)
live in `compgen/models/ffn.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # type-only; keeps this module registry-independent at runtime
    from compgen.models.attentions.base import BaseAttention
    from compgen.models.ffn import BaseFeedForward


class EncoderLayer(nn.Module):
    """
    One encoder block: self-attention (a caller-constructed attention module)
    plus a feed-forward network (a caller-constructed feed-forward module), each
    wrapped in a plain residual connection.

    Both modules are built by the caller — a task model resolves a registered
    architecture and a feed-forward variant, then constructs the modules — so
    this layer performs no architecture or variant selection or validation and
    does not initialize or re-initialize the modules it is handed. The layer
    applies no dropout of its own; any dropout lives inside the attention module
    or the feed-forward block.

    `use_norm=False` skips LayerNorm entirely (the strassen-attention paper's
    setting for Match3), matching `BackboneTransformerLayer` in the reference
    repo. `use_norm=True` applies pre-norm to both sublayers.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention: BaseAttention,
        ffn: BaseFeedForward,
        use_norm: bool = False,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()

        self.attention = attention
        self.ffn = ffn

        self.use_norm = use_norm
        if use_norm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_state: torch.Tensor,
        return_attention: bool = False,
    ):
        x = self.norm1(hidden_state) if self.use_norm else hidden_state
        attention_output, attention_weights = self.attention(hidden_state=x)
        hidden_state = hidden_state + attention_output

        x = self.norm2(hidden_state) if self.use_norm else hidden_state
        x = self.ffn(x)
        hidden_state = hidden_state + x

        if return_attention:
            return hidden_state, attention_weights
        return hidden_state
