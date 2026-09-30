"""
Task-agnostic transformer encoder layer: a registered attention architecture
plus a feed-forward stack, each wrapped in a plain residual connection.

This module composes only shared pieces (the attention registry in
`compgen/models/attentions/` and the relative-position bias in
`compgen/models/position.py`) and must not import task-specific code — task
models under `compgen/models/tasks/` assemble this layer with their own
task-specific embedding.

Feed-forward variants:
- ``"relu_stack"`` (default): the historical ``ffn_depth`` copies of
  ``Dropout -> Linear -> ReLU``; unchanged behavior.
- ``"gelu"``: a two-projection ``Linear(ffn_dim) -> GELU -> Dropout ->
  Linear(hidden)`` block, matching the reference fuzzy-logic transformer.

Relative-position bias: when ``use_relative_bias`` is set the layer owns a
`RelativePositionBias` and supplies it to attention as an additive
``score_bias`` computed from the runtime sequence length.
"""

from typing import Optional

import torch
import torch.nn as nn

from compgen.models.attentions import ATTENTION_CLASSES
from compgen.models.position import RelativePositionBias


def _init_attention_weights(attention: nn.Module) -> None:
    for name, param in attention.named_parameters():
        if "weight" in name:
            nn.init.xavier_normal_(param)
        elif "bias" in name:
            nn.init.constant_(param, 0.0)


class EncoderLayer(nn.Module):
    """
    One encoder block: self-attention (any architecture registered in
    `compgen.models.attentions.ATTENTION_CLASSES`) plus a feed-forward network,
    each wrapped in a plain residual connection.

    `use_norm=False` skips LayerNorm entirely (the strassen-attention paper's
    setting for Match3), matching `BackboneTransformerLayer` in the reference
    repo. `use_norm=True` applies pre-norm to both sublayers.
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_type: str,
        num_heads: int = 1,
        qk_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        dropout_rate: float = 0.0,
        attention_dropout_rate: Optional[float] = None,
        use_attention_dropout: bool = True,
        attention_bias: bool = False,
        ffn_depth: int = 3,
        ffn_type: str = "relu_stack",
        ffn_dim: Optional[int] = None,
        use_norm: bool = False,
        eps: float = 1e-6,
        use_relative_bias: bool = False,
        relative_bias_num_buckets: int = 16,
        relative_bias_max_distance: int = 16,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
    ):
        super().__init__()
        if attention_type not in ATTENTION_CLASSES:
            raise ValueError(f"Unsupported attention type: {attention_type}")
        if ffn_type not in ("relu_stack", "gelu"):
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        # Attention may use a different dropout rate than the residual/FFN path.
        if attention_dropout_rate is None:
            attention_dropout_rate = dropout_rate

        self.attention_type = attention_type
        self.attention = ATTENTION_CLASSES[attention_type](
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
            dropout_rate=attention_dropout_rate,
            dtype=dtype,
            bias=attention_bias,
            device=device,
            use_dropout=use_attention_dropout,
        )
        _init_attention_weights(self.attention)

        ffn_dim = hidden_dim if ffn_dim is None else ffn_dim
        self.ffn_type = ffn_type
        if ffn_type == "relu_stack":
            self.ffn = nn.ModuleList([
                nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                    nn.ReLU(),
                )
                for _ in range(ffn_depth)
            ])
        else:  # "gelu"
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim, dtype=dtype, device=device),
                nn.GELU(approximate="tanh"),
                nn.Dropout(dropout_rate),
                nn.Linear(ffn_dim, hidden_dim, dtype=dtype, device=device),
            )

        self.use_norm = use_norm
        if use_norm:
            self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)
            self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, dtype=dtype, device=device)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.relative_bias = None
        if use_relative_bias:
            self.relative_bias = RelativePositionBias(
                num_heads=num_heads,
                num_buckets=relative_bias_num_buckets,
                max_distance=relative_bias_max_distance,
                dtype=dtype,
                device=device,
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        batch_mask: Optional[dict] = None,
        score_bias: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        if score_bias is None and self.relative_bias is not None:
            score_bias = self.relative_bias(hidden_state.shape[1], hidden_state.device)

        x = self.norm1(hidden_state) if self.use_norm else hidden_state
        attention_output, attention_weights = self.attention(
            hidden_state=x, batch_mask=batch_mask, score_bias=score_bias
        )
        hidden_state = hidden_state + self.dropout1(attention_output)

        x = self.norm2(hidden_state) if self.use_norm else hidden_state
        for layer in self.ffn:
            x = layer(x)
        hidden_state = hidden_state + self.dropout2(x)

        if return_attention:
            return hidden_state, attention_weights
        return hidden_state
