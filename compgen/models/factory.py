"""
Model-layer factory: build a fully constructed task model from a typed config.

``build_model(config)`` turns a :class:`~compgen.models.configs.ModelConfig`
into a ready ``torch.nn.Module``: it selects and constructs the attention module
and the feed-forward module from the config's enum members and hands them, with
the task-specific inputs, to the matching task model, which builds its embedding,
head, and initialization. The caller supplies no sub-modules.

Selection is by typed enum member only — a raw string is never used as a
selector — so each concrete class stays in its own module while the config stays
declarative. The mappings below are keyed by enum members, not strings. Runtime
``dtype``/``device`` are left to the task models' existing defaults.
"""

from __future__ import annotations

from typing import Dict, Type

import torch.nn as nn

from compgen.models.attentions import (
    LinearAttention,
    SoftmaxAttention,
    StrassenAttention,
    ThirdOrderAttention,
)
from compgen.models.configs import (
    AttentionArchitecture,
    FeedForwardVariant,
    FuzzyLogicConfig,
    Match3Config,
    ModelConfig,
)
from compgen.models.ffn import GeluFFN, ReluStackFFN
from compgen.models.tasks.fuzzy_logic import FuzzyLogicModel
from compgen.models.tasks.match3 import Match3Model

# Enum member -> concrete class. Keys are enum members, never strings.
_ATTENTION_CLASS_BY_ARCHITECTURE: Dict[AttentionArchitecture, Type[nn.Module]] = {
    AttentionArchitecture.SOFTMAX: SoftmaxAttention,
    AttentionArchitecture.LINEAR: LinearAttention,
    AttentionArchitecture.STRASSEN: StrassenAttention,
    AttentionArchitecture.THIRD_ORDER: ThirdOrderAttention,
}

_FFN_CLASS_BY_VARIANT: Dict[FeedForwardVariant, Type[nn.Module]] = {
    FeedForwardVariant.RELU_STACK: ReluStackFFN,
    FeedForwardVariant.GELU: GeluFFN,
}


def _require_enum_selection(config: ModelConfig) -> None:
    """Fail loudly if a raw string (or anything non-enum) reaches selection."""
    if not isinstance(config.attention, AttentionArchitecture):
        raise TypeError(
            "attention must be an AttentionArchitecture enum member, not "
            f"{type(config.attention).__name__}"
        )
    if not isinstance(config.ffn, FeedForwardVariant):
        raise TypeError(
            "ffn must be a FeedForwardVariant enum member, not "
            f"{type(config.ffn).__name__}"
        )
    if config.attention not in _ATTENTION_CLASS_BY_ARCHITECTURE:
        raise ValueError(f"Unsupported attention architecture: {config.attention!r}")
    if config.ffn not in _FFN_CLASS_BY_VARIANT:
        raise ValueError(f"Unsupported feed-forward variant: {config.ffn!r}")


def build_attention(config: ModelConfig) -> nn.Module:
    """Construct the attention module named by the config's architecture enum."""
    attention_cls = _ATTENTION_CLASS_BY_ARCHITECTURE[config.attention]
    return attention_cls(
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
    )


def build_feed_forward(config: ModelConfig) -> nn.Module:
    """Construct the feed-forward module named by the config's variant enum."""
    if config.ffn is FeedForwardVariant.GELU:
        return GeluFFN(
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            dropout_rate=config.dropout_rate,
        )
    if config.ffn is FeedForwardVariant.RELU_STACK:
        return ReluStackFFN(
            hidden_dim=config.hidden_dim,
            ffn_depth=config.ffn_depth,
            dropout_rate=config.dropout_rate,
        )
    raise TypeError(
        "ffn must be a FeedForwardVariant enum member, not "
        f"{type(config.ffn).__name__}"
    )


def build_model(config: ModelConfig) -> nn.Module:
    """Build and return the task model described by ``config``.

    Dispatches on the configuration type: a :class:`Match3Config` builds the
    Match3 classifier, a :class:`FuzzyLogicConfig` builds the fuzzy-logic
    regression model. The attention and feed-forward modules are constructed
    here from the config's enum members; the task model builds the
    task-specific embedding and head and applies its initialization policy.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError(
            "config must be a ModelConfig instance, not "
            f"{type(config).__name__}"
        )
    _require_enum_selection(config)

    attention = build_attention(config)
    ffn = build_feed_forward(config)

    if isinstance(config, Match3Config):
        return Match3Model(
            hidden_dim=config.hidden_dim,
            attention=attention,
            ffn=ffn,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            embedding_norm_scalar=config.embedding_norm_scalar,
            use_norm=config.use_norm,
            share_layers=config.share_layers,
            initializer=config.initializer,
        )

    if isinstance(config, FuzzyLogicConfig):
        return FuzzyLogicModel(
            input_dim=config.input_dim,
            attention=attention,
            ffn=ffn,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            use_norm=config.use_norm,
            initializer=config.initializer,
        )

    raise TypeError(
        f"Unsupported model configuration type: {type(config).__name__}"
    )