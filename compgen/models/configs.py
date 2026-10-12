"""
Model configuration for the model-layer factory.

This module defines the typed selection enums and the per-task configuration
dataclasses that ``compgen.models.factory.build_model`` consumes. Selection is
by enum member only — a raw string is never used to pick an attention
architecture or a feed-forward variant — so a config is declarative and
type-checked, while the concrete classes stay in their own modules.

- :class:`AttentionArchitecture`: the sequence-composable attention
  architectures a model may use. ``triangular`` is intentionally absent: it is
  pair-level ``(B, N, N, C)`` and not sequence-composable through
  ``EncoderLayer``.
- :class:`FeedForwardVariant`: the two feed-forward blocks.

Per-task configuration types (:class:`Match3Config`, :class:`FuzzyLogicConfig`)
share the common knobs on :class:`ModelConfig` and add their task-specific
inputs. The config's type — not a task-type field — selects which task model the
factory builds. The initialization policy is carried as a concrete
:class:`~compgen.models.initialization.WeightInitializer` subclass (a class, not
a string or enum), matching the model layer's no-string-scheme rule.

Runtime ``dtype``/``device`` are deliberately not configuration knobs in this
change; the factory relies on the task models' existing defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Type

from compgen.models.initialization import (
    ReferenceInitializer,
    WeightInitializer,
    XavierInitializer,
)


class AttentionArchitecture(Enum):
    """Sequence-composable attention architectures a configured model may use."""

    SOFTMAX = auto()
    LINEAR = auto()
    STRASSEN = auto()
    THIRD_ORDER = auto()


class FeedForwardVariant(Enum):
    """Feed-forward blocks a configured model may use."""

    RELU_STACK = auto()
    GELU = auto()


@dataclass
class ModelConfig:
    """Common model knobs shared by every task configuration.

    The field types are the variation surface: attention architecture and head
    dimensions, feed-forward variant with width/depth, stack size and shape,
    regularization/normalization, and the initialization policy class. Subclasses
    add the task-specific inputs and override defaults where the task differs.
    """

    hidden_dim: int = 128
    num_layers: int = 1
    num_heads: int = 1
    qk_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    attention: AttentionArchitecture = AttentionArchitecture.SOFTMAX
    ffn: FeedForwardVariant = FeedForwardVariant.RELU_STACK
    ffn_dim: Optional[int] = None
    ffn_depth: int = 3
    dropout_rate: float = 0.0
    use_norm: bool = False
    initializer: Type[WeightInitializer] = XavierInitializer

    def __post_init__(self) -> None:
        # Selection is enum-only: reject raw strings (and anything else) up front
        # so a typo cannot silently drive architecture selection.
        if not isinstance(self.attention, AttentionArchitecture):
            raise TypeError(
                "attention must be an AttentionArchitecture enum member, not "
                f"{type(self.attention).__name__}"
            )
        if not isinstance(self.ffn, FeedForwardVariant):
            raise TypeError(
                "ffn must be a FeedForwardVariant enum member, not "
                f"{type(self.ffn).__name__}"
            )


@dataclass
class Match3Config(ModelConfig):
    """Configuration for the Match3 classifier task model."""

    # Match3's hand-crafted embedding divides its features by this scalar squared.
    embedding_norm_scalar: float = 1.0
    # Share one whole encoder layer across the stack (attention and feed-forward
    # are shared regardless; this additionally shares the normalization layer).
    share_layers: bool = False


@dataclass
class FuzzyLogicConfig(ModelConfig):
    """Configuration for the fuzzy-logic regression task model.

    ``input_dim`` is the fuzzy-logic input width (``num_variables + 1``); the 4
    default matches the reference 3-variable task. ``ffn_dim=None`` means the
    feed-forward width follows ``hidden_dim``.
    """

    input_dim: int = 4
    num_layers: int = 2
    num_heads: int = 8
    use_norm: bool = True
    ffn: FeedForwardVariant = FeedForwardVariant.GELU
    initializer: Type[WeightInitializer] = ReferenceInitializer