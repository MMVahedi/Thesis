"""
Public model-layer surface: the config-driven model factory and its types.

``build_model`` turns a typed configuration into a fully constructed task model
(see :mod:`compgen.models.factory`); the configuration dataclasses and selection
enums live in :mod:`compgen.models.configs`.
"""

from compgen.models.configs import (
    AttentionArchitecture,
    FeedForwardVariant,
    FuzzyLogicConfig,
    Match3Config,
    ModelConfig,
)
from compgen.models.factory import build_model

__all__ = [
    "AttentionArchitecture",
    "FeedForwardVariant",
    "FuzzyLogicConfig",
    "Match3Config",
    "ModelConfig",
    "build_model",
]