"""
Reusable model-interpretability tooling.

Task-agnostic analytics for attention weights and representation vectors:

- `attention`: turn attention weights into per-example vectors and read a
  specific token's attention distribution.
- `projection`: PCA projection of representation vectors to a small number of
  components (two by default).
- `plotting`: scatter and heatmap figure helpers (returned, not saved).
- `capture`: run a model and capture its per-layer attention and hidden states.

The layer operates on plain tensors and imports no task-specific code, so the
same functions work for the fuzzy-logic, Match3, and future task models.
"""

from compgen.analytics.attention import (
    attention_code,
    attention_row_features,
    stack_layers,
    token_attention_row,
)
from compgen.analytics.capture import CaptureResult, capture
from compgen.analytics.plotting import plot_attention, plot_projection
from compgen.analytics.projection import PCAProjector, ProjectionResult, pca_project

__all__ = [
    "CaptureResult",
    "PCAProjector",
    "ProjectionResult",
    "attention_code",
    "attention_row_features",
    "capture",
    "pca_project",
    "plot_attention",
    "plot_projection",
    "stack_layers",
    "token_attention_row",
]
