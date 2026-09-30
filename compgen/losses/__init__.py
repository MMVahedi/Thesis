"""
Reusable training objectives.

- `SharedTermContrastiveLoss` / `shared_term_contrastive`: supervised
  contrastive loss where examples sharing a term are positives.
- `positive_partner_counts`: per-example positive-partner accounting.
- `combined_loss`: model-agnostic task + contrastive combination.
"""

from compgen.losses.combined import combined_loss
from compgen.losses.contrastive import (
    SharedTermContrastiveLoss,
    positive_partner_counts,
    shared_term_contrastive,
)

__all__ = [
    "SharedTermContrastiveLoss",
    "combined_loss",
    "positive_partner_counts",
    "shared_term_contrastive",
]
