"""
Model-agnostic combined task + shared-term contrastive loss.

The task model computes its own task loss and exposes the representation
vectors the contrastive term acts on; this module only combines them, so it does
not reach into any model's internals. With a zero contrastive coefficient the
contrastive term is skipped entirely and reported as zero, matching the
reference baseline behavior.
"""

from typing import Optional

import torch

from compgen.losses.contrastive import positive_partner_counts, shared_term_contrastive


def combined_loss(
    task_loss: torch.Tensor,
    term_ids: torch.Tensor,
    representations: Optional[torch.Tensor] = None,
    lambda_contrastive: float = 0.0,
    temperature: float = 1.0,
) -> tuple:
    """Return (total, task_loss, contrastive_loss, average_positive_partners).

    `representations` may be None when `lambda_contrastive` is zero (the
    contrastive term is not needed then).
    """
    if lambda_contrastive > 0:
        if representations is None:
            raise ValueError("representations are required when lambda_contrastive > 0")
        contrastive, average_positives = shared_term_contrastive(
            representations, term_ids, temperature
        )
    else:
        contrastive = task_loss.new_zeros(())
        _, counts = positive_partner_counts(term_ids)
        average_positives = counts.float().mean()

    total = task_loss + lambda_contrastive * contrastive
    return total, task_loss, contrastive, average_positives
