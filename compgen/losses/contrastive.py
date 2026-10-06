"""
Shared-term supervised contrastive objective.

Two examples are positives when their term-identifier sets share at least one
term. The loss is the reference InfoNCE form over cosine similarities scaled by
a temperature, each example excluded from its own negatives and anchors without
any positive partner skipped:

    L_i = -1/|P(i)| sum_{p in P(i)} log[
        exp(cos(z_i, z_p) / T) / sum_{a != i} exp(cos(z_i, z_a) / T) ]

This module is task-agnostic: it operates on caller-supplied representation
vectors and term identifiers and knows nothing about any model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_partner_counts(term_ids: torch.Tensor) -> tuple:
    """Return (positive_mask, counts).

    `positive_mask[i, j]` is True when examples i and j share at least one term
    (and i != j); `counts[i]` is the number of positive partners of example i.

    `term_ids` has shape (N, K): N examples, one term index per column.
    """
    # term_ids[:, None, :, None] -> (N, 1, K, 1)
    # term_ids[None, :, None, :] -> (1, N, 1, K)
    # Broadcast comparison -> (N, N, K, K), where `same_term[i, j, k, l]` is
    # True when term k of example i equals term l of example j.
    same_term = term_ids[:, None, :, None] == term_ids[None, :, None, :]

    # Collapse both term axes -> (N, N), where `shares_term[i, j]` is True when
    # examples i and j share at least one term.
    shares_term = same_term.any(dim=(2, 3))

    # (N, N) mask with the diagonal cleared: an example trivially shares a term
    # with itself.
    positive_mask = shares_term
    positive_mask.fill_diagonal_(False)

    # Positive partners per example -> (N,).
    return positive_mask, positive_mask.sum(dim=1)


def shared_term_contrastive(
    representations: torch.Tensor, term_ids: torch.Tensor, temperature: float = 1.0
) -> tuple:
    """Shared-term InfoNCE loss and the mean positive-partner count."""
    positives, num_positives = positive_partner_counts(term_ids)
    # Identity matrix as a boolean mask: eye[i, j] is True only when i == j.
    eye = torch.eye(len(term_ids), dtype=torch.bool, device=term_ids.device)

    z = F.normalize(representations, dim=1)
    similarity = z @ z.T / temperature
    # Exclude each example from its own negatives.
    similarity = similarity.masked_fill(eye, float("-inf"))
    log_probability = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

    per_anchor = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1)) / num_positives.clamp_min(1)
    valid = num_positives > 0
    loss = per_anchor[valid].mean() if valid.any() else representations.sum() * 0.0
    return loss, num_positives.float().mean()


class SharedTermContrastiveLoss(nn.Module):
    """Module wrapper around `shared_term_contrastive` holding the temperature."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, representations: torch.Tensor, term_ids: torch.Tensor) -> tuple:
        return shared_term_contrastive(representations, term_ids, self.temperature)
