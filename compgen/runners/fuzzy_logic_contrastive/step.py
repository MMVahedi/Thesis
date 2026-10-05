"""Training step for the fuzzy-logic contrastive experiment.

The step computes the regression task loss on the response (last) token and,
when the contrastive coefficient is positive, adds the shared-term contrastive
term over the final-layer response-token attention code across heads. It reuses
the package's model, loss, and combined-loss primitives rather than defining any
of them here.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F

from compgen.losses import combined_loss
from compgen.models.tasks.fuzzy_logic import FuzzyLogicModel


@dataclass
class BatchOutput:
    """Everything one forward pass produces for a batch."""

    total_loss: torch.Tensor
    task_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    average_positives: torch.Tensor
    prediction: torch.Tensor  # response-token prediction, shape (batch, 1)
    target: torch.Tensor  # regression target, shape (batch, 1)


def forward_batch(
    model: torch.nn.Module,
    batch,
    lambda_contrastive: float = 0.0,
    temperature: float = 1.0,
) -> BatchOutput:
    """Run one batch and compute the task (+ optional contrastive) loss.

    With a positive ``lambda_contrastive`` the model is asked for attention and
    the final-layer, last-token attention code across heads is passed to the
    package's combined loss; with a zero coefficient the contrastive term is
    skipped entirely, matching the reference baseline behavior.
    """
    if lambda_contrastive > 0:
        prediction, attentions = model(batch.x, return_attention=True)
        codes: Optional[torch.Tensor] = FuzzyLogicModel.response_token_codes(attentions)
    else:
        prediction = model(batch.x)
        codes = None

    response = prediction[:, -1]
    task_loss = F.mse_loss(response, batch.y)
    total, task, contrastive, positives = combined_loss(
        task_loss,
        batch.term_ids,
        representations=codes,
        lambda_contrastive=lambda_contrastive,
        temperature=temperature,
    )
    return BatchOutput(
        total_loss=total,
        task_loss=task,
        contrastive_loss=contrastive,
        average_positives=positives,
        prediction=response,
        target=batch.y,
    )


def make_step_fn(
    lambda_contrastive: float = 0.0, temperature: float = 1.0
) -> Callable[[torch.nn.Module, object], Tuple[torch.Tensor, dict]]:
    """Build a ``step_fn(model, batch) -> (loss, metrics)`` for ``compgen.training.fit``."""

    def step_fn(model: torch.nn.Module, batch) -> Tuple[torch.Tensor, dict]:
        output = forward_batch(model, batch, lambda_contrastive, temperature)
        metrics = {
            "contrastive_loss": float(output.contrastive_loss.detach()),
            "avg_positive_partners": float(output.average_positives.detach()),
        }
        return output.total_loss, metrics

    return step_fn