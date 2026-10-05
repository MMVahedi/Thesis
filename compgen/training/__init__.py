"""
Reusable, task-agnostic training and evaluation scaffold.

`fit` runs epochs over caller-supplied data and a step function, handling
optimizer/scheduler setup, device, seeding, evaluation, best-checkpoint
selection, early stopping, and run recording through `compgen.tracking`.

See `compgen/training/README.md` for usage.
"""

from compgen.training.loop import (
    EpochRecord,
    FitResult,
    fit,
    to_device,
)

__all__ = [
    "EpochRecord",
    "FitResult",
    "fit",
    "to_device",
]
