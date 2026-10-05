"""
Generic, task-agnostic training and evaluation loop.

`fit` runs epochs over caller-supplied data and a caller-supplied step
function, handling everything the tasks share: optimizer/scheduler setup,
device placement, seeding, gradient clipping, per-epoch evaluation,
best-checkpoint selection, optional early stopping, and run recording through
`compgen.tracking`.

The one task-specific seam is the step function::

    def step_fn(model, batch) -> (loss, metrics)

`loss` is a scalar tensor to backpropagate; `metrics` is a mapping of extra
numbers to average over the epoch (it may be empty). The step functions own
batch placement on the device — use `to_device` for the common case:

    def step_fn(model, batch):
        batch = to_device(batch, device)
        ...
        return loss, {"accuracy": acc}

Evaluation reuses `step_fn` unless `eval_fn(model, batch) -> metrics` is given;
the evaluation loss is reported as ``"loss"`` in the evaluation metrics.

Returned history per epoch: training loss and step metrics, plus evaluation
metrics on evaluated epochs. The result also carries the best epoch/metric and
a deep-copied best ``state_dict``.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from compgen import tracking

StepFn = Callable[[nn.Module, Any], "tuple[torch.Tensor, Optional[Mapping[str, float]]]"]
EvalFn = Callable[[nn.Module, Any], Mapping[str, float]]


@dataclass
class EpochRecord:
    """Metrics recorded for one training epoch."""

    epoch: int
    train_loss: float
    train_metrics: Dict[str, float] = field(default_factory=dict)
    eval_metrics: Optional[Dict[str, float]] = None
    improved: bool = False


@dataclass
class FitResult:
    """Outcome of a training run."""

    history: List[EpochRecord]
    best_epoch: Optional[int] = None
    best_metric: Optional[float] = None
    stopped_early: bool = False
    best_state_dict: Optional[Dict[str, Any]] = None


def to_device(obj: Any, device: Union[str, torch.device]) -> Any:
    """Recursively move tensors in dicts/lists/tuples to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [to_device(value, device) for value in obj]
        return type(obj)(moved) if isinstance(obj, tuple) else moved
    return obj


def _resolve_optimizer(optimizer, model: nn.Module) -> torch.optim.Optimizer:
    if optimizer is None:
        return torch.optim.AdamW(model.parameters(), lr=1e-3)
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer
    if callable(optimizer):
        resolved = optimizer(model)
        if not isinstance(resolved, torch.optim.Optimizer):
            raise TypeError("optimizer factory must return a torch.optim.Optimizer")
        return resolved
    raise TypeError("optimizer must be an nn.Optimizer, a factory, or None")


def _resolve_scheduler(scheduler, optimizer: torch.optim.Optimizer):
    if scheduler is None:
        return None
    if callable(scheduler) and not hasattr(scheduler, "step"):
        return scheduler(optimizer)
    return scheduler


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_batches(model, data, step_fn, optimizer=None, grad_clip=None) -> "tuple[float, Dict[str, float]]":
    """Run one pass over ``data``; backprop only when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total_loss, num_batches = 0.0, 0
    metric_sums: Dict[str, float] = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in data:
            if training:
                optimizer.zero_grad()
            out = step_fn(model, batch)
            loss, metrics = out if isinstance(out, tuple) else (out, None)
            if training:
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += float(loss.detach())
            num_batches += 1
            for key, value in (metrics or {}).items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
    if num_batches == 0:
        raise ValueError("data source produced no batches")
    return total_loss / num_batches, {k: v / num_batches for k, v in metric_sums.items()}


def _best_metric_value(metrics: Mapping[str, float], name: str) -> float:
    if name not in metrics:
        raise ValueError(
            f"metric_for_best={name!r} not found in metrics {sorted(metrics)}"
        )
    return float(metrics[name])


def fit(
    model: nn.Module,
    train_data: Sequence[Any],
    step_fn: StepFn,
    *,
    eval_data: Optional[Sequence[Any]] = None,
    eval_fn: Optional[EvalFn] = None,
    optimizer: Any = None,
    scheduler: Any = None,
    epochs: int = 1,
    device: Union[str, torch.device] = "cpu",
    grad_clip: Optional[float] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    eval_every: int = 1,
    metric_for_best: str = "loss",
    higher_is_better: bool = False,
    early_stopping_patience: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    track: bool = True,
    run_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> FitResult:
    """Train ``model`` for ``epochs`` over ``train_data`` with ``step_fn``.

    See the module docstring for the step-function contract. Returns a
    `FitResult` with the per-epoch history and the best checkpoint.
    """
    if seed is not None:
        _seed_everything(seed, deterministic)

    model.to(device)
    optimizer = _resolve_optimizer(optimizer, model)
    scheduler = _resolve_scheduler(scheduler, optimizer)
    eval_step = eval_fn if eval_fn is not None else step_fn

    result = FitResult(history=[])
    best_value: Optional[float] = None
    epochs_without_improvement = 0

    def is_improvement(value: float) -> bool:
        if best_value is None:
            return True
        return value > best_value if higher_is_better else value < best_value

    if track and tracking_uri is not None:
        tracking.configure(tracking_uri=tracking_uri)
    run_context = (
        tracking.start_run(run_name=run_name, tags=None)
        if track
        else _nullcontext()
    )

    with run_context:
        if track:
            base_params = {
                "epochs": epochs,
                "device": str(device),
                "grad_clip": float(grad_clip) if grad_clip is not None else 0.0,
                "seed": seed if seed is not None else -1,
                "eval_every": eval_every,
                "metric_for_best": metric_for_best,
                "higher_is_better": higher_is_better,
                "early_stopping_patience": (
                    early_stopping_patience if early_stopping_patience is not None else -1
                ),
                "num_parameters": sum(p.numel() for p in model.parameters()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if params:
                base_params.update(params)
            tracking.log_params(base_params)

        for epoch in range(epochs):
            train_loss, train_metrics = _run_batches(
                model, train_data, step_fn, optimizer=optimizer, grad_clip=grad_clip
            )
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                train_metrics=train_metrics,
                eval_metrics=None,
            )

            tracked_metrics: Mapping[str, float] = {"loss": train_loss, **train_metrics}
            if eval_data is not None and epoch % eval_every == 0:
                eval_loss, eval_metrics = _run_batches(model, eval_data, eval_step)
                record.eval_metrics = {"loss": eval_loss, **eval_metrics}
                tracked_metrics = record.eval_metrics

            value = _best_metric_value(tracked_metrics, metric_for_best)
            if is_improvement(value):
                best_value = value
                result.best_epoch = epoch
                result.best_metric = value
                result.best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                record.improved = True
                if checkpoint_path is not None:
                    _save_checkpoint(result.best_state_dict, checkpoint_path)
            else:
                epochs_without_improvement += 1

            result.history.append(record)

            if track:
                logged = {f"train_{k}": v for k, v in record.train_metrics.items()}
                logged["train_loss"] = train_loss
                if record.eval_metrics is not None:
                    logged.update(record.eval_metrics)
                tracking.log_metrics(logged, step=epoch)

            if scheduler is not None:
                scheduler.step()

            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                result.stopped_early = True
                break

        if track:
            best_model = copy.deepcopy(model)
            if result.best_state_dict is not None:
                best_model.load_state_dict(result.best_state_dict)
            tracking.log_model(best_model, name="model")

    return result


def _save_checkpoint(state_dict: Mapping[str, Any], path: str) -> None:
    from pathlib import Path

    target = Path(path)
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, target)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
