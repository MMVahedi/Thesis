# training

Reusable, task-agnostic training and evaluation scaffold. `fit` runs epochs
over caller-supplied data and a step function and handles everything the tasks
share: optimizer/scheduler setup, device, seeding, gradient clipping,
evaluation, best-checkpoint selection, early stopping, and run recording
through [`compgen.tracking`](../tracking/README.md).

## Quickstart

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from compgen.training import fit, to_device

device = "cpu"
model = nn.Linear(1, 1)
x = torch.linspace(-2, 2, 16).unsqueeze(1)
y = 2 * x + 1
batches = list(zip(x.unsqueeze(1).unbind(0), y.unsqueeze(1).unbind(0)))


def step_fn(model, batch):
    xb, yb = to_device(batch, device)
    return F.mse_loss(model(xb), yb), {}  # (loss, metrics)


result = fit(
    model, batches, step_fn,
    epochs=20, device=device, seed=0,
    eval_data=batches,                 # enables per-epoch evaluation
    track=True, run_name="linear-demo",
    params={"task": "linear"},
)
print(result.best_epoch, result.best_metric)
```

## The step function

The only task-specific piece is the step function:

```python
def step_fn(model, batch) -> (loss, metrics)
```

- `loss` is a scalar tensor to backpropagate.
- `metrics` is a mapping of extra numbers averaged over the epoch (may be `{}`).

The step functions own batch placement on the device — use `to_device` for the
common case. Evaluation reuses `step_fn` unless `eval_fn(model, batch) ->
metrics` is passed; the evaluation loss is reported as `"loss"`.

## `fit` options

| Argument | Meaning |
|---|---|
| `train_data` | Sequence of batches (a `DataLoader` or a list). Re-iterated each epoch. |
| `step_fn` | `(model, batch) -> (loss, metrics)`. |
| `eval_data`, `eval_fn` | Evaluation batches and optional custom eval function. |
| `optimizer` | An `nn.Optimizer`, a factory `f(model) -> optimizer`, or `None` (defaults to AdamW `lr=1e-3`). |
| `scheduler` | An `nn.LRScheduler`, a factory `f(optimizer) -> scheduler`, or `None`. Stepped per epoch. |
| `epochs`, `device`, `grad_clip`, `seed`, `deterministic` | Standard loop controls. |
| `eval_every` | Evaluate every N epochs (default 1). |
| `metric_for_best`, `higher_is_better` | Which evaluated metric selects the best checkpoint. |
| `early_stopping_patience` | Stop after N evaluations without improvement. |
| `checkpoint_path` | Also save the best `state_dict` to disk. |
| `track`, `run_name`, `tracking_uri`, `params` | Run recording via `compgen.tracking` (`track=False` disables). |

## Result

`fit` returns a `FitResult` with:

- `history`: one `EpochRecord` per epoch — `train_loss`, `train_metrics`, and
  `eval_metrics` on evaluated epochs;
- `best_epoch`, `best_metric`, `best_state_dict`: the best tracked-metric epoch
  and a deep copy of its `state_dict`;
- `stopped_early`: whether early stopping fired.

## Tracking

With `track=True` (the default) the run records the configuration and
hyperparameters, per-epoch training/evaluation metrics as stepped metrics, and
the best (or final) model. Recording goes through `compgen.tracking`, which
defaults to the local `sqlite:///mlflow.db` store; see
[`compgen/tracking/README.md`](../tracking/README.md).
