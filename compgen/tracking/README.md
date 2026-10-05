# tracking

Reusable, task-agnostic experiment tracking backed by MLflow. It gives the
project one stable import surface for recording runs — parameters, metrics
(with steps), artifacts, and PyTorch models — so training and model-building
code never calls `mlflow` directly.

## Quickstart

```python
import torch.nn as nn

from compgen.tracking import configure, start_run, log_params, log_metrics, log_model

configure()  # defaults to the local sqlite:///mlflow.db store
model = nn.Linear(4, 1)

with start_run(run_name="demo"):
    log_params({"lr": 1e-3, "epochs": 3})
    for epoch in range(3):
        log_metrics({"train_loss": 1.0 / (epoch + 1)}, step=epoch)
    log_model(model, name="model")
```

`start_run` is a context manager, so the run ends even if the body raises.

## Backend

The tracking URI is resolved in this order (first match wins):

1. an explicit `configure(tracking_uri=...)`;
2. the `MLFLOW_TRACKING_URI` environment variable;
3. the local SQLite store `sqlite:///mlflow.db`, with artifacts under
   `mlartifacts/`.

No tracking server is required. Both `mlflow.db` and `mlartifacts/` are
gitignored. Run from the repository root so the local store lands in one place,
or pass an absolute URI / set `MLFLOW_TRACKING_URI` for a shared store.

To browse runs: `venv/bin/mlflow ui` (then open the printed URL).

## API

| Function | Purpose |
|---|---|
| `configure(tracking_uri=None, experiment_name="compgen")` | Select the backend and experiment. |
| `start_run(run_name=None, tags=None, nested=False)` | Context manager for a run. |
| `log_params(mapping)` | Log parameters (non-primitive values are stringified). |
| `log_metrics(mapping, step=None)` | Log metrics; `step` makes repeated keys a time series. |
| `log_artifact(path)` / `log_artifacts(directory)` | Log a file / a directory. |
| `log_model(model, name="model")` | Log a `torch.nn.Module`; never mutates it. |

## Model logging

`log_model` uses the MLflow PyTorch flavor and defaults to
`serialization_format="pickle"`, because the safe `"pt2"` graph format requires
an example input to trace:

```python
log_model(model, name="model", input_example=torch.zeros(1, 4), serialization_format="pt2")
```

Logging does not change the model's parameters or training/eval state, so it is
safe to call mid-training.
