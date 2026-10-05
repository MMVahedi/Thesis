"""
MLflow-backed experiment tracking for compgen.

A thin, task-agnostic wrapper over MLflow. It owns the project's tracking
defaults and exposes a stable import surface so callers never touch ``mlflow``
directly: record a run with `start_run`, then log parameters, metrics,
artifacts, and models inside it.

Backend resolution (first match wins):

1. an explicit ``tracking_uri`` passed to `configure`;
2. the ``MLFLOW_TRACKING_URI`` environment variable;
3. a local SQLite database at ``sqlite:///mlflow.db`` (with artifacts
   under ``mlartifacts/``), so no tracking server is required.

The experiment name defaults to ``"compgen"``. `start_run` configures the
backend automatically on first use.

Typical use::

    from compgen.tracking import start_run, log_params, log_metrics, log_model

    with start_run(run_name="baseline"):
        log_params({"lr": 1e-3, "epochs": 10})
        for epoch in range(10):
            log_metrics({"train_loss": 1.0 / (epoch + 1)}, step=epoch)
        log_model(model, name="model")

Logging never mutates the model: `log_model` serializes a snapshot and leaves
parameters and training/eval state untouched.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT = "compgen"
_DEFAULT_LOCAL_STORE = "sqlite:///mlflow.db"
_DEFAULT_ARTIFACT_DIR = "mlartifacts"

_configured = False


def configure(
    tracking_uri: Optional[str] = None,
    experiment_name: str = DEFAULT_EXPERIMENT,
    artifact_location: Optional[str] = None,
) -> str:
    """Point MLflow at a backend and select the experiment.

    Resolution order is explicit ``tracking_uri`` > ``MLFLOW_TRACKING_URI`` >
    the local ``sqlite:///mlflow.db`` store. For the local default, run
    artifacts are stored under ``mlartifacts/`` in the working directory.
    Returns the effective tracking URI.
    """
    global _configured
    uri = tracking_uri
    if uri is None:
        uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri is None:
        uri = _DEFAULT_LOCAL_STORE
    mlflow.set_tracking_uri(uri)

    if artifact_location is None and uri.startswith("sqlite:///"):
        artifact_location = str(Path.cwd() / _DEFAULT_ARTIFACT_DIR)
    _ensure_experiment(experiment_name, artifact_location)

    _configured = True
    return mlflow.get_tracking_uri()


def _ensure_experiment(experiment_name: str, artifact_location: Optional[str]) -> None:
    """Create the experiment (with a chosen artifact location) if it is new."""
    client = MlflowClient()
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


@contextmanager
def start_run(
    run_name: Optional[str] = None,
    tags: Optional[Mapping[str, Any]] = None,
    nested: bool = False,
) -> Iterator[Any]:
    """Start an MLflow run, guaranteeing it is ended when the block exits.

    The backend is configured on first use with the default resolution rules.
    """
    if not _configured:
        configure()
    with mlflow.start_run(
        run_name=run_name, tags=dict(tags) if tags is not None else None, nested=nested
    ) as run:
        yield run


def _coerce_param(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def log_params(params: Mapping[str, Any]) -> None:
    """Log a mapping of parameters (non-primitive values are stringified)."""
    mlflow.log_params({key: _coerce_param(value) for key, value in params.items()})


def log_metrics(metrics: Mapping[str, float], step: Optional[int] = None) -> None:
    """Log a mapping of metrics; ``step`` makes repeated keys a time series."""
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)


def log_artifact(path: Any) -> None:
    """Log a single file as an artifact."""
    mlflow.log_artifact(str(path))


def log_artifacts(directory: Any) -> None:
    """Log every file under a directory as artifacts."""
    mlflow.log_artifacts(str(directory))


def log_model(
    model: Any,
    name: str = "model",
    input_example: Any = None,
    serialization_format: str = "pickle",
) -> None:
    """Log a ``torch.nn.Module`` with the MLflow PyTorch flavor.

    Does not mutate the model: parameters and training/eval state are unchanged.
    ``serialization_format`` defaults to ``"pickle"`` because the newer ``"pt2"``
    format requires an ``input_example`` to trace the graph, which a generic
    caller may not have; pass ``input_example`` and ``serialization_format="pt2"``
    to opt into the traced format.
    """
    mlflow.pytorch.log_model(
        model,
        name=name,
        input_example=input_example,
        serialization_format=serialization_format,
    )
