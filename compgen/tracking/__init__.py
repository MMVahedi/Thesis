"""
Reusable, MLflow-backed experiment tracking.

Task-agnostic wrapper over MLflow: run lifecycle plus logging of parameters,
metrics (with steps), artifacts, and PyTorch models. Defaults to a local
``mlruns/`` store and honors an explicit URI or ``MLFLOW_TRACKING_URI``.

See `compgen/tracking/README.md` for usage.
"""

from compgen.tracking.tracking import (
    DEFAULT_EXPERIMENT,
    configure,
    log_artifact,
    log_artifacts,
    log_metrics,
    log_model,
    log_params,
    start_run,
)

__all__ = [
    "DEFAULT_EXPERIMENT",
    "configure",
    "log_artifact",
    "log_artifacts",
    "log_metrics",
    "log_model",
    "log_params",
    "start_run",
]
