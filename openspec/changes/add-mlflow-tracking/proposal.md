# Proposal

## Why

The library can build and evaluate models, but it has no experiment tracking: there is no training loop in `compgen/`, and nothing records a run's hyperparameters, metrics, artifacts, or model. As the thesis moves from building models to training and comparing them, results must be reproducible and comparable. MLflow is the chosen tracker; this change adds the plumbing, the runnable training scaffold, the pinned dependency, the docs, and an `AGENTS.md` convention so every future implementation logs its runs through the same layer.

## What Changes

- **Introduce a new `compgen/tracking/` layer** — a thin, task-agnostic wrapper over MLflow for the run lifecycle, parameters, metrics, artifacts, and PyTorch models, imported as `compgen.tracking.*`.
- **Local-first tracking with an environment override** — default to a local SQLite store (`sqlite:///mlflow.db`, artifacts under `mlartifacts/`), overridable by an explicit argument or the `MLFLOW_TRACKING_URI` environment variable; no tracking server is required.
- **Introduce a new `compgen/training/` layer** — a generic, task-agnostic training/evaluation scaffold parameterized by caller-supplied step functions, handling optimizer/scheduler setup, device, seeding, epochs, evaluation, and best-checkpoint selection, and logging the run through the tracking layer.
- **Pin the tracking dependency** — add `mlflow==3.16.1` to the root `requirements.txt` (verified to resolve without changing the installed `torch`/`numpy`/`pandas`).
- **Keep run output untracked** — `.gitignore` covers the MLflow local run output (`mlflow.db`, `mlartifacts/`).
- **Document and enforce the convention** — a minimal usage doc under the tracking layer, and an `AGENTS.md` entry so future training/model-building implementations log parameters, metrics, artifacts, and models via `compgen.tracking` and use the `compgen.training` scaffold.
- **Untouched** — the sealed `compgen/experiments/` archive (it stays self-contained and does not import `compgen`), the model/dataset/losses/analytics layers, and all existing default behavior.

## Capabilities

### New Capabilities

- `tracking-layer`: the task-agnostic MLflow-backed experiment-tracking surface (run lifecycle, params, metrics, artifacts, model logging), its local-first/override backend rules, and its documentation-and-conventions contract.
- `training-layer`: the generic, task-agnostic training and evaluation scaffold that runs epochs over caller-supplied data and step functions, selects the best checkpoint, and records runs through the tracking layer.

### Modified Capabilities

- `runtime-dependencies`: the single pinned root manifest gains `mlflow` as a direct runtime dependency (exact `==` pin), while the torch/CUDA anchoring, conflict-free-install, and experiment-isolation guarantees are preserved.
- `repo-structure`: the "generated artifacts stay untracked" contract is extended to cover MLflow local run output (`mlflow.db`, `mlartifacts/`).

## Impact

- **Code**: new `compgen/tracking/` package (`__init__.py`, the MLflow wrapper, `README.md`) and new `compgen/training/` package (`__init__.py`, the training/evaluation loop, `README.md`). No existing package changes its default behavior.
- **Dependencies**: `mlflow==3.16.1` added to root `requirements.txt`. Verified by a resolution dry-run to add only MLflow and its transitive dependencies, leaving `torch==2.14.0+cu132`, `numpy`, and `pandas` unchanged; the retained experiment keeps its own requirements untouched.
- **Config/artifacts**: `.gitignore` gains the MLflow run directories; tracking defaults to a local store so no external service is needed.
- **Docs**: `AGENTS.md` gains the tracking/training layer entries and the "log every training run" convention; the root `README.md` layout gains the two layers; each new layer carries a short README.
- **API**: purely additive. No existing constructor, function signature, or default output changes.
- **Not touched**: `compgen/experiments/` (sealed, self-contained), `compgen/models/`, `compgen/datasets/`, `compgen/losses/`, `compgen/analytics/`.
