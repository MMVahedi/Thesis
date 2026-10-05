# Design

## Context

See `proposal.md` — Why. Current state that shapes the approach:

- The package has layers `compgen/models/`, `compgen/datasets/`, `compgen/losses/`, `compgen/analytics/`, plus the sealed `compgen/experiments/` archive. There is **no training loop** anywhere in `compgen/` (the archive's `train.py` was removed when the model/data/loss were extracted) and **no tracking** of any kind.
- House layering pattern: each layer is a package with a small `__init__.py` public surface, module docstrings stating the contract, and (for `datasets`, and now `analytics`) a short `README.md`. Layer specs are separate capabilities (`model-layer`, `dataset-layer`, `losses-layer`, `analytics-layer`, `runtime-dependencies`, `repo-structure`).
- `requirements.txt` pins exact `==` versions and is documented in `AGENTS.md`; `.gitignore` already covers `data/`, `*.jsonl`, `*.zip`, logs, checkpoints, and `*.pt`/`*.ckpt`, but not MLflow output.
- All models are `torch.nn.Module`s with different output contracts: `Match3Model.forward(batch[, return_attention])` returns `(y_hat, hidden_state[, attentions])`; `FuzzyLogicModel.forward(x, return_attention=False)` returns `prediction` (or `(prediction, attentions)`). Losses live in `compgen/losses/` and take caller-supplied representations/targets, not a fixed model.
- `mlflow==3.16.1` is the current release. A resolution dry-run (`pip install --dry-run mlflow==3.16.1`) showed it would add MLflow and its transitive dependencies **without** changing the installed `torch==2.14.0+cu132`, `numpy`, `pandas`, or `scikit-learn`.
- MLflow 3.x uses `name=` on the model-logging API (the older `artifact_path=` argument is deprecated).

## Goals / Non-Goals

**Goals:**
- One task-agnostic MLflow wrapper (`compgen.tracking`) with a local-first file store default and an explicit/env override, covering run lifecycle, params, metrics (with steps), artifacts, and PyTorch models.
- One task-agnostic training/evaluation engine (`compgen.training`) that runs epochs over caller-supplied data and a caller-supplied step function, handles optimizer/scheduler/device/seeding, evaluates, selects the best checkpoint, and records the run through `compgen.tracking`.
- A pinned `mlflow` dependency, gitignored run directories, short usage READMEs for both layers, and `AGENTS.md` conventions that make logging the default expectation for future implementations.

**Non-Goals:**
- No task-specific training functions, default loss semantics, or dataset wiring for Match3/fuzzy-logic (the engine is parameterized, not specialized).
- No hyperparameter sweeping/orchestration, no remote tracking-server provisioning, and no MLflow autologging.
- No changes to the sealed `compgen/experiments/` archive, the model/dataset/losses/analytics layers, or any existing default output.
- No distributed/multi-GPU training and no serving/deployment.

## Decisions

1. **Two sibling layers, `compgen/tracking/` and `compgen/training/`, not one module or one combined layer.**
   *Why*: tracking (an MLflow-backed logging backend) and training (an optimization loop) are distinct behavioral contracts and map onto separate capability specs and READMEs, consistent with the existing one-package-per-layer organization. *Alternatives considered*: a single `compgen/tracking.py` — rejected, it breaks the package-per-layer convention and mixes backend plumbing with the loop; one combined `compgen/training/` that owns MLflow — rejected, it couples a reusable logging backend to one consumer.

2. **`compgen/tracking/` = `__init__.py` (public surface) + `tracking.py` (wrapper) + `README.md`.** The wrapper exposes: `configure(tracking_uri=None, experiment_name="compgen")`, a `start_run(run_name=None, tags=None, nested=False)` context manager, `log_params(mapping)`, `log_metrics(mapping, step=None)`, `log_artifact(path)`, `log_artifacts(directory)`, and `log_model(model, name="model")`. Each is a thin delegation to MLflow — no custom storage, no reimplementation.
   *Why*: MLflow already provides the mechanics; the layer adds the project defaults and a stable import surface so callers never touch `mlflow` directly. *Alternatives considered*: exposing `mlflow` directly — rejected, it scatters backend calls across the codebase and defeats the "consider tracking in future implementations" convention.

3. **Backend resolution: explicit argument → `MLFLOW_TRACKING_URI` → local `sqlite:///mlflow.db`.**
   `configure` (and `start_run` when needed) resolves the URI in that order; an explicit argument wins over the environment. The experiment name defaults to `compgen`. The local default is an SQLite database (`sqlite:///mlflow.db`) with artifacts under `mlartifacts/`. *Why*: local-first means no server is needed to start tracking, while the env override supports a shared/remote store later without code changes; SQLite is MLflow's recommended local backend now that the filesystem (`mlruns/`) backend is in maintenance mode and raises by default. *Alternatives considered*: requiring a server URI — rejected by the user's choice; the `mlruns/` filesystem backend — rejected because MLflow 3.16 raises on it unless `MLFLOW_ALLOW_FILE_STORE=true`, making it a poor default.

4. **Metrics are logged with `mlflow.log_metric(key, value, step=...)` per key, not `mlflow.log_metrics`.** *Why*: `log_metrics` has no `step`, so stepped metrics (per-epoch) would collapse to a single point; the per-key form preserves time series, which the spec requires. Params use `mlflow.log_params`.

5. **Model logging uses the MLflow PyTorch flavor with `name=`, and must not mutate the model.**
   `log_model` calls `mlflow.pytorch.log_model(model, name=...)` (3.x signature) without changing parameters or train/eval state. *Why*: the PyTorch flavor reloads to a `torch.nn.Module`, matching the spec; `name=` avoids the deprecated `artifact_path=`. *Alternative considered*: `mlflow.log_artifact` on a `state_dict` — rejected, it loses the reload-as-module guarantee.

6. **`compgen/training/` = `__init__.py` + `loop.py` (the engine) + `README.md`.** The engine is parameterized by a caller-supplied `step_fn(model, batch) -> (loss, metrics)` and an optional `eval_fn(model, batch) -> metrics`, configuration for optimizer (an `nn.Optimizer` instance or a factory with kwargs), optional scheduler factory, `epochs`, `device`, `grad_clip`, `seed`, `eval_every`, `metric_for_best`, `early_stopping_patience`, `checkpoint_path`, `track` (default `True`), and `run_name`. `fit(...)` returns a per-epoch history.
   *Why*: the two task models have incompatible output/target contracts and the losses take caller-supplied representations, so a universal default loss is impossible; the step function is the minimal task-specific seam while the loop owns everything generic. *Alternatives considered*: a default supervised step that assumes a fixed output shape — rejected as wrong for both current tasks; task-specific train functions — rejected, they duplicate the loop and couple layers.

7. **Device handling: the engine moves the model to `device`; `step_fn`/`eval_fn` receive `(model, batch)` and are responsible for moving the batch.** *Why*: batches are heterogeneous (a dict for Match3, a tensor or attribute-bearing object for fuzzy-logic), and only the task code knows how to place them; forcing a generic mover would guess. A small `to_device` helper may be offered, but the contract is that the step functions own batch placement. *Alternative considered*: auto-moving every tensor in a dict recursively — rejected as surprising and lossy for non-tensor batch fields.

8. **Best-checkpoint selection keeps the best model in memory via a deep-copied `state_dict`, and logs it as the run's model artifact at the end; `checkpoint_path` optionally also writes it to disk.** Early stopping uses `metric_for_best` and `early_stopping_patience`. *Why*: models in this repo are small (tens of thousands of parameters), so an in-memory best snapshot is simplest and avoids per-epoch disk churn; the disk option remains for large runs. *Alternative considered*: always writing checkpoints — rejected as unnecessary I/O and artifact clutter for current model sizes.

9. **Seeding: `fit` seeds `random`, `numpy`, and `torch` (and optionally enables deterministic cuDNN when asked).** *Why*: the spec requires same-seed reproducibility of recorded losses; best-effort seeding is the standard approach, with the honest caveat that full CUDA determinism is not guaranteed.

10. **`mlflow==3.16.1` pinned exactly in `requirements.txt`; `.gitignore` adds `mlflow.db` and `mlartifacts/`.** *Why*: matches the repo's exact-pin rule and keeps default local runs from being committed. The dry-run above is the evidence no anchored package moves. *Alternative considered*: a range pin — rejected by the existing manifest contract.

11. **`track=True` by default in `fit`, with an explicit disable.** *Why*: the request is that future implementations "consider" tracking; defaulting to on makes logging the path of least resistance while `track=False` supports throwaway/test runs. Tracking goes through `compgen.tracking`, so `fit` never imports `mlflow` directly.

12. **Docs: `compgen/tracking/README.md` and `compgen/training/README.md` carry minimal runnable examples; `AGENTS.md` gains both layer entries with the convention that future training/model-building code logs through `compgen.tracking` and uses `compgen.training`; the root `README.md` layout gains the two rows.** *Why*: matches how `datasets`/`analytics` are documented and satisfies the specs' documentation requirements.

## Risks / Trade-offs

- [MLflow is a heavy import and pulls many transitive dependencies] → Verified via dry-run that it resolves without disturbing the anchored `torch`/`numpy`/`pandas`; MLflow is imported only by `compgen.tracking`, so unrelated imports (`compgen.models`, `compgen.analytics`) stay light. The dependency is deliberate and pinned.
- [MLflow 3.x deprecates older model-logging arguments] → Use the 3.x `name=` argument and pin 3.16.1; implementation verifies reload works on this version.
- [Local `sqlite:///mlflow.db` is relative to the working directory, so runs from different directories go to different stores] → Document that runs are launched from the repo root; support an absolute explicit URI or `MLFLOW_TRACKING_URI` for a fixed store. `.gitignore` covers the default DB and artifact directory.
- [`step_fn` leaves device/target handling to the caller] → Documented contract with a worked example in the training README; a shared `to_device` helper is provided for the common case.
- [Keeping the best model in memory is bounded by model size] → Current models are small; `checkpoint_path` remains available for larger models.
- [Full CUDA determinism is not guaranteed by seeding] → Spec requires same-seed reproducibility of recorded losses under identical conditions; seeding is best-effort and deterministic cuDNN is opt-in. Documented, not overpromised.
- [Importing `compgen.training` transitively imports MLflow] → Intended: training records through tracking; the dependency is pinned and required.
- [Training/tracking APIs are new and may evolve] → Additive change; nothing existing depends on them, so later refinement is non-breaking.

## Migration Plan

Additive, single change, no data migration: pin `mlflow==3.16.1` in `requirements.txt` → install it → add `compgen/tracking/` → add `compgen/training/` → extend `.gitignore` → update `README.md`/`AGENTS.md` → verify (dry-run install check, a short tracked run that records params/metrics/model, and existing-layer import checks). Rollback is `git revert` of the change commit plus `pip uninstall mlflow`; no existing default behavior or dependency changes on rollback.

## Open Questions

- Whether to add MLflow autologging (framework-level automatic logging) later. Deferrable: it would layer on top of the explicit logging API and would not change these specs or the layer boundaries.
