# Tasks

## 1. Dependency and ignore rules

- [x] 1.1 Add `mlflow==3.16.1` to the root `requirements.txt` (as a direct, exactly pinned entry, keeping the existing torch/CUDA anchoring and comments intact), then install it. Verify: `venv/bin/pip install -r requirements.txt` succeeds, `venv/bin/python -c "import mlflow; print(mlflow.__version__)"` prints `3.16.1`, `venv/bin/python -c "import torch,numpy; print(torch.__version__, numpy.__version__)"` still prints `2.14.0+cu132` and the previously pinned numpy, and `venv/bin/pip check` reports no broken requirements.
- [x] 1.2 Extend `.gitignore` with the MLflow local run output (`mlflow.db`, `mlartifacts/`). Verify `git check-ignore mlflow.db mlartifacts/` reports both paths ignored and `git status --short` shows no tracking-run files as untracked.

## 2. Tracking layer

- [x] 2.1 Create `compgen/tracking/__init__.py` and `compgen/tracking/tracking.py` implementing the public surface (`configure`, `start_run`, `log_params`, `log_metrics`, `log_artifact`, `log_artifacts`, `log_model`) as thin delegations to MLflow, with module docstrings. Verify `venv/bin/python -c "import compgen.tracking"` succeeds and `venv/bin/python -c "import compgen.tracking, sys; print([m for m in sys.modules if m.startswith('compgen')])"` shows no task model/dataset module imported.
- [x] 2.2 Implement backend resolution (explicit argument, then `MLFLOW_TRACKING_URI`, then local `sqlite:///mlflow.db`) and the configurable experiment name. Verify with a temp store: a run records to the local SQLite store by default; setting `MLFLOW_TRACKING_URI` redirects it; an explicit URI wins when both are set.
- [x] 2.3 Implement parameter, stepped-metric, and artifact logging. Verify: logged params and latest metric values are read back via an MLflow client; the same metric logged at steps 0,1,2 yields a 3-point history; a logged file and a logged directory both appear in the run's artifacts.
- [x] 2.4 Implement `log_model` via the MLflow PyTorch flavor using the 3.x `name=` argument, without mutating the model. Verify a small `nn.Module` logs and reloads as a `torch.nn.Module`, and a full `state_dict` clone is bit-identical before and after logging.
- [x] 2.5 Implement the `start_run` context manager so the run always ends (normal exit and exception). Verify after a run that raises inside the context, the exception propagates and no active MLflow run remains.

## 3. Tracking documentation and conventions

- [x] 3.1 Write `compgen/tracking/README.md` with a minimal, runnable usage example (configure, start a run, log params/metrics/artifacts/a model). Verify the example runs as written from the repository root and records a run.
- [x] 3.2 Add the tracking layer to `README.md` (layout row) and to `AGENTS.md` (layout entry plus the convention that future training/model-building implementations log their runs through `compgen.tracking`). Verify every tracking path named in these docs exists and the documented import snippet runs.

## 4. Training layer

- [x] 4.1 Create `compgen/training/__init__.py` and `compgen/training/loop.py` implementing the task-agnostic `fit` engine parameterized by `step_fn(model, batch) -> (loss, metrics)` (and an optional `eval_fn`). Verify on a toy linear regression: `fit` runs the requested number of epochs, the returned history contains the per-epoch training loss, and the loss decreases; and importing `compgen.training` imports no task model/dataset module.
- [x] 4.2 Implement the configuration surface (optimizer instance or factory, optional scheduler factory, `device`, `grad_clip`, `seed`) with seeding of `random`/`numpy`/`torch`. Verify: two identical seeded runs produce matching loss histories; gradient clipping is applied when configured (a step's gradient norm is bounded); an optimizer instance is used as-is.
- [x] 4.3 Implement evaluation at a configurable interval, `metric_for_best` best-checkpoint retention (in-memory `state_dict`, optional `checkpoint_path`), and optional early stopping. Verify: eval metrics appear in the history; the retained best checkpoint matches the best tracked-metric epoch; with a non-improving metric, training stops before the epoch budget and reports early stop.
- [x] 4.4 Wire run recording through `compgen.tracking`: log the run configuration at start, per-epoch training/evaluation metrics as stepped metrics, and the best (or final) model artifact at the end, with `track=False` disabling it. Verify: a tracked toy run's params/metrics/model are readable from the run; a `track=False` run creates no run directory/active run while still completing.

## 5. Training documentation and conventions

- [x] 5.1 Write `compgen/training/README.md` with a minimal, runnable example and add the training layer to `README.md` and `AGENTS.md` (entry plus the convention to use the engine and log through `compgen.tracking`). Verify the example completes a short run from the repo root and every documented training path exists.

## 6. Integration verification

- [x] 6.1 Run a tracked end-to-end training pass on a real task model: build a small Match3 model and batch, wrap a supervised `step_fn`, run a few epochs through `fit`, and confirm the MLflow run records the params, per-epoch metrics, and a reloadable model artifact; also confirm the existing `compgen.analytics` layer still imports and works.
- [x] 6.2 Confirm isolation and hygiene: `grep -rnE "import compgen|from compgen" compgen/experiments/` returns no matches, and `git status --short` shows no generated datasets, checkpoints, archives, or `mlflow.db`/`mlartifacts/` content as untracked.
- [x] 6.3 Validate the change: `openspec validate add-mlflow-tracking --strict` passes.
