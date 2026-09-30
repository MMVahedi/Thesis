# Tasks

## 1. Analytics package scaffolding

- [x] 1.1 Create the `compgen/analytics/` package with `__init__.py` and the four modules `capture.py`, `attention.py`, `projection.py`, `plotting.py`, each carrying a module docstring stating its contract; re-export the public functions from `__init__.py`. Verify from the repo root: `venv/bin/python -c "import compgen.analytics"` succeeds and `venv/bin/python -c "import compgen.analytics as a; print(sorted(n for n in dir(a) if not n.startswith('_')))"` lists the public functions.
- [x] 1.2 Enforce the task-agnostic contract: analytics modules import only `torch`, `numpy`, `sklearn`, `matplotlib`, and each other — never a task model or task-specific code. Verify `rg -n "compgen\.models\.tasks|compgen\.datasets|compgen\.losses" compgen/analytics/` returns no matches, and that `venv/bin/python -c "import compgen.analytics"` succeeds without importing any task model.

## 2. Model capture helper

- [x] 2.1 Implement `capture.py`'s `capture(model, batch, device=None)`: put the model in `eval()` for the duration (restoring its previous training flag after), run under `torch.no_grad()`, register forward hooks on every `BaseAttention` submodule to record attention weights in registration order and forward-pre hooks to record the hidden state entering each block, and return a `CaptureResult(attentions, hidden_states)`. Verify by running over a small Match3 batch and a small fuzzy-logic batch: `len(attentions) == num_layers`, every attention has shape `(B, H, N, N)`, `len(hidden_states) == len(attentions)`, the model's `training` flag is unchanged, and a cloned parameter is bit-identical before and after.
- [x] 2.2 Document the `BaseAttention`-only capture assumption in the module docstring and handle a model with no shared attention modules by raising a clear error. Verify: calling `capture` on a module with no `BaseAttention` children raises an error naming the missing shared attention modules.

## 3. Attention vectors and per-token inspection

- [x] 3.1 Implement `attention.py`'s `attention_code(attention, query_token=-1, key_token=-1) -> (B, H)`, `attention_row_features(attention, query_token=-1) -> (B, H*N)`, and `stack_layers(attentions, reducer)` (concatenating per-layer features so layer identity is preserved). Verify that for a fuzzy-logic model `attention_code(attentions[-1])` matches the existing `FuzzyLogicModel.response_token_codes(attentions)` output exactly under a fixed seed, and that the row/stack functions return the documented shapes.
- [x] 3.2 Implement `token_attention_row(attentions, layer, query_token, batch_index=0, average_heads=False)` returning the raw `(H, N)` row (or `(N,)` when `average_heads=True`), with out-of-range `layer`/`query_token`/`batch_index` raising an error that names the invalid value. Verify: per-head and averaged rows each sum to 1 within tolerance on a fuzzy-logic attention tensor, and an out-of-range `query_token` raises an error mentioning that index.

## 4. PCA projection

- [x] 4.1 Implement `projection.py`'s `PCAProjector(n_components=2)` (`.fit`, `.transform`, `.fit_transform`, storing components/mean/explained-variance) and the `pca_project(vectors, n_components=2) -> ProjectionResult` convenience, with a deterministic sign convention (each component sign-fixed so its largest-absolute loading is positive). Verify: `(B, D)` vectors project to `(B, 2)` coordinates with a length-2 explained-variance ratio; `n_components=3` yields 3 columns; two runs on the same input give identical arrays; requesting more components than samples raises an error naming the mismatch.
- [x] 4.2 Confirm the projection consumes attention-derived vectors: run `attention_code` on captured attention and feed the `(B, H)` result into `pca_project`, verifying `(B, 2)` coordinates (using `H >= 2`, e.g. the 8-head fuzzy-logic model).

## 5. Plotting helpers

- [x] 5.1 Implement `plotting.py`'s `plot_projection(coordinates, labels=None, cmap="viridis", ax=None)` and `plot_attention(attention, ax=None)`, each returning a `matplotlib.figure.Figure` and accepting an optional axis; plotting must create no files. Verify: each returns a `Figure`, the projection figure has one point per coordinate (matching the number of input rows when `labels` is supplied), and calling them adds no new files to the working directory.

## 6. Match3 attention-returning path

- [x] 6.1 Add an optional `return_attention=False` argument to `Match3Model.forward`; when `True`, call each encoder layer with `return_attention=True`, collect the attentions, and return `(y_hat, hidden_state, attentions)`. Verify with a small batch: the default call returns exactly `(y_hat, hidden_state)` with the same shapes and (under a fixed seed) the same values as before the change, and `return_attention=True` returns a 3-tuple whose attention list has one `(B, H, N, N)` tensor per layer.
- [x] 6.2 Confirm the new path integrates with the analytics layer: captured Match3 attention and `return_attention=True` Match3 attention both feed `attention_code`/`token_attention_row` and `pca_project` without task-specific glue (use `H >= 2`, e.g. the 2-head Match3 config).

## 7. Documentation

- [x] 7.1 Add the analytics layer to `README.md` (layout row plus a short usage note showing capture → attention code → PCA → plot) and to `AGENTS.md`'s layout section, describing the modules and the `compgen.analytics.*` import style. Verify that every repository-relative path mentioned for the analytics layer in both files exists (`capture.py`, `attention.py`, `projection.py`, `plotting.py`, `__init__.py`) and that the documented import snippets run as written.

## 8. Integration verification

- [x] 8.1 Run an end-to-end interpretability pass from the repo root for both tasks: build a small fuzzy-logic batch and a small Match3 batch, `capture` each model's attention, derive per-example vectors with `attention_code`, project them with `pca_project` to `(B, 2)`, and produce both figures with the plotting helpers; confirm all outputs are finite and the documented shapes hold.
- [x] 8.2 Confirm experiment isolation and artifact hygiene are untouched: `rg -n "import compgen|from compgen" compgen/experiments/` returns no matches, `git status` shows no generated artifacts (datasets, checkpoints, `.zip`) added, and `venv/bin/python compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/analyze_attention.py --help` still parses.
- [x] 8.3 Validate the change: `openspec validate add-analytics-layer --strict` passes.
