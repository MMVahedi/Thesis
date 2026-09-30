# Design

## Context

See `proposal.md` — Why. Current state that shapes the approach:

- The package has four layers (`compgen/models/`, `compgen/datasets/`, `compgen/losses/`, and the sealed `compgen/experiments/` archive) but no analysis layer. The only interpretation code is `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/analyze_attention.py`, which is intentionally sealed (imports nothing from `compgen`) and bound to one saved-run format: it loads `attention_codes.npz`, runs `sklearn` t-SNE plus logistic term-decoding, and writes files into a run directory.
- Attention weights are already surfaced by the model layer: `EncoderLayer.forward(..., return_attention=True)` returns `(hidden_state, attention_weights)` and `SoftmaxAttention` (and siblings) return `(att, att_weights)` with `att_weights` of shape `(B, H, N, N)`. `FuzzyLogicModel.forward(..., return_attention=True)` returns `(prediction, attentions)` where `attentions` is a list of per-layer `(B, H, N, N)` tensors; `Match3Model.forward` currently returns only `(y_hat, hidden_state)`.
- `FuzzyLogicModel.response_token_codes(attentions)` already reduces the final layer to a `(B, H)` per-example code by taking `attentions[-1][:, :, -1, -1]` — the last query token's attention to the last key. This is exactly the "attention scores as vectors" pattern the analytics layer generalizes.
- Admission of a new layer follows the house pattern: a sibling package under `compgen/`, a dedicated capability spec, and `README.md`/`AGENTS.md` entries (as with `dataset-layer`, `losses-layer`, `model-layer`).
- `torch`, `scikit-learn`, and `matplotlib` are already pinned in the root `requirements.txt`; the experiment already relies on `sklearn` (PCA/TSNE/LogisticRegression) and `matplotlib`. No new dependency is needed.

## Goals / Non-Goals

**Goals:**
- A reusable, task-agnostic analytics layer under `compgen/analytics/`, importable as `compgen.analytics.*`, whose core functions take tensors and never import a task model.
- Turn attention weights into per-example vectors (generalizing `response_token_codes`), inspect one query token's attention row, project vectors to 2D via PCA (configurable component count, deterministic signs), and optionally plot.
- A capture helper that runs any shared-attention-composed model (fuzzy-logic, Match3, future tasks) in inference mode and returns per-layer attention (and the hidden states entering each layer).
- Give `Match3Model` an attention-returning path so attention is available from the model API for parity with `FuzzyLogicModel`.

**Non-Goals:**
- No non-linear dimensionality reduction (t-SNE/UMAP), clustering, classifier probes, or decoding analyses — the experiment's t-SNE/decoding stays in the sealed archive; this change ships PCA only.
- No training, checkpoint loading, or dataset generation in the analytics layer.
- No changes to the sealed `compgen/experiments/` archive (or its frozen notebook), the attention registry/architectures, `compgen/datasets/`, or `compgen/losses/`.
- No new runtime dependency and no change to any model's default output or numerics.

## Decisions

1. **New sibling package `compgen/analytics/` with four focused modules plus `__init__.py`:**
   - `capture.py` — run a model and capture attention/hidden states.
   - `attention.py` — attention → per-example vectors, and per-token attention inspection.
   - `projection.py` — PCA fitting/transform and the 2D result type.
   - `plotting.py` — scatter and heatmap figure helpers.
   *Why*: mirrors the existing layer organization (each layer is a package with small modules and a public `__init__`), and keeps model-running (`capture`) separate from pure tensor math (`attention`, `projection`). *Alternatives considered*: a single `analytics.py` module — rejected, it would mix torch capture, numpy/sklearn math, and plotting; nesting under `compgen/models/analysis/` — rejected by the user's placement choice, analysis is not model definition.

2. **Core functions are model-agnostic: they accept tensors.** `attention.py` and `projection.py` import neither `torch` task models nor `compgen.models.tasks`; they operate on `(B, H, N, N)` attention tensors and `(B, D)` vectors. *Why*: matches the `losses-layer` precedent (caller-supplied representations) and makes the tools reusable across tasks; it also keeps the spec-level contract about tensors meaningful.

3. **Attention → vector API generalizes `response_token_codes`.**
   - `attention_code(attention, query_token=-1, key_token=-1) -> (B, H)`: for a single `(B, H, N, N)` tensor, gathers `[query_token, key_token]` per head; defaults reproduce the existing last-token-to-last-token code.
   - `attention_row_features(attention, query_token=-1) -> (B, H*N)`: the full attention row of the query token, flattened over heads and keys.
   - `stack_layers(attentions, reducer="code"|"row") -> (B, L*H)` / `(B, L*H*N)`: concatenate per-layer features so layer identity is preserved in the vector.
   *Why*: one small family covers both the existing `(B, H)` code and the richer full-row feature, and the default arguments make the existing behavior a special case. *Alternative considered*: a single function with many booleans — rejected in favor of three named functions with clear output contracts.

4. **Per-token inspection returns the raw attention row.** `token_attention_row(attentions, layer, query_token, batch_index=0, average_heads=False)` accepts a single tensor or a list/stack of layers, selects the layer and batch example, and returns `(H, N)` (or `(N,)` when `average_heads=True`). Out-of-range `query_token`/`batch_index`/`layer` raise an error naming the invalid value. *Why*: the user's "find the attention matrix on a specific token" is a lookup, and returning the original weights (no renormalization) keeps it faithful. *Alternative considered*: returning a plotted figure directly — rejected; inspection returns data, plotting is separate.

5. **PCA via `scikit-learn`, wrapped in a small result type and with a deterministic sign fix.**
   - `PCAProjector(n_components=2)` with `.fit(vectors)`, `.transform(vectors)`, `.fit_transform(vectors)`, storing `components_`, `mean_`, `explained_variance_ratio_`.
   - `pca_project(vectors, n_components=2) -> ProjectionResult` convenience that fits and returns a `ProjectionResult(coordinates, explained_variance_ratio, projector)` dataclass.
   - Input is a torch tensor or numpy array; output coordinates are numpy arrays. Requesting more components than there are vectors (or samples ≤ 0) raises an error naming the mismatch.
   - **Deterministic signs**: after fitting, each component is sign-flipped so its largest-absolute-magnitude loading is positive, making coordinates reproducible across runs and platforms.
   *Why*: `scikit-learn` is already a pinned dependency and the experiment already uses it; re-implementing SVD in torch would duplicate a solved problem. The sign fix addresses PCA's inherent sign ambiguity so the "same input → same coordinates" scenario holds. *Alternative considered*: `torch.linalg.svd`/`torch.pca_lowrank` — rejected to avoid re-deriving whitening and solver behavior already provided and pinned.

6. **Capture helper uses forward hooks on shared attention modules, not per-task code.**
   - `capture(model, batch, device=None) -> CaptureResult(attentions, hidden_states)`:
     - sets the model to `eval()` for the duration (restoring its previous training flag afterward), runs under `torch.no_grad()`, and passes the raw batch straight to `model.forward`;
     - registers a forward hook on every module that is an instance of `BaseAttention`, recording the attention weights (the hook output's second element for the softmax/linear family) in module registration order;
     - registers a forward-pre-hook on the same modules to record the hidden state entering each attention block (the layer input).
   *Why*: works for both `FuzzyLogicModel` and `Match3Model` (and any future model built from `EncoderLayer`) without requiring `return_attention` support and without importing a task model — the hook targets the shared base class. Inference mode disables dropout so captured weights are the clean evaluation-time distributions. *Alternatives considered*: require `return_attention=True` on every task model — rejected as more invasive and still task-coupled; capture in the caller — rejected, it is the one piece worth sharing. *Trade-off*: hook capture relies on `BaseAttention` subclasses being the only attention modules in the model (currently true); this is documented as an assumption.

7. **`Match3Model.forward` gains `return_attention=False` (additive).** When `True`, each encoder layer is called with `return_attention=True`, the attentions are collected, and the method returns `(y_hat, hidden_state, attentions)`; when `False` it returns `(y_hat, hidden_state)` exactly as today. `FuzzyLogicModel` already returns `(prediction, attentions)`. *Why*: gives the model-layer API parity and a hook-free path, per the user's choice; keeping the default false guarantees no existing caller changes. *Alternative considered*: rely solely on the capture helper — rejected because the user asked for direct model access, and a plain return path is simpler for notebook use.

8. **Plotting returns figures and never writes files.** `plot_projection(coordinates, labels=None, cmap="viridis", ax=None)` and `plot_attention(attention, ax=None)` build and return `matplotlib.figure.Figure` objects; labels/values drive point coloring. *Why*: figure-returning helpers compose with notebooks and leave backend/saving to the caller (the experiment archive keeps its own save-to-disk convention). *Alternative considered*: a `save_path` argument — rejected as filesystem policy that belongs to callers.

9. **Documentation.** Module docstrings state each module's contract; `README.md` gains an `compgen/analytics/` layout row and a short usage note, and `AGENTS.md`'s layout section lists the analytics layer, mirroring how the other layers are documented. No separate layer README is added. *Why*: matches existing doc conventions and the `analytics-layer` spec's documentation requirement.

## Risks / Trade-offs

- [Hook capture depends on `BaseAttention` being the only attention module in a model] → Capture targets `BaseAttention` instances only and records in registration order; documented in the module docstring. Chain-of-attention in the current models is unambiguous (one base attention per encoder layer).
- [PCA sign/solver nondeterminism across runs or platforms] → Fixed sign convention (largest-absolute loading positive) plus a full/deterministic solver path; a scenario asserts identical coordinates on repeated runs.
- [Importing `matplotlib`/`scikit-learn` at package import adds weight] → Both are already pinned runtime dependencies; `plotting` is the only module that imports `matplotlib`, and the package `__init__` can import it lazily if import cost surfaces.
- [Capture in `eval()` mode differs from training-time attention (dropout on)] → Intended: interpretation should read clean distributions; the helper restores the model's prior training flag and does not touch parameters, and callers who want training-time weights call the model directly.
- [`Match3Model.forward` signature grows] → Additive with a `False` default; the default output and all existing constructions are unchanged, verified by a regression check on the default path.
- [Attention weights can be task-specific in meaning (Match3 classification vs fuzzy-logic regression)] → The analytics layer treats them as generic tensors and does not interpret semantics; labeling is left to callers (e.g. passing `term_ids` as plot colors).

## Migration Plan

Single additive change, no phased rollout and no data migration: add `compgen/analytics/` (modules + `__init__`), add the optional `return_attention` path to `Match3Model`, then update `README.md`/`AGENTS.md`. Rollback is `git revert` of the change commit; nothing existing depends on the new layer, and no existing default output changes. The sealed experiment archive is untouched.
