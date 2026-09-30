# Proposal

## Why

The library can now train models and produce attention weights and hidden states, but the only interpretation code lives in the sealed `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/analyze_attention.py` archive, which imports nothing from `compgen` and is tied to one saved-run format (t-SNE over a fixed `attention_codes.npz`). The research needs a reusable, task-agnostic analytics layer so the same interpretability workflows — projecting vectors to 2D, inspecting what a specific token attends to — can run directly on live model outputs for both the fuzzy-logic and Match3 tasks, and for any future task, without copying code out of an experiment.

## What Changes

- **Introduce a new `compgen/analytics/` layer** as a sibling of `models/`, `datasets/`, and `losses/`, for reusable model-interpretability tooling.
- **Model-agnostic core, tensor-in API.** The analytics functions take already-extracted tensors — attention weights `(B, H, N, N)`, per-example representation vectors `(B, D)` — so they know nothing about a specific task model.
- **Dimensionality reduction to 2D.** A PCA projector that fits on representation vectors and returns 2D coordinates plus explained-variance ratios, with deterministic, reproducible component signs. Vectors can come from attention (e.g. a token's per-head attention row) or from hidden states.
- **Per-token attention inspection.** Utilities to select a query token and read its attention distribution over key positions, per head or averaged, for one layer or all layers, for a chosen batch example.
- **Representation-vector extraction from attention.** Generalize the existing response-token code (`FuzzyLogicModel.response_token_codes`) into reusable functions that turn attention weights into per-example vectors, so PCA/t-SNE can be applied to attention-derived features for any task.
- **Optional model capture.** A thin helper that runs a model over a batch and captures per-layer attention weights (and hidden states), working for both the fuzzy-logic and Match3 models by hooking the shared attention modules — no task-specific code.
- **Optional plotting helpers.** Lightweight 2D scatter of projected vectors (with optional per-point labels/coloring) and attention heatmaps, returning matplotlib figures rather than saving files.
- **`Match3Model` gains an attention-returning path.** `Match3Model.forward` accepts an optional `return_attention=False` (mirroring `FuzzyLogicModel`) so attention weights are available directly from the model API, not only through hooks. **Additive**, existing default output is unchanged.
- **Docs.** `README.md` and `AGENTS.md` describe the new analytics layer, its modules, and how it is used.

## Capabilities

### New Capabilities

- `analytics-layer`: reusable, task-agnostic model-interpretability tooling — attention-to-vector extraction, per-token attention inspection, dimensionality reduction (PCA to 2D), and optional plotting — plus a capture helper and its documentation contract.

### Modified Capabilities

- `model-layer`: adds a requirement that task models can return their attention weights on demand (Match3Model gains an optional `return_attention` path with unchanged default output), alongside the existing public-API-stability guarantees.

## Impact

- **Code**: new `compgen/analytics/` package (`capture`, `attention`, `projection`, `plotting`, `__init__`). Additive change to `compgen/models/tasks/match3.py` (optional `return_attention` in `forward`).
- **API**: purely additive. `Match3Model.forward` with no new argument returns exactly what it does today; `return_attention=True` returns `(y_hat, hidden_state, attentions)`. No existing constructor parameters or selectable names change.
- **Dependencies**: no new runtime dependency — uses the already-pinned `torch` and `scikit-learn`, with `matplotlib` only for the optional plotting helpers.
- **Docs**: `README.md` and `AGENTS.md` gain an analytics-layer entry; the sealed experiment archive and its frozen notebook are untouched.
- **Behavior**: no change to any existing model's default output, training, or dataset behavior. The analytics layer is read-only with respect to models (capture runs inference only).
- **Not touched**: `compgen/experiments/` (sealed, self-contained), `compgen/datasets/`, `compgen/losses/`, and the attention registry/architectures.
