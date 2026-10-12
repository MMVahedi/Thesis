# Proposal

## Why

Weight initialization is scattered across the model layer: `BaseAttention._init_weights` and the `TokenClassifier` constructor each reimplement the same xavier-normal / zero-bias pass inline, the reference T5-style scheme lives in `compgen/models/initialization.py` and is applied only by `FuzzyLogicModel`, and the feed-forward variants, regression head, and task embeddings silently keep PyTorch's defaults. There is no single place to see or change how weights start, so a newly added module can easily drift out of an existing policy. Centralizing initialization in one class makes the policy explicit and lets every parameterized module obtain it from one controlled source.

## What Changes

- Add a `WeightInitializer` policy class in `compgen/models/initialization.py` that centralizes the schemes the model layer uses, exposed as shared instances: `XAVIER_INIT` (xavier-normal weights, zero biases), `REFERENCE_INIT` (the existing T5-style truncated-normal scheme for the fuzzy-logic transformer), and `DEFAULT_INIT` (explicit no-op that leaves PyTorch's defaults).
- Every parameterized model-layer module SHALL accept an optional initializer at construction and use it for its own parameters: attention architectures (default `XAVIER_INIT`), `TokenClassifier` (default `XAVIER_INIT`), `RegressionHead`, `ReluStackFFN`/`GeluFFN`, the task embeddings, and the encoder layer's normalization parameters (default `DEFAULT_INIT`).
- Task models SHALL accept an initializer and apply it across their composed sub-modules: `FuzzyLogicModel` defaults to `REFERENCE_INIT` (replacing its inline `init_reference_weights(self)` call) and `Match3Model` defaults to `DEFAULT_INIT`.
- Remove the inline xavier loop from `TokenClassifier` and route attention's `_init_weights` through the injected initializer; retain `init_reference_weights` as a thin wrapper for compatibility.
- Update the `AGENTS.md` and root `README.md` model-layer description to point at the centralized initializer.
- **No numeric behavior change**: every module keeps the scheme it uses today, including `FuzzyLogicModel`'s whole-model reference re-initialization; only the code path moves into one class.

## Capabilities

### New Capabilities
<!-- None: this is a reorganization of existing model-layer behavior. -->

### Modified Capabilities
- `model-layer`: add a "Centralized weight initialization" requirement (one policy class; each parameterized module obtains and uses it) and amend "Attention architectures initialize their own weights" so its scheme comes from the shared initializer rather than a hardcoded inline pass.

## Impact

- Code: `compgen/models/initialization.py`, `compgen/models/attentions/base.py` and the five architecture modules, `compgen/models/heads.py`, `compgen/models/ffn.py`, `compgen/models/encoder.py`, `compgen/models/embeddings/{fuzzy_logic,match3}.py`, and `compgen/models/tasks/{fuzzy_logic,match3}.py`.
- API: constructors gain an optional `initializer` keyword (backward compatible); `init_reference_weights` is kept as a thin wrapper, so no caller breaks.
- Docs: `AGENTS.md` and the root `README.md`.
- No dependency or data-flow change; no tracked checkpoints depend on initialization order beyond the already-accepted fixed-seed drift.
