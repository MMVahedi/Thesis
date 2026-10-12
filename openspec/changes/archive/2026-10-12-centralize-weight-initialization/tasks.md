# Tasks

## 1. Initializer policy class

- [x] 1.1 In `compgen/models/initialization.py`, add a `WeightInitializer` class that wraps an initialization scheme callable (with a name) and applies it to a module via `__call__`/`apply`; add the `_init_xavier_style` scheme (xavier-normal on `nn.Linear`/`nn.Embedding` weights, zero their biases), keep the existing reference scheme, add an explicit no-op `_init_default_style`, and expose the shared instances `XAVIER_INIT`, `REFERENCE_INIT`, and `DEFAULT_INIT`; verify each instance applies to a tiny `nn.Sequential` and produces the expected weights (xavier/zero, reference truncated-normal, and unchanged for the no-op).
- [x] 1.2 Keep `init_reference_weights` as a thin wrapper delegating to `REFERENCE_INIT`, and update the module docstring to describe the centralized policy rather than a single reference helper; verify importing `init_reference_weights` still works and that applying it and `REFERENCE_INIT` to identical modules yields identical weights.
- [x] 1.3 Verify scheme centralization: search `compgen/models/` for `xavier_normal_`, `trunc_normal_`, and `nn.init.` and confirm the schemes are defined only in `initialization.py` (no inline initialization loop remains in any other model-layer module once groups 2–7 are done).

## 2. Attention architectures

- [x] 2.1 In `compgen/models/attentions/base.py`, add an `initializer: WeightInitializer = XAVIER_INIT` constructor parameter to `BaseAttention`, store it, and make `_init_weights` delegate to `self.initializer(self)`; verify each of the five concrete architectures still constructs and its projection weights follow xavier-normal with zero biases, with no external initialization call.
- [x] 2.2 Update the `base.py` module/class docstrings to say the architecture initializes itself using the injected shared initializer (default xavier) rather than a fixed inline scheme; verify the docstrings no longer describe a hardcoded loop.
- [x] 2.3 Verify the initializer is injectable: construct each architecture with a non-default initializer (e.g. `REFERENCE_INIT`) and confirm its projection parameters follow that scheme instead of xavier-normal.

## 3. Prediction heads

- [x] 3.1 In `compgen/models/heads.py`, add an `initializer: WeightInitializer = XAVIER_INIT` parameter to `TokenClassifier`, replace its inline xavier/zero loop with a delegation to the injected initializer, and update its docstring; verify a constructed classifier's `dense`/`out_proj` weights match an explicit xavier-normal draw and its biases are zero, and that a non-default initializer changes the scheme.
- [x] 3.2 Add an `initializer: WeightInitializer = DEFAULT_INIT` parameter to `RegressionHead` and apply it at the end of `__init__`, updating the docstring to state the default keeps PyTorch's defaults so the `LayerNorm` is untouched; verify a default-constructed regression head's parameters equal PyTorch defaults, while a supplied `XAVIER_INIT` changes the linear weights.

## 4. Feed-forward variants

- [x] 4.1 In `compgen/models/ffn.py`, add an `initializer: WeightInitializer = DEFAULT_INIT` parameter to `BaseFeedForward`, store it, and add a `_init_weights`/apply call that each variant (`ReluStackFFN`, `GeluFFN`) invokes after building its blocks; update the module docstring; verify both variants construct and forward at unchanged width and that their default-constructed weights equal the previous PyTorch defaults (no extra random draws).
- [x] 4.2 Verify injection: construct each variant with a supplied `XAVIER_INIT` and confirm its `nn.Linear` weights now follow xavier-normal with zero biases.

## 5. Task embeddings

- [x] 5.1 In `compgen/models/embeddings/fuzzy_logic.py` and `compgen/models/embeddings/match3.py`, add an `initializer: WeightInitializer = DEFAULT_INIT` parameter and apply it after the projection is built (for Match3, guard the absent `in_proj`); verify both embeddings construct and forward with unchanged output shapes and that their default weights equal PyTorch defaults.
- [x] 5.2 Verify injection: construct each embedding with a supplied `XAVIER_INIT` and confirm its projection weights follow xavier-normal.

## 6. Encoder layer

- [x] 6.1 In `compgen/models/encoder.py`, add an `initializer: WeightInitializer = DEFAULT_INIT` parameter and apply it to the layer's own normalization parameters after they are built; update the docstring to note the layer still never initializes or re-initializes the attention/feed-forward modules it is handed; verify passing an attention module with known weights to `EncoderLayer` leaves its `state_dict()` unchanged, and that with `use_norm=False` the layer has no parameters.

## 7. Task models

- [x] 7.1 In `compgen/models/tasks/fuzzy_logic.py`, add `initializer: WeightInitializer = REFERENCE_INIT`, replace the `init_reference_weights(self)` call with `self.initializer(self)`, and update the docstring/imports; verify a default-constructed `FuzzyLogicModel` has all its Linear weights following the reference truncated-normal scheme (attention included) and that parameter count and forward output shapes are unchanged.
- [x] 7.2 In `compgen/models/tasks/match3.py`, add `initializer: WeightInitializer = DEFAULT_INIT` and end `__init__` with `self.initializer(self)`, updating the docstring; verify a default-constructed `Match3Model` keeps the previous per-module schemes (attention/classifier xavier; FFN/embedding defaults) and that parameter count and forward output shapes are unchanged.
- [x] 7.3 Verify a non-default model-level initializer overrides the composition: construct each task model with a supplied initializer (e.g. `XAVIER_INIT` for fuzzy-logic) and confirm the whole model's Linear weights follow that scheme.

## 8. Documentation

- [x] 8.1 Update the model-layer description in `AGENTS.md` and the root `README.md` to describe the centralized initialization policy (one class and shared instances in `compgen/models/initialization.py`, modules obtaining it by constructor injection, task models applying it across their composition); verify every repository-relative model-layer path mentioned still exists and no text still describes initialization as ad-hoc per module.

## 9. Integration verification

- [x] 9.1 From the repo root with a fixed seed, build one `Match3Model` and one `FuzzyLogicModel` before and after the change and confirm total parameter counts and forward output shapes match; record that exact fixed-seed weight values are expected to match where the parameter set and traversal order are unchanged.
- [x] 9.2 Build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` (no `initializer` argument) and confirm it constructs and forwards successfully, proving the defaults keep the runner working unchanged.
