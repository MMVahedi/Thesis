# Tasks

## 1. Attention architectures initialize themselves

- [x] 1.1 Add an `_init_weights()` method to `BaseAttention` (`compgen/models/attentions/base.py`) that applies xavier-normal initialization to every `weight` parameter and zeros every `bias` parameter (the scheme currently in `encoder._init_attention_weights`); verify the method runs without error on a constructed architecture.
- [x] 1.2 Call `self._init_weights()` at the end of `__init__` in each registered architecture — `SoftmaxAttention`, `LinearAttention`, `StrassenAttention`, `ThirdOrderAttention`, `TriangularAttention` — after their projections are built; verify from the repo root that constructing each architecture (by name via `build_attention`) yields projection weights that match an explicit xavier-normal draw and zero biases, with no external initialization call.
- [x] 1.3 Document the self-initialization in the `attentions/base.py` module/class docstring and note it where an architecture builds projections; verify the docstrings state that a constructed architecture is already initialized.

## 2. Encoder layer becomes a dropout-free, non-initializing composition

- [x] 2.1 In `compgen/models/encoder.py`, remove the module-level `_init_attention_weights` helper and its call, remove the `dropout_rate` parameter and the `self.dropout1`/`self.dropout2` modules, and change `forward` so both residual connections add the sub-module output directly; verify an `EncoderLayer` built with a constructed attention and FFN forwards a hidden state of the same shape, returns per-layer attention weights of shape `(batch, num_heads, length, length)`, and exposes no `nn.Dropout` module.
- [x] 2.2 Update the `encoder.py` module and `EncoderLayer` docstrings to state that the layer applies no dropout and does not initialize the modules it is handed (each module initializes itself); verify the docstrings no longer mention `dropout1`/`dropout2`, a dropout rate, or `_init_attention_weights`.
- [x] 2.3 Verify the encoder stays registry-independent and side-effect-free: from the repo root, confirm `compgen/models/encoder.py` neither imports the attention registry, the feed-forward registry/module, nor any task module, and that passing an attention module with known weights to `EncoderLayer` leaves those weights byte-identical.

## 3. Task models construct the feed-forward directly and stop passing dropout

- [x] 3.1 Update `compgen/models/tasks/match3.py` and `compgen/models/tasks/fuzzy_logic.py` to stop passing `dropout_rate` to `EncoderLayer` while continuing to pass it when constructing the feed-forward module and to the classifier/head; verify both models construct, forward with the same output shapes as before, and report unchanged total parameter counts.
- [x] 3.2 Update `compgen/models/tasks/match3.py` so `Match3Model` instantiates `ReluStackFFN(hidden_dim=..., ffn_depth=..., dropout_rate=..., dtype=..., device=...)` directly inside its per-layer construction (replacing the `build_ffn("relu_stack", ...)` call); verify construction with `ffn_depth` values including the default, that `share_layers=True` still shares one layer, that a forward returns probabilities `(batch, length, 1)` and hidden states `(batch, length, hidden_dim)`, and that the reference-config parameter count is unchanged.
- [x] 3.3 Update `compgen/models/tasks/fuzzy_logic.py` so `FuzzyLogicModel` drops its `ffn_type` parameter and instantiates `GeluFFN(hidden_dim=..., ffn_dim=..., dropout_rate=..., dtype=..., device=...)` directly per layer (replacing the `build_ffn(ffn_type, ...)` call); verify the model constructs, that each layer's `ffn` is a `GeluFFN`, that a forward returns per-token predictions plus per-layer attention of shape `(batch, num_heads, length, length)`, and that the reference-config parameter count is unchanged.
- [x] 3.4 Update the `Match3Model`/`FuzzyLogicModel` docstrings (and the `FuzzyLogicModel` parameter list) so they say dropout lives in the feed-forward block and classifier and that the feed-forward variant is constructed directly (no factory, no `ffn_type`); verify no docstring claims encoder residual dropout or a feed-forward factory.

## 4. Remove the feed-forward factory

- [x] 4.1 Delete `build_ffn` and `_FFN_CLASSES` from `compgen/models/ffn.py`, keeping the `BaseFeedForward`/`ReluStackFFN`/`GeluFFN` classes, and update the module docstring to describe direct construction instead of the factory; verify `ReluStackFFN` and `GeluFFN` still construct and forward at unchanged width and that `compgen/models/ffn.py` exposes no `build_ffn`.
- [x] 4.2 Update `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` to stop passing `ffn_type` to `FuzzyLogicModel` (it may still pass `ffn_dim`); verify the runner's model builds and forwards successfully.

## 5. Documentation

- [x] 5.1 Update `AGENTS.md` (and the root `README.md` only if it describes the encoder's dropout/initialization, the feed-forward factory, or the task models' FFN selection) so the model-layer description states that the encoder applies no dropout and does not initialize its inputs, that attention architectures self-initialize, that `compgen/models/ffn.py` provides the `ReluStackFFN`/`GeluFFN` classes without a factory, and that task models instantiate the feed-forward module directly; verify every repository-relative model-layer path mentioned still exists.

## 6. Integration verification

- [x] 6.1 From the repo root with a fixed seed, build one `Match3Model` and one `FuzzyLogicModel` and confirm total parameter counts and a forward pass's output shapes match the pre-change revision; confirm neither model's encoder layers contain any `nn.Dropout`, that `FuzzyLogicModel` no longer accepts `ffn_type`, and that `compgen/models/ffn.py` exposes no `build_ffn`; record that fixed-seed weight values may differ (expected, due to initialization order).
- [x] 6.2 Confirm initialization ownership: snapshot an already-constructed attention module's `state_dict()`, construct an `EncoderLayer` with it, and confirm the state dict is unchanged; separately confirm a freshly constructed architecture's weights are the xavier/zero scheme.
- [x] 6.3 Smoke-run the string-based config path: build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` and confirm it constructs and forwards successfully.
