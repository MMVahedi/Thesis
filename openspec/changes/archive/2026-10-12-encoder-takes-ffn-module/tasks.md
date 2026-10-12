# Tasks

## 1. Feed-forward module family

- [x] 1.1 Create `compgen/models/ffn.py` with a shared `BaseFeedForward`, a `ReluStackFFN` (a `ModuleList` of `Sequential(Dropout, Linear, ReLU)` blocks, `ffn_depth` of them) and a `GeluFFN` (`Sequential(Linear(ffn_dim), GELU(approximate="tanh"), Dropout, Linear(hidden))`), and a `build_ffn(name, *, hidden_dim, ffn_dim=None, ffn_depth=3, dropout_rate=0.0, dtype=torch.float64, device="cpu")` factory that validates the name, resolves `ffn_dim` to `hidden_dim` when `None`, and returns the constructed module; verify from the repo root that each variant constructs and forwards a hidden state of the same shape, that `build_ffn("relu_stack", ...)` returns a `ReluStackFFN` honoring `ffn_depth`, that `build_ffn("gelu", ..., ffn_dim=256)` returns a `GeluFFN`, and that an unsupported name raises `ValueError` naming the unsupported variant.
- [x] 1.2 Write the `ffn.py` module docstring documenting the two variants (`relu_stack`, `gelu`) and the shared factory, matching the encoder's previous descriptions; verify the docstring names both variants and the factory.

## 2. Encoder layer consumes a constructed feed-forward module

- [x] 2.1 In `compgen/models/encoder.py`, replace the `ffn_type`/`ffn_depth`/`ffn_dim` constructor parameters with an `ffn: nn.Module` parameter, set `self.ffn = ffn`, replace the variant branch with nothing, and change `forward` to call `self.ffn(x)` once instead of iterating; verify an `EncoderLayer` built with a constructed attention and each FFN variant forwards a hidden state of the same shape and still returns per-layer attention weights of shape `(batch, num_heads, length, length)`.
- [x] 2.2 Update the `encoder.py` module and `EncoderLayer` docstrings to state that the feed-forward block is also a caller-constructed module and that the layer performs no feed-forward variant selection; verify the docstrings no longer describe `ffn_type`/`ffn_depth`/`ffn_dim` or a variant branch.
- [x] 2.3 Verify the encoder layer stays registry-independent: from the repo root, confirm `compgen/models/encoder.py` neither imports `compgen.models.ffn` nor any task module, and that importing `compgen.models.encoder` succeeds without importing any task model or the feed-forward module.

## 3. Task models construct the feed-forward module

- [x] 3.1 Update `compgen/models/tasks/fuzzy_logic.py` so `FuzzyLogicModel` keeps its `ffn_type`/`ffn_dim` parameters, constructs the feed-forward module per layer via `build_ffn(ffn_type, hidden_dim=..., ffn_dim=..., dropout_rate=..., dtype=..., device=...)`, and passes it into `EncoderLayer`; verify the model constructs with `ffn_type="gelu"`, that each layer's `ffn` is a `GeluFFN`, that a forward returns per-token predictions plus per-layer attention of shape `(batch, num_heads, length, length)`, and that the reference-config parameter count is unchanged from before the change.
- [x] 3.2 Update `compgen/models/tasks/match3.py` so `Match3Model` keeps its `ffn_depth` parameter, constructs the feed-forward module via `build_ffn("relu_stack", hidden_dim=..., ffn_depth=..., dropout_rate=..., dtype=..., device=...)` inside its per-layer construction, and passes it into `EncoderLayer`; verify construction with `ffn_depth` values including the default, that `share_layers=True` still shares one layer, that a forward returns probabilities `(batch, length, 1)` and hidden states `(batch, length, hidden_dim)`, and that the reference-config parameter count is unchanged from before the change.
- [x] 3.3 Update the `FuzzyLogicModel` and `Match3Model` module docstrings (if they describe how the feed-forward variant is selected) to state that the model constructs the selected feed-forward variant and hands the module to the shared encoder layer; verify no docstring still says the encoder layer selects the feed-forward variant by name.

## 4. Documentation

- [x] 4.1 Update `AGENTS.md` (and the root `README.md` only if it describes the encoder's feed-forward selection) so the model-layer description states that `EncoderLayer` consumes caller-constructed attention and feed-forward modules, and that the feed-forward variants live in the new `compgen/models/ffn.py` with a shared factory; verify every repository-relative model-layer path mentioned still exists and the add-an-architecture/add-a-task conventions remain accurate.

## 5. Integration verification

- [x] 5.1 From the repo root with a fixed seed, build one `Match3Model` (relu_stack) and one `FuzzyLogicModel` (gelu) and compare against the pre-change revision: total parameter counts and a forward pass's output shapes must match; confirm `EncoderLayer` no longer exposes `ffn_type`/`ffn_depth`/`ffn_dim`, and record that `state_dict()` keys under `ffn.*` may differ (expected and accepted).
- [x] 5.2 Confirm feed-forward forward-equivalence by constructing an `EncoderLayer` with the new FFN classes and the pre-change FFN layout sharing copied weights, and checking identical outputs for both `relu_stack` and `gelu` on the same input.
- [x] 5.3 Smoke-run the string-based config path: build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` (which passes `ffn_type="gelu"`, `ffn_dim`) and confirm it constructs and forwards successfully.
