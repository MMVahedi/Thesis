# Tasks

## 1. Registry construction helper

- [x] 1.1 Add `build_attention(name, *, hidden_dim, num_heads=None, qk_head_dim=None, v_head_dim=None, dtype=torch.float64, device="cpu")` to `compgen/models/attentions/__init__.py` that validates `name` against `ATTENTION_CLASSES` (raising the existing "Unsupported attention type" error for unknown names) and returns a constructed attention module; verify from the repo root that `build_attention("softmax", hidden_dim=8, num_heads=2)` returns a `SoftmaxAttention`, that `build_attention("standard", ...)` raises `ValueError` naming the unsupported architecture, and that `ATTENTION_CLASSES` still exposes all five names.
- [x] 1.2 Extend the `attentions/__init__.py` module docstring to note that name→module construction is offered by the shared helper, in addition to the name→class registry; verify the docstring names the retained five architectures and no task-local mapping is introduced.

## 2. Encoder layer consumes a constructed attention module

- [x] 2.1 In `compgen/models/encoder.py`, replace the `attention_type`/`num_heads`/`qk_head_dim`/`v_head_dim` constructor parameters with an `attention: nn.Module` parameter, set `self.attention = attention`, keep the `_init_attention_weights(self.attention)` call, and remove the `ATTENTION_CLASSES` import and the name-validation block; verify an `EncoderLayer` built with a constructed `SoftmaxAttention` forwards a hidden state of the same shape and returns per-layer attention weights of shape `(batch, num_heads, length, length)`.
- [x] 2.2 Update the `encoder.py` module and `EncoderLayer` docstrings to state that the layer consumes a caller-constructed attention module and performs no architecture-name selection, and to note that the passed module is (re-)initialized by the layer; verify the docstrings no longer describe accepting or resolving an architecture name.
- [x] 2.3 Verify the encoder layer is registry-independent: from the repo root, search `compgen/models/encoder.py` and confirm it neither imports the attention registry nor any task module, and that importing `compgen.models.encoder` succeeds without importing any task model.

## 3. Task models construct attention and pass the module

- [x] 3.1 Update `compgen/models/tasks/fuzzy_logic.py` so `FuzzyLogicModel` keeps its `attention_type: str` parameter, constructs the attention module per layer via `build_attention(...)` using the model's head/dimension/dtype/device configuration, and passes the constructed module into `EncoderLayer`; verify the model constructs with each registered sequence-composable name (softmax, linear) with each layer's attention being an instance of the selected class, that one forward returns per-token predictions plus per-layer attention of shape `(batch, num_heads, length, length)`, and that the reference-config parameter count is unchanged from before the change.
- [x] 3.2 Update `compgen/models/tasks/match3.py` so `Match3Model` keeps its `attention_type: str` parameter, builds the attention module via `build_attention(...)` inside its per-layer construction, and passes it into `EncoderLayer`; verify construction with `"softmax"` and `"strassen"`, that `share_layers=True` still shares one layer across the stack, that a forward returns probabilities `(batch, length, 1)` and hidden states `(batch, length, hidden_dim)`, and that a fixed-seed forward matches the pre-change output.
- [x] 3.3 Update the `FuzzyLogicModel` and `Match3Model` module docstrings to describe constructing the selected registered architecture and handing the module to the shared encoder layer; verify no docstring still says the encoder layer selects the architecture by name.

## 4. Documentation

- [x] 4.1 Update `AGENTS.md` (and the root `README.md` only if it describes the encoder layer's attention selection) so the model-layer description states that `EncoderLayer` consumes a constructed attention module while task models resolve registered names via the shared registry/helper; verify every repository-relative model-layer path mentioned still exists and the add-an-architecture/add-a-task conventions remain accurate.

## 5. Integration verification

- [x] 5.1 From the repo root with a fixed seed, build one `Match3Model` and one `FuzzyLogicModel` with a registered attention on the current tree and on the pre-change revision (e.g. via `git stash` or `git worktree`), then compare: total parameter counts, a forward pass's output shapes, and `state_dict()` key lists (confirming no `attention_type` key existed and `layers.<i>.attention.*` keys are unchanged).
- [x] 5.2 Smoke-run the string-based config path to confirm it is unaffected: build a model through `compgen/runners/fuzzy_logic_contrastive/run.py:build_model` (or an equivalent one-liner passing `attention_type="softmax"`) and confirm it constructs and forwards successfully.
