# Tasks

## 1. Fuzzy-logic task model takes a constructed attention module

- [x] 1.1 In `compgen/models/tasks/fuzzy_logic.py`, replace the `attention_factory: Callable[[], nn.Module]` parameter with `attention: nn.Module`, pass that object into each `EncoderLayer`, remove the now-unused `Callable` import, and update the class/module docstring to describe a supplied module rather than a factory. Verify by constructing `FuzzyLogicModel` with a `SoftmaxAttention` object (succeeds), and by confirming that passing `attention_factory=` raises `TypeError`.
- [x] 1.2 Verify the module is shared and the forward contract holds: construct a 2-layer model with a `SoftmaxAttention` object and assert `model.layers[0].attention is model.layers[1].attention is attention`; run a forward pass and confirm per-layer attention tensors of shape `(B, num_heads, N, N)` and unchanged prediction shape.

## 2. Match3 task model takes a constructed attention module

- [x] 2.1 In `compgen/models/tasks/match3.py`, replace the `attention_factory: Callable[[], nn.Module]` parameter with `attention: nn.Module`, use it inside `make_layer()`, remove the now-unused `Callable` import, and update the docstring. Verify by constructing `Match3Model` with a `SoftmaxAttention` object (succeeds), and by confirming that passing `attention_factory=` raises `TypeError`.
- [x] 2.2 Verify sharing interacts correctly with `share_layers`: with `share_layers=False` assert every `model.layers[i].attention is attention` while the per-layer `ffn` modules differ; with `share_layers=True` assert `model.layers[i] is model.layers[0]`. Run a forward pass and confirm the `(y_hat, hidden_state)` shapes are unchanged, including the `return_attention=True` path.

## 3. Integration checks

- [x] 3.1 Verify model-level initialization and optimization still work over a shared module: construct each task model with its default initializer, confirm all Linear/Embedding parameters are finite and follow the model's scheme, and run one optimizer step over `model.parameters()` (shared attention parameters must be deduplicated) without error.
- [x] 3.2 Record the known consequences for a follow-up: confirm this change modifies only the two task models, that the out-of-scope callers (`compgen/runners/fuzzy_logic_contrastive/run.py`, `notebooks/match3_depth_vs_strassen.ipynb`, `notebooks/match3_strassen_vs_standard_own_stack.ipynb`, and the `README.md` example) still pass `attention_factory=` and now break, and that the fuzzy-logic shared-attention change lowers its parameter count from the documented 151,009 reference.
