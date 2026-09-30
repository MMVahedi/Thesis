# Tasks

## 1. Shared attention base

- [x] 1.1 Create `compgen/models/attentions/base.py` with `BaseAttention(nn.Module)` that resolves configuration (`hidden_dim`, `num_heads`, optional `qk_head_dim`/`v_head_dim` with the equal-dimension default, `scaler`, `mask_padding_value`, `dropout_rate`, `dtype`, `device`, `use_dropout`) and provides shared helpers: head split/merge, a rank-adapting `construct_mask`, and output-width reconciliation back to `hidden_dim` (adding an output projection only when the aggregated width differs). Verify from the repo root: `venv/bin/python -c "from compgen.models.attentions.base import BaseAttention"` succeeds, and a tiny subclass returns `hidden_dim`-wide output for the default and for independent QK/V head dims.
- [x] 1.2 Document the base's configuration resolution (including "only one dimension given -> both take that value" and "no output projection in the default equal case") in the module docstring; verify by reading the docstring and asserting the default path adds no output-projection parameters.

## 2. Softmax mixer (rename) and registry

- [x] 2.1 Add `compgen/models/attentions/softmax.py` implementing `SoftmaxAttention` on `BaseAttention` (scaled QK scores, additive mask, max-subtraction for stability, softmax over keys, dropout, value aggregation), and delete `compgen/models/attentions/standard.py`; verify the migrated module runs one forward pass and, under a fixed seed and default head/dimension settings, matches the pre-change `standard` numerics (smoke test run before deleting the old module, recording the reference output).
- [x] 2.2 Update `compgen/models/attentions/__init__.py` so `ATTENTION_CLASSES` exposes `softmax`, `linear` (added in group 3), `strassen`, `triangular`, `third_order` and no longer exposes `standard`; update its docstring to the new name set and the retained-but-pair-level note for `triangular`; verify: `venv/bin/python -c "from compgen.models.attentions import ATTENTION_CLASSES; assert 'standard' not in ATTENTION_CLASSES and 'softmax' in ATTENTION_CLASSES"` and constructing each present name succeeds.

## 3. Linear mixer

- [x] 3.1 Add `compgen/models/attentions/linear.py` implementing `LinearAttention` on `BaseAttention` with the same projections, scaling, head split, dropout, and output reconciliation as `softmax` but no final softmax normalization, normalizing (softmax-path) masking replaced by zero-fill so masked positions contribute no weight; verify with a focused check: for the same module configuration and input, an unnormalized softmax-path output (softmax step removed) equals the linear output, and masked positions contribute zero weight.
- [x] 3.2 Register `linear` (already listed in 2.2) and verify both sibling mixers construct with a non-default head count and distinct QK/V head dimensions and return `hidden_dim`-wide output.

## 4. Migrate the high-order variants

- [x] 4.1 Refactor `strassen.py`, `third_order.py`, and `triangular.py` onto `BaseAttention`, mapping score-producing matrices to the QK width and value-producing matrices to the V width (design Decision 4), keeping each variant's contraction equations, normalization style, and returned-weight convention; verify each constructs with the default equal head dimension and runs a forward pass, and that `triangular` still takes/returns pair-level `(B, N, N, C)` (construction-level check only, as it is not sequence-composable).
- [x] 4.2 Verify the default equal-dimension path of each migrated variant is structurally unchanged (same projection shapes, scaler, and contraction labels as the pre-change code) and record any accepted divergence; verify `triangular` and the other variants remain in `ATTENTION_CLASSES` and constructible.

## 5. Encoder and task configuration pass-through

- [x] 5.1 Add optional `qk_head_dim`/`v_head_dim` pass-through to `compgen/models/encoder.py:EncoderLayer` and forward them to the registry constructor; verify an `EncoderLayer` built with and without them constructs and that omitting them reproduces today's behavior.
- [x] 5.2 Add the same optional pass-through to `compgen/models/tasks/match3.py:Match3Model`; verify `Match3Model(attention_type="softmax")` and `"linear"` construct with and without the new kwargs, that `attention_type="standard"` raises an error naming the unsupported architecture, and that `forward(batch)` still returns probabilities `(B, N, 1)` and final hidden states `(B, N, hidden_dim)`.

## 6. Notebooks and documentation

- [x] 6.1 Update `notebooks/match3_depth_vs_strassen.ipynb` and `notebooks/match3_strassen_vs_standard_own_stack.ipynb`, replacing `"standard"`/label/model-name strings with `"softmax"` (and `standard_depth{d}` -> `softmax_depth{d}`); verify by extracting and compiling the notebooks' code cells and confirming no `"standard"` architecture reference remains (`rg -n '"standard"' notebooks/` returns nothing).
- [x] 6.2 Update `README.md` and `AGENTS.md` to the new architecture name set, the shared base, the head/QK/V-dimension configuration, and the softmax/linear pair, while stating the high-order variants are retained; verify every model-layer path or name the docs mention matches the code (e.g. `softmax.py`, `linear.py`, `base.py` exist; `standard.py` is gone).
- [x] 6.3 Update the `attentions/__init__.py` and `embeddings/match3.py` docstrings if they reference the old `standard` name; verify no stale `standard` architecture reference remains in `compgen/models/` (`rg -n 'standard' compgen/models/` only matches unrelated words if any, and no `attention_type` example uses it).

## 7. Integration verification

- [x] 7.1 Run the notebook sanity path from the repo root: build a small Match3 batch with `match3_collate_fn` from `compgen/datasets/torch_datasets/match3.py`, construct `Match3Model` for `softmax`, `linear`, and one retained high-order variant, and confirm finite outputs with documented shapes under the float64 defaults.
- [x] 7.2 Confirm experiment isolation is untouched: run `python self_test.py` inside `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` (must pass) and verify `git status` shows no generated artifacts (datasets, checkpoints, zips) tracked or newly added.
- [x] 7.3 Confirm the full registry round-trips and validate the change: `venv/bin/python -c "from compgen.models.attentions import ATTENTION_CLASSES; print(sorted(ATTENTION_CLASSES))"` prints `['linear', 'softmax', 'strassen', 'third_order', 'triangular']`, and `openspec validate modularize-attention-blocks --strict` passes.
