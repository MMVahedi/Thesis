# Tasks

## 1. Remove the attention registry and builder

- [x] 1.1 In `compgen/models/attentions/__init__.py`, delete `ATTENTION_CLASSES` and `build_attention`, and remove the now-unused `typing`/`torch` imports. Rewrite the module docstring to state there is no name→architecture registry or construction helper and that callers import and instantiate the concrete classes. Verify with `venv/bin/python -c "import compgen.models.attentions as a; assert not hasattr(a, 'ATTENTION_CLASSES') and not hasattr(a, 'build_attention'); print(a.SoftmaxAttention, a.LinearAttention)"`.
- [x] 1.2 Confirm the concrete classes are still reachable both from the package and their submodules, and that no `ATTENTION_CLASSES`/`build_attention` reference remains in code: `grep -rn "ATTENTION_CLASSES\|build_attention" --include='*.py' compgen scripts notebooks` returns no matches once tasks 3 and 4 are done (re-run there).

## 2. Switch task models to a caller-supplied attention factory

- [x] 2.1 In `compgen/models/tasks/match3.py`, replace `attention_type: str` with `attention_factory: Callable[[], nn.Module]` (add the `Callable` import, drop the `build_attention` import, drop the now-dead `qk_head_dim`/`v_head_dim` parameters), call `attention_factory()` in `make_layer`, and change `self.name` to a configuration-only label without the architecture. Verify construction: build a `Match3Model` with `attention_factory=lambda: SoftmaxAttention(hidden_dim=128, num_heads=2, dtype=torch.float64)` and assert each layer's attention is a `SoftmaxAttention`.
- [x] 2.2 In `compgen/models/tasks/fuzzy_logic.py`, apply the same change (drop the `attention_type` default and the dead `qk_head_dim`/`v_head_dim` parameters, add `attention_factory`, call it per layer, drop the builder import, adjust `self.name`). Verify with `attention_factory=lambda: LinearAttention(hidden_dim=128, num_heads=8, qk_head_dim=16, v_head_dim=16, dtype=torch.float64)` that the model constructs and each encoder layer's attention is a `LinearAttention`.
- [x] 2.3 Verify one fresh module per layer and that the models reject the old surface: a two-layer model built from a counting factory reports two distinct attention objects, and constructing either model with `attention_type="softmax"` raises `TypeError`.

## 3. Update the fuzzy-logic contrastive runner

- [x] 3.1 In `compgen/runners/fuzzy_logic_contrastive/config.py`, remove the `ATTENTION_CLASSES` import and the `_SEQUENCE_COMPOSABLE` tuple built from it; keep `attention_types: Tuple[str, ...]` and `Variant.attention_type` as labels. Replace the registry-based `__post_init__` check with a check against the runner's own supported-label set. Verify `ExperimentConfig()` still validates and `ExperimentConfig().variants()` returns four variants (softmax/linear × λ=0/positive).
- [x] 3.2 In `compgen/runners/fuzzy_logic_contrastive/run.py`, add the explicit label→concrete-class table (`softmax` → `SoftmaxAttention`, `linear` → `LinearAttention`), build a zero-arg factory from the variant's label plus the resolved model/device configuration, and pass `attention_factory=...` to `FuzzyLogicModel` instead of `attention_type=...`. Verify a tiny direct `run_experiment` (or `venv/bin/python -m compgen.runners.fuzzy_logic_contrastive --epochs 1` with small defaults) completes and produces the four expected result files.
- [x] 3.3 Confirm the sweep path is untouched and still drives the runner: `venv/bin/python scripts/sweep_fuzzy_logic.py --dry-run` lists the grid combinations, and `scripts/grids/contrastive.json` / `scripts/grids/contrastive_seeds.json` still parse as valid `attention_types` overrides.

## 4. Update notebooks and documentation

- [x] 4.1 Update `notebooks/match3_depth_vs_strassen.ipynb` and `notebooks/match3_strassen_vs_standard_own_stack.ipynb` to build and pass an `attention_factory` (e.g. `lambda: StrassenAttention(...)` / `lambda: SoftmaxAttention(...)`) instead of `attention_type=`, keeping their per-architecture loops and labels. Verify each notebook's code cells compile and the changed construction cells reference `attention_factory`.
- [x] 4.2 Update `AGENTS.md` and `README.md` to remove the `ATTENTION_CLASSES`/`build_attention` wording and describe direct construction (callers import and instantiate attention classes; task models take a factory). Verify every model-layer path still named in both docs exists and that `grep -rn "ATTENTION_CLASSES\|build_attention" AGENTS.md README.md` returns nothing.

## 5. Integration verification

- [x] 5.1 Repo-wide check that the old surface is gone: `grep -rn "attention_type\|ATTENTION_CLASSES\|build_attention" --include='*.py' --include='*.ipynb' compgen scripts notebooks` shows only the runner's string labels (config/CLI/grid/`Variant`/metrics keys) and no `build_attention`/`ATTENTION_CLASSES` hits.
- [x] 5.2 Run the fuzzy-logic contrastive experiment end to end with small defaults and confirm four tracked runs complete and report softmax and linear each at a zero and a positive contrastive coefficient, with metrics/logging unchanged from before the change.
