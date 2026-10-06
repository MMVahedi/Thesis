# Tasks

## 1. Analytics: full attention-matrix reduction

- [x] 1.1 Add `attention_matrix_features(attention)` to `compgen/analytics/attention.py`, returning `(B, H * N * N)` by flattening heads, query positions, and key positions and reusing the existing shape check; verify on a random `(2, 3, 4, 4)` tensor that the result is `(2, 48)` and equals `attention.reshape(2, 3 * 4 * 4)`
- [x] 1.2 Extend `stack_layers` to accept `reducer="matrix"` (allowed set `code`/`row`/`matrix`) and concatenate each layer's full-matrix features; verify `stack_layers([t, t], reducer="matrix")` has shape `(2, 2 * 3 * 4 * 4)`, that `code`/`row` outputs are unchanged, and that an unknown reducer still raises `ValueError`
- [x] 1.3 Export `attention_matrix_features` from `compgen/analytics/__init__.py` and update the module docstring plus the analytics example in `README.md` to mention the full-matrix reduction; verify `from compgen.analytics import attention_matrix_features` succeeds and every documented analytics path exists

## 2. Experiment configuration

- [x] 2.1 Add `contrastive_representation: str = "code"` to `ExperimentConfig` in `compgen/runners/fuzzy_logic_contrastive/config.py`, validated in `__post_init__` against `("matrix", "row", "code")`; verify the default config has `contrastive_representation == "code"` and that an unknown value raises `ValueError`
- [x] 2.2 Document the option next to the other contrastive settings in the config module, and confirm the default config still yields exactly four variants differing only in attention type and λ

## 3. Step-function representation selection

- [x] 3.1 Add a `representation: str = "code"` argument to `forward_batch` and extract `stack_layers(attentions[-1], reducer=representation)` when `lambda_contrastive > 0`, passing the result to `combined_loss`; verify on a sampled batch that each of `matrix`/`row`/`code` yields a finite loss that backpropagates and that the feature widths are `H * N * N`, `H * N`, and `H` respectively
- [x] 3.2 Thread `representation` through `make_step_fn`; verify that with `code` the extracted tensor equals `attentions[-1][:, :, -1, -1]` (matching the previous behavior) and that with `lambda_contrastive == 0` the contrastive term is still skipped

## 4. Orchestration, CLI, tracking, and output

- [x] 4.1 Pass `config.contrastive_representation` into the step function in `run.py` and include it in the tracked run parameters and the run name; verify a tracked run records the representation
- [x] 4.2 Include the representation in the per-variant metrics filename (`{variant.name}_{representation}.json`) so different representations do not overwrite each other; verify two runs with different representations write distinct files
- [x] 4.3 Add `--contrastive-representation` with `choices=("matrix", "row", "code")` and the config default to `__main__.py`; verify `--contrastive-representation matrix` is accepted and an invalid value is rejected by argparse

## 5. Integration checks

- [x] 5.1 Run the experiment with a positive λ for each representation and verify all variants train and evaluate to completion and each recorded representation matches the requested one
- [x] 5.2 Run the default configuration and verify it reproduces the previous `code` behavior (same extracted representation and metric schema), with no change to the four-variant count
