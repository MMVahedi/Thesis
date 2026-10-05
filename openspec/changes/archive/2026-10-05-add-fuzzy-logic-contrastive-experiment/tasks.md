# Tasks

## 1. Runner package and configuration

- [x] 1.1 Create the runner package `compgen/runners/__init__.py` and `compgen/runners/fuzzy_logic_contrastive/__init__.py`, and verify `venv/bin/python -c "import compgen.runners.fuzzy_logic_contrastive"` succeeds from the repository root
- [x] 1.2 Implement `config.py` with an `ExperimentConfig` dataclass (task, dataset sizes, batch size, epochs, λ, temperature, seed, attention types) and a grid builder, and verify the default config yields exactly four variants differing only in attention type and λ
- [x] 1.3 Document `compgen/runners/` as the home for package-composing runnable entry points in `AGENTS.md` and the root `README.md`, and verify every repository-relative path documented for the runner exists

## 2. Step function and contrastive wiring

- [x] 2.1 Implement `step.py` with a `step_fn(model, batch)` returning `(loss, metrics)`: mean-squared regression loss plus, when λ > 0, the shared-term contrastive term over `FuzzyLogicModel.response_token_codes` via `compgen.losses.combined_loss`, and verify on a sampled batch that the loss is finite and backpropagates
- [x] 2.2 Verify that with λ = 0 the step loss equals the plain mean-squared loss and the reported contrastive term is zero
- [x] 2.3 Verify that with λ > 0 the step loss equals the task loss plus λ times the contrastive loss computed independently with `shared_term_contrastive`

## 3. Training and evaluation orchestration

- [x] 3.1 Implement `run.py` to build the on-the-fly generator, pre-sample a deterministic train batch list, construct each variant's model, and train it with `compgen.training.fit` (AdamW lr 1e-3, weight decay 0.1, epoch-granularity scheduler, seed from config), verifying one variant completes one epoch and writes a metrics JSON
- [x] 3.2 Add post-training evaluation over the train/test/id/ood splits computing MSE, MAE, the per-example reference R², and within-±0.05 accuracy, and verify all four splits are reported in the metrics JSON
- [x] 3.3 Verify determinism: run one variant twice with the same seed and configuration and confirm the recorded metrics are identical

## 4. Command-line entry point

- [x] 4.1 Implement `__main__.py` exposing flags for task, dataset sizes, batch size, epochs, λ, temperature, seed, device, and output directory, and verify `venv/bin/python -m compgen.runners.fuzzy_logic_contrastive --epochs 1` runs all four variants and writes four result records

## 5. Integration checks

- [x] 5.1 With the small defaults, run the full four-variant experiment end to end and verify four result records are produced with no errors
- [x] 5.2 Verify that across the four variants only the attention type and λ differ (compare the saved configuration of each run) and that no file under `compgen/experiments/` imports the enclosing package
