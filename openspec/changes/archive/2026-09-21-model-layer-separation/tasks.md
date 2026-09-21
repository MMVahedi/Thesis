# Tasks

## 1. Architecture registry

- [x] 1.1 Move the architecture registry to `compgen/models/attentions/__init__.py` and register all four variants (standard, strassen, triangular, third_order) under their existing string names; verify from the repo root: `python -c "from compgen.models.attentions import ATTENTION_CLASSES; assert set(ATTENTION_CLASSES) == {'standard','strassen','triangular','third_order'}"` and that constructing each class with `hidden_dim=16, num_heads=2` succeeds.

## 2. Shared scaffolding modules

- [x] 2.1 Create `compgen/models/encoder.py` containing the task-agnostic encoder layer (moved verbatim from `tasks/match3.py:Match3EncoderLayer` with its attention-weight-init helper), validating `attention_type` against the shared registry; verify from the repo root that it imports and that `rg "import|from"` over it shows no imports from `compgen.models.tasks` or `compgen.models.embeddings`.
- [x] 2.2 Create `compgen/models/heads.py` containing the per-token classification head (moved verbatim from `tasks/match3.py:TokenClassifier` with its init); verify from the repo root that it imports and constructs with `hidden_dim=16`, and that its imports reference no task modules.

## 3. Slim task model

- [x] 3.1 Rewrite `compgen/models/tasks/match3.py` as thin glue: import `Match3Embedding`, the shared encoder layer, the shared head, and the registry; keep `Match3Model`'s constructor parameters, attribute paths (`embedding`, `layers`, `classifier`), and `forward(batch)` output contract identical; update its docstring to say all registered architectures are supported; verify `rg -n "ATTENTION_CLASSES\s*=" compgen/models/tasks/match3.py` finds no local registry definition and no local encoder/head class definitions remain.
- [x] 3.2 Behavior smoke test from the repo root (CPU-only venv; local machine has no GPU/torch install): construct `Match3Model` once per registered architecture name (each succeeds) and once with an unknown name (raises `ValueError` naming it); run one forward pass per sequence-contract variant (standard, strassen, third_order) on a small batch built by `match3_collate_fn` from `compgen/datasets/torch_datasets/match3.py` and check outputs are finite with the documented shapes (probabilities `(batch, seq_len, 1)` and hidden states `(batch, seq_len, hidden_dim)`); verify triangular at construction level only (its pair-level `(B, N, N, C)` contract is not sequence-composable — document that in the registry docstring). GPU-dependent checks are run on the GPU server and do not block this task.

## 4. Documentation

- [x] 4.1 Update `AGENTS.md` and the `README.md` layout table to describe the model layer (architectures + shared registry, shared encoder/heads scaffolding, per-task `embeddings/` + thin `tasks/` files and the add-a-task convention); verify every repository-relative path the two docs mention for the model layer exists on disk.
- [x] 4.2 Update the `tasks/match3.py` docstring (remove the "restricted to comparing exactly two attention mechanisms" wording) and re-check `embeddings/match3.py`'s docstring still points at the real generator path; verify by reading both docstrings.

## 5. Integration verification

- [x] 5.1 Prove notebooks are unaffected: extract the import and model-construction code cells from both notebooks in `notebooks/` and compile/execute them from the repo root (as the `repo-structure` import-discipline scenario does); verify zero edits to the notebooks were needed (`git diff --stat notebooks/` empty).
- [x] 5.2 Confirm experiment isolation is untouched: run `python self_test.py` inside `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` (it must pass without repository-root code) and verify `git status` reports no generated artifacts (datasets, checkpoints, zips) as tracked or newly added.