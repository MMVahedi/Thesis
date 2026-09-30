# Proposal

## Why

The fuzzy-logic attention experiment under `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` was developed as a standalone codebase before the `compgen/` package existed. Its model, data generator, and shared-term contrastive loss are useful beyond that one experiment, but today they only exist as duplicated, experiment-local modules. Consolidating them into the package turns one-off experiment code into reusable building blocks and gives the contrastive objective a permanent home instead of living inside a single experiment.

## What Changes

- Extract the PyTorch experiment's fuzzy-logic model into the package as a thin task model plus a task embedding, composed from **new reusable shared blocks** added to the model layer (relative-position bias, a GELU feed-forward path, a regression head, and T5-style initialization).
- Extract the experiment's on-the-fly fuzzy-logic data generator into the dataset layer as a new torch-facing on-the-fly dataset that preserves the train/test/id/ood function split and the batched tensor bundle.
- Introduce a new `compgen/losses/` package holding the shared-term supervised contrastive loss, its positive-partner helper, and a model-agnostic combined task+contrastive training loss.
- **BREAKING** Remove the JAX experiment `compgen/experiments/fuzzy_logic_attention_contrastive/` entirely.
- **BREAKING** Delete the now-superseded files from the PyTorch experiment (`model.py`, `data.py`, `losses.py`) and the harness files that only existed to drive them (`train.py`, `self_test.py`, `make_notebook.py`); keep the self-contained `analyze_attention.py`, `summarize_results.py`, the committed notebook, `config.py`, `README.md`, and packaging files isolated in place.
- Update `AGENTS.md`, `README.md`, and the affected specs so they describe the consolidated package and no longer reference the JAX experiment or the deleted experiment files.

## Capabilities

### New Capabilities
- `dataset-layer`: the two standing dataset conventions (offline JSONL generator + torch reader) plus the new on-the-fly torch fuzzy-logic dataset with train/test/id/ood splits and a batched tensor bundle.
- `losses-layer`: the new `compgen/losses/` package — shared-term supervised contrastive loss, positive-partner counting, and a model-agnostic combined task+contrastive training loss.

### Modified Capabilities
- `model-layer`: adds the fuzzy-logic task model and embedding, the new reusable shared blocks (relative-position bias, GELU feed-forward, regression head, initialization helper), and the documentation contract for them.
- `repo-structure`: the experiment inventory changes (JAX experiment removed, PyTorch experiment reduced to self-contained analysis), the experiment self-containment wording, and the documentation/import-discipline references to the removed experiment.
- `runtime-dependencies`: removes the now-obsolete JAX-experiment references and adjusts the manifest-exclusion wording that named that experiment.

## Impact

- **Code**: new `compgen/losses/` package; new `compgen/datasets/` on-the-fly module; new `compgen/models/embeddings/fuzzy_logic.py` and `compgen/models/tasks/fuzzy_logic.py`; new shared model blocks in `compgen/models/`; deleted experiment directories/files.
- **Breaking**: the JAX experiment no longer exists; the PyTorch experiment loses its runnable train/self-test harness and is retained only as an analysis archive.
- **Docs/specs**: `AGENTS.md`, `README.md`, `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/README.md`, and the `repo-structure` / `runtime-dependencies` specs must stop describing the removed experiment.
- No dependency changes: everything extracted is PyTorch, which the root manifest already pins.
