# Proposal

## Why

The model layer currently mixes two concerns: task-agnostic architectures and task-specific glue. The architecture registry (`ATTENTION_CLASSES`) lives inside the task file `compgen/models/tasks/match3.py:20`, so adding a new attention architecture requires editing task code, and every future task file would grow its own private registry. Likewise, `Match3EncoderLayer` and `TokenClassifier` are behaviorally task-agnostic but live inside the Match3 task file, so a second task would copy ~100 lines of boilerplate that then diverges. The thesis is about to add more architectures (triangular and third-order attention are implemented but not yet usable from task models) and more tasks, so this coupling is the growth bottleneck.

## What Changes

- **Move the architecture registry to a task-agnostic home**: `compgen/models/attentions/__init__.py` exports `ATTENTION_CLASSES` covering all four implemented variants (standard, strassen, triangular, third_order). Task code and notebooks select architectures by string from this one registry; adding an architecture becomes: new module in `attentions/` + one registry entry, visible to every task automatically.
- **Extract task-agnostic scaffolding**: the encoder layer (attention + FFN + optional LayerNorm + residual) moves to `compgen/models/encoder.py`; the per-token classification head moves to `compgen/models/heads.py`. These become the reusable composition pieces for any task model.
- **Slim the task file**: `compgen/models/tasks/match3.py` keeps only genuinely task-specific glue — `Match3Embedding` (imported from `embeddings/`) plus a thin `Match3Model` composition (~40 lines instead of ~185).
- **Relax architecture restriction (deliberate behavior change)**: `Match3Model` currently raises `ValueError` for attention types outside {standard, strassen}; after this change it accepts any architecture in the neutral registry. Which architectures a study compares becomes a notebook/experiment config concern, not a task-model constraint.
- **Stable public API**: `Match3Model`'s constructor signature and `forward(batch)` contract (the batch dict produced by `match3_collate_fn`) are unchanged, so the two notebooks and the datasets layer keep working without edits.
- **Docs updated**: `AGENTS.md` and `README.md` layout sections describe the new modules (registry, encoder, heads) and the convention for adding a task (one file in `embeddings/` + one thin file in `tasks/`).

## Capabilities

### New Capabilities

- `model-layer`: How the model layer inside `compgen/models/` is organized for growth — a single task-agnostic architecture registry, reusable encoder/head scaffolding, and thin task-specific model files.

### Modified Capabilities

- *(none — `repo-structure` requirements are unaffected; all package paths and the import discipline stay valid)*

## Impact

- **Code**: `compgen/models/attentions/__init__.py` (registry), new `compgen/models/encoder.py` and `compgen/models/heads.py`, slimmed `compgen/models/tasks/match3.py`. `compgen/models/embeddings/match3.py` unchanged. Experiment suites untouched (they never import `compgen`, so the self-contained rule holds).
- **API**: `Match3Model` public API unchanged (notebooks unaffected). Removed internal names: `Match3EncoderLayer`, task-file-local `ATTENTION_CLASSES`. New public names: `EncoderLayer`, `TokenClassifier` (relocated), `ATTENTION_CLASSES` (relocated).
- **Behavior**: `Match3Model(attention_type="triangular"|"third_order")` now constructs instead of raising.
- **Docs**: `AGENTS.md`, `README.md`, and the `tasks/match3.py` docstring updated to the new structure.
- **Out of scope / follow-ups**: shared notebook training runner, graduating a fuzzy-logic model into the library, task-centric reorganization, notebook placement — all explicitly deferred.