# Proposal

## Why

The model currently injects sequence position only through a learned relative-position bias added to attention scores (`compgen/models/position.py`, owned by `EncoderLayer`). For the current refactor we want a simpler, well-understood absolute positional signal: a fixed (sinusoidal) positional encoding added to the input embedding. This removes an entire bucketed, per-head attention-bias subsystem whose behavior is harder to reason about and leaves a single, conventional source of position information. The relative-bias idea is still worth keeping on record for possible future testing.

## What Changes

- Add a fixed sinusoidal absolute positional encoding and add it to the fuzzy-logic input embedding (the learned `Linear` embedding matrix), so positions enter once at the input rather than as a learned per-head attention bias.
- **BREAKING** Remove the relative-position-bias functionality entirely: delete `compgen/models/position.py`’s `RelativePositionBias` and T5-style bucketing, and drop the `use_relative_bias` / `relative_bias_num_buckets` / `relative_bias_max_distance` options and wiring from `EncoderLayer`, `FuzzyLogicModel`, the experiment config, and the runner.
- Keep the generic additive `score_bias` parameter on attention and on `EncoderLayer` — it is an architecture-neutral interface and is not relative-position-specific.
- Update `AGENTS.md` and affected module docstrings so they describe absolute positional encoding and no longer reference the removed relative-bias block.
- Record the removed relative-bias idea in a markdown note under `results/` (what it was, why it was replaced, how it could be tested again). Because `results/` is gitignored, add a targeted exception so the note is tracked.

## Capabilities

### New Capabilities
<!-- None: this change modifies an existing capability. -->

### Modified Capabilities
- `model-layer`: The reusable model-blocks requirement changes from a relative-position bias block to a fixed absolute positional encoding; the fuzzy-logic task model requirement changes from relative-position bias to absolute positional encoding in its input embedding.

## Impact

- `compgen/models/position.py` — replaced: relative-position bias removed; fixed positional encoding introduced.
- `compgen/models/encoder.py` — relative-bias options, import, ownership, and `score_bias` fallback removed.
- `compgen/models/embeddings/fuzzy_logic.py` — fixed positional encoding added to the embedding output.
- `compgen/models/tasks/fuzzy_logic.py` — relative-bias constructor parameters and pass-through removed.
- `compgen/runners/fuzzy_logic_contrastive/config.py` and `run.py` — relative-bias configuration removed.
- `compgen/models/attentions/{softmax,linear}.py` — docstring examples referencing relative-position bias updated (behavior unchanged).
- `AGENTS.md` — description of the model layer’s position handling updated.
- `results/` — new tracked markdown note; `.gitignore` exception to keep it under version control.
