# Proposal

## Why

Initialization is currently spread across every parameterized module: each attention architecture, prediction head, feed-forward block, embedding, and the encoder layer accepts an `initializer` and initializes itself, while the task models additionally apply one over the composition. A model that composes many modules therefore has initialization logic in many places, and every sub-level constructor carries an internal concern it should not know about. Initialization belongs to the assembled model.

## What Changes

- Move weight initialization entirely to the final task model: each task model keeps a single initialization method that applies its policy across the whole composed model, and sub-level modules no longer initialize.
- **BREAKING**: remove the `initializer` parameter and all self-initialization from sub-level modules — `BaseAttention` and the five architectures, `TokenClassifier`, `RegressionHead`, `ReluStackFFN`/`GeluFFN`, both task embeddings, and `EncoderLayer`.
- Each task model applies one uniform `WeightInitializer` scheme to its whole model: fuzzy-logic keeps `ReferenceInitializer`; Match3 switches to `XavierInitializer`. Match3 still gets xavier on its attention and classifier (as today), but its feed-forward blocks and embedding now also become xavier instead of PyTorch defaults — a deliberate numeric change.
- Keep the `WeightInitializer` class hierarchy in `compgen/models/initialization.py` as the model-level policies, and keep `init_reference_weights` as a thin compatibility wrapper.
- Remove the now-obsolete requirement that attention architectures initialize their own weights.

## Capabilities

### New Capabilities
<!-- None: this relocates an existing model-layer concern to the model level. -->

### Modified Capabilities
- `model-layer`: remove "Centralized weight initialization" (per-module self-initialization) and "Attention architectures initialize their own weights", and add "Model-level weight initialization" requiring initialization only at the task-model level, through one method with a uniform scheme, with no sub-level initialization.

## Impact

- Code: `compgen/models/initialization.py` (docstrings), `compgen/models/attentions/base.py` and the five architecture modules, `compgen/models/heads.py`, `compgen/models/ffn.py`, `compgen/models/encoder.py`, `compgen/models/embeddings/{fuzzy_logic,match3}.py`, and `compgen/models/tasks/{fuzzy_logic,match3}.py`.
- API: sub-level constructors lose the `initializer` keyword (breaking change); the task models keep an `initializer` keyword.
- Numerics: schemes are preserved for fuzzy-logic (reference everywhere) and Match3 moves its feed-forward blocks and embedding from PyTorch defaults to xavier; as with the earlier initializer changes, exact fixed-seed values shift for both models because initialization now happens once at the model level instead of during each sub-module's construction.
- Docs: `AGENTS.md` and the root `README.md` model-layer description.
