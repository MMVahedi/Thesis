# Proposal

## Why

The previous change made `EncoderLayer` consume a caller-constructed attention module, but its feed-forward network is still built internally from a `ffn_type` string plus `ffn_depth`/`ffn_dim` integers, with a `if ffn_type == ...` branch in the layer. This is the same string-indirection the attention work removed: the encoder block selects and validates a feed-forward variant even though the caller already knows which one it wants. Moving the feed-forward into its own class, constructed by the caller and passed into the block, makes the layer a symmetric composition of two caller-supplied modules and removes the remaining variant selection from it.

## What Changes

- Add a feed-forward module family under the model layer: a shared base plus one class per existing variant — a residual ReLU stack (`relu_stack`) and a two-projection GELU block (`gelu`) — and a `build_ffn` factory that resolves a variant name to a constructed module (mirroring `build_attention`).
- `EncoderLayer` SHALL accept an already-constructed feed-forward module and SHALL use it as its feed-forward sublayer. It SHALL no longer accept `ffn_type`, `ffn_depth`, or `ffn_dim`, SHALL perform no feed-forward variant selection or validation, and SHALL depend on no feed-forward registry.
- The feed-forward submodule attribute path stays `self.ffn`, but moving the block into a class may change the state-dict keys beneath `ffn.*` (accepted by the user; existing checkpoints are not a constraint).
- The task models (`Match3Model`, `FuzzyLogicModel`) SHALL keep their `ffn_type`/`ffn_depth`/`ffn_dim` parameters, construct the feed-forward module from those, and pass the constructed module into each `EncoderLayer`. Their public signatures are unchanged.
- **BREAKING** (internal library API): `EncoderLayer` no longer accepts `ffn_type`, `ffn_depth`, or `ffn_dim`. Its only direct consumers are the two task models, which are updated in the same change.

No behavior of the models' `forward` paths, output shapes, parameter counts (for the supported variants), attention-returning contract, or the config/notebook/runner string surface changes.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: the encoder layer's composition contract now covers the feed-forward sublayer too. The layer consumes a caller-constructed feed-forward module and no longer selects a variant, while task models keep selecting the variant by name and constructing the module.

## Impact

- `compgen/models/ffn.py` (new): feed-forward variant classes and the `build_ffn` factory.
- `compgen/models/encoder.py`: `EncoderLayer` accepts a constructed feed-forward module; drops `ffn_type`/`ffn_depth`/`ffn_dim` and the variant branch.
- `compgen/models/tasks/match3.py`, `compgen/models/tasks/fuzzy_logic.py`: construct the feed-forward module from their retained parameters and pass it to `EncoderLayer`; public APIs unchanged.
- `compgen/runners/fuzzy_logic_contrastive/run.py` and `config.py`: unchanged (they still pass `ffn_type`/`ffn_dim` to `FuzzyLogicModel`).
- Docs (`AGENTS.md`, `README.md`) that describe the encoder layer's feed-forward variants.
- Accepted: feed-forward state-dict keys under `ffn.*` may change.
