# Proposal

## Why

`EncoderLayer` currently takes an attention *architecture name* as a string, validates it against `ATTENTION_CLASSES`, and looks the class up again to construct it (`compgen/models/encoder.py:61-74`). This string indirection makes the encoder block depend on the task-agnostic registry and forces every composition site that already knows the architecture to re-encode it as a string that the block then re-checks. Passing the attention object directly removes that re-checking and makes the block a pure module that uses the attention it is given.

## What Changes

- `EncoderLayer` SHALL accept an already-constructed attention module (an `nn.Module`) instead of an architecture-name string. It SHALL use that module as its attention and SHALL no longer look up or validate architecture names.
- `EncoderLayer` SHALL no longer import `ATTENTION_CLASSES`, drop the `attention_type`/`self.attention_type` attribute, and drop the `num_heads`, `qk_head_dim`, and `v_head_dim` construction parameters (the caller now supplies them to the attention constructor).
- The task models (`Match3Model`, `FuzzyLogicModel`) SHALL keep taking an `attention_type` string, resolve it to a registered architecture class, construct the attention module, and pass the constructed module to each `EncoderLayer`. Their public signatures and the string-selection boundary are unchanged.
- Add one registry helper (`build_attention`) that turns an architecture name plus construction config into a constructed attention module, so the name validation happens once instead of at every composition site. Unknown names still fail with an error naming the unsupported architecture.
- Attention weight initialization (`_init_attention_weights`) stays owned by `EncoderLayer`, so Match3 initialization behavior is preserved.

No behavior of the models' `forward` paths, attention-returning contract, output shapes, or the string-selection/config/notebook surface changes.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: the composition contract between the shared encoder layer and attention changes. The encoder layer now consumes a constructed attention module rather than an architecture name, while task models remain responsible for resolving registered names and constructing attention.

## Impact

- `compgen/models/encoder.py`: `EncoderLayer` constructor/service surface changes (accepts a module, drops name validation and head-dimension kwargs).
- `compgen/models/tasks/match3.py`, `compgen/models/tasks/fuzzy_logic.py`: construct the attention module from the retained `attention_type` string and hand it to `EncoderLayer`; public APIs unchanged.
- `compgen/models/attentions/__init__.py`: adds a `build_attention` helper (registry and names retained).
- Docs (`AGENTS.md`, `README.md`) that describe the encoder layer's attention selection.
- No change to the runner/config/notebook string surface, grid JSONs, or the fuzzy-logic-contrastive experiment spec.
