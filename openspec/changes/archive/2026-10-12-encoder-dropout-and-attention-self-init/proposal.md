# Proposal

## Why

`EncoderLayer` still owns two concerns that do not belong to a pure composition block:

1. It applies its own residual dropout (`dropout1` after attention, `dropout2` after the feed-forward), driven by a `dropout_rate` parameter. Now that the feed-forward and attention are caller-constructed modules with their own dropout, the encoder holding a third dropout is redundant and surprising.
2. It (re-)initializes the attention module it is handed via `_init_attention_weights`. The module was constructed by the caller and should already know how to initialize itself; a consumer re-initializing someone else's module is a side effect the block should not have.

Making the encoder a plain composition (add residual, no dropout, no initialization) and pushing weight initialization into the attention base class removes the last responsibilities the block took on behalf of its inputs.

The feed-forward construction helper `build_ffn` is the same kind of indirection: it takes a variant name plus construction arguments and returns an object. A caller that needs a feed-forward block should build the object it needs directly, so this change also removes `build_ffn`.

## What Changes

- **Remove dropout from `EncoderLayer`:** drop the `dropout_rate` parameter and the `dropout1`/`dropout2` modules; both residual connections add the sub-module output directly.
- **Move attention weight initialization into the attention family:** `BaseAttention` gains the shared initialization (xavier-normal on projection weights, zero on biases — today's scheme), and each registered architecture initializes its own weights on construction. `EncoderLayer` no longer initializes a passed attention module; `build_attention` does not initialize either.
- **Task models stop passing `dropout_rate` to `EncoderLayer`.** They still pass it when constructing the feed-forward module (feed-forward-internal dropout) and to their classifiers, so feed-forward and classifier dropout stay configurable.
- **Remove the `build_ffn` factory.** Delete `build_ffn` (and its `_FFN_CLASSES` mapping) from `compgen/models/ffn.py`; `Match3Model` instantiates `ReluStackFFN` directly and `FuzzyLogicModel` instantiates `GeluFFN` directly. `FuzzyLogicModel` drops its `ffn_type` parameter.
- **BREAKING (behavior):** Match3 no longer has residual dropout on the encoder's residual paths; only the feed-forward block and classifier apply dropout. The `model-layer` requirement that says "residual and feed-forward dropout SHALL remain configurable" is amended to drop the residual clause.
- **BREAKING (API):** `FuzzyLogicModel` no longer accepts `ffn_type` (the contrastive runner stops passing it).

The encoder's public `forward` output contract (hidden state, optional per-layer attention weights), parameter counts for attention, and the config/notebook string surface are unchanged (`FuzzyLogicModel` loses the `ffn_type` argument).

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: (a) the encoder layer applies no dropout and no longer initializes its inputs; (b) attention architectures initialize their own weights on construction; (c) the task-model dropout requirement is amended to drop the residual-dropout clause; (d) the feed-forward factory is removed — the model layer no longer offers a name-keyed `build_ffn`, and task models instantiate the feed-forward variant directly.

## Impact

- `compgen/models/encoder.py`: `EncoderLayer` loses `dropout_rate`/`dropout1`/`dropout2` and `_init_attention_weights` (plus its call); `forward` becomes dropout-free.
- `compgen/models/attentions/base.py`: gains the shared `_init_weights`; `compgen/models/attentions/{softmax,linear,strassen,third_order,triangular}.py` call it after building their projections.
- `compgen/models/tasks/match3.py`, `compgen/models/tasks/fuzzy_logic.py`: stop passing `dropout_rate` to `EncoderLayer`; instantiate `ReluStackFFN`/`GeluFFN` directly; `FuzzyLogicModel` drops its `ffn_type` parameter.
- `compgen/models/ffn.py`: `build_ffn` and `_FFN_CLASSES` are removed (the `ReluStackFFN`/`GeluFFN` classes stay).
- `compgen/runners/fuzzy_logic_contrastive/run.py`: stops passing `ffn_type` to `FuzzyLogicModel`.
- Docs (`AGENTS.md`, `README.md`) that describe the encoder's dropout, where attention is initialized, or the feed-forward factory/variant selection.
- Accepted behavior change: Match3 loses residual dropout (its `dropout_rate` now only affects the feed-forward block and classifier).
