# Proposal

## Why

The task models treat their two sub-modules inconsistently: the feed-forward block is a plain constructed module the model builds and hands to `EncoderLayer`, while attention is passed in as a zero-argument factory that the model must call per layer. Since `EncoderLayer` already consumes a constructed attention object, the factory is a redundant indirection at the task-model boundary. Making attention a constructed object as well — the same shape as the feed-forward module — removes that indirection and gives both sub-modules one uniform construction surface.

## What Changes

- **BREAKING**: `Match3Model` and `FuzzyLogicModel` replace their `attention_factory: Callable[[], nn.Module]` constructor parameter with `attention: nn.Module`, a caller-supplied, already-constructed attention module.
- The single supplied attention module is shared across every encoder layer (all layers reference the same module, so attention parameters are shared). The feed-forward block and normalization remain per-layer as today; `Match3Model.share_layers` still shares the whole layer when enabled.
- The task models no longer call a factory; `EncoderLayer`, the attention classes, and the initialization policy are unchanged.
- Call sites outside the model layer — the contrastive runner, the two Match3 notebooks, and the README example — are deliberately left on the old factory surface in this change and will need a follow-up update. `AGENTS.md`/`README.md` prose describing the factory is likewise out of scope here.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: The task-model composition requirements change from "accept a caller-supplied, zero-argument attention factory and call it once per encoder layer" to "accept a caller-supplied, already-constructed attention module and share that one module across its encoder layers, mirroring how it consumes the feed-forward module".

## Impact

- Code: `compgen/models/tasks/match3.py`, `compgen/models/tasks/fuzzy_logic.py` (constructor surface, layer construction, unused `Callable` import).
- Unchanged: `compgen/models/encoder.py` (already takes a constructed attention object), `compgen/models/attentions/*` (architecture classes), `compgen/models/initialization.py`.
- Breaking for callers: `compgen/runners/fuzzy_logic_contrastive/run.py:build_model`, `notebooks/match3_depth_vs_strassen.ipynb`, `notebooks/match3_strassen_vs_standard_own_stack.ipynb`, and the `README.md` example. These pass `attention_factory=` and will raise `TypeError` until a follow-up change updates them.
- Documentation divergence: `AGENTS.md`/`README.md` prose and the README example still describe the removed factory. They are out of scope here and will be stale until a follow-up; the `model-layer` documentation requirement only mandates describing direct attention construction, which remains true, so no spec requirement is contradicted.
- Numerics: sharing one attention module across layers changes the fuzzy-logic model's per-layer attention parameters (previously independent). This is an accepted consequence of the chosen "share it" design.
