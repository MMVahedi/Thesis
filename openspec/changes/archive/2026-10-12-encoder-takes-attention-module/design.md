# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `EncoderLayer.__init__` (`compgen/models/encoder.py`) takes `attention_type: str`, `num_heads`, `qk_head_dim`, `v_head_dim`, validates the name against `ATTENTION_CLASSES`, and constructs `ATTENTION_CLASSES[attention_type](hidden_dim=..., num_heads=..., qk_head_dim=..., v_head_dim=..., dtype=..., device=...)`. It then applies `_init_attention_weights`.
- The only direct consumers of `EncoderLayer` are `Match3Model` and `FuzzyLogicModel` (verified by grep). Notebooks, the contrastive runner, and `scripts/sweep_fuzzy_logic.py` construct task models only, always passing `attention_type` strings.
- `Match3Model` builds layers through a `make_layer()` closure and supports `share_layers`; `FuzzyLogicModel` builds `num_layers` independent layers and applies `init_reference_weights` over the whole model after construction.

## Goals / Non-Goals

**Goals:**
- Make the encoder block consume a constructed attention module and stop resolving/validating architecture names.
- Keep the string→architecture selection exactly where it is today: in the task models (and, transitively, configs/notebooks).
- Preserve all observable behavior: forward paths, output shapes, attention-returning contract, parameter counts, and state-dict attribute paths.

**Non-Goals:**
- Removing or changing the `ATTENTION_CLASSES` registry, config files, grid JSONs, or the notebooks' string surface.
- Changing the fuzzy-logic-contrastive experiment or its spec.
- Introducing a plugin/registration mechanism beyond what exists.

## Decisions

1. **`EncoderLayer` takes a constructed `nn.Module`, not a class or a name.** Per the selected direction, the caller builds the attention object and the block uses it as-is. *Alternative considered*: accept the class and instantiate per layer - rejected by the chosen direction; it keeps instantiation ownership in the block rather than the caller.

2. **Task models keep `attention_type: str` and construct the module.** They gain one line each: resolve the name, construct the attention with the model's head/dimension/dtype/device configuration, and pass it to `EncoderLayer`. This confines the change to the library composition boundary and leaves configs/notebooks/runners untouched. *Alternative considered*: change task-model signatures to classes - rejected by the chosen scope.

3. **Add a single `build_attention(name, *, hidden_dim, num_heads, qk_head_dim, v_head_dim, dtype, device)` helper in `compgen/models/attentions/__init__.py`.** It validates the name once (raising the same "Unsupported attention type" error) and returns a constructed module. *Why*: the request's motivation is that name-checking is repeated at every composition site; centralizing resolution/validation in the registry removes the duplication now that two task models construct attention. *Alternative considered*: inline the `ATTENTION_CLASSES[name](...)` call in each task model - simpler, but reintroduces two copies of the name check.

4. **`EncoderLayer` drops `attention_type`, `num_heads`, `qk_head_dim`, and `v_head_dim` from its constructor and drops the `ATTENTION_CLASSES` import.** The block's remaining job is the attention/feed-forward/residual composition. Head and dimension configuration now belongs to the attention constructor at the call site. *Why*: keeping unused pass-through parameters would preserve the confusing surface the change removes. No consumer outside the two task models exists.

5. **Attention weight initialization stays in `EncoderLayer`.** `_init_attention_weights(self.attention)` continues to run on the passed module. *Why*: `Match3Model` relies on this xavier init and does not initialize attention itself; keeping it in the block preserves Match3 behavior with no change to the task model. `FuzzyLogicModel`'s trailing `init_reference_weights(self)` continues to override it, as today. *Trade-off*: the block re-initializes a module it did not construct; documented as a known side effect (see Risks).

6. **Each task model constructs a distinct attention module per encoder layer** (inside its existing per-layer construction), so per-layer weights stay independent exactly as today. `share_layers=True` in `Match3Model` continues to share the whole `EncoderLayer` (and therefore its single attention instance). The submodule attribute name `self.attention` is unchanged, so state-dict keys (`layers.N.attention.*`) remain identical.

## Risks / Trade-offs

- [State-dict compatibility] Removing `self.attention_type` and the pass-through params could change attribute paths → `self.attention` and its internal `query`/`key`/`value` paths are unchanged; `attention_type` was never a buffer/parameter (not in state_dict). Verify by comparing a `state_dict()` key list before/after for one Match3 and one FuzzyLogic config.
- [Re-initialization surprise] `EncoderLayer` re-inits a caller-supplied module → acceptable today because both task models want that initialization; note it in the `EncoderLayer` docstring so a future caller that passes a pre-trained attention knows to expect it.
- [Match3 behavior drift] Moving attention construction to the task model could change initialization order → both paths construct then `_init_attention_weights`, so order and result are unchanged; verify a fixed-seed Match3 forward matches before/after.
- [Parameter count drift] Reordering construction could alter random draws → `build_attention` constructs with the same kwargs in the same order the layer did; verify total parameter counts are identical for the reference Match3 and fuzzy-logic configs.

## Migration Plan

Internal library API change only. The public surfaces of `Match3Model` and `FuzzyLogicModel`, plus all configs, notebooks, runners, and scripts, are unchanged, so no caller migration is required. Rollback is reverting the two task models and `EncoderLayer` together.

## Open Questions

None.
