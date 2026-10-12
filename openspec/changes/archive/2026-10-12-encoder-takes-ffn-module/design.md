# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `EncoderLayer.__init__` (`compgen/models/encoder.py`) takes `ffn_type: str`, `ffn_depth: int`, `ffn_dim: Optional[int]`, validates `ffn_type` against `("relu_stack", "gelu")`, and builds `self.ffn`: a `ModuleList` of `Sequential(Dropout, Linear, ReLU)` copies for `"relu_stack"`, or a two-projection `Sequential(Linear, GELU, Dropout, Linear)` for `"gelu"`. `forward` does `for layer in self.ffn: x = layer(x)`.
- The attention side already follows the constructed-module pattern: `EncoderLayer` takes `attention: nn.Module`, and task models build it with `compgen.models.attentions.build_attention`.
- `Match3Model` passes `ffn_depth`; `FuzzyLogicModel` passes `ffn_type="gelu"` and `ffn_dim`; the contrastive runner sets `ffn_type="gelu"`, `ffn_dim` on the model. No caller constructs `EncoderLayer` directly except the two task models.
- There is no `ffn.*` checkpoint tracked in the repo (`*.pt`/`*.ckpt` are gitignored).

## Goals / Non-Goals

**Goals:**
- Make the encoder block a symmetric composition of two caller-constructed modules (attention and feed-forward).
- Move the two feed-forward variants into dedicated classes with a shared `build_ffn` factory, mirroring `build_attention`.
- Keep task-model signatures (`ffn_type`/`ffn_depth`/`ffn_dim`) and the runner/config/notebook surface unchanged.
- Preserve forward behavior for both variants and total parameter counts.

**Non-Goals:**
- Preserving the exact `state_dict` keys under `ffn.*` (accepted; no tracked checkpoints).
- Preserving the exact random weight values of a fixed-seed Match3 build (construction/initialization order may shift; see Risks).
- Changing the attention side, the attention registry, or the string surface of configs/notebooks.

## Decisions

1. **New module `compgen/models/ffn.py` with a base plus one class per variant.** `BaseFeedForward` holds the shared construction context; `ReluStackFFN` and `GeluFFN` each implement `forward`. *Alternative considered*: a single configurable class with a variant argument - rejected in favor of one class per variant, which keeps each variant's structure explicit and matches the chosen direction.

2. **`build_ffn(name, *, hidden_dim, ffn_dim=None, ffn_depth=3, dropout_rate=0.0, dtype, device)` in `compgen/models/ffn.py`.** It validates the name (raising the existing "Unsupported ffn_type" error), resolves `ffn_dim` to `hidden_dim` when `None`, and returns the constructed module. *Why*: mirrors `build_attention` and keeps the one name check in a single place. *Alternative considered*: a name→class registry `FFN_CLASSES` - unnecessary for two variants; the factory is enough, and the spec requires selectable-by-name construction, not an exposed registry.

3. **`EncoderLayer` takes `ffn: nn.Module` and drops `ffn_type`/`ffn_depth`/`ffn_dim`.** `self.ffn = ffn`; `forward` calls `self.ffn(x)` once instead of iterating, since each variant class owns its internal sequencing. *Why*: removes the last variant-selection branch from the block. The submodule attribute stays `self.ffn`, so the top-level path is unchanged; the keys *beneath* it change because the internal layout moves into a class (accepted).

4. **The encoder does not initialize the feed-forward module, and does not re-initialize it.** Today the FFN relies on default `nn.Linear` initialization (unlike attention's `_init_attention_weights`), and `FuzzyLogicModel`'s trailing `init_reference_weights` overrides it. The FFN class is used exactly as constructed, preserving that behavior.

5. **Task models keep their FFN parameters and construct the module.** `FuzzyLogicModel` builds `GeluFFN` via `build_ffn(ffn_type, hidden_dim, ffn_dim, ...)`; `Match3Model` builds `ReluStackFFN` via `build_ffn("relu_stack", hidden_dim, ffn_depth=..., dropout_rate=..., ...)`. Each passes the module into `EncoderLayer`. Public signatures are unchanged.

6. **Dropout split is unchanged.** `dropout_rate` continues to drive both the FFN's internal dropout (now inside the constructed module) and the encoder's residual `dropout1`/`dropout2`; the task models pass it to both `build_ffn` and `EncoderLayer`.

## Risks / Trade-offs

- [State-dict keys under `ffn.*` change] Moving the block into a class relocates internal module paths (e.g. `ffn.0.1.weight` becomes a class-relative path) → accepted by the user; no tracked checkpoints exist. Mitigation: keep the `self.ffn` attribute name; document the key change and provide no automatic remap.
- [Fixed-seed weight drift for Match3] Python evaluates all constructor arguments (attention, then feed-forward) before `EncoderLayer.__init__` runs `_init_attention_weights`, so the xavier attention init now draws after FFN construction instead of before → parameter *counts*, shapes, and architecture are unchanged, but exact random weights under a fixed seed can differ. Mitigation: treat parity as count/shape/behavior, not byte-identical values; note it in verification. `FuzzyLogicModel` is unaffected in kind because its trailing `init_reference_weights` re-initializes all Linears.
- [Forward equivalence] Switching from `for layer in self.ffn` to `self.ffn(x)` could change results if the class sequencing differs → `ReluStackFFN` iterates its `ModuleList` of `Sequential(Dropout, Linear, ReLU)` exactly as before and `GeluFFN` is the same `Sequential`; verify equal outputs against the pre-change path with copied weights.
- [Parameter-count drift] Reordering construction could change counts → variants build the same `nn.Linear`s with the same shapes; verify total parameter counts are identical for the reference Match3 (relu_stack) and fuzzy-logic (gelu) configs.

## Migration Plan

Internal library API change. `Match3Model`/`FuzzyLogicModel` public surfaces and all configs/notebooks/runners/scripts are unchanged. Rollback is reverting `encoder.py`, the two task models, and removing `ffn.py`. Any external checkpoint that stored `ffn.*` keys would need remapping (none are tracked).

## Open Questions

None.
