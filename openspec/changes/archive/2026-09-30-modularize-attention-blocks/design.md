# Design

## Context

See proposal.md — Why. Current state that shapes the approach:

- Four independent modules in `compgen/models/attentions/`: `standard.py`, `strassen.py`, `triangular.py`, `third_order.py`. Each hardcodes `head_dim = hidden_dim // num_heads`, repeats `construct_mask`, and repeats the projection/reshape/dropout wiring. None accepts an independent QK vs V width.
- `compgen/models/encoder.py:EncoderLayer` constructs the architecture by string from `ATTENTION_CLASSES` and hardcodes the attention kwargs it forwards (`hidden_dim`, `num_heads`, `dropout_rate`, `dtype`, `device`, `use_dropout`). `compgen/models/tasks/match3.py:Match3Model` builds the stack; only `standard`/`strassen` are used by the notebooks.
- The established QK/V-width pattern already exists in `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/model.py:StandardAttention`: separate `query`/`key`/`value` linears (`qk_dim`/`v_dim`), `qk_head_dim = qk_dim // num_heads`, `v_head_dim = v_dim // num_heads`, and an output projection `Linear(v_dim, emb_dim)` back to the residual width. The experiment is sealed (must not be imported) but is a valid reference for the shape contract.
- Masking differs per variant: `standard` uses an additive `(B, 1, H, N, N)` mask; `strassen` a shared `(B, 1, N, N)` mask applied to three matrices; `triangular` a `(B, N, N, N, 1)` mask; `third_order` an `(B, 1, N, N*N)` mask.
- `triangular` is pair-level: it takes `(B, N, N, C)` and returns pair representations — it is not sequence-composable through `EncoderLayer` (documented in the registry). It is retained but not usable from `Match3Model` forward.
- No tracked checkpoints exist (`*.pt` is gitignored), so state-dict key stability is a convenience, not a hard constraint.

## Goals / Non-Goals

**Goals:**
- One shared configuration surface for all architectures: `num_heads` plus independently settable QK and V head dimensions, defaulting to a single equal head dimension.
- A shared base that owns mask construction, head splitting/merging, scaling, dropout, and output-width reconciliation, so each architecture module contains only its score function and aggregation.
- Preserve all five architectures (including the three high-order/legacy ones) as registered, constructible, migrated-on-base modules.
- Default construction reproduces the previous single-head-dimension behavior for the retained variants.

**Non-Goals:**
- No new high-order architectures, no architecture-search/config-builder layer.
- No changes to the experiment suites (`compgen/experiments/`, sealed) or the dataset layer.
- No deletion or permanent sidelining of the high-order variants.
- No change to dtype defaults (`torch.float64`) or the `opt_einsum.contract`-based contractions.

## Decisions

1. **Shared base in `compgen/models/attentions/base.py`, class `BaseAttention(nn.Module)`.**
   *Why*: a single home for the configuration resolution and mechanics every variant needs; it lives with the architectures so the registry package stays the architecture sub-package's public face. *Alternatives considered*: a mixin-free helper module of free functions — rejected because head/dim resolution is per-instance state that must be shared by subclasses; putting the base under `compgen/models/` — rejected, it belongs with the architectures.

2. **Dimension configuration: `num_heads` explicit; `qk_head_dim` and `v_head_dim` optional.**
   Resolution rules, applied in the base constructor:
   - neither given -> `qk_head_dim = v_head_dim = hidden_dim // num_heads` (previous behavior);
   - only one given -> both take that value;
   - both given -> used independently.
   No secret third default: the equal case is the documented default, not a fallback. *Why*: matches the user's "independent but with a default where QK and V share the head dimension," and keeps the existing constructor semantics valid.

3. **Output width is reconciled back to `hidden_dim`.** The value path produces `num_heads * v_head_dim`; when that differs from `hidden_dim` (or when the architecture's aggregation does not already return `hidden_dim`), the base applies an output projection `Linear(num_heads * v_head_dim, hidden_dim)`. In the default equal-dimension case no output projection is added, so the residual width and the previous parameterization are preserved. *Why*: `EncoderLayer`'s residual adds attention output to `hidden_dim` hidden states; the attention must therefore return `hidden_dim` regardless of V width.

4. **Branches get distinct projections where widths differ.** Following the experiment's pattern, an architecture whose QK and V widths are independent uses separate `query`/`key` and `value` projections (`Linear(hidden_dim, num_heads * qk_head_dim)` and `Linear(hidden_dim, num_heads * v_head_dim)`). The multi-projection variants map their existing matrices onto the same split: for `strassen`/`third_order`, the score-producing matrices (`a,b,c` / `q_i,k_j,k_k`) use the QK width and the value-producing ones (`v1,v2` / `v_j,v_k`) use the V width; for `triangular`, the two key projections use the QK width and the two value projections use the V width. In the default equal case this reduces to the current shapes. *Why*: one consistent mapping rule instead of per-variant ad hoc rules.

5. **`standard` is renamed to `softmax` (module `standard.py` -> `softmax.py`, class `StandardAttention` -> `SoftmaxAttention`), and a new `linear` module is added.** The registry becomes `{"softmax", "linear", "strassen", "triangular", "third_order"}`, with no `standard` key. *Why*: the user chose a hard rename; exposing the softmax/linear pair as first-class sibling mixers is the point of the reorganization.

6. **`linear` is `softmax` with the final normalization removed, and uses a multiplicative mask.** Same projections, scaling, head split, dropout, and output reconciliation as `softmax`; `att_weights = scores` (no softmax, no max-subtraction-for-stability). Because an additive `-1e4` mask would still contribute `-1e4` weights in the unnormalized sum, `linear` masks by zeroing (or `masked_fill(..., 0)`) rather than adding a large negative. *Why*: the user's definition is "looks like softmax attention but without softmax at the last step"; zeroing is the correct masking semantics once normalization is gone. This masking difference is the one deliberate behavioral divergence between the two mixers.

7. **Mask construction is centralized in the base as a rank-adapter.** The base exposes `construct_mask` that expands a `(B, N)` padding mask to a requested rank/layout (e.g. `(B, 1, H, N, N)`, `(B, 1, N, N)`, `(B, N, N, N, 1)`, `(B, 1, N, N*N)`) so each variant calls one helper with its target shape. The additive/multiplicative choice stays at the call site (softmax-family add the pad value; `linear` zeroes). *Why*: the existing variants differ only in target rank and additive-vs-multiplicative, both of which are parameterizable without changing numerics.

8. **`EncoderLayer` and `Match3Model` gain optional pass-through kwargs, not a new signature.** `EncoderLayer` accepts `qk_head_dim`/`v_head_dim` (both optional) and forwards them to the registry constructor; `Match3Model` accepts the same and forwards to each `EncoderLayer`. Omitting them reproduces current behavior; existing notebook constructions remain valid (additive optional kwargs). *Why*: keeps the task API's existing parameters stable while exposing the new knobs.

9. **Notebook migration is mechanical.** `"standard"` -> `"softmax"` in architecture names, label strings, and the derived model-name strings (e.g. `standard_depth{d}` -> `softmax_depth{d}`) in both notebooks. No training logic changes.

10. **Registry docstring and docs restated.** `attentions/__init__.py` docstring lists the new name set and notes `triangular` is retained but pair-level (not sequence-composable). `README.md` and `AGENTS.md` model-layer lines are updated to the new names and the head/dimension config.

## Risks / Trade-offs

- [Migrated variants' parameterization changes, so old state dicts may not load] -> No tracked checkpoints exist; preserve attribute names where cheap, and call out in tasks that no checkpoint compatibility is promised.
- [Numerical drift in `softmax`/legacy variants during migration] -> Keep the default equal-dimension path structurally identical (same projections, scaling, mask rank, `opt_einsum` equations); verify with a smoke test comparing a migrated `softmax` forward against the pre-change `standard` forward under an identical seed before deleting the old code path.
- [`linear` output magnitude grows with sequence length (no normalization)] -> Documented as intended; it is the user's requested definition, not a defect.
- [`triangular` remains non-composable through `EncoderLayer`] -> Unchanged from today; documented in the registry, and tasks verify it at construction level only (as the prior change did).
- [Hard rename silently breaks any external caller using `"standard"`] -> The registry rejects unknown names with an error naming them; notebooks and docs are updated in the same change; the breaking change is marked in the proposal.
- [Extra output projection when V width differs could surprise downstream shape expectations] -> Base always returns `hidden_dim`, so `EncoderLayer`/task contracts are unaffected.

## Migration Plan

Single cohesive refactor, no phased rollout: base + resolution -> migrate `softmax` (rename) -> add `linear` -> migrate `strassen`/`triangular`/`third_order` -> registry/docstrings -> encoder/task pass-through -> notebooks and docs. Rollback is `git revert` of the change commit; the only externally visible break (the `standard` name) is intentional and localized. No data or dataset migration is involved.
