# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `compgen/models/initialization.py` holds the abstract `WeightInitializer` base plus `XavierInitializer`, `ReferenceInitializer`, and `DefaultInitializer` (each implementing `_init`), and the `init_reference_weights` wrapper. `WeightInitializer.__call__` runs `module.apply(self._init)`, i.e. it already initializes a whole module tree.
- Every parameterized sub-level module accepts `initializer: Type[WeightInitializer]` and self-initializes: `BaseAttention` (stores `initializer()` and `_init_weights` calls it; the five architectures call `self._init_weights()`), `TokenClassifier` (`XavierInitializer`), `RegressionHead` (`DefaultInitializer`), `BaseFeedForward`/`ReluStackFFN`/`GeluFFN` (`DefaultInitializer`), both task embeddings (`DefaultInitializer`), and `EncoderLayer` (`DefaultInitializer`, applied to its two `nn.LayerNorm`s).
- `FuzzyLogicModel` takes `initializer: Type[WeightInitializer] = ReferenceInitializer` and applies it across the assembled model (so every Linear/Embedding, attention included, ends reference). `Match3Model` takes `initializer = DefaultInitializer` (a no-op) and relies on the sub-level self-initialization: attention and classifier xavier, feed-forward and embedding PyTorch defaults.
- The `model-layer` spec has "Centralized weight initialization" (class hierarchy, every module self-initializes) and "Attention architectures initialize their own weights" (self-init + injectable policy).

## Goals / Non-Goals

**Goals:**
- One initialization method per task model, applying one uniform `WeightInitializer` policy to the whole assembled model.
- No initialization concern in any sub-level module: no `initializer` parameter, no self-initialization.
- Keep the `WeightInitializer` class hierarchy as the model-level policies.

**Non-Goals:**
- Preserving Match3's exact per-module numerics (its feed-forward blocks and embedding move from PyTorch defaults to xavier).
- Adding new schemes or changing parameter counts, architectures, or forward behavior.
- Introducing a shared task-model base class for the initialization method (two models do not justify one).

## Decisions

1. **Remove initialization from every sub-level module.** Delete the `initializer` parameter, the `self.initializer` attribute, and the self-initialization call from `BaseAttention` (and its `_init_weights`) and the five architectures, `TokenClassifier`, `RegressionHead`, `BaseFeedForward`/`ReluStackFFN`/`GeluFFN` (and `_init_weights`), both embeddings, and `EncoderLayer`. Remove the now-unused `initialization.py` imports from each. *Why*: the request is that sub-level modules have no initialization concern; leaving an inert parameter would keep the concern visible. *Trade-off*: breaking constructor signatures, updated in-package in the same change.

2. **Each task model owns one initialization method with a uniform policy.** Keep `initializer: Type[WeightInitializer]` on `Match3Model` and `FuzzyLogicModel`, add a single private method (e.g. `_init_weights(self) -> None`) that runs `self.initializer(self)`, and call it at the end of `__init__`. *Why*: `WeightInitializer` already traverses the whole model tree, so one method initializes a model of any size; keeping the method on each task model matches "a model ... will have just one initialization method" without adding an abstraction for two classes. *Alternative considered*: a shared `ModelInitialization` mixin — rejected as unneeded indirection for two models.

3. **Default schemes.** `FuzzyLogicModel` keeps `ReferenceInitializer` (numerically identical: it already applied reference across the whole model). `Match3Model` changes its default from the no-op `DefaultInitializer` to `XavierInitializer`, so attention and classifier stay xavier and the feed-forward blocks and embedding become xavier rather than PyTorch defaults. *Why*: with a single uniform scheme, xavier is the closest match to Match3's current scheme (attention/classifier are xavier today); `DefaultInitializer` would instead revert attention/classifier to PyTorch defaults. *Trade-off*: an accepted numeric change for Match3 (see Risks).

4. **Keep the class hierarchy and the wrapper.** `XavierInitializer`/`ReferenceInitializer`/`DefaultInitializer`, the abstract base, and `init_reference_weights` are unchanged in shape; only the docstrings change to describe model-level application. *Why*: the user chose to keep the hierarchy; it is the model-level policy set.

5. **Spec: replace the per-module requirement, remove the attention one.** REMOVE "Centralized weight initialization" and ADD "Model-level weight initialization" (a MODIFIED block must keep the old requirement's scenarios, and its "each module initializes itself" and "pre-change default scheme is preserved" scenarios describe the behavior being undone); REMOVE "Attention architectures initialize their own weights". *Why*: the per-module and self-init contracts no longer hold, so they are removed and replaced by the model-level requirement rather than amended in place.

## Risks / Trade-offs

- [Fixed-seed value shift for both models] Moving initialization to a single model-level method removes the RNG draws the sub-level constructors used to consume, so that method runs from a different RNG position; fixed-seed state-dict values shift for **both** models even where the scheme is unchanged (fuzzy-logic stays reference-everywhere). Match3 additionally changes scheme (feed-forward/embedding to xavier) → accepted and stated in the proposal and the spec's uniform-default scenario; verify schemes, parameter counts, and forward shapes are preserved rather than byte-identical weights.
- [Breaking sub-level signatures] Removing the `initializer` keyword from sub-level constructors changes their public surface → only in-package call sites exist (the attention factory in `run.py` never passed one) and are updated together; recorded as BREAKING in the proposal.
- [Deleting `_init_weights` call sites] The five architectures call `BaseAttention._init_weights`; missing one leaves a stray call → remove the base method and all five call sites together; verify each architecture constructs (uninitialized by itself) and the task model then initializes it.
- [Uninitialized standalone module] A sub-module used outside a task model now keeps PyTorch's constructor defaults → that is the intended behavior; document it.

## Migration Plan

Internal API change: sub-level constructors lose `initializer`, both task models keep it and add a single initialization method. In-package call sites and docs are updated in the same change. `init_reference_weights` remains. Rollback is reverting `initialization.py` docstrings, the sub-level modules, the two task models, and the docs together. No tracked checkpoints are affected beyond the accepted Match3 numeric change.

## Open Questions

None.
