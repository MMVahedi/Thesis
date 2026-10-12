# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `compgen/models/initialization.py` defines three module-level scheme functions (`_init_xavier_style`, `_init_reference_style`, `_init_default_style`), one `WeightInitializer(scheme: Callable[[nn.Module], None], name: str)` class whose `__call__` runs `module.apply(self._scheme)`, and three singleton instances `XAVIER_INIT`, `REFERENCE_INIT`, `DEFAULT_INIT`. The `name` string is used only by `__repr__`; no lookup is string-keyed today.
- Every parameterized module and both task models type their parameter as `initializer: WeightInitializer` and default to one of the singletons: attention (`XAVIER_INIT`), `TokenClassifier` (`XAVIER_INIT`), `RegressionHead`, `ReluStackFFN`/`GeluFFN`, both task embeddings, and `EncoderLayer` (`DEFAULT_INIT`), `FuzzyLogicModel` (`REFERENCE_INIT`), `Match3Model` (`DEFAULT_INIT`).
- `init_reference_weights(module)` is a thin wrapper over `REFERENCE_INIT`, with no remaining callers in the package.
- The `model-layer` spec's "Centralized weight initialization" requirement (added by the immediately preceding change) describes the policy behaviorally without naming classes or specifying the representation.

## Goals / Non-Goals

**Goals:**
- Represent each initialization scheme as its own class under one abstract base, with no string name/type and no string-keyed selection.
- Have modules and task models receive the policy **class** and apply it, replacing the singleton-instance imports.
- Preserve each module's default scheme and the exact numerical result.

**Non-Goals:**
- Adding new schemes (the set stays xavier, reference, no-op).
- Changing any module's scheme, parameter set, state-dict layout, or forward behavior.
- Introducing a class/name registry or any string-to-class mapping.

## Decisions

1. **Abstract base + one subclass per scheme.** Make `WeightInitializer` an abstract base (via `abc.ABC`/`abstractmethod`) whose `__call__` does `module.apply(self._init)` and returns the module, and whose `_init(module)` is the abstract per-module hook. Add `XavierInitializer`, `ReferenceInitializer`, and `DefaultInitializer` subclasses, each implementing `_init` with the scheme currently in the corresponding free function (the free functions are removed). `__repr__` uses the concrete class name. *Why*: a subclass is the natural, type-checkable home for a scheme; the base keeps the traversal/`apply` behavior in one place. *Alternatives considered*: keep the callable+`name` (rejected — the string is the thing being removed); an enum of scheme tags (rejected — still a string/value discriminator, not a class).

2. **Modules receive the policy class, not an instance or a string.** Each parameter becomes `initializer: Type[WeightInitializer] = <Subclass>` (e.g. `XavierInitializer`); `__init__` instantiates it once (`self.initializer = initializer()`) and applies it. *Why*: matches "classes for initialization, not a string for type" and removes the shared singleton instances; the type itself names the scheme. *Alternatives considered*: keep passing instances with `= XavierInitializer()` defaults (rejected — the user asked to remove the singleton instances, and class defaults read as the scheme type); pass the base class and resolve by attribute (rejected — more indirection for no gain).

3. **Remove the singleton instances.** Delete `XAVIER_INIT`/`REFERENCE_INIT`/`DEFAULT_INIT`; update every import and default to reference the class. *Why*: the user chose removal; with class defaults the constants are redundant. *Trade-off*: a breaking change for any external importer, but the only consumers are in-package call sites updated here.

4. **Keep `init_reference_weights` as a thin wrapper.** It delegates to `ReferenceInitializer` so the symbol stays importable. *Why*: same low-cost compatibility rationale as the preceding change. *Alternative considered*: delete it (rejected — needless breakage).

5. **Amend the existing `model-layer` requirement rather than add a parallel one.** The "Centralized weight initialization" requirement is the one that governs the policy, so it is MODIFIED to describe the class hierarchy and the no-strings rule, adding a scenario that each scheme is a distinct class and no string identifies a scheme. *Why*: one authoritative requirement per concern; a second requirement would duplicate it.

## Risks / Trade-offs

- [Numeric drift] Reordering or renaming schemes could change which random draws land on which parameter → keep each scheme's body and traversal exactly; verify fixed-seed state-dict checksums are byte-identical to the current revision.
- [Breaking import change] Removing `XAVIER_INIT`/`REFERENCE_INIT`/`DEFAULT_INIT` breaks external importers → only in-package call sites exist and are updated together; note it in the proposal as BREAKING.
- [Class vs instance confusion] A caller could pass `XavierInitializer()` (an instance) where a class is expected → the constructor would fail on `initializer()`; document the class-based contract in each docstring and the module docstring.
- [Abstract base bypass] A future subclass that forgets `_init` → `abstractmethod` prevents instantiation, so it fails loudly at construction.

## Migration Plan

Internal, mostly backward-compatible API change with three removed names. All packages' modules, both task models, and the docs are updated in the same change; `run.py:build_model` needs no change because it never passed an initializer. `init_reference_weights` remains. Rollback is reverting `initialization.py`, the modules that reference it, and the docs together. No tracked checkpoints depend on the initializer representation.

## Open Questions

None.
