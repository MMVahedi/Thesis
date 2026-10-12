# Proposal

## Why

The centralized initializer introduced by the previous change is a single `WeightInitializer` class constructed from a scheme callable plus a string `name` (e.g. `"xavier"`), and each scheme is reachable only through an opaque module-level singleton. A string identifying the initialization type can be misspelled and carries no type-level structure: the schemes are indistinguishable by type, extending the set means supplying another same-shaped callable, and the `name` field is used only by `__repr__`. Representing each scheme as its own class makes the policy set explicit, self-describing, and type-checkable.

## What Changes

- Replace the string-named `WeightInitializer(scheme, name)` with a class hierarchy: an abstract `WeightInitializer` base plus the concrete subclasses `XavierInitializer`, `ReferenceInitializer`, and `DefaultInitializer`, each owning its scheme. No string `name` field and no string-keyed scheme lookup remain.
- **BREAKING**: remove the module-level singleton instances `XAVIER_INIT`, `REFERENCE_INIT`, and `DEFAULT_INIT`; modules and task models reference the policy **class** instead.
- Every parameterized module and task model that accepted an initializer now accepts the policy class (e.g. `initializer: type[WeightInitializer] = XavierInitializer`) and applies it, rather than importing a singleton instance or a string.
- Keep `init_reference_weights` as a thin compatibility wrapper, now over `ReferenceInitializer`.
- Numeric and externally observable behavior is unchanged: each module keeps its current default scheme (attention and the classification head xavier; regression head, feed-forward, embeddings, and encoder norms PyTorch defaults; the fuzzy-logic task model reference).

## Capabilities

### New Capabilities
<!-- None: this reshapes an existing model-layer policy rather than introducing a capability. -->

### Modified Capabilities
- `model-layer`: amend "Centralized weight initialization" so the schemes are a class hierarchy — one class per scheme, no string identifier or string-keyed lookup — and each parameterized module receives the policy class rather than a singleton instance or a string.

## Impact

- Code: `compgen/models/initialization.py`, `compgen/models/attentions/base.py` and the five architecture modules, `compgen/models/heads.py`, `compgen/models/ffn.py`, `compgen/models/encoder.py`, `compgen/models/embeddings/{fuzzy_logic,match3}.py`, and `compgen/models/tasks/{fuzzy_logic,match3}.py`.
- API: the three singleton names are removed and the `WeightInitializer` constructor surface changes from a callable+string to an abstract base with subclasses; internal call sites are updated in the same change.
- Docs: `AGENTS.md` and the root `README.md` model-layer description.
- No dependency or data-flow change; no numeric behavior change (defaults preserve each module's scheme).
