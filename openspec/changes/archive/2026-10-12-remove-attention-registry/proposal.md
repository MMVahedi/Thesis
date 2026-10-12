# Proposal

## Why

The attention package currently owns a name→class registry (`ATTENTION_CLASSES`) plus a name→constructed-module factory (`build_attention`) that hides which attention a model actually uses behind a string. This adds an indirection and a central registry to maintain, and it lets a typo in a config string silently drive architecture selection. The intent is to make attention construction explicit: whoever needs an attention module constructs the object directly.

## What Changes

- **BREAKING**: Remove `build_attention` from `compgen/models/attentions/__init__.py`.
- **BREAKING**: Remove the `ATTENTION_CLASSES` name→class registry. No module exposes a name-to-architecture mapping; attention classes are imported and instantiated directly.
- **BREAKING**: `Match3Model` and `FuzzyLogicModel` no longer take `attention_type: str`. They take a zero-argument `attention_factory: Callable[[], nn.Module]`, called once per encoder layer, and each layer's attention is whatever object the factory returns.
- Callers (the fuzzy-logic contrastive runner and its config, the fuzzy-logic sweep script, notebooks) construct attention objects themselves — e.g. `lambda: SoftmaxAttention(...)` — instead of passing architecture names.
- Update `AGENTS.md` and `README.md` so the documented model-layer layout matches reality (no registry, no builder; caller constructs attention).
- The shared attention base, the individual attention classes, and the encoder layer's "consumes a constructed attention module" contract are unchanged.

## Capabilities

### New Capabilities
<!-- None. -->

### Modified Capabilities
- `model-layer`: The single task-agnostic architecture registry and the builder helper are removed; task models now receive a caller-supplied attention factory rather than resolving a registered architecture name themselves. Requirements that name the registry, name-based selection, or an architecture-construction helper are rewritten around direct construction and factories.

## Impact

- Code: `compgen/models/attentions/__init__.py`, `compgen/models/tasks/match3.py`, `compgen/models/tasks/fuzzy_logic.py`, `compgen/runners/fuzzy_logic_contrastive/{config,run,__main__}.py`, `scripts/sweep_fuzzy_logic.py`, `notebooks/match3_strassen_vs_standard_own_stack.ipynb`.
- Docs: `AGENTS.md`, `README.md`.
- Breaking for any external caller that constructs a task model with `attention_type=` or imports `build_attention`/`ATTENTION_CLASSES`.
- No change to the attention classes' math, the encoder layer, the loss/training/tracking layers, or the runner's externally observed four-variant comparison (still softmax and linear, each with zero and positive contrastive coefficient).
