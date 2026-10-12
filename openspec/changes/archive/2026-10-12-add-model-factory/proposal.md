# Proposal

## Why

Producing a model variation today means every caller hard-codes construction: import the concrete attention class, build an attention module, pick the feed-forward class, and forward a dozen constructor arguments into `Match3Model`/`FuzzyLogicModel`. There is no single entry point that turns a declared configuration into a ready task model, so comparing variations (different attention, feed-forward, depth/width, initialization) duplicates construction logic at each call site and makes sweeps awkward. A model-layer factory that accepts a typed configuration and returns a constructed model gives one place to declare a variation.

## What Changes

- Add a model-layer factory (`compgen/models/factory.py`) exposing `build_model(config) -> nn.Module`, plus the configuration types it consumes (`compgen/models/configs.py`): typed per-task config dataclasses and selection enums.
- Add `AttentionArchitecture` and `FeedForwardVariant` **enums** used by the config. Raw strings SHALL NOT select an architecture; a config carries an enum member (e.g. `AttentionArchitecture.SOFTMAX`, `FeedForwardVariant.GELU`).
- Add per-task config dataclasses — `Match3Config` and `FuzzyLogicConfig` — covering the variation knobs: attention architecture and head dimensions (`num_heads`, `qk_head_dim`, `v_head_dim`), feed-forward variant with width/depth (`ffn_dim`/`ffn_depth`), stack size/shape (`num_layers`, `hidden_dim`), regularization and layer sharing (`dropout_rate`, `use_norm`, `share_layers`), and the initialization policy. The config type selects the task, so no task-type field is exposed.
- `build_model` dispatches on the config type, constructs the attention module and feed-forward module from the enum members, builds the task-specific embedding/head, and returns the composed task model. Sub-modules keep their current construction surface; only the task model gains a config entry point.
- **BREAKING (convention)**: The model layer now intentionally exposes an architecture-selection surface and a construction helper. The `model-layer` requirements that forbid any name-to-architecture mapping, any construction helper, and any model-level factory are rewritten: selection is by typed enum (never raw string), and the factory is the single sanctioned constructor. The existing "callers construct attention directly, no registry" requirements are replaced rather than layered on top.
- Update `AGENTS.md` and the root `README.md` so the documented model-layer layout includes the factory and enum-based selection.
- Out of scope: the stale call sites (`compgen/runners/fuzzy_logic_contrastive/run.py` and both Match3 notebooks) still pass the removed `attention_factory=` surface; this change does not touch them and does not require the factory be adopted by them yet.

## Capabilities

### New Capabilities
<!-- None: the factory is part of the model layer's organization and is specified there. -->

### Modified Capabilities
- `model-layer`: Replaces the requirements forbidding a name-to-architecture mapping, a construction helper, and any model factory with requirements for a typed-enum, config-driven model factory; adds config/factory construction requirements while keeping attention/feed-forward/initializer classes registry-free and forbidding raw-string selection.

## Impact

- Code (new): `compgen/models/configs.py` (selection enums + per-task config dataclasses), `compgen/models/factory.py` (`build_model`).
- Code (possibly touched): `compgen/models/__init__.py` and/or a package `__init__` re-export for the factory and configs; `compgen/models/tasks/*.py` remain the composition surface and are not required to change beyond what the factory needs.
- Docs: `AGENTS.md`, `README.md` (model-layer layout and add-a-task convention).
- Unchanged: attention classes, `BaseAttention`, `EncoderLayer`, `ffn.py` classes, `heads.py`, `initialization.py` classes, the task models' existing constructor surfaces, datasets/losses/training/tracking layers, and the frozen experiment archive.
- Not addressed here: the runner and notebooks that still use `attention_factory=` (tracked as a separate follow-up); no existing model numerics change.