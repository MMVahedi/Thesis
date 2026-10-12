# Design

## Context

See `proposal.md` — Why. Current state and constraints that shape the approach:

- After a recent series of changes the model layer deliberately had **no** name→architecture registry and **no** construction helper: `compgen/models/attentions/__init__.py` re-exports the concrete classes and forbids name-based selection, `compgen/models/ffn.py` exposes two directly instantiated classes, and `EncoderLayer` consumes already-constructed attention and feed-forward modules. This change intentionally reverses that stance for the model level, with typed enums instead of strings.
- `Match3Model` and `FuzzyLogicModel` accept a caller-supplied, already-constructed attention module (one instance shared across every encoder layer) but still hard-code their feed-forward class: `Match3Model` builds `ReluStackFFN`, `FuzzyLogicModel` builds `GeluFFN`. Their constructors expose `ffn_depth`/`ffn_dim` and `num_heads`, but `num_heads` is used only for `self.name` — the attention module carries its own head count and dimensions.
- Initialization is a class hierarchy (`WeightInitializer` + concrete subclasses) with no string identifier; each task model applies one policy over the whole composed model.
- `compgen/models/embeddings/*` and `compgen/models/heads.py` provide the task-specific input representation and heads the task models compose.
- The runner (`compgen/runners/fuzzy_logic_contrastive/run.py`) and both Match3 notebooks still call the removed `attention_factory=` surface and are already broken; updating them is explicitly out of scope for this change.

## Goals / Non-Goals

**Goals:**
- One model-layer entry point, `build_model(config) -> nn.Module`, that turns a typed configuration into a fully constructed task model (attention, feed-forward, embedding, head, initialization) with no caller-constructed sub-modules.
- Architecture and feed-forward selection by typed enum members only, never raw strings.
- Per-task configuration types exposing the variation knobs (attention + head dims, feed-forward variant + width/depth, layers/width, dropout/norm/share, initializer).
- Task models accept a caller-supplied, constructed feed-forward module, mirroring the attention module they already accept.

**Non-Goals:**
- Adopting the factory in the runner or notebooks (stale call sites tracked separately).
- Exposing runtime `dtype`/`device` as configuration knobs (not selected for this change); the factory relies on the models' existing defaults.
- Changing attention math, `BaseAttention`, `EncoderLayer`, the heads, the dataset/loss/training/tracking layers, or the frozen experiment archive.
- A compatibility shim for the old registry/`attention_factory`/string surfaces.

## Decisions

### 1. A free-function factory with explicit per-task dispatch
`compgen/models/factory.py` exposes `build_model(config) -> nn.Module`. It dispatches on the configuration type with an explicit `isinstance` branch (one branch per task), builds the attention module and feed-forward module from the config's enum members, and calls the task model with the assembled sub-modules.

- Why a free function: the project already favors explicit, direct construction; a two-branch `isinstance` dispatch is the least magical version of "config in, model out".
- Alternatives considered: a `.build()` method on each config (spreads assembly across config classes but avoids any central dispatch point); a `{config_type: builder}` mapping (registry-like, rejected). A central dispatch means adding a task also edits the factory — an accepted trade-off (see Risks).

### 2. Selection by typed enums, mapped in the factory
`compgen/models/configs.py` defines `AttentionArchitecture` and `FeedForwardVariant` enums. The factory holds enum→concrete-class mappings (keys are enum members, not strings) and constructs the modules. Only sequence-composable attention architectures are enumerated (`softmax`, `linear`, `strassen`, `third_order`); `triangular` is omitted because it is pair-level and not sequence-composable through `EncoderLayer`.

- Why enums and not classes/factories in the config: the user requires a typed selector rather than raw strings, and enums serialize cleanly to future CLI/JSON sweep surfaces without being strings.
- Alternatives considered: raw strings (explicitly rejected — reintroduces the typo-driven selection the project removed); class references in the config (no selection surface at all, but the user asked for enum-based config).

### 3. Per-task configuration dataclasses
`Match3Config` and `FuzzyLogicConfig` are dataclasses. A shared base holds the common knobs; each subclass adds the task-specific inputs (e.g. `input_dim` for fuzzy-logic; `embedding_norm_scalar` for Match3) and flags (e.g. `share_layers` for Match3). The config type selects the task — there is no task-type field.

- Common knobs: `attention` (`AttentionArchitecture`), `num_heads`, `qk_head_dim`, `v_head_dim`, `hidden_dim`, `num_layers`, `ffn` (`FeedForwardVariant`), `ffn_dim`, `ffn_depth`, `dropout_rate`, `use_norm`, `initializer`.
- The initializer is the concrete `WeightInitializer` subclass (a class, per the existing no-string-scheme rule), not an enum.

### 4. Task models accept a constructed feed-forward module
`Match3Model` and `FuzzyLogicModel` replace their `ffn_depth`/`ffn_dim` construction parameters with an `ffn: nn.Module` parameter and use that module as the feed-forward of every encoder layer, exactly mirroring the existing caller-supplied `attention: nn.Module`. The factory constructs the module from `FeedForwardVariant` and passes it.

- Consequence: the feed-forward module (and its parameters) is now shared across layers instead of distinct per layer, matching the attention sharing semantics adopted earlier. This is a deliberate, accepted numeric change; a future change could add a per-layer factory if distinctness is needed.
- Alternatives considered: keeping the task-native FFN and exposing only width/depth (drops the requested variant knob); an ffn factory (opposite direction from the attention refactor that removed the attention factory).

### 5. Initialization stays class-based and model-owned
The config carries the initializer class; the factory forwards it to the task model, which continues to apply one policy over the whole model. No string/enum scheme identity is introduced, so the existing initialization requirement is untouched.

## Risks / Trade-offs

- **Reversing the no-registry convention invites string selection again** → Selection is enum-only, the delta spec forbids raw-string selectors, and the factory docstring states the rule.
- **Sharing one feed-forward module across layers changes numerics** versus today's per-layer feed-forward → Documented as an accepted consequence; a follow-up can add per-layer construction if a study needs it.
- **Adding a task now touches the central factory**, contradicting the documented "add a task = embedding + thin task file" convention → The factory uses plain `isinstance` branches (no registry); `AGENTS.md`/`README.md` are updated to state the third step, and the change is scoped to two tasks.
- **Factory/config can drift from the task models' constructor defaults** → The factory passes every value explicitly, and a task verifies factory-built output matches a directly constructed equivalent.
- **No dtype/device in config** → The factory uses the models' existing defaults (`float64`, `cpu`); exposing runtime device/precision is deliberately deferred.

## Migration Plan

1. Add `AttentionArchitecture` and `FeedForwardVariant` enums plus `Match3Config`/`FuzzyLogicConfig` in `compgen/models/configs.py`.
2. Switch `Match3Model` and `FuzzyLogicModel` to a caller-supplied `ffn: nn.Module` (drop `ffn_depth`/`ffn_dim`).
3. Add `build_model` in `compgen/models/factory.py` with the enum→class mappings and per-task branches; re-export from `compgen/models/__init__.py` if that package exposes names.
4. Update `AGENTS.md` and the root `README.md` to describe the factory, enum selection, and the revised add-a-task convention.
5. Rollback: the change is a normal commit; reverting restores the previous task-model surfaces and removes the factory/configs.

## Open Questions

- Exposing `dtype`/`device` as configuration knobs is deferred; if GPU runs need the factory soon, add them as optional config fields without changing the specs.
- Whether the later runner adoption (separate change) needs a string→enum adaptation layer for its CLI/JSON grids is deferred to that change.