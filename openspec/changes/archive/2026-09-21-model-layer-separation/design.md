# Design

## Context

See proposal.md — Why. Current state that shapes the approach:

- All `__init__.py` files in the package are empty, so `attentions/__init__.py` is free to become the registry without import-order surprises.
- The only external consumers of the model layer are the two notebooks, each importing exactly `from compgen.models.tasks.match3 import Match3Model`; nothing else imports `Match3EncoderLayer`, `TokenClassifier`, or the task-local registry (verified by grep).
- All four attention variants share essentially the same constructor signature (`hidden_dim, num_heads, dropout_rate, dtype, device, use_dropout`), so one registry keyed by string works uniformly.
- Experiment suites never import `compgen`; this change does not touch them, so their self-containment rule is unaffected.
- The repo has no test suite or build system (see AGENTS.md); verification is by import/compile checks and smoke tests.

## Goals / Non-Goals

**Goals:**
- One neutral home for architecture registration; new architecture = one file + one registry line.
- Shared, task-agnostic encoder layer and classification head; new task model = task embedding + thin composition.
- Zero required edits to notebooks, datasets layer, or experiment suites.

**Non-Goals:**
- No shared notebook training/eval runner (deferred follow-up change).
- No fuzzy-logic model graduating into the library (deferred; experiment suites stay sealed).
- No renaming of `attentions/` to `architectures/` and no task-centric bundling (`compgen/tasks/<task>/{generator,dataset,embedding,model}`) — the user's stated layering is layer-centric and growth is expected to be architecture-sweep-heavy; task-centric reorganization would churn every import path and the `repo-structure` spec examples for a benefit that only appears at 3+ tasks. Revisit if a third task lands.
- No behavior change to the attention variants themselves (dtype defaults, `opt_einsum` contractions, `batch_mask` handling stay byte-for-byte).

## Decisions

1. **Registry lives in `compgen/models/attentions/__init__.py`, registering all four variants.**
   *Why*: the package init is the natural public face of the architectures sub-package; a separate `registry.py` or a root-level `compgen/models/registry.py` adds a file with no benefit. *Alternatives considered*: dynamic discovery (module scanning / entry points) — rejected as overkill for a research repo; registry keyed by class object instead of string — rejected because notebooks pass serializable string configs.
2. **String-keyed dict stays the selection mechanism.** `EncoderLayer` validates the `attention_type` string against the registry, replacing the validation currently done in the task file. *Rationale*: keeps notebook configs serializable and keeps validation at the single composition point.
3. **Restriction relaxation is deliberate.** Today `Match3Model(attention_type="triangular")` raises; after the change it constructs. The old restriction reproduced a reference-paper study scope, which belongs in the study config (notebook), not the task model. `tasks/match3.py`'s docstring ("restricted to comparing exactly two") is updated to say the task supports all registered architectures.
4. **Scaffolding placement: `compgen/models/encoder.py` and `compgen/models/heads.py`, siblings of `attentions/`.** `EncoderLayer` may import from `attentions` (it selects from the registry); neither may import anything from `tasks/` or `embeddings/`. This fixes the dependency direction: task → shared → architectures, never shared → task. *Alternatives considered*: a single `components.py` — rejected (two focused files are more discoverable); putting scaffolding under `tasks/base.py` — rejected (scaffolding is task-agnostic; placing it under tasks/ would blur exactly the line being drawn).
5. **Code moves are behavior-preserving.** `Match3EncoderLayer` → `EncoderLayer` with identical parameters (`use_norm=False`, `ffn_depth=3`, residual wiring, dtype/device defaults) and its attention-init helper; `TokenClassifier` moves unchanged with its xavier init. Model attribute paths (`embedding`, `layers`, `classifier`) are unchanged, so `.pt` state-dict keys (attribute-path based, not class-name based) remain loadable for any checkpoints kept outside git.
6. **Embeddings stay in `models/embeddings/`.** Task-specific embeddings remain one file per task there, not folded into `tasks/`, preserving the documented generator↔embedding cross-reference and the existing directory meaning. The add-a-task convention becomes: `embeddings/<task>.py` + thin `tasks/<task>.py`.

## Risks / Trade-offs

- [Notebook relying on `ValueError` for non-registered architectures] → Grep confirms both notebooks use only `standard`/`strassen`; the relaxation is additive.
- [Circular imports after registry moves into `attentions/__init__.py`] → Attention modules import only `torch`/`opt_einsum` and never the registry or tasks; `encoder.py` imports the registry one-way. Verified by importing from the repo root after the move.
- [Silent numeric drift when moving code] → The encoder layer keeps identical module structure and defaults; smoke test constructs every registered variant and runs one forward pass, checking output shapes and finiteness under the float64 defaults.
- [Docs drift after refactor] → AGENTS.md/README path checks are part of the task list, mirroring the `repo-structure` documentation requirement.

## Migration Plan

Single cohesive refactor, no phased rollout: registry in place → scaffolding modules extracted → task file slimmed → docs updated. Rollback is `git revert` of the one change commit; no data, config, or serialization migration exists (no tracked checkpoints or artifacts are affected).

## Open Questions

None. The layer-centric-vs-task-centric fork was resolved from the user's stated layering and expected growth mix; it is recorded above as a non-goal rather than left open.