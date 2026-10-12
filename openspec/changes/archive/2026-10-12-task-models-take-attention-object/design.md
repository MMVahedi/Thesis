# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `Match3Model.__init__` (`compgen/models/tasks/match3.py`) takes `attention_factory: Callable[[], nn.Module]` and calls it inside its `make_layer()` closure; `FuzzyLogicModel.__init__` (`compgen/models/tasks/fuzzy_logic.py`) takes the same parameter and calls it once per layer in a list comprehension. Both also construct their feed-forward block (`ReluStackFFN` / `GeluFFN`) inline and hand the constructed object to `EncoderLayer`.
- `EncoderLayer.__init__` (`compgen/models/encoder.py`) already accepts a constructed `attention: BaseAttention` and a constructed `ffn`; it neither constructs nor initializes them. `self.attention` is the only attribute that matters to state-dict keys (`layers.N.attention.*`).
- Both task models end `__init__` with `self.initializer(self)`, one model-level `apply` over the assembled model; there is no per-layer initialization.
- Callers passing a factory: `compgen/runners/fuzzy_logic_contrastive/run.py:build_model`, two Match3 notebooks, and the README example. These are out of scope for this change (see proposal.md - Impact).

## Goals / Non-Goals

**Goals:**
- Give the task models one uniform sub-module surface: both attention and feed-forward are caller-provided/built constructed objects, with no per-layer factory call.
- Keep `EncoderLayer`, the attention classes, and the initialization policy untouched.
- Preserve the attention-returning contract, forward behavior, dtype/device handling, and `layers.N.attention` state-dict attribute paths.

**Non-Goals:**
- Updating the runner, notebooks, or docs (deferred follow-up).
- Preserving per-layer independent attention parameters, or the fuzzy-logic reference parameter count (see Risks).
- Introducing any attention registry, class-based selection, or plugin mechanism.

## Decisions

1. **Replace the factory with a single constructed module (`attention: nn.Module`), keeping the second positional slot.** In both task models, swap the parameter name/type and use `attention` directly wherever `attention_factory()` was called. *Why*: this is the requested surface and it matches the feed-forward object pattern exactly. *Alternative considered*: accept the attention class plus its configuration and construct per layer inside the model - rejected by the chosen direction, which supplies the object from the caller.

2. **Share the one supplied module across every encoder layer.** `FuzzyLogicModel` passes the same `attention` to each `EncoderLayer`; `Match3Model.make_layer()` uses the same `attention` for every layer, so `share_layers=True` continues to share the whole layer while `share_layers=False` now shares only the attention module (FFN and norm stay per-layer). *Why*: with a single object and no factory there is no other source of per-layer modules, and the user chose sharing over cloning. *Alternative considered*: `copy.deepcopy` the module per layer - rejected (not chosen; it would also silently diverge from "receive one object").

3. **Keep `EncoderLayer` unchanged.** It already consumes constructed objects, so no shared scaffolding changes. *Why*: smallest blast radius; the encoder's requirements already match the new surface.

4. **Remove the now-unused `Callable` import** from both task-model files; keep `Type` for the initializer parameter. *Why*: no factory type remains.

## Risks / Trade-offs

- [Breaking caller API] `attention_factory=` becomes `attention=`; the runner, notebooks, and README example raise `TypeError`. → Out of scope per the chosen "library only" scope; documented in proposal.md - Impact as a required follow-up.
- [Fuzzy-logic numerics and parameter count] Two layers previously held two independent attention modules; now they share one, reducing the model's parameter count and breaking the documented 151,009-parameter reference reproduction. → Accepted consequence of the chosen "share it" design; must be called out in any place that cites the reference count (a follow-up docs task, not this change).
- [Initialization re-draws the shared module] `self.initializer(self)` runs `module.apply`, which visits the shared attention once per parent layer, so its parameters are re-drawn N times (last draw wins). The same already happens today for `Match3Model(share_layers=True)`, whose shared layer is visited N times. → No code change needed; note it so the behavior is not mistaken for a bug.
- [Parameter duplication in `state_dict()`] The shared module's parameters now appear under multiple `layers.N.attention.*` keys pointing at the same storage; `model.parameters()` still deduplicates, so the optimizer sees each parameter once. → Expected for tied weights; verify a forward/backward pass and an optimizer `step` succeed after the change.
