# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `compgen/models/initialization.py` defines `_init_reference_style` (T5-style: Linear weights get truncated-normal with `std = sqrt(1/fan_in)/0.8796...` and a `±2·std` clamp, linear biases are zeroed, embedding weights get normal `std = sqrt(1/num_embeddings)`) and the free function `init_reference_weights(module)` that applies it via `module.apply`.
- `BaseAttention._init_weights` (`compgen/models/attentions/base.py`) iterates `named_parameters()` and applies `nn.init.xavier_normal_` to any parameter whose name contains `weight` and zeros any containing `bias`; all five concrete architectures call it at the end of their `__init__`. Attention projections are `nn.Linear` with `bias=False`, plus `out_proj` (also Linear, bias-free) when the value width differs from `hidden_dim`.
- `TokenClassifier.__init__` (`compgen/models/heads.py`) reimplements the same xavier/zero pass inline. `RegressionHead` deliberately keeps PyTorch defaults (its `nn.LayerNorm` must stay untouched).
- `ReluStackFFN`/`GeluFFN` (`compgen/models/ffn.py`) and both task embeddings (`compgen/models/embeddings/*.py`) do not initialize explicitly — they keep PyTorch defaults.
- `EncoderLayer` (`compgen/models/encoder.py`) owns optional `nn.LayerNorm` parameters but never initializes the attention/feed-forward modules it is handed.
- `FuzzyLogicModel.__init__` (`compgen/models/tasks/fuzzy_logic.py`) ends with `init_reference_weights(self)`, re-initializing every Linear/Embedding in the assembled model — including the attention module it obtained from the caller-supplied factory. `Match3Model` performs no model-level initialization.
- The task models are the only callers of `init_reference_weights`; the isolated archive under `compgen/experiments/` imports nothing from `compgen`, and `run.py:build_model` constructs `FuzzyLogicModel` without any initialization argument.

Net effect: attention and the classification head duplicate a scheme; the reference scheme lives in one module but is applied only by one task model; the feed-forward, regression head, and embeddings keep defaults with no single place recording that as a policy.

## Goals / Non-Goals

**Goals:**
- Put every initialization scheme the model layer uses behind one policy object so there is a single place to control initialization.
- Have each parameterized module obtain the policy through constructor injection and initialize itself with it.
- Preserve each module's current numeric scheme and application order, including `FuzzyLogicModel`'s whole-model reference re-initialization and `Match3Model`'s absence of one.

**Non-Goals:**
- Unifying modules onto a single shared scheme (explicitly out of scope — schemes are preserved).
- Changing parameter counts, module shapes, state-dict layouts, or forward behavior.
- Changing the attention-factory surface or threading an initializer through it.
- Touching the isolated archive under `compgen/experiments/`.

## Decisions

1. **One `WeightInitializer` class that wraps a scheme callable.** In `compgen/models/initialization.py`, add `WeightInitializer(scheme, name)` whose `__call__(module)` (and `apply(module)`) runs `scheme(module)` and returns it. Expose three module-level instances: `XAVIER_INIT` (new `_init_xavier_style`), `REFERENCE_INIT` (the existing `_init_reference_style`), and `DEFAULT_INIT` (`_init_default_style`, an explicit no-op). *Why*: a single class plus instances is directly injectable, identity-comparable, and keeps the scheme functions small and independently testable. *Alternatives considered*: a subclass per scheme — more types than needed; a string/enum scheme key resolved everywhere — spreads lookup instead of centralizing a control object.

2. **Constructor injection with a per-module default.** Every parameterized module gains an optional `initializer: WeightInitializer = <its current scheme>` keyword and uses it during `__init__`: attention (`XAVIER_INIT`), `TokenClassifier` (`XAVIER_INIT`), `RegressionHead` (`DEFAULT_INIT`), `BaseFeedForward`/`ReluStackFFN`/`GeluFFN` (`DEFAULT_INIT`), task embeddings (`DEFAULT_INIT`), and `EncoderLayer` for its normalization weights (`DEFAULT_INIT`). *Why*: the default reproduces today's behavior exactly (the no-op default adds no random draws), while a caller can override per module. *Alternative considered*: a process-global initializer object modules consult — rejected (hidden global state, harder to override per module).

3. **Leaf modules self-initialize; task models apply one initializer across the composed model.** `FuzzyLogicModel` gains `initializer: WeightInitializer = REFERENCE_INIT` and replaces `init_reference_weights(self)` with `self.initializer(self)`; `Match3Model` gains `initializer: WeightInitializer = DEFAULT_INIT` and ends with `self.initializer(self)`. *Why*: attention is produced by a caller-supplied, zero-argument factory the task model cannot inject into, and today `FuzzyLogicModel` re-initializes the entire assembled model (attention included). A model-level apply is the only way to preserve that exactly; the no-op default preserves Match3's behavior and adds no draws. *Alternative considered*: pass the initializer down into each child constructor only — rejected because it would leave fuzzy-logic attention on the xavier scheme, changing its numerics.

4. **Keep `init_reference_weights` as a thin wrapper.** It delegates to `REFERENCE_INIT` and stays importable, so no external caller breaks; `FuzzyLogicModel` uses the initializer directly. *Alternative considered*: delete the function — rejected (needless breakage for a public-ish symbol).

5. **Schemes match parameters by module type and keep today's application order.** The xavier scheme applies to `nn.Linear` (and, for generality, `nn.Embedding`) weights and zeros their biases; the reference scheme keeps its existing `module.apply` traversal; `DEFAULT_INIT` changes nothing. *Why*: `isinstance` targeting avoids accidentally xavier-initializing a `LayerNorm`, and reusing each scheme's existing traversal keeps fixed-seed draw order stable. *Alternative considered*: keep name-substring matching — works today but would mis-handle a future normalized module that carries `weight`/`bias` names.

6. **Remove the duplicated inline loops.** `TokenClassifier.__init__` drops its loop and calls `self.initializer(self)`; `BaseAttention._init_weights` delegates to `self.initializer(self)`, keeping the concrete architectures' existing `self._init_weights()` call sites. *Why*: the duplication is the thing being centralized; keeping the call sites avoids touching all five architecture files beyond the constructor signature.

## Risks / Trade-offs

- [Fixed-seed weight drift] Reordering parameter traversal inside a scheme changes which random draws land on which parameter → keep each scheme's existing traversal order; verify schemes, counts, and shapes rather than byte-identical weights, matching prior accepted drift.
- [A module forgets to initialize] A new parameterized module that omits the injected call would be silently uninitialized → the spec's per-module scenarios and a constructor-level verification for each module guard this.
- [No-op default mistaken for a bug] `DEFAULT_INIT` intentionally does nothing → document it explicitly and name it as the deliberate "keep PyTorch defaults" policy.
- [Two initialization layers] Leaf modules self-initialize and a task model may then re-apply a model-level initializer → this mirrors today's behavior (attention self-inits, then fuzzy-logic re-inits everything); document it so the ordering is not surprising.
- [LayerNorm stays default] The schemes do not touch `nn.LayerNorm`, so `EncoderLayer`'s initializer is effectively a no-op today → acceptable and behavior-preserving; it leaves room for a future scheme without changing the interface.

## Migration Plan

Internal, backward-compatible API addition: constructors gain an optional `initializer` keyword whose default reproduces the current scheme, and `init_reference_weights` remains. `run.py:build_model` needs no change — it omits the argument and receives the defaults. Rollback is reverting `initialization.py`, `attentions/base.py`, `heads.py`, `ffn.py`, `encoder.py`, the two embeddings, the two task models, and the docs together. No tracked checkpoints depend on these initialization details.

## Open Questions

None.
