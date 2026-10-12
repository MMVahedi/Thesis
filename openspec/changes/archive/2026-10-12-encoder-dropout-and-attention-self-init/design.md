# Design

## Context

See proposal.md - Why. Current state that shapes the approach:

- `EncoderLayer.__init__` (`compgen/models/encoder.py`) takes `attention: BaseAttention`, `ffn: BaseFeedForward`, `dropout_rate`, and builds `self.dropout1`/`self.dropout2` (`nn.Dropout`). `forward` applies them on the two residual connections. It also calls the module-level `_init_attention_weights(self.attention)` (xavier on weights, zero on biases).
- All five concrete architectures build their projections *after* `BaseAttention.__init__()`: `SoftmaxAttention`/`LinearAttention` create `query`/`key`/`value`; `StrassenAttention` (`w_a`/`w_b`/`w_c`/`w_v1`/`w_v2`), `ThirdOrderAttention` (`w_qi`/`w_kj`/`w_kk`/`w_vj`/`w_vk`), and `TriangularAttention` (`left_k`/`right_k`/`left_v`/`right_v`/`query`) create theirs. `BaseAttention` also creates `out_proj` when the value width differs from `hidden_dim`.
- `Match3Model` and `FuzzyLogicModel` pass `dropout_rate` to `EncoderLayer` and to `build_ffn` / their classifier. `FuzzyLogicModel` additionally calls `init_reference_weights(self)` at the end, which overrides all Linear initializations.
- `build_ffn` (`compgen/models/ffn.py`) is a name→constructed-module factory: `Match3Model` calls `build_ffn("relu_stack", ...)` and `FuzzyLogicModel` calls it with its `ffn_type` string. The contrastive runner passes `ffn_type="gelu"` to `FuzzyLogicModel`.
- `model-layer` spec line: "The Match3 task model SHALL NOT expose an attention-dropout constructor parameter; its residual and feed-forward dropout SHALL remain configurable."

## Goals / Non-Goals

**Goals:**
- Make `EncoderLayer` a pure composition: add residual, no dropout, no initialization of its inputs.
- Give attention architectures a self-contained initialization owned by `BaseAttention`.
- Preserve encoder `forward` output shapes, the attention-returning contract, and parameter counts; keep feed-forward and classifier dropout configurable.
- Remove the `build_ffn` name→object factory so task models instantiate the feed-forward variant directly.

**Non-Goals:**
- Preserving Match3's residual dropout (explicitly removed).
- Preserving exact fixed-seed Match3 weight values (initialization order shifts; see Risks).
- Changing the `build_attention` factory's surface or the config/notebook string surface.

## Decisions

1. **`BaseAttention` owns the initialization; concrete architectures invoke it.** Add `BaseAttention._init_weights()` — xavier-normal on `weight` parameters, zero on `bias` parameters (the scheme currently in `encoder._init_attention_weights`). Each concrete architecture calls `self._init_weights()` at the end of its `__init__`, after building its projections (and after `BaseAttention.__init__`, which may have created `out_proj`). *Why*: the method is implemented once in the base and inherited; the call site must be after the subclass builds its parameters, which only the subclass knows. *Alternative considered*: auto-invoke after subclass construction via `__init_subclass__` wrapping — rejected as too implicit for a small, fixed set of architectures.

2. **`EncoderLayer` no longer initializes the passed attention** and drops the module-level `_init_attention_weights` helper. A constructed attention is ready to use. *Why*: the caller owns construction and now initialization; a consumer re-initializing an input is the side effect being removed.

3. **`EncoderLayer` drops `dropout_rate` and both `nn.Dropout` modules.** `forward` becomes `hidden_state = hidden_state + attention_output` and `hidden_state = hidden_state + self.ffn(x)`. *Why*: the feed-forward's dropout already lives in the FFN class, and the block should not hold a third dropout. *Alternative considered*: keep `dropout1` only — rejected; the user asked to remove encoder dropout.

4. **Task models stop passing `dropout_rate` to `EncoderLayer`** but keep passing it when constructing the feed-forward module and to their classifiers. *Why*: feed-forward and classifier dropout remain configurable; only the encoder's residual dropout disappears.

5. **Amend the `model-layer` requirements** so Match3's configurable dropout is "feed-forward dropout" (residual clause removed), the "attention-dropout parameter is gone" scenario refers to the feed-forward dropout rate, and the feed-forward requirement no longer promises a name-keyed factory. *Why*: the spec must match the new behavior; the change authorizes the amendment.

6. **Remove the `build_ffn` factory; task models instantiate the feed-forward classes directly.** `Match3Model` builds `ReluStackFFN(hidden_dim, ffn_depth=..., dropout_rate=..., ...)` and `FuzzyLogicModel` builds `GeluFFN(hidden_dim, ffn_dim=..., dropout_rate=..., ...)`; `FuzzyLogicModel` drops its `ffn_type` parameter and the contrastive runner (`run.py:build_model`) stops passing it. *Why*: the factory is the same name→object indirection being removed elsewhere — a caller should construct the object it needs. *Alternative considered*: keep `build_ffn` and only stop the encoder relying on variant selection — rejected; the user asked to remove the factory.

## Risks / Trade-offs

- [Match3 loses residual dropout] A real training-behavior change for Match3 → the proposal and spec amendment make it explicit; `dropout_rate` still drives the FFN blocks and classifier.
- [Fixed-seed weight drift] Attention initialization moves from the encoder (after it built both sub-modules) to inside attention construction → parameter *counts*, shapes, and initialization *scheme* are unchanged, but exact random draws under a fixed seed differ for Match3. `FuzzyLogicModel` re-initializes via `init_reference_weights`. Mitigation: verify counts/shapes and the scheme (xavier/zero), not byte-identical weights.
- [FuzzyLogicModel API change] Dropping `ffn_type` removes a small public parameter → the contrastive runner is the only external caller and is updated in this change; verify no other caller passes `ffn_type`.
- [Parameter-count parity] Removing dropout and relocating init should not change parameter counts → verify total parameters for the reference Match3 and fuzzy-logic configs are unchanged.
- [Subclass forgets the init call] A future architecture that omits `self._init_weights()` at the end of its `__init__` would be uninitialized → the spec requirement and a constructor-level verification for every registered architecture guard this.

## Migration Plan

Internal library API change plus accepted behavior/API changes (`FuzzyLogicModel` loses `ffn_type`; Match3 loses residual dropout). The contrastive runner is updated in the same change; other configs, notebooks, and scripts are unchanged. Rollback is reverting `encoder.py`, `attentions/base.py`, the five architecture files, `ffn.py`, the two task models, and the runner together. No tracked checkpoints depend on these initialization details.

## Open Questions

None.
