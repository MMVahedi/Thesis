# Design

## Context

See `proposal.md` for motivation. The relevant current state:

- `BaseAttention` owns the shared attention surface and currently carries
  `dropout_rate`, `bias`, `use_dropout`, and `mask_padding_value`, plus the
  `_flatten_padding_mask`/`construct_mask` helpers and the `_drop_weights`
  helper. All five variants accept the same four knobs and each `forward`
  contains a `batch_mask` branch.
- Mask plumbing is the only consumer of `construct_mask`: `match3_collate_fn`
  builds a `{"attention_mask": ..., "mask": None}` mapping, `Match3Model`
  forwards it, and `EncoderLayer` relays it.
- The pinned consumer is the fuzzy-logic runner (`compgen/runners/`): its
  config sets `attention_dropout_rate=0.0` and `attention_bias=False`, and
  `FuzzyLogicModel.forward` already passes `batch_mask=None`. The on-the-fly
  fuzzy-logic sampler produces fixed-length batches, so no padding exists to
  mask.
- The isolated `compgen/experiments/` archive must not import `compgen` and is
  out of bounds.

## Goals / Non-Goals

**Goals:**

- Make attention a pure `(hidden_state[, score_bias]) -> (output, weights)`
  transform with no mask, no dropout, and bias-free projections.
- Remove the now-unsupported configuration knobs from the whole call chain so
  misuse fails loudly rather than silently no-oping.
- Keep the fuzzy-logic runner and the shared encoder/task composition runnable
  and consistent.

**Non-Goals:**

- No change to the attention math (scores, scaling, normalization) beyond
  deleting the mask/dropout steps.
- No change to feed-forward, residual, embedding, or head dropout; those stay.
- No new masking model (e.g. fixed-length batching or causal masking).
- No update to the two Match3 notebooks.

## Decisions

### Remove mask handling entirely rather than keep an optional path

Delete `construct_mask`, `_flatten_padding_mask`, and `mask_padding_value`, and
drop `batch_mask` from every attention `forward`. The pinned consumer never
masks, and the optional path was never exercised by it.

- Alternatives considered: keep `batch_mask` as an optional argument that is
  ignored (rejected — silent no-op hides misuse and keeps dead code); keep the
  mask machinery only in `BaseAttention` (rejected — still untested weight).
- Consequence: `forward` signatures change from
  `forward(hidden_state, batch_mask=None, score_bias=None)` to
  `forward(hidden_state, score_bias=None)`.

### Remove `dropout_rate`, `use_dropout`, and `bias` from attention

Attention weights are no longer dropped out and the Q/K/V (and reconciliation)
projections are always bias-free. Remove `_drop_weights` and the
`nn.Dropout` module held by attention.

- Alternatives considered: keep `use_dropout` as a toggle (rejected — the pinned
  runner uses zero attention dropout; the toggle adds an API branch with no
  caller); keep `bias` (rejected — the reference and both task models construct
  attention bias-free).
- Note: `EncoderLayer`'s residual and feed-forward dropout (`dropout_rate`) and
  the task models' head/embedding dropout are untouched.

### Keep `score_bias`

The additive `score_bias` path stays: it is orthogonal to masking/dropout, is
used by the softmax mixer's interface, and was explicitly left out of scope.

### Remove mask plumbing from the encoder and task models

`EncoderLayer.forward` loses `batch_mask`; `EncoderLayer.__init__` loses
`attention_dropout_rate`, `use_attention_dropout`, and `attention_bias`.
`FuzzyLogicModel` loses `attention_dropout_rate`/`attention_bias` and stops
passing `batch_mask=None`; `Match3Model` loses `use_attention_dropout` and stops
passing `batch_mask`. The Match3 collate no longer emits a `batch_mask` entry.

- Alternatives considered: keep `EncoderLayer`'s attention-dropout knobs and
  compute nothing (rejected — inconsistent surface).

### Leave the Match3 notebooks untouched

They still pass `use_attention_dropout`/`dropout_rate` to `Match3Model` and will
break. This is deliberate per the agreed scope (library callers only); the
notebooks are updated in a separate change.

## Risks / Trade-offs

- **Match3 correctness on padded sequences degrades** (padded positions are now
  attended) → Accepted: Match3 is not the pinned consumer; the loss of masking
  is the point of the change. A future change can reintroduce masking or adopt
  fixed-length batching if Match3 becomes active again.
- **Parameter-count / initialization changes from dropping projection bias** →
  Attention projections had `bias=False` already in the pinned runner, so no
  numeric change there; bias-bearing constructions elsewhere change shape. No
  tests exist to update; a construction smoke check covers it.
- **The two notebooks break** → Documented; out of scope, updated separately.
- **External configs passing removed runner fields** (`attention_bias`,
  `attention_dropout_rate`) break → The runner is the only in-repo consumer and
  is updated in the same change; the removal is intentional and breaking.

## Migration Plan

1. Simplify `BaseAttention` and all five attention variants (remove knobs,
   mask helpers, `_drop_weights`, and `batch_mask` branches; update docstrings).
2. Update `EncoderLayer` to the new attention surface and drop mask plumbing.
3. Update `FuzzyLogicModel` and `Match3Model` call sites.
4. Drop `batch_mask` from the Match3 collate.
5. Drop `attention_dropout_rate`/`attention_bias` from the runner config and
   model construction.
6. Smoke-check: construct and run one forward pass for each registered
   sequence-composable architecture via the fuzzy-logic model, and construct
   the Match3 model over a collated batch.

Rollback: the change is API removal with no data migration; revert the commit.

## Open Questions

- Whether Match3 should later move to fixed-length batching (or regain a masking
  path) is deferred and does not affect this change's specs or tasks.
