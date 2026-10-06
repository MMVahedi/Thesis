# Design

## Context

See `proposal.md` for motivation. Current state and constraints that shape the
approach:

- `FuzzyLogicModel.forward(..., return_attention=True)` returns `(prediction,
  attentions)`, where `attentions` is a list of per-layer softmax weight
  tensors of shape `(B, H, N, N)`. Its `response_token_codes(attentions)` is
  hard-wired to `attentions[-1][:, :, -1, -1]` — the final layer's last-token
  self-attention across heads, shape `(B, H)`.
- `compgen/analytics/attention.py` is the package's task-agnostic home for
  attention-to-vector reductions. It already provides `attention_code`
  (`(B, H)`, the current form) and `attention_row_features` (`(B, H * N)`), and
  `stack_layers(attentions, reducer, ...)` which concatenates per-layer
  features with reducers `"code"` and `"row"`. Flattening the full `(B, H, N,
  N)` matrix is not yet provided.
- The runner is split as `config.py` (`ExperimentConfig` + the four-variant
  grid), `step.py` (`forward_batch` / `make_step_fn`: model → representation →
  `combined_loss`), `run.py` (orchestration, evaluation, tracking, per-variant
  JSON), and `__main__.py` (flags derived from the config).
- The contrastive loss (`combined_loss` → `shared_term_contrastive`) is
  model-agnostic: it consumes any `(B, D)` representation plus `term_ids`, so
  all three forms slot in without touching the loss.
- The existing capability spec constrains extraction to the final encoder
  layer with a fixed form; the analytics spec already covers the `(B, H)` and
  `(B, H * N)` forms.

## Goals / Non-Goals

**Goals:**

- Make the contrastive representation a first-class configuration choice with
  three forms — full flattened matrix (`matrix`), response-token row (`row`),
  and response-self-attention per head (`code`) — defaulting to `code`.
- Reuse the analytics layer for all extraction so the runner holds no
  representation logic of its own and the two existing forms keep one code path.
- Keep the experiment's four-model comparison shape and the default run's
  behavior unchanged (one representation per run).

**Non-Goals:**

- No mode that trains all three representations in one invocation.
- No change to the loss, model, dataset, or training engine; no per-layer
  stacking of the representation (the final layer only, as today).
- No new dependency; no change to the isolated `compgen/experiments/` archive.

## Decisions

**1. Represent the choice as a validated string field.**
`ExperimentConfig` gains `contrastive_representation: str = "code"`, validated
in `__post_init__` against `("matrix", "row", "code")`. Rationale: the config
already validates string sets (`attention_types` against the attention
registry), and the CLI can present `choices` directly. Alternative: a Python
`Enum` — more ceremony for a three-value option and inconsistent with the
existing string-based config.

**2. Add the full-matrix reduction to the analytics layer.**
Add `attention_matrix_features(attention) -> (B, H * N * N)` and extend
`stack_layers` to accept a `"matrix"` reducer, alongside the existing `"code"`
and `"row"`. Rationale: the analytics layer is the specified home for
attention-to-vector reductions and already owns the other two forms; keeping all
three there preserves the layer boundary and gives one tested path. Alternative:
flatten inline in `step.py` — duplicates a reduction and contradicts the layout
the analytics capability defines.

**3. Select the representation in `step.py`, applied to the final layer.**
`forward_batch` gains a `representation` argument and extracts
`stack_layers(attentions[-1], reducer=representation)` — a single layer, so the
result is purely that layer's features (`(B, H)`, `(B, H * N)`, or
`(B, H * N * N)`). The result is passed unchanged as `representations` to
`combined_loss`. Rationale: `step.py` already owns the model → representation →
loss wiring, and a single-layer `stack_layers` call dispatches to the right
reducer uniformly. `FuzzyLogicModel.response_token_codes` is left in place as a
model-level convenience, but the runner no longer depends on it, so no
model-layer behavior changes. Alternative: generalize `response_token_codes`
with a mode — pushes experiment concerns into the shared model layer and would
change model-layer behavior for no gain.

**4. Flow the option through the existing seams.**
`make_step_fn(representation, ...)` bakes it in as it does `lambda` and
`temperature`; `run_variant` passes `config.contrastive_representation`; the CLI
adds `--contrastive-representation` with `choices=("matrix", "row", "code")` and
default `code`. Rationale: consistent with how every other study knob travels.

**5. Keep different representations from overwriting each other.**
Include the representation in the tracked run name and the per-variant output
filename (`{variant.name}_{representation}.json`). Rationale: results are
otherwise written to `{variant.name}.json` regardless of representation, so
running a second representation silently overwrites the first. The four-model
count and the `Variant` grid are unchanged; only the artifact/run label carries
the extra field.

## Risks / Trade-offs

- [The `matrix` representation is `H * N * N`-dimensional (8·16·16 = 2048) and sequence-length-dependent, and every attention row sums to one, adding a shared constant that cosine similarity can be dominated by] → It is opt-in; `code` (default) is unaffected. The sequence-length dependence is inherent to flattening and is recorded here rather than hidden; the option exists precisely to test it.
- [For linear attention the "weights" are unnormalized scores, so the three forms are not normalized alike across architectures] → Same extraction applies to both architectures; results are reported per architecture as before.
- [`stack_layers` reducer validation must keep rejecting unknown names and leave `"code"`/`"row"` output unchanged] → Add `"matrix"` to the allowed set; existing reducers return the same shapes.
- [Adding a representation suffix to output filenames changes the names of the default run's artifacts] → Unavoidable for safe multi-representation runs; the file contents, the four-variant count, and the metric schema are unchanged.
- [No test suite exists] → Verify with a short smoke run that each representation trains a positive-λ variant end to end, produces the expected feature width, and that the default `code` run reproduces the previous behavior.

## Migration Plan

Additive configuration with an unchanged default: an existing run without
overrides behaves as before, except per-variant JSON files gain a
`_{representation}` suffix. No rollback steps beyond reverting the change.

## Open Questions

- Whether `matrix` or `row` yields a more useful shared-term signal than `code`
  is the empirical question the option enables; it does not affect the specs,
  the approach, or the task breakdown.
