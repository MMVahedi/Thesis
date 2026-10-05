# Proposal

## Why

The package already contains every ingredient for the fuzzy-logic contrastive
experiment — the reference model with selectable attention (`model-layer`), the
on-the-fly dataset with term ids (`dataset-layer`), the shared-term contrastive
loss (`losses-layer`), and the attention-code reduction (`analytics-layer`) —
but there is no runnable entry point that ties them together. The original
training script was deleted during the experiment reorganization, so `combined_loss`
currently has zero callers and the reference result cannot be reproduced or
extended.

This change adds a runnable experiment that composes those pieces and compares
four models — softmax and linear attention, each with and without the proposed
contrastive term on the final-layer response-token attention code — so the
effect of the contrastive objective on both attention architectures can be
measured.

## What Changes

- Add a runnable experiment entry point inside the package that composes the
  existing model, on-the-fly dataset, contrastive loss, training engine, and
  tracking, without adding new model, data, or loss behavior.
- The experiment runs four models: `attention_type` in `{softmax, linear}` ×
  contrastive coefficient in `{0, λ}` (a single positive λ).
- The contrastive term acts on the final-layer, last (response/output) token
  attention code across heads — the diagonal
  `attention[-1][:, :, -1, -1]` — combined with the regression task loss.
- Default to a small configuration (small train/eval dataset, one batch size)
  so the comparison can be run as a quick study; everything configurable.
- Report regression metrics (MSE, MAE, reference R², within-±0.05) on the
  train/test/id/ood splits plus contrastive diagnostics (contrastive loss,
  average positive partners), logged through `compgen.tracking`.
- Adjust the organizing contract so a runnable entry point that composes the
  package has a defined home under the package, distinct from the isolated
  `compgen/experiments/` archives.

## Capabilities

### New Capabilities

- `fuzzy-logic-contrastive-experiment`: a runnable fuzzy-logic experiment that
  composes the package's model, dataset, loss, training, and tracking layers to
  train and evaluate the four model variants (softmax/linear × without/with the
  response-token attention-code contrastive term) on the reference task splits,
  with deterministic, configurable, small defaults.

### Modified Capabilities

- `repo-structure`: the requirement describing where the experiment suite lives
  is extended to recognize runnable entry points that compose the package (as
  opposed to the isolated, non-importing archives under `compgen/experiments/`),
  and to state where those entry points live.

## Impact

- New code under `compgen/` (a runner subpackage); no change to existing model,
  dataset, loss, analytics, or training behavior.
- Documentation: `AGENTS.md` and the root `README.md` gain the new runner
  location, per the documentation-matches-layout requirement.
- No new runtime dependencies; runs in the existing `venv/` (PyTorch, CPU or the
  local RTX 2060).
- `compgen/experiments/` and its isolated archive are untouched.
