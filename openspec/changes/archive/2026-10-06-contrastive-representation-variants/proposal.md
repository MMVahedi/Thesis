# Proposal

## Why

The experiment's contrastive term is hard-wired to a single representation: the
final encoder layer's last (response/output) token attention to itself, one
feature per head (`FuzzyLogicModel.response_token_codes`). The reference
"attention as a hypernetwork" idea admits several plausible ways to reduce a
layer's attention to a per-example vector, and it is not yet known which one
produces the most useful shared-term code for the fuzzy-logic compositional
task. With only one form available, the alternatives cannot be compared without
editing the code.

This change makes the contrastive representation a configuration choice among
three forms and keeps the current response-self-attention form as the default,
so the alternatives can be studied without changing the experiment's shape.

## What Changes

- Add a `contrastive_representation` configuration option to the experiment,
  with exactly three values:
  - `matrix` — flatten the final layer's full attention scores per example
    (`(B, H, N, N)` → `(B, H·N·N)`);
  - `row` — the response token's attention row over key positions
    (`(B, H, N)` → `(B, H·N)`);
  - `code` — the response token's attention to itself, one feature per head
    (`(B, H)`).
- Default the option to `code` (the current behavior), so an unmodified run is
  unchanged.
- Extract the selected representation from the final encoder layer and pass it
  unchanged to the existing shared-term contrastive loss; the four-model
  comparison shape (softmax/linear × λ = 0/λ > 0) and everything else stay the
  same, with one representation per run.
- Add the one missing task-agnostic reduction (flattening the full attention
  matrix) to the analytics layer, where the other attention-to-vector
  reductions already live, together with the matching layer-stacking reducer.
- Expose the option through the command-line flags and record it with each
  tracked run so runs are comparable and reproducible.

## Capabilities

### New Capabilities

<!-- None: this extends an existing capability. -->

### Modified Capabilities

- `fuzzy-logic-contrastive-experiment`: the contrastive-representation
  requirement changes from a single fixed response-token code to a selectable
  representation among three forms, defaulting to the response token's
  per-head self-attention; the choice becomes part of the exposed
  configuration.
- `analytics-layer`: the attention-derived representation reduction gains a
  full attention-matrix flattening form (in addition to the per-head
  single-value and per-token row forms already specified).

## Impact

- Code: `compgen/runners/fuzzy_logic_contrastive/config.py` (new option and
  default), `step.py` (representation selection), `run.py` (wiring/logging),
  `__main__.py` (flag), and `compgen/analytics/attention.py` (full-matrix
  reduction and reducer).
- No change to model, dataset, loss, training, or tracking behavior;
  `FuzzyLogicModel.response_token_codes` is retained.
- No new runtime dependencies; runs in the existing `venv/` (PyTorch, CPU or
  the local RTX 2060).
- The isolated `compgen/experiments/` archive is untouched.
