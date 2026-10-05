# Design

## Context

See `proposal.md` for motivation. Current state that shapes the approach:

- `FuzzyLogicModel` already accepts any registered `attention_type` and exposes
  `response_token_codes(attentions)`, which returns `attentions[-1][:, :, -1, -1]`
  — the final layer's last-token attention across heads, shape `(batch, heads)`.
- `FuzzyLogicOnlineGenerator.sample(...)` returns a `FuzzyLogicBatch` carrying
  `x`, `y`, `base_mse`, `latents`, and `term_ids`, and is deterministic per
  `(split, batch_index, seed)`.
- `compgen.losses.combined_loss(task_loss, term_ids, representations, lambda, T)`
  combines a task loss with the shared-term contrastive term and reports the
  average positive-partner count. It currently has no caller.
- `compgen.training.fit(model, train_data, step_fn, ...)` iterates a supplied
  sequence of batches, handles optimizer/scheduler/seeding/checkpointing and
  logs through `compgen.tracking`. It steps the scheduler once per epoch.
- `repo-structure` requires library code under one top-level package, and
  `compgen/experiments/` is reserved for isolated archives that must not import
  the package. The new runner composes the package, so it needs a different home.

The deleted reference `train.py` (recoverable from git) established the split
metrics (MSE, MAE, paper R², within-±0.05, contrastive loss, positive count) and
the hyperparameters: AdamW lr 1e-3, 100 warmup steps, cosine decay to 1e-4,
weight decay 0.1, sequence length 16.

## Goals / Non-Goals

**Goals:**

- A runnable entry point under `compgen/runners/` that composes the existing
  model, dataset, loss, training, and tracking layers without duplicating any of
  them.
- Reproduce the four-model comparison (softmax/linear × λ=0/λ>0) with a single
  small, fast default configuration, fully configurable.
- Report the reference split metrics plus contrastive diagnostics and log runs.

**Non-Goals:**

- No new model, dataset, loss, or training-engine behavior.
- No HyLA, no Match3, no attention-code clustering/analysis.
- No large sweeps (multiple dataset sizes, batch sizes, or λ values) and no
  changes to the isolated `compgen/experiments/` archive.

## Decisions

**1. Home: `compgen/runners/fuzzy_logic_contrastive/`.** Chosen over a top-level
`experiments/` (forbidden by `repo-structure`, and would shadow a generic name)
and over a notebook (harder to run unattended and to keep deterministic). The
package layout will be:

```
compgen/runners/
  __init__.py
  fuzzy_logic_contrastive/
    __init__.py
    config.py    # ExperimentConfig + the four-variant grid
    step.py      # step_fn: task loss + optional contrastive term
    run.py       # orchestration: build, train via fit, evaluate splits, log
    __main__.py  # `python -m compgen.runners.fuzzy_logic_contrastive` CLI
```

**2. Feed the engine pre-sampled batch lists.** `fit` iterates a supplied
sequence, but the dataset is an on-the-fly sampler. For a small study we
pre-sample a deterministic list of `num_train // batch_size` training batches and
a fixed evaluation batch list once, then pass them to `fit`. Alternative
(re-sample each epoch by wrapping the generator in a `Dataset`/iterable) adds an
epoch hook the engine does not expose; pre-sampling keeps determinism and stays
within the engine.

**3. Contrastive representation and loss wiring.** The step function calls the
model with `return_attention=True`, extracts `response_token_codes(attentions)`
(final layer, last token, across heads), and passes it to `combined_loss` with
the batch's `term_ids`. When `lambda == 0` the contrastive term is skipped by the
loss itself (reported as zero), while still computing the positive-partner count.
The loss code is not modified.

**4. Training and evaluation.** Train with `compgen.training.fit` using one
optimizer (AdamW, lr 1e-3, weight decay 0.1) and an epoch-granularity scheduler.
Because `fit` evaluates a single `eval_data`, use the train split during fitting
and run a dedicated post-training pass over train/test/id/ood with the best
state dict to produce the reported metrics. Alternative (per-step warmup + cosine
exactly as the reference) would require extending the engine, which is out of
scope for this small comparison; the divergence is noted under Risks.

**5. Metrics and tracking.** Compute MSE, MAE, the per-example reference R²
(`1 - error² / base_mse`), within-±0.05 accuracy, contrastive loss, and average
positive partners; write a per-variant metrics JSON and log the run through
`compgen.tracking` with the attention type and λ as parameters (used as the run
name).

**6. Defaults (configurable).** `task=logic_4var_2term`, `num_train=2000`,
`num_eval=1000`, `batch_size=64`, `epochs=1`, `lambda_contrastive=0.05`,
`temperature=1.0`, `seed=2024`, `attention_types=[softmax, linear]`. Model
hyperparameters reuse the reference config (2 layers, hidden 128, 8 heads,
q/k and v width 16, GELU FFN 256, relative-position bias, attention dropout 0.1)
so only architecture and λ vary across the four.

**7. Documentation.** Update `AGENTS.md` and the root `README.md` to list
`compgen/runners/` as the home for package-composing runnable entry points,
distinct from the isolated `compgen/experiments/` archive, as the
documentation-matches-layout requirements demand.

## Risks / Trade-offs

- [Epoch-granularity schedule diverges from the reference's per-step warmup/cosine] → Acceptable for a small comparative study; record lr schedule as a parameter and revisit if results look schedule-sensitive.
- [Small defaults (2000 examples, 1 epoch) may underfit, muting the contrastive effect] → Defaults are configurable; increase `num_train`/`epochs` without code changes.
- [Contrastive codes for linear attention are unnormalized scores, unlike softmax's normalized weights] → Inherent to the two architectures and faithful to the paper's hypernetwork framing; keep the same extraction for both and report per-architecture results separately.
- [Batch composition changes both the number of negatives and the probability of shared-term positives] → Fixed seed and identical batch size/order across the four variants keep the comparison controlled.
- [No test suite exists] → Verify with a short smoke run of the four variants and a repeat-run determinism check.

## Open Questions

- The single positive `lambda` (default 0.05) and `epochs` (default 1) are
  starting points; if the effect is too weak to read, raise them. Neither changes
  the specs, approach, or tasks.
