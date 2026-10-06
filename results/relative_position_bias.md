# Relative-position bias (removed)

This note records a design we removed from the model layer so the idea is not
lost. It is **not active code** — the package no longer implements it. It is kept
here for possible future experimentation.

## What it was

Position used to be injected inside attention by a learned **relative-position
bias**, in `compgen/models/position.py` (`RelativePositionBias`):

- An `nn.Embedding(num_buckets, num_heads)` table with one learned bias value
  per head.
- For each `(query, key)` pair it looked up a bucket for the pair's distance
  `query − key`, using T5-style bucketing: the nearest `num_buckets // 2`
  distances got their own bucket, farther distances were spread logarithmically
  and saturated at `max_distance`.
- The result, shape `(1, num_heads, N, N)`, was added to the attention scores
  before masking and normalization.

It was owned by `EncoderLayer` and configured through `use_relative_bias`,
`relative_bias_num_buckets` (default 16), and `relative_bias_max_distance`
(default 16), plumbed through `FuzzyLogicModel`, `ExperimentConfig`, and
`run.build_model`.

## Why it was replaced

The refactor moved to a single, conventional **fixed absolute positional
encoding**: a parameter-free sinusoidal vector (`sin`/`cos` at geometrically
spaced frequencies), added once to the fuzzy-logic input embedding
(`FuzzyLogicEmbedding`). Reasons:

- **Simplicity**: one deterministic, well-understood block beats a bucketed,
  per-head learned bias whose behavior is harder to interpret.
- **No parameters / no length bound**: the sinusoidal encoding adds nothing to
  the checkpoint and works for any sequence length.

The trade-off, and the reason this note exists: the absolute encoding is added
**once at the input**, whereas the relative bias acted on attention scores at
**every layer**. If long-range or deep-stack position use turns out to matter
for a task, the relative variant is worth re-testing.

## How to re-test it (if needed)

1. Reintroduce a bias module (learned table + T5-style bucketing) producing an
   additive `(1, num_heads, N, N)` tensor. The attention mixers still accept an
   optional additive `score_bias`, so no attention changes are required — pass
   it through `EncoderLayer`'s `score_bias` argument.
2. Add a config switch on the task model/experiment to choose between the fixed
   absolute encoding (input embedding) and the learned relative bias (attention
   scores), or run both as a comparison.
3. Compare on a position-sensitive task using the existing regression and
   contrastive metrics.
