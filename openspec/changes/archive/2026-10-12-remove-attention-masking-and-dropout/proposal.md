# Proposal

## Why

The fuzzy-logic task needs neither padding nor causal masking — its on-the-fly
sampler produces fixed-length batches with no padding — yet every attention
architecture still carries mask machinery plus three optional configuration
knobs (`dropout_rate`, `bias`, `use_dropout`). This dead weight complicates the
attention API and hides the fact that, for fuzzy-logic, attention is a pure
hidden-state transform. We simplify attention now, while the only pinned
consumer (the fuzzy-logic runner) never uses any of it.

## What Changes

- **BREAKING** Remove padding-mask handling from `BaseAttention` and all five
  attention variants (`softmax`, `linear`, `strassen`, `triangular`,
  `third_order`): drop `construct_mask`, `_flatten_padding_mask`,
  `mask_padding_value`, and the per-`forward` `batch_mask` branch. Attention
  becomes a pure function of the hidden state (plus the optional `score_bias`).
- **BREAKING** Remove the `batch_mask` argument from attention `forward`, from
  `EncoderLayer.forward`, and from the task models' forward paths. The Match3
  collate function stops emitting its `batch_mask` mapping.
- **BREAKING** Remove the `dropout_rate`, `bias`, and `use_dropout` constructor
  parameters from `BaseAttention` and every attention variant, together with the
  `_drop_weights` helper. Attention no longer applies dropout and its
  projections are always bias-free.
- Update the library callers to the new attention surface: `EncoderLayer`
  (drop `attention_dropout_rate`, `use_attention_dropout`, `attention_bias`),
  `FuzzyLogicModel` (drop `attention_dropout_rate`, `attention_bias`; stop
  passing `batch_mask=None`), `Match3Model` (drop `use_attention_dropout`;
  stop passing `batch_mask`), and the fuzzy-logic runner `config.py`/`run.py`
  (drop `attention_dropout_rate`, `attention_bias`).
- Keep the optional additive `score_bias` path unchanged; it is out of scope.
- Leave the two Match3 notebooks untouched in this change (deliberate: only
  library callers are updated).

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: the shared attention interface drops its masking, dropout, and
  bias knobs and becomes a hidden-state-only transform; the encoder layer stops
  wiring attention dropout/bias/mask, and task models stop threading a batch
  mask.
- `dataset-layer`: the Match3 collate function no longer produces a batch-mask
  mapping, since the model no longer accepts one.

## Impact

Affected code:

- `compgen/models/attentions/base.py` — remove mask helpers, `mask_padding_value`,
  `dropout_rate`/`bias`/`use_dropout`, `_drop_weights`; bias-free projections.
- `compgen/models/attentions/{softmax,linear,strassen,third_order,triangular}.py`
  — drop the removed params, the mask branch, and dropout calls; update
  docstrings.
- `compgen/models/encoder.py` — drop attention dropout/bias wiring and
  `batch_mask` from `forward`.
- `compgen/models/tasks/fuzzy_logic.py`, `compgen/models/tasks/match3.py` — drop
  removed attention params and the `batch_mask` argument.
- `compgen/datasets/torch_datasets/match3.py` — stop emitting `batch_mask`.
- `compgen/runners/fuzzy_logic_contrastive/config.py` and `run.py` — drop
  `attention_dropout_rate` and `attention_bias`.

Behavioral impact:

- Match3 loses padding masking, so padded positions are no longer excluded from
  attention. This is an accepted consequence of the simplification; the
  fuzzy-logic runner is the pinned consumer and is unaffected.
- The two Match3 notebooks still pass removed parameters and will break until
  updated separately.
- Isolated archive under `compgen/experiments/` is not touched (it must not
  import the package).
