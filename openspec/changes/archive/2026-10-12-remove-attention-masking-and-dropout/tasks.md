# Tasks

## 1. Simplify the attention layer

- [x] 1.1 In `compgen/models/attentions/base.py`, remove the `dropout_rate`, `bias`, `use_dropout`, and `mask_padding_value` parameters; make the Q/K/V and output projections bias-free; remove the attention `nn.Dropout`, `_drop_weights`, `_flatten_padding_mask`, and `construct_mask`. Verify the module imports and `BaseAttention` no longer references `dropout`, `mask_padding_value`, or `construct_mask`.
- [x] 1.2 Update `softmax.py` (`SoftmaxAttention`): drop the removed constructor params and the `batch_mask`/mask branch, change `forward` to `forward(self, hidden_state, score_bias=None)`, and remove the `_drop_weights` call. Verify a forward pass in train mode returns weights equal to the softmax of the (unmasked, un-dropped) scaled scores.
- [x] 1.3 Update `linear.py` (`LinearAttention`): same parameter and signature removal, delete the zeroing mask branch, and remove `_drop_weights`. Verify a forward pass returns the scaled-score weights with no masking or dropout.
- [x] 1.4 Update the retained high-order variants `strassen.py`, `third_order.py`, and `triangular.py`: drop the removed params, delete their `batch_mask` branches and `_drop_weights` calls, and change `forward` to `forward(self, hidden_state, score_bias=None)`. Verify each constructs and runs one forward pass on its expected input shape.
- [x] 1.5 Update the module and class docstrings in `base.py` and the five variants to remove masking/dropout/bias wording and reflect the hidden-state-only transform. Verify no docstring still claims masking, dropout, or a padding value.

## 2. Update the encoder layer

- [x] 2.1 In `compgen/models/encoder.py`, remove `attention_dropout_rate`, `use_attention_dropout`, and `attention_bias` from `EncoderLayer.__init__`; construct attention with only the structural parameters; remove `batch_mask` from `forward` and stop passing it (and `score_bias` handling stays). Verify the layer constructs and a forward pass returns a hidden state of the same shape as its input.
- [x] 2.2 Update the `encoder.py` docstring to drop mention of attention dropout/bias and the forwarded batch mask. Verify the docstring no longer references removed parameters.

## 3. Update the task models

- [x] 3.1 In `compgen/models/tasks/fuzzy_logic.py`, remove `attention_dropout_rate` and `attention_bias` from `FuzzyLogicModel.__init__` and the `EncoderLayer` construction, and stop passing `batch_mask=None` in `forward`. Verify the model constructs and one forward pass returns per-token predictions plus per-layer attention of shape `(batch, num_heads, length, length)`.
- [x] 3.2 In `compgen/models/tasks/match3.py`, remove `use_attention_dropout` and stop passing `batch_mask` in `forward`; keep `dropout_rate` for the residual/feed-forward and head paths. Verify the model constructs and a forward pass over a collated batch returns probabilities and final hidden states.

## 4. Update the dataset collate

- [x] 4.1 In `compgen/datasets/torch_datasets/match3.py`, remove the `batch_mask` entry from `match3_collate_fn`'s returned mapping (leaving inputs, position, labels, and geometry). Verify the returned mapping has no `batch_mask` key and its inputs/labels have the expected padded batch shape.

## 5. Update the fuzzy-logic runner

- [x] 5.1 In `compgen/runners/fuzzy_logic_contrastive/config.py`, remove the `attention_dropout_rate` and `attention_bias` fields (and their comments).
- [x] 5.2 In `compgen/runners/fuzzy_logic_contrastive/run.py`, stop passing `attention_dropout_rate` and `attention_bias` to `FuzzyLogicModel`. Verify `ExperimentConfig()` still validates and `build_model` constructs for both `softmax` and `linear`.

## 6. Integration checks

- [x] 6.1 Construct the fuzzy-logic model for each registered sequence-composable architecture (`softmax`, `linear`) and run one forward pass; verify it matches the reference configuration (two layers, hidden 128, 8 heads) and returns attention weights of the expected shape.
- [x] 6.2 Run the fuzzy-logic contrastive runner for a single small configuration (small `num_train`/`num_eval`, `epochs=1`) and verify it completes end to end without error.
- [x] 6.3 Confirm the isolated `compgen/experiments/` archive and the two Match3 notebooks were not modified, and record that the notebooks are deliberately left broken until a follow-up change.
