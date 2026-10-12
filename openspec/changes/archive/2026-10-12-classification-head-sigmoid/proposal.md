# Proposal

## Why

`TokenClassifier` currently returns raw logits, and `Match3Model.forward` applies `torch.sigmoid` afterwards — so the head is named a classifier but does not itself produce a probability, and every consumer must know to add the sigmoid. The head should own the full binary-classification readout: compute the logit and apply the sigmoid, returning a probability in `(0, 1)` directly. This keeps the non-linearity colocated with the linear readout it belongs to instead of leaking into the task model.

## What Changes

- **BREAKING**: `TokenClassifier.forward` returns per-token probabilities instead of raw logits — it applies `sigmoid` to its single output logit and returns `(batch, length, 1)` values bounded within `[0, 1]`.
- The sigmoid application stays inside `compgen/models/heads.py`; it is not moved into the task model or any other module.
- `Match3Model.forward` stops applying its own `torch.sigmoid` and consumes the head's probability output directly. Its public output contract (per-token probabilities plus final hidden states) is unchanged, so existing notebooks that train with `nn.BCELoss()` are unaffected.
- `RegressionHead` is unchanged: regression has no sigmoid/probability step.
- Because the head no longer exposes raw logits, a caller can no longer feed them to `BCEWithLogitsLoss`; binary classification training uses `BCELoss` on the returned probabilities.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `model-layer`: the per-token classification head's output contract changes from raw logits to probabilities, and the Match3 task model's forward no longer applies its own sigmoid.

## Impact

- `compgen/models/heads.py` — `TokenClassifier.forward` applies `sigmoid`; docstring updated to describe the probability output and the loss implication.
- `compgen/models/tasks/match3.py` — removes the `torch.sigmoid(logits)` step and forwards the head's probabilities; no change to the returned tuple shape or `return_attention` behavior.
- Training callers — any trainer that fed `TokenClassifier` output to `BCEWithLogitsLoss` must switch to `BCELoss`; the Match3 notebooks already use `nn.BCELoss()` and need no change.
- `openspec/specs/model-layer/spec.md` — gains a requirement fixing the classification head's sigmoid/probability behavior.
