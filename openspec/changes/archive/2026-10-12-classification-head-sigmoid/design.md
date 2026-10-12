# Design

## Context

`compgen/models/heads.py` currently defines `TokenClassifier` returning raw logits; `Match3Model.forward` applies `torch.sigmoid` itself (`compgen/models/tasks/match3.py:96`) and the Match3 notebooks train with `nn.BCELoss()` on those probabilities. The `model-layer` spec does not yet fix the classification head's output contract. `RegressionHead` is a separate, sigmoid-free path. See proposal.md - Why for motivation.

Constraints:
- The head must remain task-agnostic and must not import task-specific code.
- `Match3Model`'s public output (per-token predictions, final hidden states, optional per-layer attention) must not change, so existing notebooks keep working.
- `TokenClassifier` has exactly one consumer in the repository (`Match3Model`; confirmed by grep for `TokenClassifier`, `sigmoid`, and `BCELoss`/`BCEWithLogitsLoss`).

## Goals / Non-Goals

**Goals:**
- The classification head produces the complete binary readout as a probability bounded within `[0, 1]` (mathematically interior to `(0, 1)`; float64 may round to the endpoints for large logits).
- The sigmoid lives in the head only; the task model contains no normalization step.
- Match3 and its notebooks keep their current observable behavior (per-token probabilities).

**Non-Goals:**
- Changing `RegressionHead` or the fuzzy-logic task model.
- Adding a logits-vs-probabilities toggle or a dual-output contract.
- Preserving a raw-logit output for `BCEWithLogitsLoss`.
- Changing the head's parameters, attribute names, or initialization (so state-dict layout is untouched).

## Decisions

### Return probabilities only, with the sigmoid inside the head
`TokenClassifier.forward` applies `torch.sigmoid` to `out_proj`'s output as its last step and returns that tensor. Alternatives considered: (a) return both logits and probabilities, retaining the numerically stable `BCEWithLogitsLoss` path; (b) an opt-in flag selecting sigmoid or logits. Both were rejected in favor of the explicit user decision to have the head emit probabilities only, matching the head's role as the classifier readout and the notebooks' existing `BCELoss` usage.

### Sigmoid stays in the head, not the task model
`heads.py` owns the sigmoid; `match3.py` drops its `torch.sigmoid(logits)` line and assigns the head output directly to `y_hat`. This keeps the readout self-contained and matches the user's boundary: the non-linearity is not relocated into the task model or elsewhere under `models/`.

### Loss contract moves to probability-input BCE
Since the head no longer exposes logits, binary-classification training consumes the head output with a probability-input objective (`BCELoss`), which the Match3 notebooks already use. The stable fused `BCEWithLogitsLoss` path is intentionally given up (see Risks). The head output stays `(batch, length, 1)`, so loss and metric shapes are unchanged.

## Risks / Trade-offs

- **Loss of numerical stability** → `BCEWithLogitsLoss` computes sigmoid+cross-entropy in a fused, stable form; applying sigmoid in the head removes that option, so `BCELoss` on saturated/confidently-wrong predictions can produce the `log(0)` gradient issues the fused loss avoids. Accepted as the cost of the chosen contract; the notebooks already operate this way. Reintroducing a logit output later would require a follow-up change.
- **Silently wrong external use** → any caller feeding head output to `BCEWithLogitsLoss` would now be passing probabilities where logits are expected. Grep confirms `Match3Model` is the only consumer, and it is updated in the same change; notebooks use `BCELoss`.
- **Checkpoint compatibility** → parameters, names, and init are unchanged, so existing state dicts still load; only the forward output changes.
