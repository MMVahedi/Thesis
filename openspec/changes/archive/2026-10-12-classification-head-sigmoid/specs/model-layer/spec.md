# Spec Delta

## ADDED Requirements

### Requirement: Classification head returns probabilities
The per-token binary classification head SHALL apply a sigmoid to its single output logit and return per-token probabilities bounded within the closed interval `[0, 1]`, rather than raw logits. (Sigmoid is mathematically interior to `(0, 1)`; in floating point it may round to the endpoints for large-magnitude logits.) The sigmoid SHALL be part of the classification head and SHALL NOT be applied again by any task model that composes it. Because the head exposes probabilities and not logits, binary-classification training SHALL consume the head output with a binary cross-entropy objective that expects probabilities. The scalar regression head SHALL remain unchanged and SHALL NOT apply a sigmoid to its output.

#### Scenario: Classification head emits probabilities
- **WHEN** the per-token classification head is applied to hidden states of shape `(batch, length, hidden)`
- **THEN** it returns a tensor of shape `(batch, length, 1)` whose values all lie within the closed interval `[0, 1]`

#### Scenario: Task model applies no additional sigmoid
- **WHEN** the Match3 task model's `forward` is called with a collated batch
- **THEN** its returned per-token predictions lie within `[0, 1]` and equal the classification head's output applied to the final hidden states, with no extra normalization applied in the task model

#### Scenario: Regression head is unaffected
- **WHEN** the scalar regression head is applied to hidden states
- **THEN** it returns one scalar per token whose values are not constrained to `(0, 1)`
