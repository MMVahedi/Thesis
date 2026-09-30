# Spec Delta

## ADDED Requirements

### Requirement: Task models expose attention weights on demand
A task model in the model layer SHALL be able to return its per-layer attention weights on demand through its `forward` method, without changing its default output. `Match3Model.forward` SHALL accept an optional `return_attention` argument (default `False`); when `False` it SHALL return exactly the previous output, and when `True` it SHALL additionally return the attention weights of every encoder layer in order, each of shape `(batch, num_heads, length, length)`. This mirrors the fuzzy-logic task model's existing attention-returning path.

#### Scenario: Default output is unchanged
- **WHEN** the Match3 task model's `forward` is called without `return_attention` (or with it `False`)
- **THEN** it returns only the per-token probabilities and final hidden states, exactly as before

#### Scenario: Attention weights are returned on request
- **WHEN** the Match3 task model's `forward` is called with `return_attention=True`
- **THEN** it returns the per-token probabilities, the final hidden states, and one attention-weight tensor per encoder layer, each of shape `(batch, num_heads, length, length)`

#### Scenario: Fuzzy-logic and Match3 agree on the attention contract
- **WHEN** the fuzzy-logic task model and the Match3 task model are each asked for attention weights
- **THEN** both provide per-layer attention weights with the same `(batch, heads, query, key)` layout that the analytics layer consumes
