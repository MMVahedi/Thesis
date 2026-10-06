# Spec Delta

## MODIFIED Requirements

### Requirement: Attention-derived representation vectors

The analytics layer SHALL provide a function that reduces attention weights of shape `(B, H, N, N)` to per-example representation vectors in one of three forms: the attention value at a selected query and key position per head, giving shape `(B, H)`; the selected query token's attention over key positions flattened over heads and keys, giving shape `(B, H * N)`; or the full attention matrix flattened over heads, query positions, and key positions, giving shape `(B, H * N * N)`. The query token SHALL default to the last token. It SHALL accept a single attention tensor or a stack of layers and, when given a stack, MAY concatenate each layer's contribution.

#### Scenario: A token's attention becomes one vector per example
- **WHEN** attention weights `(B, H, N, N)` are reduced for a chosen query token
- **THEN** the result has one vector per example whose length is determined by the head count (and, when requested, the key length)

#### Scenario: The default query token is the last position
- **WHEN** the reduction is called without an explicit query token
- **THEN** it uses the last sequence position as the query token

#### Scenario: Layers can be stacked
- **WHEN** a list of per-layer attention tensors is passed
- **THEN** each layer contributes its own features and the per-layer contributions can be distinguished in the result

#### Scenario: The full attention matrix flattens to one vector per example
- **WHEN** the full-matrix reduction is requested for attention weights `(B, H, N, N)`
- **THEN** each example's vector contains one feature per head, query position, and key position, giving shape `(B, H * N * N)`
