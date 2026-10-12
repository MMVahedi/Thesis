# Spec Delta

## REMOVED Requirements

### Requirement: Public task API stays stable
**Reason**: The attention simplification deliberately changes the Match3 task model's public surface: its constructor no longer accepts `use_attention_dropout`, and its forward input contract no longer requires (or carries) a batch-mask mapping. The stability guarantee this requirement stated was specific to the earlier architecture-name reorganization and no longer holds.
**Migration**: Construct task models without the removed attention parameters and call their forward paths with a collated batch that has no `batch_mask` entry. The new surface is defined by the added "Task models consume an unmasked batch" requirement.

## MODIFIED Requirements

### Requirement: Softmax and linear attention mixers
The registry SHALL expose two sibling mixers that share the same projection, scaling, and multi-head splitting: a softmax attention whose attention weights are normalized with softmax, and a linear attention whose attention weights are the scaled scores without the final softmax normalization. Neither mixer SHALL apply masking or dropout to its weights.

#### Scenario: Linear is softmax without the final normalization
- **WHEN** the softmax and linear architectures are constructed with identical configuration and applied to the same input, and the softmax mixer's normalization is removed
- **THEN** the linear mixer's output matches the unnormalized softmax path

#### Scenario: Both mixers are selectable
- **WHEN** the registry is imported from the repository root
- **THEN** both `softmax` and `linear` are selectable names that construct attention modules

## ADDED Requirements

### Requirement: Attention is a hidden-state-only transform
Every attention architecture in the registry SHALL compute its output from the token hidden states alone. Attention `forward` SHALL accept the hidden states and the optional additive `score_bias`, and SHALL NOT accept a mask argument. Attention construction SHALL NOT accept a dropout rate, a dropout toggle, a projection-bias flag, or a padding-mask value. Attention weights SHALL be neither masked nor dropped out.

#### Scenario: No mask argument is accepted
- **WHEN** an attention architecture's `forward` is called with hidden states and an optional additive score bias
- **THEN** it returns the attention output and attention weights, and passing a batch-mask argument is rejected

#### Scenario: Removed configuration knobs are rejected
- **WHEN** any registered attention architecture is constructed with `dropout_rate`, `use_dropout`, `bias`, or `mask_padding_value`
- **THEN** construction fails because the parameter is not accepted

#### Scenario: No dropout is applied to attention weights
- **WHEN** an attention architecture runs in training mode with dropout once configured externally
- **THEN** the returned attention weights equal the unmasked, un-dropped normalization of its scores

#### Scenario: Head and dimension configuration is preserved
- **WHEN** each registered architecture is constructed with a hidden dimension, a head count, and optionally distinct query/key and value head dimensions
- **THEN** construction succeeds with the same head and dimension behavior as before the simplification

### Requirement: Task models consume an unmasked batch
The fuzzy-logic and Match3 task models SHALL construct their attention without dropout or bias configuration, and their forward paths SHALL NOT accept, read, or forward a batch mask. The Match3 task model SHALL NOT expose an attention-dropout constructor parameter; its residual and feed-forward dropout SHALL remain configurable.

#### Scenario: Fuzzy-logic model forward takes no mask
- **WHEN** the fuzzy-logic task model's forward is called with inputs
- **THEN** it produces predictions and, on request, per-layer attention weights, without any mask being supplied or forwarded to attention

#### Scenario: Match3 model forward takes no mask
- **WHEN** the Match3 task model's forward is called with a collated batch
- **THEN** it produces per-token probabilities and final hidden states without reading a batch-mask entry from the batch

#### Scenario: Match3 attention-dropout parameter is gone
- **WHEN** the Match3 task model is constructed with `use_attention_dropout`
- **THEN** construction fails because the parameter is not accepted, while the residual/feed-forward dropout rate remains configurable

#### Scenario: Attention-returning contract is unchanged
- **WHEN** the fuzzy-logic or Match3 task model is asked for attention weights
- **THEN** each still returns per-layer weights of shape `(batch, num_heads, length, length)`
