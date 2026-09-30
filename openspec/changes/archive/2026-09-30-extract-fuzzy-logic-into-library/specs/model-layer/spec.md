# Spec Delta

## ADDED Requirements

### Requirement: Reusable shared model blocks
The model layer SHALL provide reusable, task-agnostic blocks that compose into full sequence models: a relative-position bias block that produces an additive attention bias indexed by head and query/key position from a sequence length, feed-forward block support for both the existing residual ReLU stack and a two-projection GELU residual block, and a regression head that maps token hidden states to a scalar prediction after a final normalization.

#### Scenario: Relative-position bias is head-indexed and additive
- **WHEN** the relative-position bias block is evaluated for a sequence length and head count
- **THEN** it returns a tensor broadcastable over batch with one bias value per head and query/key position

#### Scenario: Both feed-forward variants are available
- **WHEN** an encoder layer is constructed with each supported feed-forward variant
- **THEN** construction succeeds and a forward pass returns a hidden state of the same shape as its input

#### Scenario: Regression head reduces to one scalar per token
- **WHEN** the regression head is applied to hidden states of shape `(batch, length, hidden)`
- **THEN** it returns a tensor of shape `(batch, length, 1)`

### Requirement: Attention accepts an additive score bias
The sequence-composable attention architectures registered in the shared registry (softmax and linear) SHALL accept an optional additive score bias and add it to the scaled query/key scores before normalization, so a relative-position or other bias can be incorporated without reimplementing attention outside the registry.

#### Scenario: Bias is added to the scores
- **WHEN** the same attention is applied with and without a non-zero additive bias and all other inputs are equal
- **THEN** the resulting attention weights differ and keep the same shape

### Requirement: Fuzzy-logic regression task model
The model layer SHALL provide a fuzzy-logic regression task model that composes a task-specific input embedding, a configurable stack of encoder layers whose attention is chosen from the shared registry, and the regression head, and SHALL accept any registered sequence-composable attention architecture name.

#### Scenario: Model reproduces the reference architecture
- **WHEN** the fuzzy-logic task model is constructed with the reference configuration (two layers, hidden width 128, feed-forward width 256, 8 heads, query/key and value width 16, relative-position bias, GELU feed-forward)
- **THEN** it produces per-token scalar predictions and exposes one attention-weight matrix per layer

#### Scenario: Registered architectures are accepted
- **WHEN** the fuzzy-logic task model is constructed with each registered sequence-composable attention name
- **THEN** every construction succeeds and the selected architecture appears in the encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the fuzzy-logic task model is constructed with a name absent from the shared registry
- **THEN** construction fails with an error naming the unsupported architecture
