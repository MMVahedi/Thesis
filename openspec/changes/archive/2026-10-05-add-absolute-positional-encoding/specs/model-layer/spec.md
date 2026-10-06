# Spec Delta

## REMOVED Requirements

### Requirement: Reusable shared model blocks
**Reason**: This requirement is re-expressed to drop the relative-position bias block, which is removed, and to introduce a fixed absolute positional encoding block. The feed-forward and regression-head blocks it also covered are unchanged and retained under the replacement requirement.
**Migration**: Use the added "Reusable model blocks and fixed positional encoding" requirement. Relative-position bias has no replacement and is no longer available; position is now supplied by the fixed absolute positional encoding added to the fuzzy-logic input embedding.

## ADDED Requirements

### Requirement: Reusable model blocks and fixed positional encoding
The model layer SHALL provide reusable, task-agnostic blocks that compose into full sequence models: a fixed absolute positional encoding block that produces a deterministic position vector for each sequence position, feed-forward block support for both the existing residual ReLU stack and a two-projection GELU residual block, and a regression head that maps token hidden states to a scalar prediction after a final normalization. The model layer SHALL NOT provide a relative-position bias block.

#### Scenario: Positional encoding is deterministic and position-indexed
- **WHEN** the positional encoding block is evaluated for a sequence length and hidden width
- **THEN** it returns a tensor with one position vector per sequence position at the given hidden width, broadcastable over batch, and containing no learned parameters

#### Scenario: Both feed-forward variants are available
- **WHEN** an encoder layer is constructed with each supported feed-forward variant
- **THEN** construction succeeds and a forward pass returns a hidden state of the same shape as its input

#### Scenario: Regression head reduces to one scalar per token
- **WHEN** the regression head is applied to hidden states of shape `(batch, length, hidden)`
- **THEN** it returns a tensor of shape `(batch, length, 1)`

## MODIFIED Requirements

### Requirement: Fuzzy-logic regression task model
The model layer SHALL provide a fuzzy-logic regression task model that composes a task-specific input embedding, a configurable stack of encoder layers whose attention is chosen from the shared registry, and the regression head, and SHALL accept any registered sequence-composable attention architecture name. The task-specific input embedding SHALL add a fixed absolute positional encoding to its projected token representations.

#### Scenario: Model reproduces the reference architecture
- **WHEN** the fuzzy-logic task model is constructed with the reference configuration (two layers, hidden width 128, feed-forward width 256, 8 heads, query/key and value width 16, fixed absolute positional encoding, GELU feed-forward)
- **THEN** it produces per-token scalar predictions and exposes one attention-weight matrix per layer

#### Scenario: Registered architectures are accepted
- **WHEN** the fuzzy-logic task model is constructed with each registered sequence-composable attention name
- **THEN** every construction succeeds and the selected architecture appears in the encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the fuzzy-logic task model is constructed with a name absent from the shared registry
- **THEN** construction fails with an error naming the unsupported architecture

#### Scenario: Position enters once through the input embedding
- **WHEN** the fuzzy-logic task model processes a batch
- **THEN** a fixed position signal is added to the embedded tokens before the encoder stack, and no relative-position bias is added to attention scores
