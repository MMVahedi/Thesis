# Spec Delta

## ADDED Requirements

### Requirement: Task models accept a caller-supplied attention module
A task model SHALL accept a caller-supplied, already-constructed attention module and SHALL use that same module as the attention of every encoder layer it composes, so the module and its parameters are shared across the layers. The task model SHALL NOT restrict which attention may be supplied, SHALL NOT perform any name-based allow-list or validation, and SHALL use exactly the module supplied. Which architectures a particular study compares SHALL be decided by the calling notebook or experiment configuration, not restricted inside the task model.

#### Scenario: Every caller-supplied attention module composes
- **WHEN** the Match3 or fuzzy-logic task model is constructed with a supplied attention module of a given attention architecture
- **THEN** construction succeeds and the resulting encoder stack exposes that attention architecture

#### Scenario: One module is shared across encoder layers
- **WHEN** a task model with multiple encoder layers is constructed with a supplied attention module
- **THEN** every encoder layer's attention is that same module instance, so its parameters are shared, while each layer's feed-forward block remains distinct

#### Scenario: Task model performs no architecture validation
- **WHEN** a task model is constructed with a supplied attention module
- **THEN** the task model performs no name-based lookup or rejection and uses exactly the module supplied

## MODIFIED Requirements

### Requirement: Task models compose shared scaffolding
The encoder layer (attention block plus feed-forward block with optional normalization and residual connections) and the per-token classification head SHALL be importable, reusable modules that depend on no task-specific code. Each task model file SHALL contain only its task-specific input representation and thin composition of the shared pieces. A task model SHALL accept a caller-supplied, already-constructed attention module and SHALL pass that constructed module into each of its encoder layers. The task model SHALL NOT resolve, select, or validate an attention architecture name itself.

#### Scenario: Shared scaffolding is task-agnostic
- **WHEN** the shared encoder-layer and classification-head modules are imported from the repository root and their imports are inspected
- **THEN** neither imports any task-specific module, and either can be used without importing any task model

#### Scenario: Task file is thin glue
- **WHEN** a task model file under the model layer is inspected
- **THEN** it defines (or imports) only its task-specific embedding and composes the shared encoder layer and shared head, obtaining attention solely from the caller-supplied attention module and defining no encoder or head of its own

#### Scenario: Task model constructs the attention it selected
- **WHEN** a task model is constructed with a caller-supplied, already-constructed attention module and the resulting encoder stack is inspected
- **THEN** each encoder layer's attention is that supplied module, and the task model's public construction surface takes attention as a constructed module rather than as an architecture name or a factory

### Requirement: Fuzzy-logic regression task model
The model layer SHALL provide a fuzzy-logic regression task model that composes a task-specific input embedding, a configurable stack of encoder layers whose attention comes from a caller-supplied, already-constructed attention module shared across the stack, and the regression head, and SHALL accept any attention module a caller supplies. The task-specific input embedding SHALL add a fixed absolute positional encoding to its projected token representations.

#### Scenario: Model reproduces the reference architecture
- **WHEN** the fuzzy-logic task model is constructed with an attention module configured for the reference (8 heads, query/key and value width 16) and the reference task configuration (two layers, hidden width 128, feed-forward width 256, fixed absolute positional encoding, GELU feed-forward)
- **THEN** it produces per-token scalar predictions and exposes one attention-weight matrix per layer

#### Scenario: Registered architectures are accepted
- **WHEN** the fuzzy-logic task model is constructed with a supplied attention module for each provided sequence-composable attention architecture
- **THEN** every construction succeeds and the selected architecture appears in the encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the fuzzy-logic task model is constructed with an architecture name string instead of an attention module
- **THEN** construction fails because the model takes attention as a caller-supplied module, not an architecture name

#### Scenario: Position enters once through the input embedding
- **WHEN** the fuzzy-logic task model processes a batch
- **THEN** a fixed position signal is added to the embedded tokens before the encoder stack, and no relative-position bias is added to attention scores

## REMOVED Requirements

### Requirement: Task models accept a caller-supplied attention factory
**Reason**: Replaced by a caller-supplied, already-constructed attention module. The factory was a redundant indirection: `EncoderLayer` already consumes a constructed attention object, and the feed-forward sub-module is likewise consumed as a constructed object, so the task-model boundary no longer needs a per-layer construction callable.

**Migration**: Construct the attention module at the call site and pass it to the task model instead of a factory. A single module is now used for every encoder layer (its parameters are shared), rather than one module being produced per layer. Callers passing `attention_factory=` must switch to supplying a constructed attention module.
