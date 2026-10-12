# Spec Delta

## ADDED Requirements

### Requirement: Encoder layer consumes a constructed attention module
The shared encoder layer SHALL accept an already-constructed attention module and SHALL use that module as its self-attention. It SHALL NOT accept an attention architecture name, SHALL NOT look up or validate architecture names, and SHALL depend on no architecture registry. Resolving a registered architecture name to its class and constructing the module SHALL be the responsibility of the task model or other caller that composes the encoder layer.

#### Scenario: Constructed module becomes the layer's attention
- **WHEN** an encoder layer is constructed with a constructed attention module and applied to a hidden state
- **THEN** the layer uses that exact module for self-attention and returns a hidden state of the same shape as its input

#### Scenario: Encoder layer takes no architecture name
- **WHEN** the encoder layer's construction surface is inspected
- **THEN** it accepts no architecture-name string and performs no registry lookup or architecture-name validation

#### Scenario: Encoder layer is registry-independent
- **WHEN** the shared encoder-layer module is imported from the repository root and its imports are inspected
- **THEN** it imports neither the architecture registry nor any task-specific module

## MODIFIED Requirements

### Requirement: Task models compose shared scaffolding
The encoder layer (attention block plus feed-forward block with optional normalization and residual connections) and the per-token classification head SHALL be importable, reusable modules that depend on no task-specific code. Each task model file SHALL contain only its task-specific input representation and thin composition of the shared pieces. A task model SHALL resolve its selected registered architecture name to the architecture class, construct the attention module with the model's configuration, and pass the constructed module into each encoder layer.

#### Scenario: Shared scaffolding is task-agnostic
- **WHEN** the shared encoder-layer and classification-head modules are imported from the repository root and their imports are inspected
- **THEN** neither imports any task-specific module, and either can be used without importing any task model

#### Scenario: Task file is thin glue
- **WHEN** a task model file under the model layer is inspected
- **THEN** it defines (or imports) only its task-specific embedding and composes the shared encoder layer and shared head, constructing the selected registered architecture and passing the constructed module into the encoder layer — with no duplicated encoder or head definitions

#### Scenario: Task model constructs the attention it selected
- **WHEN** a task model is constructed from a registered architecture name and the resulting encoder stack is inspected
- **THEN** each encoder layer's attention is a constructed module of that architecture, and the task model's public construction surface still selects the architecture by its registered name
