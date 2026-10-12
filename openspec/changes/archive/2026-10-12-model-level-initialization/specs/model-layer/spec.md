# Spec Delta

## ADDED Requirements

### Requirement: Model-level weight initialization
The model layer SHALL centralize weight initialization at the task-model level. The initialization schemes SHALL be classes owned by the model layer — an abstract weight-initialization policy base class with one concrete subclass per scheme (a xavier scheme, the reference truncated-normal scheme, and a no-op that preserves PyTorch's defaults) — and no string name or string-keyed mapping SHALL identify or select a scheme. A task model SHALL own a single initialization method that applies one uniform policy to the whole model it composes. Sub-level modules — attention architectures, feed-forward variants, prediction heads, task embeddings, and the encoder layer — SHALL NOT accept an initialization policy and SHALL NOT initialize their own weights.

#### Scenario: Initialization schemes live in one place
- **WHEN** the model layer is inspected for how the initialization schemes are defined
- **THEN** the schemes are defined once in the model layer's initialization module, and no attention, feed-forward, head, embedding, or task-model module contains an inline xavier-normal or truncated-normal initialization loop of its own

#### Scenario: Each scheme is a distinct class, not a string
- **WHEN** the model layer's initialization module is inspected for how a scheme is represented and selected
- **THEN** each scheme is a distinct concrete subclass of the abstract policy base class, and no string name, string type field, or string-keyed lookup is used to identify or select a scheme

#### Scenario: Only the task model initializes its modules
- **WHEN** a task model is constructed
- **THEN** exactly one initialization method applies the model's single policy across every sub-module it composes

#### Scenario: Sub-level modules carry no initialization concern
- **WHEN** any attention architecture, feed-forward variant, prediction head, task embedding, or encoder layer is inspected
- **THEN** its construction surface accepts no initialization policy and it applies no self-initialization to its parameters

#### Scenario: A non-default initializer overrides the model's scheme
- **WHEN** a task model is constructed with a supplied non-default policy class
- **THEN** all of the model's Linear and Embedding parameters follow that scheme

#### Scenario: Each model applies one uniform default scheme
- **WHEN** the Match3 task model and the fuzzy-logic task model are each constructed with their default initializer
- **THEN** the fuzzy-logic task model applies the reference scheme across its whole composed model and the Match3 task model applies the xavier scheme across its whole composed model

## REMOVED Requirements

### Requirement: Centralized weight initialization
**Reason**: Initialization is no longer per-module. It moves to the task-model level, where one uniform policy initializes the whole composed model, so the requirement's "each module accepts a policy and initializes itself" contract no longer holds. Replaced by the "Model-level weight initialization" requirement.
**Migration**: Construct sub-level modules without an initializer. The task models now apply a single `WeightInitializer` policy across the whole model after construction.

### Requirement: Attention architectures initialize their own weights
**Reason**: Attention architectures no longer accept an initialization policy or initialize themselves; the composing task model initializes the whole model.
**Migration**: Construct the attention module without an initializer; the task model applies its single policy across the assembled model after construction.
