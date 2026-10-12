# Spec Delta

## MODIFIED Requirements

### Requirement: Centralized weight initialization
The model layer SHALL centralize the initialization schemes it uses as a class hierarchy: an abstract weight-initialization policy base class with one concrete subclass per scheme — a xavier scheme, the reference truncated-normal scheme, and a no-op that preserves PyTorch's defaults — and SHALL NOT identify or select a scheme by a string name or a string-keyed mapping. Each parameterized model-layer module — attention architectures, feed-forward variants, prediction heads, task embeddings, and the encoder layer's normalization parameters — SHALL accept the policy class it should use at construction and initialize its own parameters with it, so no module reimplements an initialization scheme inline. A task model that composes these modules SHALL accept a policy class, default to the scheme it used before this change, and apply it across the model it composes.

#### Scenario: Initialization schemes live in one place
- **WHEN** the model layer is inspected for how modules choose their weight-initialization scheme
- **THEN** the schemes are defined once in the model layer's initialization module, and no attention, feed-forward, head, embedding, or task-model module contains an inline xavier-normal or truncated-normal initialization loop of its own

#### Scenario: Each scheme is a distinct class, not a string
- **WHEN** the model layer's initialization module is inspected for how a scheme is represented and selected
- **THEN** each scheme is a distinct concrete subclass of the abstract policy base class, and no string name, string type field, or string-keyed lookup is used to identify or select a scheme

#### Scenario: Each module initializes itself through the injected policy
- **WHEN** each parameterized model-layer module is constructed with its default initializer and its parameters are inspected without any external initialization call
- **THEN** that module's parameters follow the scheme of the policy class it received

#### Scenario: A non-default initializer overrides the scheme
- **WHEN** a parameterized module is constructed with a supplied non-default initializer and its parameters are inspected
- **THEN** the module's parameters follow the supplied initializer's scheme instead of its default scheme

#### Scenario: Each module's pre-change default scheme is preserved
- **WHEN** attention architectures, the classification head, the regression head, the feed-forward variants, the task embeddings, and the fuzzy-logic task model are each constructed without an explicit initializer, as before this change
- **THEN** attention and the classification head use xavier-normal weights with zero biases, the fuzzy-logic task model uses the reference truncated-normal scheme across its composed modules, and the regression head, feed-forward variants, task embeddings, and encoder normalization parameters keep PyTorch's defaults

#### Scenario: Task models apply the policy across their composition
- **WHEN** the fuzzy-logic task model and the Match3 task model are each constructed with their default initializer
- **THEN** the fuzzy-logic task model applies the reference scheme across its composed sub-modules, and the Match3 task model leaves each sub-module's own scheme unchanged
