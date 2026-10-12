# Spec Delta

## ADDED Requirements

### Requirement: Centralized weight initialization
The model layer SHALL expose a single weight-initialization policy that owns the initialization schemes the model layer uses, so that no module reimplements an initialization scheme inline. Each parameterized model-layer module — attention architectures, feed-forward variants, prediction heads, task embeddings, and the encoder layer's normalization parameters — SHALL accept an initializer at construction and initialize its own parameters with it. A task model that composes these modules SHALL accept an initializer, default to the scheme it used before this change, and apply it across the model it composes.

#### Scenario: Initialization schemes live in one place
- **WHEN** the model layer is inspected for how modules choose their weight-initialization scheme
- **THEN** the schemes are defined once in the model layer's initialization module, and no attention, feed-forward, head, embedding, or task-model module contains an inline xavier-normal or truncated-normal initialization loop of its own

#### Scenario: Each module initializes itself through the injected policy
- **WHEN** each parameterized model-layer module is constructed with its default initializer and its parameters are inspected without any external initialization call
- **THEN** that module's parameters follow the scheme associated with the initializer it received

#### Scenario: A non-default initializer overrides the scheme
- **WHEN** a parameterized module is constructed with a supplied non-default initializer and its parameters are inspected
- **THEN** the module's parameters follow the supplied initializer's scheme instead of its default scheme

#### Scenario: Each module's pre-change default scheme is preserved
- **WHEN** attention architectures, the classification head, the regression head, the feed-forward variants, the task embeddings, and the fuzzy-logic task model are each constructed without an explicit initializer, as before this change
- **THEN** attention and the classification head use xavier-normal weights with zero biases, the fuzzy-logic task model uses the reference truncated-normal scheme across its composed modules, and the regression head, feed-forward variants, task embeddings, and encoder normalization parameters keep PyTorch's defaults

#### Scenario: Task models apply the policy across their composition
- **WHEN** the fuzzy-logic task model and the Match3 task model are each constructed with their default initializer
- **THEN** the fuzzy-logic task model applies the reference scheme across its composed sub-modules, and the Match3 task model leaves each sub-module's own scheme unchanged

## MODIFIED Requirements

### Requirement: Attention architectures initialize their own weights
The model layer SHALL initialize every attention architecture's weights at construction using the shared weight-initialization policy — by default xavier-normal on its projection weights and zero on its biases — so a constructed attention module is ready to use without any external initialization call. An attention architecture SHALL accept the initialization policy to use at construction. The encoder layer SHALL NOT initialize or re-initialize a passed attention module.

#### Scenario: A constructed architecture is already initialized
- **WHEN** an attention architecture is constructed and its weights are inspected without any external initialization being called
- **THEN** its projection weights follow the shared policy's default xavier-normal scheme and its biases are zero

#### Scenario: A non-default policy changes the scheme
- **WHEN** an attention architecture is constructed with a supplied non-default initialization policy and its parameters are inspected
- **THEN** its projection parameters follow that policy's scheme instead of the default xavier-normal scheme

#### Scenario: Passing a constructed module does not re-initialize it
- **WHEN** an already-constructed attention module with known weights is passed to an encoder layer, and the encoder layer's construction is inspected
- **THEN** the encoder layer leaves those weights unchanged and performs no initialization of the passed module
