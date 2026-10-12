# Spec Delta

## REMOVED Requirements

### Requirement: Attention is constructed directly by callers
**Reason**: Superseded by typed-enum, config-driven model construction. The model layer now intentionally exposes architecture selection and a construction helper (the model factory), so a requirement forbidding any name-to-architecture mapping or construction helper no longer holds. Selection stays typed (an enum), so raw strings remain banned.

**Migration**: Declare the architecture with the model-layer `AttentionArchitecture` enum inside a model config and obtain the constructed model from the factory. Callers that bypass the factory may still import a concrete attention class and instantiate it directly; no string-keyed lookup is reintroduced.

## MODIFIED Requirements

### Requirement: Feed-forward variants are constructible modules
The model layer SHALL provide the residual ReLU-stack feed-forward and the two-projection GELU feed-forward as constructible modules. The model factory SHALL select the variant from a typed `FeedForwardVariant` enum carried in a model config and construct the corresponding module, which the task model passes into each encoder layer. The model layer SHALL NOT provide a raw-string-keyed feed-forward factory, and there SHALL be no feed-forward variant-name (string) mapping or lookup. Direct construction of a feed-forward class by an explicit caller SHALL remain possible.

#### Scenario: Both variants forward at unchanged width
- **WHEN** a feed-forward module is constructed for each supported variant and applied to a hidden state
- **THEN** each returns a tensor of the same shape as its input

#### Scenario: Task model constructs the feed-forward directly
- **WHEN** a model is built through the factory with a feed-forward variant selected by a `FeedForwardVariant` enum member and its encoder stack is inspected
- **THEN** each encoder layer's feed-forward is a directly instantiated module of that variant, constructed without a string-keyed name-to-module factory

#### Scenario: No feed-forward factory is exposed
- **WHEN** the model layer's feed-forward module is searched for a string-keyed name-to-module construction function
- **THEN** no such string-keyed factory exists and no feed-forward variant is selected by a raw string

### Requirement: Task models compose shared scaffolding
The encoder layer (attention block plus feed-forward block with optional normalization and residual connections) and the per-token classification head SHALL be importable, reusable modules that depend on no task-specific code. Each task model file SHALL contain only its task-specific input representation and thin composition of the shared pieces. A task model SHALL accept caller-supplied, already-constructed attention and feed-forward modules and SHALL pass those constructed modules into each of its encoder layers. The task model SHALL NOT resolve, select, validate, or construct an attention architecture or a feed-forward variant itself.

#### Scenario: Shared scaffolding is task-agnostic
- **WHEN** the shared encoder-layer and classification-head modules are imported from the repository root and their imports are inspected
- **THEN** neither imports any task-specific module, and either can be used without importing any task model

#### Scenario: Task file is thin glue
- **WHEN** a task model file under the model layer is inspected
- **THEN** it defines (or imports) only its task-specific embedding and composes the shared encoder layer and shared head, obtaining attention and feed-forward solely from the caller-supplied modules and defining no encoder or head of its own

#### Scenario: Task model constructs the attention it selected
- **WHEN** a task model is constructed with caller-supplied, already-constructed attention and feed-forward modules and the resulting encoder stack is inspected
- **THEN** each encoder layer's attention is that supplied attention module and its feed-forward is that supplied feed-forward module, and the task model's public construction surface takes attention and feed-forward as constructed modules rather than as names, variants, or factories

### Requirement: Fuzzy-logic regression task model
The model layer SHALL provide a fuzzy-logic regression task model that composes a task-specific input embedding, a configurable stack of encoder layers whose attention and feed-forward come from caller-supplied, already-constructed modules shared across the stack, and the regression head, and SHALL accept any attention and feed-forward module a caller supplies. The task-specific input embedding SHALL add a fixed absolute positional encoding to its projected token representations.

#### Scenario: Model reproduces the reference architecture
- **WHEN** the fuzzy-logic task model is constructed with a supplied attention module configured for the reference (8 heads, query/key and value width 16), a supplied GELU feed-forward module, and the reference task configuration (two layers, hidden width 128, feed-forward width 256, fixed absolute positional encoding)
- **THEN** it produces per-token scalar predictions and exposes one attention-weight matrix per layer

#### Scenario: Registered architectures are accepted
- **WHEN** the fuzzy-logic task model is constructed with supplied attention and feed-forward modules for each provided sequence-composable attention architecture and each supported feed-forward variant
- **THEN** every construction succeeds and the selected architecture and variant appear in the encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the fuzzy-logic task model is constructed with an architecture name string instead of an attention module
- **THEN** construction fails because the model takes attention as a caller-supplied module, not an architecture name

#### Scenario: Position enters once through the input embedding
- **WHEN** the fuzzy-logic task model processes a batch
- **THEN** a fixed position signal is added to the embedded tokens before the encoder stack, and no relative-position bias is added to attention scores

### Requirement: Documentation matches the model-layer layout
`AGENTS.md` and the root `README.md` SHALL describe the model layer's organization: where attention architectures and their individual classes live, where shared scaffolding lives, and the convention for adding a task (task-specific embedding plus thin task file plus a config/factory branch), including that the model layer exposes a config-driven model factory whose architecture and feed-forward selection is by typed enum (never a raw string). Every model-layer path they document SHALL exist.

#### Scenario: Documented model-layer paths exist
- **WHEN** every repository-relative path mentioned for the model layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form, and the add-a-task convention matches the actual directory structure

#### Scenario: Documentation describes direct attention construction
- **WHEN** `AGENTS.md` and the root `README.md` are inspected for how attention and feed-forward are selected
- **THEN** they describe a config-driven model factory selecting architectures by typed enum, and mention no string-keyed architecture registry or raw-string construction helper

## ADDED Requirements

### Requirement: Config-driven model factory
The model layer SHALL provide a model factory that accepts a single configuration object and returns a fully constructed task model, without the caller constructing the attention module, feed-forward module, embedding, or head. The factory SHALL construct the attention module and feed-forward module from the configuration, build the task-specific embedding and head, apply the task model's single initialization policy, and return the composed task model. The factory SHALL support the Match3 classifier task model and the fuzzy-logic regression task model. Adding a task SHALL require adding its configuration type and its factory branch rather than editing the existing task branches.

#### Scenario: Factory builds a Match3 model
- **WHEN** the factory is given a Match3 configuration
- **THEN** it returns a constructed Match3 task model whose encoder layers use the configured attention and feed-forward

#### Scenario: Factory builds a fuzzy-logic model
- **WHEN** the factory is given a fuzzy-logic configuration
- **THEN** it returns a constructed fuzzy-logic task model whose encoder layers use the configured attention and feed-forward

#### Scenario: Returned model is ready to run
- **WHEN** the factory returns a model
- **THEN** the model is a `torch.nn.Module` with all sub-modules constructed and initialized, usable in a forward pass without the caller supplying any sub-module

#### Scenario: Factory construction matches direct construction
- **WHEN** a model is built through the factory from a configuration and the same architecture is constructed directly with equivalent arguments
- **THEN** the two models expose the same architecture configuration (stack depth, widths, head count, attention architecture, feed-forward variant)

### Requirement: Typed enum architecture selection
Attention architecture selection and feed-forward variant selection in a model config SHALL use typed enum members defined by the model layer (`AttentionArchitecture` and `FeedForwardVariant`). The factory SHALL map each enum member to its concrete class. A raw string SHALL NOT be accepted or used to select an attention architecture or a feed-forward variant anywhere in the model layer.

#### Scenario: Each attention architecture is selectable by enum
- **WHEN** the factory is given a configuration selecting each supported sequence-composable attention architecture by enum member
- **THEN** construction succeeds and the resulting model's encoder stack uses that attention architecture

#### Scenario: Raw string is not accepted as a selector
- **WHEN** a configuration or factory call supplies an attention architecture or feed-forward variant as a raw string instead of an enum member
- **THEN** selection fails rather than silently resolving a string name

#### Scenario: Each feed-forward variant is selectable by enum
- **WHEN** the factory is given a configuration selecting each supported feed-forward variant by enum member
- **THEN** construction succeeds and the resulting model's encoder layers use that variant

### Requirement: Per-task model configuration
The model layer SHALL provide a configuration type for each task model it builds (a Match3 configuration and a fuzzy-logic configuration). Each configuration SHALL expose the model's variation knobs: the attention architecture and head dimensions (head count plus independently configurable query/key and value head dimensions), the feed-forward variant with its width or depth, the stack size and shape (number of layers and hidden width), regularization and layer-sharing settings (feed-forward dropout, normalization, layer sharing where the task supports it), and the task model's initialization policy. The configuration's type SHALL determine which task model is built; no separate task-type field SHALL be required.

#### Scenario: Knobs drive the built model
- **WHEN** a configuration supplies a value for each of its variation knobs and the model is built through the factory
- **THEN** the resulting model reflects every supplied value (stack depth and width, attention architecture and head dimensions, feed-forward variant and width/depth, dropout, normalization, layer sharing)

#### Scenario: Configuration type selects the task
- **WHEN** a Match3 configuration and a fuzzy-logic configuration are each passed to the factory
- **THEN** the factory builds the corresponding task model without a task-type string being supplied

#### Scenario: Initialization policy is carried as a class
- **WHEN** a configuration supplies its initialization policy
- **THEN** the policy is a `WeightInitializer` subclass applied by the built model, and no string identifies or selects the scheme