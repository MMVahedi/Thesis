# model-layer Specification

## Purpose

Governs how the model layer inside the code package is organized so new attention architectures and new task models can be added without cross-editing layers: one task-agnostic architecture registry, reusable encoder/head scaffolding, and thin task-specific model files.

## Requirements

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

### Requirement: Shared configurable attention interface
All attention architectures in the model layer SHALL be built on one shared configuration interface that exposes the number of attention heads and independently configurable query/key and value head dimensions. When only a single head dimension is supplied, the query/key and value head dimensions SHALL default to the same value; when distinct values are supplied, the architecture SHALL use them independently. Supplying only a hidden dimension and a head count SHALL reproduce the previous single-head-dimension behavior.

#### Scenario: Head count is configurable for every architecture
- **WHEN** each attention architecture is constructed with a given head count and hidden dimension
- **THEN** construction succeeds and the resulting module reports that head count, splitting its query/key and value projections across that many heads

#### Scenario: Query/key and value head dimensions default to equal
- **WHEN** an architecture is constructed with a hidden dimension and head count but no explicit per-branch dimension
- **THEN** the query/key head dimension and value head dimension are equal, matching the previous behavior

#### Scenario: Query/key and value head dimensions are independently settable
- **WHEN** an architecture is constructed with distinct query/key and value head dimensions
- **THEN** construction succeeds and the resulting module uses those distinct dimensions per head

#### Scenario: Interface applies to softmax and linear alike
- **WHEN** the softmax and linear architectures are each constructed with a non-default head count and distinct query/key and value head dimensions
- **THEN** both constructions succeed and both expose the requested head and dimension configuration

### Requirement: Softmax and linear attention mixers
The model layer SHALL provide two sibling mixers that share the same projection, scaling, and multi-head splitting: a softmax attention whose attention weights are normalized with softmax, and a linear attention whose attention weights are the scaled scores without the final softmax normalization. Neither mixer SHALL apply masking or dropout to its weights.

#### Scenario: Linear is softmax without the final normalization
- **WHEN** the softmax and linear architectures are constructed with identical configuration and applied to the same input, and the softmax mixer's normalization is removed
- **THEN** the linear mixer's output matches the unnormalized softmax path

#### Scenario: Both mixers are selectable
- **WHEN** the softmax and linear attention classes are imported from the repository root and instantiated directly
- **THEN** both construct attention modules

### Requirement: Attention is a hidden-state-only transform
Every attention architecture in the model layer SHALL compute its output from the token hidden states alone. Attention `forward` SHALL accept the hidden states and the optional additive `score_bias`, and SHALL NOT accept a mask argument. Attention construction SHALL NOT accept a dropout rate, a dropout toggle, a projection-bias flag, or a padding-mask value. Attention weights SHALL be neither masked nor dropped out.

#### Scenario: No mask argument is accepted
- **WHEN** an attention architecture's `forward` is called with hidden states and an optional additive score bias
- **THEN** it returns the attention output and attention weights, and passing a batch-mask argument is rejected

#### Scenario: Removed configuration knobs are rejected
- **WHEN** any attention architecture is constructed with `dropout_rate`, `use_dropout`, `bias`, or `mask_padding_value`
- **THEN** construction fails because the parameter is not accepted

#### Scenario: No dropout is applied to attention weights
- **WHEN** an attention architecture runs in training mode with dropout once configured externally
- **THEN** the returned attention weights equal the unmasked, un-dropped normalization of its scores

#### Scenario: Head and dimension configuration is preserved
- **WHEN** each attention architecture is constructed with a hidden dimension, a head count, and optionally distinct query/key and value head dimensions
- **THEN** construction succeeds with the same head and dimension behavior as before the simplification

### Requirement: Task models consume an unmasked batch
The fuzzy-logic and Match3 task models SHALL construct their attention without dropout or bias configuration, and their forward paths SHALL NOT accept, read, or forward a batch mask. The Match3 task model SHALL NOT expose an attention-dropout constructor parameter; its feed-forward dropout SHALL remain configurable.

#### Scenario: Fuzzy-logic model forward takes no mask
- **WHEN** the fuzzy-logic task model's forward is called with inputs
- **THEN** it produces predictions and, on request, per-layer attention weights, without any mask being supplied or forwarded to attention

#### Scenario: Match3 model forward takes no mask
- **WHEN** the Match3 task model's forward is called with a collated batch
- **THEN** it produces per-token probabilities and final hidden states without reading a batch-mask entry from the batch

#### Scenario: Match3 attention-dropout parameter is gone
- **WHEN** the Match3 task model is constructed with `use_attention_dropout`
- **THEN** construction fails because the parameter is not accepted, while the feed-forward dropout rate remains configurable

#### Scenario: Attention-returning contract is unchanged
- **WHEN** the fuzzy-logic or Match3 task model is asked for attention weights
- **THEN** each still returns per-layer weights of shape `(batch, num_heads, length, length)`

### Requirement: Documentation matches the model-layer layout
`AGENTS.md` and the root `README.md` SHALL describe the model layer's organization: where attention architectures and their individual classes live, where shared scaffolding lives, and the convention for adding a task (task-specific embedding plus thin task file plus a config/factory branch), including that the model layer exposes a config-driven model factory whose architecture and feed-forward selection is by typed enum (never a raw string). Every model-layer path they document SHALL exist.

#### Scenario: Documented model-layer paths exist
- **WHEN** every repository-relative path mentioned for the model layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form, and the add-a-task convention matches the actual directory structure

#### Scenario: Documentation describes direct attention construction
- **WHEN** `AGENTS.md` and the root `README.md` are inspected for how attention and feed-forward are selected
- **THEN** they describe a config-driven model factory selecting architectures by typed enum, and mention no string-keyed architecture registry or raw-string construction helper

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

### Requirement: Attention accepts an additive score bias
The sequence-composable attention architectures provided by the model layer (softmax and linear) SHALL accept an optional additive score bias and add it to the scaled query/key scores before normalization, so a relative-position or other bias can be incorporated without reimplementing attention outside the model layer.

#### Scenario: Bias is added to the scores
- **WHEN** the same attention is applied with and without a non-zero additive bias and all other inputs are equal
- **THEN** the resulting attention weights differ and keep the same shape

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

### Requirement: Task models expose attention weights on demand
A task model in the model layer SHALL be able to return its per-layer attention weights on demand through its `forward` method, without changing its default output. `Match3Model.forward` SHALL accept an optional `return_attention` argument (default `False`); when `False` it SHALL return exactly the previous output, and when `True` it SHALL additionally return the attention weights of every encoder layer in order, each of shape `(batch, num_heads, length, length)`. This mirrors the fuzzy-logic task model's existing attention-returning path.

#### Scenario: Default output is unchanged
- **WHEN** the Match3 task model's `forward` is called without `return_attention` (or with it `False`)
- **THEN** it returns only the per-token probabilities and final hidden states, exactly as before

#### Scenario: Attention weights are returned on request
- **WHEN** the Match3 task model's `forward` is called with `return_attention=True`
- **THEN** it returns the per-token probabilities, the final hidden states, and one attention-weight tensor per encoder layer, each of shape `(batch, num_heads, length, length)`

#### Scenario: Fuzzy-logic and Match3 agree on the attention contract
- **WHEN** the fuzzy-logic task model and the Match3 task model are each asked for attention weights
- **THEN** both provide per-layer attention weights with the same `(batch, heads, query, key)` layout that the analytics layer consumes

### Requirement: Encoder layer consumes a constructed attention module
The shared encoder layer SHALL accept an already-constructed attention module and SHALL use that module as its self-attention. It SHALL NOT accept an attention architecture name, SHALL NOT look up or validate architecture names, and SHALL depend on no architecture registry. Constructing the attention module SHALL be the responsibility of the task model or other caller that composes the encoder layer.

#### Scenario: Constructed module becomes the layer's attention
- **WHEN** an encoder layer is constructed with a constructed attention module and applied to a hidden state
- **THEN** the layer uses that exact module for self-attention and returns a hidden state of the same shape as its input

#### Scenario: Encoder layer takes no architecture name
- **WHEN** the encoder layer's construction surface is inspected
- **THEN** it accepts no architecture-name string and performs no registry lookup or architecture-name validation

#### Scenario: Encoder layer is registry-independent
- **WHEN** the shared encoder-layer module is imported from the repository root and its imports are inspected
- **THEN** it imports neither any attention-architecture registry nor any task-specific module

### Requirement: Encoder layer consumes a constructed feed-forward module
The shared encoder layer SHALL accept an already-constructed feed-forward module and SHALL use that module as its feed-forward sublayer. It SHALL NOT accept a feed-forward variant name, depth, or width configuration, SHALL NOT select or validate a feed-forward variant, and SHALL depend on no feed-forward registry. Constructing the feed-forward module from the selected variant and configuration SHALL be the responsibility of the task model or other caller that composes the encoder layer.

#### Scenario: Constructed module becomes the layer's feed-forward
- **WHEN** an encoder layer is constructed with a constructed feed-forward module and applied to a hidden state
- **THEN** the layer uses that module for its feed-forward sublayer and returns a hidden state of the same shape as its input

#### Scenario: Encoder layer takes no feed-forward variant configuration
- **WHEN** the encoder layer's construction surface is inspected
- **THEN** it accepts no feed-forward variant name, depth, or width parameter and performs no feed-forward variant selection or validation

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

### Requirement: Encoder layer applies no dropout
The encoder layer SHALL apply no dropout to its residual paths: its attention and feed-forward residual connections SHALL add the sub-module outputs directly. Any dropout SHALL live inside the sub-modules the layer is given (the attention module and the feed-forward block), and the feed-forward block's dropout SHALL remain configurable through its construction.

#### Scenario: Residual connections add without dropout
- **WHEN** an encoder layer with constructed sub-modules processes a hidden state in training mode
- **THEN** its residual connections add the attention and feed-forward outputs without applying any dropout

#### Scenario: Feed-forward dropout lives in the block
- **WHEN** a feed-forward block is constructed with a dropout rate and handed to the encoder layer
- **THEN** that dropout is applied inside the block and the encoder layer holds no dropout module

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

### Requirement: Classification head returns probabilities
The per-token binary classification head SHALL apply a sigmoid to its single output logit and return per-token probabilities bounded within the closed interval `[0, 1]`, rather than raw logits. (Sigmoid is mathematically interior to `(0, 1)`; in floating point it may round to the endpoints for large-magnitude logits.) The sigmoid SHALL be part of the classification head and SHALL NOT be applied again by any task model that composes it. Because the head exposes probabilities and not logits, binary-classification training SHALL consume the head output with a binary cross-entropy objective that expects probabilities. The scalar regression head SHALL remain unchanged and SHALL NOT apply a sigmoid to its output.

#### Scenario: Classification head emits probabilities
- **WHEN** the per-token classification head is applied to hidden states of shape `(batch, length, hidden)`
- **THEN** it returns a tensor of shape `(batch, length, 1)` whose values all lie within the closed interval `[0, 1]`

#### Scenario: Task model applies no additional sigmoid
- **WHEN** the Match3 task model's `forward` is called with a collated batch
- **THEN** its returned per-token predictions lie within `[0, 1]` and equal the classification head's output applied to the final hidden states, with no extra normalization applied in the task model

#### Scenario: Regression head is unaffected
- **WHEN** the scalar regression head is applied to hidden states
- **THEN** it returns one scalar per token whose values are not constrained to `(0, 1)`

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
