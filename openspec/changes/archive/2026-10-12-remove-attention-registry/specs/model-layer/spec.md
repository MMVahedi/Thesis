# Spec Delta

## REMOVED Requirements

### Requirement: Single task-agnostic architecture registry
**Reason**: The name→class registry and the name→module builder are removed so that architecture selection is explicit: whoever needs an attention module imports the concrete class and constructs the object directly. There is no longer a single registry to own, extend, or validate against.
**Migration**: Import the concrete attention class (for example `compgen.models.attentions.softmax.SoftmaxAttention` or `compgen.models.attentions.linear.LinearAttention`) and instantiate it with the configuration you need instead of resolving a string name through the registry or the removed `build_attention` helper. The replacement behavior is defined by the added "Attention is constructed directly by callers" and "Task models accept a caller-supplied attention factory" requirements.

### Requirement: Any registered architecture is usable by a task model
**Reason**: A task model no longer accepts an architecture name and performs no registry lookup or name validation. It accepts a caller-supplied attention factory, so usability is over whatever attention the caller provides rather than over names present in a shared registry.
**Migration**: Construct task models with an attention factory that returns the desired attention module (for example `attention_factory=lambda: StrassenAttention(...)`) instead of passing `attention_type="<name>"`. The replacement behavior is defined by the added "Task models accept a caller-supplied attention factory" requirement.

## MODIFIED Requirements

### Requirement: Task models compose shared scaffolding
The encoder layer (attention block plus feed-forward block with optional normalization and residual connections) and the per-token classification head SHALL be importable, reusable modules that depend on no task-specific code. Each task model file SHALL contain only its task-specific input representation and thin composition of the shared pieces. A task model SHALL accept a caller-supplied, zero-argument attention factory, call it once per encoder layer to obtain a constructed attention module, and pass each constructed module into its encoder layer. The task model SHALL NOT resolve, select, or validate an attention architecture name itself.

#### Scenario: Shared scaffolding is task-agnostic
- **WHEN** the shared encoder-layer and classification-head modules are imported from the repository root and their imports are inspected
- **THEN** neither imports any task-specific module, and either can be used without importing any task model

#### Scenario: Task file is thin glue
- **WHEN** a task model file under the model layer is inspected
- **THEN** it defines (or imports) only its task-specific embedding and composes the shared encoder layer and shared head, obtaining attention solely from the caller-supplied factory and defining no encoder or head of its own

#### Scenario: Task model constructs the attention it selected
- **WHEN** a task model is constructed with a caller-supplied attention factory and the resulting encoder stack is inspected
- **THEN** each encoder layer's attention is a module produced by that factory, and the task model's public construction surface takes the attention as a factory rather than as an architecture name

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

### Requirement: Documentation matches the model-layer layout
`AGENTS.md` and the root `README.md` SHALL describe the model layer's organization: where attention architectures and their individual classes live, where shared scaffolding lives, and the convention for adding a task (task-specific embedding plus thin task file), including that callers construct attention directly with no name-to-architecture registry and no construction helper. Every model-layer path they document SHALL exist.

#### Scenario: Documented model-layer paths exist
- **WHEN** every repository-relative path mentioned for the model layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form, and the add-a-task convention matches the actual directory structure

#### Scenario: Documentation describes direct attention construction
- **WHEN** `AGENTS.md` and the root `README.md` are inspected for how attention is selected
- **THEN** they describe callers importing and constructing attention classes directly, and mention no name-to-architecture registry or attention construction helper

### Requirement: Attention accepts an additive score bias
The sequence-composable attention architectures provided by the model layer (softmax and linear) SHALL accept an optional additive score bias and add it to the scaled query/key scores before normalization, so a relative-position or other bias can be incorporated without reimplementing attention outside the model layer.

#### Scenario: Bias is added to the scores
- **WHEN** the same attention is applied with and without a non-zero additive bias and all other inputs are equal
- **THEN** the resulting attention weights differ and keep the same shape

### Requirement: Fuzzy-logic regression task model
The model layer SHALL provide a fuzzy-logic regression task model that composes a task-specific input embedding, a configurable stack of encoder layers whose attention comes from a caller-supplied attention factory, and the regression head, and SHALL accept any attention factory a caller supplies. The task-specific input embedding SHALL add a fixed absolute positional encoding to its projected token representations.

#### Scenario: Model reproduces the reference architecture
- **WHEN** the fuzzy-logic task model is constructed with the reference configuration (two layers, hidden width 128, feed-forward width 256, 8 heads, query/key and value width 16, fixed absolute positional encoding, GELU feed-forward)
- **THEN** it produces per-token scalar predictions and exposes one attention-weight matrix per layer

#### Scenario: Registered architectures are accepted
- **WHEN** the fuzzy-logic task model is constructed with an attention factory for each provided sequence-composable attention architecture
- **THEN** every construction succeeds and the selected architecture appears in the encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the fuzzy-logic task model is constructed with an architecture name string instead of an attention factory
- **THEN** construction fails because the model takes attention as a caller-supplied factory, not an architecture name

#### Scenario: Position enters once through the input embedding
- **WHEN** the fuzzy-logic task model processes a batch
- **THEN** a fixed position signal is added to the embedded tokens before the encoder stack, and no relative-position bias is added to attention scores

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

### Requirement: Attention architectures initialize their own weights
The model layer SHALL initialize every attention architecture's weights at construction — xavier-normal on its projection weights and zero on its biases — so a constructed attention module is ready to use without any external initialization call. The encoder layer SHALL NOT initialize or re-initialize a passed attention module.

#### Scenario: A constructed architecture is already initialized
- **WHEN** an attention architecture is constructed and its weights are inspected without any external initialization being called
- **THEN** its projection weights follow the xavier-normal scheme and its biases are zero

#### Scenario: Passing a constructed module does not re-initialize it
- **WHEN** an already-constructed attention module with known weights is passed to an encoder layer, and the encoder layer's construction is inspected
- **THEN** the encoder layer leaves those weights unchanged and performs no initialization of the passed module

## ADDED Requirements

### Requirement: Attention is constructed directly by callers
The model layer SHALL NOT expose a name-to-architecture mapping or a name-to-constructed-module factory for attention. Every caller that needs an attention module SHALL import the concrete attention class and construct the object directly with the configuration it needs. Adding a new attention architecture SHALL require only adding its module and instantiating it where it is used; no central registry entry SHALL be required.

#### Scenario: No attention registry or builder exists
- **WHEN** the attention package is inspected for a name-to-architecture mapping or a name-to-module construction function
- **THEN** neither exists, and no attention module is selected by a string name anywhere in the model layer

#### Scenario: Callers construct attention classes directly
- **WHEN** a task model or a composing experiment needs an attention module
- **THEN** it imports the concrete attention class and instantiates the object, rather than resolving a string name

### Requirement: Task models accept a caller-supplied attention factory
A task model SHALL accept a zero-argument attention factory and SHALL obtain one attention module per encoder layer by calling it. The task model SHALL NOT restrict which attention may be supplied, SHALL NOT perform any name-based allow-list or validation, and SHALL use exactly the modules the factory produces. Which architectures a particular study compares SHALL be decided by the calling notebook or experiment configuration, not restricted inside the task model.

#### Scenario: Every caller-supplied attention module constructs
- **WHEN** the Match3 or fuzzy-logic task model is constructed with an attention factory that produces a given attention architecture
- **THEN** every construction succeeds and the resulting encoder stack exposes that attention architecture

#### Scenario: One module is constructed per encoder layer
- **WHEN** a task model with multiple encoder layers is constructed from a factory that returns a fresh module on each call
- **THEN** each encoder layer holds its own attention module produced by a separate factory call

#### Scenario: Task model performs no architecture validation
- **WHEN** a task model is constructed with an attention factory
- **THEN** the task model performs no name-based lookup or rejection and uses exactly the module the factory returns
