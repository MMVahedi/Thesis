# model-layer Specification

## Purpose

Governs how the model layer inside the code package is organized so new attention architectures and new task models can be added without cross-editing layers: one task-agnostic architecture registry, reusable encoder/head scaffolding, and thin task-specific model files.

## Requirements

### Requirement: Single task-agnostic architecture registry
The model layer SHALL expose all implemented attention architectures through exactly one registry that lives with the architectures themselves and is not owned by any task. Adding a new architecture SHALL require only adding its module and registering it in that one place; task-specific code and notebooks SHALL NOT contain their own mappings of architecture names to classes. The registry SHALL expose the softmax mixer, the linear mixer, and the retained high-order variants (strassen, triangular, third_order) under stable names; the previous `standard` name SHALL no longer be selectable.

#### Scenario: All implemented variants are registered
- **WHEN** the architecture registry is imported from the repository root
- **THEN** it exposes selectable names for softmax, linear, strassen, triangular, and third_order, and selecting each name constructs an attention module

#### Scenario: Retained high-order variants remain available
- **WHEN** the registry is inspected for the high-order variants
- **THEN** strassen, triangular, and third_order are still present and constructible, and `standard` is absent

#### Scenario: Old name is rejected
- **WHEN** a task model is constructed with `attention_type="standard"`
- **THEN** construction fails with an error naming the unsupported architecture

#### Scenario: No task-local architecture registries
- **WHEN** task-specific model code under the model layer is searched for architecture-name-to-class mappings defined locally in task files
- **THEN** zero such local mappings exist; task code obtains architecture classes only from the shared registry

### Requirement: Task models compose shared scaffolding
The encoder layer (attention block plus feed-forward block with optional normalization and residual connections) and the per-token classification head SHALL be importable, reusable modules that depend on no task-specific code. Each task model file SHALL contain only its task-specific input representation and thin composition of the shared pieces.

#### Scenario: Shared scaffolding is task-agnostic
- **WHEN** the shared encoder-layer and classification-head modules are imported from the repository root and their imports are inspected
- **THEN** neither imports any task-specific module, and either can be used without importing any task model

#### Scenario: Task file is thin glue
- **WHEN** a task model file under the model layer is inspected
- **THEN** it defines (or imports) only its task-specific embedding and composes the shared encoder layer, shared head, and a registered architecture — with no duplicated encoder or head definitions

### Requirement: Any registered architecture is usable by a task model
A task model SHALL accept any architecture name present in the shared registry and SHALL reject only names absent from it. Which architectures a particular study compares SHALL be decided by the calling notebook or experiment configuration, not restricted inside the task model.

#### Scenario: Every registered variant constructs
- **WHEN** the Match3 task model is constructed once with each name in the shared registry
- **THEN** every construction succeeds and the resulting model exposes the selected architecture in its encoder stack

#### Scenario: Unknown architecture name is rejected
- **WHEN** the Match3 task model is constructed with a name not present in the shared registry
- **THEN** construction fails with an error naming the unsupported architecture

### Requirement: Public task API stays stable
The Match3 task model's public interface SHALL remain unchanged across this reorganization: its constructor parameters, its output on forward, and its input contract (the batch mapping produced by the dataset layer's collate function). Because the architecture-name rename is a deliberate breaking change to selectable names, existing notebooks SHALL be migrated from `standard` to `softmax` as part of this change; the notebook code that imports the task model, constructs it, and consumes its output is otherwise unchanged.

#### Scenario: Notebook import and usage unchanged
- **WHEN** a migrated notebook's import of the Match3 task model and its construction with softmax or strassen attention are executed from the repository root
- **THEN** the import resolves, construction succeeds with the same parameters as before, and forward over a batch from the dataset layer produces the same output structure (probabilities and final hidden states) as before

### Requirement: Shared configurable attention interface
All attention architectures in the registry SHALL be built on one shared configuration interface that exposes the number of attention heads and independently configurable query/key and value head dimensions. When only a single head dimension is supplied, the query/key and value head dimensions SHALL default to the same value; when distinct values are supplied, the architecture SHALL use them independently. Supplying only a hidden dimension and a head count SHALL reproduce the previous single-head-dimension behavior.

#### Scenario: Head count is configurable for every architecture
- **WHEN** each registered architecture is constructed with a given head count and hidden dimension
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
The registry SHALL expose two sibling mixers that share the same projection, scaling, masking, and multi-head splitting: a softmax attention whose attention weights are normalized with softmax, and a linear attention whose attention weights are the scaled scores without the final softmax normalization.

#### Scenario: Linear is softmax without the final normalization
- **WHEN** the softmax and linear architectures are constructed with identical configuration and applied to the same input, and the softmax mixer's normalization is removed
- **THEN** the linear mixer's output matches the unnormalized softmax path

#### Scenario: Both mixers are selectable
- **WHEN** the registry is imported from the repository root
- **THEN** both `softmax` and `linear` are selectable names that construct attention modules

### Requirement: Documentation matches the model-layer layout
`AGENTS.md` and the root `README.md` SHALL describe the model layer's organization: where architectures and their registry live, where shared scaffolding lives, and the convention for adding a task (task-specific embedding plus thin task file). Every model-layer path they document SHALL exist.

#### Scenario: Documented model-layer paths exist
- **WHEN** every repository-relative path mentioned for the model layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form, and the add-a-task convention matches the actual directory structure

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