# Spec Delta

## MODIFIED Requirements

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

### Requirement: Public task API stays stable
The Match3 task model's public interface SHALL remain unchanged across this reorganization: its constructor parameters, its output on forward, and its input contract (the batch mapping produced by the dataset layer's collate function). Because the architecture-name rename is a deliberate breaking change to selectable names, existing notebooks SHALL be migrated from `standard` to `softmax` as part of this change; the notebook code that imports the task model, constructs it, and consumes its output is otherwise unchanged.

#### Scenario: Notebook import and usage unchanged
- **WHEN** a migrated notebook's import of the Match3 task model and its construction with softmax or strassen attention are executed from the repository root
- **THEN** the import resolves, construction succeeds with the same parameters as before, and forward over a batch from the dataset layer produces the same output structure (probabilities and final hidden states) as before

## ADDED Requirements

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
