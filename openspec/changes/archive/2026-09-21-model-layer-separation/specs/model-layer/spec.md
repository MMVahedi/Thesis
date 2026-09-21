# Spec Delta

## Purpose

Governs how the model layer inside the code package is organized so new attention architectures and new task models can be added without cross-editing layers: one task-agnostic architecture registry, reusable encoder/head scaffolding, and thin task-specific model files.

## ADDED Requirements

### Requirement: Single task-agnostic architecture registry
The model layer SHALL expose all implemented attention architectures through exactly one registry that lives with the architectures themselves and is not owned by any task. Adding a new architecture SHALL require only adding its module and registering it in that one place; task-specific code and notebooks SHALL NOT contain their own mappings of architecture names to classes.

#### Scenario: All implemented variants are registered
- **WHEN** the architecture registry is imported from the repository root
- **THEN** it exposes selectable names for all four implemented variants (standard, strassen, triangular, third_order), and selecting each name constructs an attention module

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
The Match3 task model's public interface SHALL remain unchanged across this reorganization: its constructor parameters, its output on forward, and its input contract (the batch mapping produced by the dataset layer's collate function). Existing notebooks SHALL continue to run against the reorganized layer without modification.

#### Scenario: Notebook import and usage unchanged
- **WHEN** a notebook's existing import of the Match3 task model and its existing construction with standard or strassen attention are executed from the repository root
- **THEN** the import resolves, construction succeeds with the same parameters as before, and forward over a batch from the dataset layer produces the same output structure (probabilities and final hidden states) as before

### Requirement: Documentation matches the model-layer layout
`AGENTS.md` and the root `README.md` SHALL describe the model layer's organization: where architectures and their registry live, where shared scaffolding lives, and the convention for adding a task (task-specific embedding plus thin task file). Every model-layer path they document SHALL exist.

#### Scenario: Documented model-layer paths exist
- **WHEN** every repository-relative path mentioned for the model layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form, and the add-a-task convention matches the actual directory structure