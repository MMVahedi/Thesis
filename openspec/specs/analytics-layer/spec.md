# analytics-layer Specification

## Purpose

Gives the code package reusable, task-agnostic model-interpretability tooling: turning attention weights into per-example representation vectors, inspecting what a specific token attends to, projecting vectors to two dimensions (PCA), and optionally plotting the results — so interpretability analyses run directly on live model outputs for any task instead of living inside one sealed experiment.

## Requirements

### Requirement: Task-agnostic interpretation surface
The analytics layer SHALL operate on caller-supplied tensors — attention weights and representation vectors — and SHALL NOT import any task-specific model module. It SHALL be importable from the repository root under the package prefix, and the same functions SHALL apply to outputs of the fuzzy-logic task, the Match3 task, and any future task.

#### Scenario: Core functions take tensors, not models
- **WHEN** the attention-vector, per-token inspection, and projection functions are imported and their inputs inspected
- **THEN** each accepts plain tensors (attention weights or representation vectors) and none requires a task model instance or imports task-specific code

#### Scenario: The layer imports without a task model
- **WHEN** the analytics package is imported from the repository root without importing any task model
- **THEN** the import succeeds and the public functions are available

### Requirement: Attention-derived representation vectors
The analytics layer SHALL provide a function that reduces attention weights of shape `(B, H, N, N)` to per-example representation vectors, selecting a query token (defaulting to the last token) and combining the selected token's attention over key positions across heads, and SHALL preserve the per-head structure as separate features so the vector has shape `(B, H)` (or `(B, H * N)` when the full attention row per head is requested). It SHALL accept a single attention tensor or a stack of layers and, when given a stack, MAY concatenate each layer's contribution.

#### Scenario: A token's attention becomes one vector per example
- **WHEN** attention weights `(B, H, N, N)` are reduced for a chosen query token
- **THEN** the result has one vector per example whose length is determined by the head count (and, when requested, the key length)

#### Scenario: The default query token is the last position
- **WHEN** the reduction is called without an explicit query token
- **THEN** it uses the last sequence position as the query token

#### Scenario: Layers can be stacked
- **WHEN** a list of per-layer attention tensors is passed
- **THEN** each layer contributes its own features and the per-layer contributions can be distinguished in the result

### Requirement: Per-token attention inspection
The analytics layer SHALL provide a function that reads, for a chosen batch example, the attention distribution of a specific query token over the key positions of a given layer, returning either one row per attention head or a single row averaged over heads. Query-token, head, and layer selection SHALL be explicit arguments, and the returned rows SHALL be the original (possibly head-averaged) attention weights for that query token.

#### Scenario: One token's attention row is returned
- **WHEN** the function is called with a valid batch example, layer, and query token
- **THEN** it returns that token's attention over key positions, with one row per head by default and a single averaged row when averaging is requested

#### Scenario: Averaging over heads preserves the distribution
- **WHEN** head averaging is requested for a token whose per-head rows each sum to one
- **THEN** the averaged row also sums to one (within numerical tolerance)

#### Scenario: An out-of-range token index is rejected
- **WHEN** the function is called with a query token index outside the sequence length
- **THEN** it raises an error naming the invalid index rather than returning a silently wrong slice

### Requirement: Two-dimensional projection of representation vectors
The analytics layer SHALL provide a projection that fits a principal-component analysis on representation vectors and returns two-dimensional coordinates by default, together with the explained-variance ratio of the retained components. The component count SHALL be configurable. Projections SHALL be deterministic: the same input vectors and configuration SHALL produce the same coordinates, including a fixed sign convention for each component.

#### Scenario: Vectors project to two dimensions
- **WHEN** representation vectors of shape `(B, D)` are projected with the default configuration
- **THEN** the result contains `B` two-dimensional coordinates and the explained-variance ratio of the two components

#### Scenario: The component count is configurable
- **WHEN** a projection is requested with `k` components
- **THEN** the coordinates have `k` columns and the explained-variance ratio has `k` entries

#### Scenario: Repeated projections are identical
- **WHEN** the same vectors are projected twice with the same configuration
- **THEN** the two coordinate arrays are identical

#### Scenario: Fewer examples than requested components is rejected
- **WHEN** a projection requests more components than there are input vectors
- **THEN** it raises an error instead of returning degenerate coordinates

### Requirement: Model capture helper
The analytics layer SHALL provide a helper that runs a model over a batch in inference mode and captures its per-layer attention weights, so callers obtain attention for interpretation without task-specific code. The helper SHALL work for models composed of the shared attention modules (fuzzy-logic and Match3 alike), SHALL NOT modify model parameters, and SHALL return results that can be fed directly to the attention-vector and per-token inspection functions.

#### Scenario: Capture returns per-layer attention
- **WHEN** the helper is run over a batch for a model built from the shared attention modules
- **THEN** it returns the model's per-layer attention weights, each with shape `(B, H, N, N)`

#### Scenario: Capture works across tasks
- **WHEN** the helper is run for both the fuzzy-logic task model and the Match3 task model
- **THEN** both return captured attention weights without the caller supplying task-specific extraction code

#### Scenario: Capture does not train or mutate the model
- **WHEN** the helper runs over a batch
- **THEN** it performs inference only (dropout disabled) and leaves the model's parameters unchanged

### Requirement: Optional visualization helpers
The analytics layer SHALL provide plotting helpers that turn projected coordinates into a two-dimensional scatter figure and attention weights into a heatmap figure. The helpers SHALL accept optional per-point labels or values for coloring, SHALL return the figure objects, and SHALL NOT write to the filesystem.

#### Scenario: Coordinates produce a scatter figure
- **WHEN** two-dimensional coordinates are plotted, optionally with a coloring array
- **THEN** the helper returns a matplotlib figure containing one point per coordinate

#### Scenario: Attention produces a heatmap figure
- **WHEN** a token's attention row (or a per-head attention matrix) is plotted
- **THEN** the helper returns a matplotlib figure representing the attention values

#### Scenario: Plotting does not write files
- **WHEN** a plotting helper is called
- **THEN** it returns the figure and creates no file on disk

### Requirement: Documentation matches the analytics-layer layout
`README.md` and `AGENTS.md` SHALL describe the analytics layer: its purpose, where it lives, the modules it provides, and how it is imported and used. Every analytics-layer path they document SHALL exist.

#### Scenario: Documented analytics paths exist
- **WHEN** every repository-relative path mentioned for the analytics layer in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form and the described usage matches the actual module layout
