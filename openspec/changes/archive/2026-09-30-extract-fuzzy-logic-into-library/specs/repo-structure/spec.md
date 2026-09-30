# Spec Delta

## MODIFIED Requirements

### Requirement: Single top-level code package
All runnable library code (attention models, embeddings, tasks, dataset generators, torch datasets, losses) SHALL live under exactly one top-level Python package, the experiment suite SHALL live under `experiments/` within that package (`compgen/experiments/`), and the repository root SHALL NOT contain top-level `models/`, `datasets/`, `losses/`, or `experiments/` code directories. As a result, running Python from the repository root SHALL NOT shadow any identically named installed third-party package (notably HuggingFace `datasets`).

#### Scenario: Root run resolves third-party datasets
- **WHEN** a process is started from the repository root and `import datasets` executes in an environment where HuggingFace `datasets` is installed
- **THEN** the installed HuggingFace package is imported, not any local code

#### Scenario: All library modules reachable through the package
- **WHEN** every public module formerly under `models/` and `datasets/` is imported via the new package prefix (e.g. `compgen.models.attentions.softmax`, `compgen.datasets.generators.match3`) from the repository root
- **THEN** all imports succeed without a top-level `models/` or `datasets/` directory existing at the root

#### Scenario: Experiments relocated under the package
- **WHEN** the repository root is listed after the reorganization
- **THEN** no `experiments/` directory exists at the root, the retained experiment is found at `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/`, and no JAX experiment directory exists

### Requirement: Documentation matches the layout
`AGENTS.md` and the root `README.md` SHALL accurately describe the actual repository layout, import style, and data flow; every path they document SHALL exist, and no stale references (such as pointing at `dataset/generators/match3.py`, referencing the removed JAX experiment, or misattributing file locations) SHALL remain.

#### Scenario: Documented paths exist
- **WHEN** every repository-relative path mentioned in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form

#### Scenario: Stale module references eliminated
- **WHEN** docstrings and documentation are searched for the outdated `dataset/generators/` path and for imports not matching the new package prefix
- **THEN** zero stale references remain, and the formerly misattributed docstring points at the module's real location

#### Scenario: Removed experiment is not referenced
- **WHEN** `AGENTS.md`, the root `README.md`, and the retained experiment documentation are searched for references to the removed JAX experiment or its files
- **THEN** no such references remain

## REMOVED Requirements

### Requirement: Experiments remain self-contained
**Reason**: The PyTorch experiment loses its bundled training harness, self-test, and notebook generator as part of the library extraction, so its self-check no longer exists and the original requirement's scenarios no longer describe the retained content. The isolation contract is restated for the analysis-only experiment.
**Migration**: Use the added requirement "Retained experiment stays self-contained"; no external consumer depended on the removed self-test.

## ADDED Requirements

### Requirement: Retained experiment stays self-contained
The retained experiment directory under `compgen/experiments/` SHALL stand alone: it MUST NOT import from the enclosing package or any root code, so it remains portable on its own. Uploadable archives (`.zip`) SHALL NOT be committed; where an archive is still produced, it SHALL be reproducible on demand from the experiment directory via the documented build command and SHALL not depend on code outside that directory.

#### Scenario: Experiment isolation preserved
- **WHEN** all `.py` files and notebooks under `compgen/experiments/` are searched for imports of the enclosing package or of root `models`/`datasets`/`losses`
- **THEN** zero matches are found

#### Scenario: No committed upload archives
- **WHEN** tracked files are listed (`git ls-files`) and searched for `.zip` entries
- **THEN** zero upload archives are tracked

#### Scenario: Retained experiment files are a self-contained analysis archive
- **WHEN** the retained `.py` files in the PyTorch experiment directory are inspected
- **THEN** they contain only self-contained analysis/plotting code, and the removed model, data, loss, training, self-test, and notebook-generation modules are absent
