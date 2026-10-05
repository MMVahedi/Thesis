# Spec Delta

## MODIFIED Requirements

### Requirement: Single top-level code package
All runnable library code (attention models, embeddings, tasks, dataset generators, torch datasets, losses, and runnable experiment entry points) SHALL live under exactly one top-level Python package. Within that package, the isolated experiment archives SHALL live under `experiments/` (`compgen/experiments/`) and SHALL NOT import the enclosing package, while runnable experiment entry points that compose the package SHALL live under `runners/` (`compgen/runners/`). The repository root SHALL NOT contain top-level `models/`, `datasets/`, `losses/`, `experiments/`, or `runners/` code directories. As a result, running Python from the repository root SHALL NOT shadow any identically named installed third-party package (notably HuggingFace `datasets`).

#### Scenario: Root run resolves third-party datasets
- **WHEN** a process is started from the repository root and `import datasets` executes in an environment where HuggingFace `datasets` is installed
- **THEN** the installed HuggingFace package is imported, not any local code

#### Scenario: All library modules reachable through the package
- **WHEN** every public module formerly under `models/` and `datasets/` is imported via the new package prefix (e.g. `compgen.models.attentions.softmax`, `compgen.datasets.generators.match3`) from the repository root
- **THEN** all imports succeed without a top-level `models/` or `datasets/` directory existing at the root

#### Scenario: Experiments relocated under the package
- **WHEN** the repository root is listed after the reorganization
- **THEN** no `experiments/` directory exists at the root, the retained experiment is found at `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/`, and no JAX experiment directory exists

#### Scenario: Package-composing runners have a defined home
- **WHEN** the repository is inspected for runnable experiment entry points that import the enclosing package
- **THEN** each such entry point is found under `compgen/runners/`, and no entry point that imports the enclosing package lives under `compgen/experiments/`
