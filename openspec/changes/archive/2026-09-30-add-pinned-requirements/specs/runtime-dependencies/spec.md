# Spec Delta

## Purpose

Defines the contract for the project's local runtime environment: one pinned root dependency manifest, anchored to the host's GPU driver, CUDA runtime, and torch build, that is installable, conflict-free, and documented — so the GPU environment can be rebuilt deterministically instead of by hand.

## ADDED Requirements

### Requirement: Single pinned runtime manifest

The repository SHALL provide one root dependency manifest that lists every direct runtime dependency needed by the code package, the notebooks, and the PyTorch experiment, with exact version pins using `==`. The manifest SHALL pin `torch` to the exact CUDA build present in the local environment and SHALL declare the package index required to resolve that build. Dependencies belonging only to the isolated JAX experiment SHALL NOT appear in the root manifest.

#### Scenario: Direct dependencies are present and exactly pinned
- **WHEN** the root manifest is inspected
- **THEN** it contains exact `==` pins for `torch`, `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, and `ipython`, and no listed dependency uses an unpinned or range specifier

#### Scenario: Torch is pinned to the installed CUDA build
- **WHEN** the `torch` entry and the manifest's configured indexes are inspected
- **THEN** `torch` is pinned to the `cu132` local build installed in the local environment, and the PyTorch CUDA wheel index needed to resolve it is declared

#### Scenario: JAX-only dependencies are excluded
- **WHEN** the root manifest is searched for the JAX experiment's stack
- **THEN** no `jax`, `flax`, `optax`, `chex`, `ml_collections`, `einops`, `wandb`, `tensorflow`, `tensorflow_datasets`, `tensorflow_text`, `sentencepiece`, or `datasets` entries are present

### Requirement: Manifest is installable in the local environment

The pinned manifest SHALL install into the local virtual environment without changing the already-installed torch or numpy versions, and the resulting environment SHALL import every listed dependency and expose GPU acceleration on the host.

#### Scenario: Install succeeds without disturbing the anchored packages
- **WHEN** the manifest is installed into the local virtual environment
- **THEN** installation completes successfully and the installed `torch` and `numpy` remain at the versions they had before installation

#### Scenario: GPU smoke check passes
- **WHEN** each listed dependency is imported in the local virtual environment and `torch.cuda.is_available()` and `torch.cuda.get_arch_list()` are evaluated
- **THEN** all imports succeed, `torch.cuda.is_available()` is `True`, and the compiled architecture list contains the host GPU's `sm_75` capability

#### Scenario: No broken dependency relationships
- **WHEN** the environment is checked for dependency conflicts after installation
- **THEN** no broken or unsatisfied requirements are reported

### Requirement: Documentation reflects the dependency manifest

`AGENTS.md` SHALL describe the root dependency manifest and the Python and torch versions it targets, and SHALL NOT state that no root requirements file exists. Versions and paths it documents SHALL match the manifest and the local environment.

#### Scenario: AGENTS documents the manifest
- **WHEN** `AGENTS.md` is inspected
- **THEN** it references the root dependency manifest and no longer claims that the repository has no root requirements file

#### Scenario: Documented versions match the manifest
- **WHEN** the Python and torch versions stated in `AGENTS.md` are compared with the manifest and the local environment
- **THEN** they agree

### Requirement: Experiment isolation is preserved

The root manifest SHALL NOT become a dependency of the isolated experiment directories and SHALL NOT alter their existing requirements files. Each experiment SHALL remain independently runnable from its own directory and its own declared dependencies.

#### Scenario: Existing experiment requirements are untouched
- **WHEN** the requirements files inside each experiment directory are inspected after this change
- **THEN** their contents are unchanged by this change

#### Scenario: JAX experiment stays self-contained
- **WHEN** the JAX experiment's requirements file is inspected
- **THEN** it still declares its own full stack and the experiment does not rely on the root manifest to run
