# Spec Delta

## MODIFIED Requirements

### Requirement: Single pinned runtime manifest

The repository SHALL provide one root dependency manifest that lists every direct runtime dependency needed by the code package, the notebooks, and the retained PyTorch experiment, with exact version pins using `==`. The manifest SHALL pin `torch` to the exact CUDA build present in the local environment and SHALL declare the package index required to resolve that build. The JAX experiment's former stack SHALL NOT appear in the root manifest.

#### Scenario: Direct dependencies are present and exactly pinned
- **WHEN** the root manifest is inspected
- **THEN** it contains exact `==` pins for `torch`, `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, and `ipython`, and no listed dependency uses an unpinned or range specifier

#### Scenario: Torch is pinned to the installed CUDA build
- **WHEN** the `torch` entry and the manifest's configured indexes are inspected
- **THEN** `torch` is pinned to the `cu132` local build installed in the local environment, and the PyTorch CUDA wheel index needed to resolve it is declared

#### Scenario: JAX-only dependencies are excluded
- **WHEN** the root manifest is searched for the JAX experiment's stack
- **THEN** no `jax`, `flax`, `optax`, `chex`, `ml_collections`, `einops`, `wandb`, `tensorflow`, `tensorflow_datasets`, `tensorflow_text`, `sentencepiece`, or `datasets` entries are present

## REMOVED Requirements

### Requirement: Experiment isolation is preserved

**Reason**: One of this requirement's scenarios asserted that the JAX experiment stayed self-contained; that experiment is removed, so the requirement is restated for the single retained experiment.
**Migration**: Use the added requirement "Experiment dependencies stay isolated".

## ADDED Requirements

### Requirement: Experiment dependencies stay isolated

The root manifest SHALL NOT become a dependency of the retained experiment directory and SHALL NOT alter its existing requirements files. The experiment SHALL remain independently runnable from its own directory and its own declared dependencies.

#### Scenario: Existing experiment requirements are untouched
- **WHEN** the requirements files inside the retained experiment directory are inspected after this change
- **THEN** their contents are unchanged by this change

#### Scenario: Retained experiment ships its own requirements
- **WHEN** the retained experiment's requirements file is inspected
- **THEN** it still declares its own full stack and the experiment does not rely on the root manifest to run
