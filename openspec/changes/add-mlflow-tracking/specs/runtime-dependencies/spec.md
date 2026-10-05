# Spec Delta

## MODIFIED Requirements

### Requirement: Single pinned runtime manifest

The repository SHALL provide one root dependency manifest that lists every direct runtime dependency needed by the code package, the notebooks, the retained PyTorch experiment, and the experiment-tracking/training layers, with exact version pins using `==`. The manifest SHALL pin `torch` to the exact CUDA build present in the local environment and SHALL declare the package index required to resolve that build. The manifest SHALL include the experiment-tracking dependency (`mlflow`) as a direct, exactly pinned entry. The JAX experiment's former stack SHALL NOT appear in the root manifest.

#### Scenario: Direct dependencies are present and exactly pinned
- **WHEN** the root manifest is inspected
- **THEN** it contains exact `==` pins for `torch`, `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `ipython`, and `mlflow`, and no listed dependency uses an unpinned or range specifier

#### Scenario: The tracking dependency is pinned as a direct requirement
- **WHEN** the root manifest is searched for the experiment-tracking dependency
- **THEN** `mlflow` appears as a direct entry with an exact `==` pin and is not left as an unpinned or transitive-only dependency

#### Scenario: Torch is pinned to the installed CUDA build
- **WHEN** the `torch` entry and the manifest's configured indexes are inspected
- **THEN** `torch` is pinned to the `cu132` local build installed in the local environment, and the PyTorch CUDA wheel index needed to resolve it is declared

#### Scenario: JAX-only dependencies are excluded
- **WHEN** the root manifest is searched for the JAX experiment's stack
- **THEN** no `jax`, `flax`, `optax`, `chex`, `ml_collections`, `einops`, `wandb`, `tensorflow`, `tensorflow_datasets`, `tensorflow_text`, `sentencepiece`, or `datasets` entries are present
