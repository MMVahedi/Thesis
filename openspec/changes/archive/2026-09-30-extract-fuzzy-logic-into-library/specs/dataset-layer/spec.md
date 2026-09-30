# Spec Delta

## Purpose

Defines how datasets are produced and loaded in the code package: an offline two-layer convention (torch-free generators writing JSON Lines files, torch readers loading them) plus an on-the-fly torch dataset for the fuzzy-logic task that materializes batches directly with the reference train/test/ood function splits.

## ADDED Requirements

### Requirement: Offline generation and torch loading are separate layers
Dataset generation and torch loading SHALL remain separate layers: generators under `compgen/datasets/generators/` SHALL be torch-free and SHALL serialize examples to JSON Lines files, and readers under `compgen/datasets/torch_datasets/` SHALL only read those JSON Lines files into `torch.utils.data.Dataset` objects without containing generation logic.

#### Scenario: Generators do not import torch
- **WHEN** the modules under `compgen/datasets/generators/` are inspected
- **THEN** none of them imports `torch` or any torch submodule

#### Scenario: Readers contain no generation logic
- **WHEN** the modules under `compgen/datasets/torch_datasets/` are inspected
- **THEN** each exposes a `torch.utils.data.Dataset` that reads a `.jsonl` file produced by a generator

### Requirement: On-the-fly fuzzy-logic dataset with function splits
The dataset layer SHALL provide a torch-facing on-the-fly fuzzy-logic dataset that samples fresh batches without an intermediate JSON Lines file, partitioning the task's functions into train, test, in-distribution (`id`), and out-of-distribution (`ood`) pools using the reference 50/50 held-out in-distribution split and 25% held-out conjunctions for OOD.

#### Scenario: All four function pools are available and non-empty
- **WHEN** the on-the-fly dataset is constructed for a supported task configuration
- **THEN** it exposes train, test, id, and ood function pools, and each pool is non-empty

#### Scenario: Batches carry inputs, targets, latents, and term identity
- **WHEN** a batch is sampled for a split
- **THEN** it returns inputs of shape `(batch, sequence_length, num_variables + 1)`, regression targets, the sampled formulas ("latents"), their term identifiers, and a per-example baseline mean-squared error

#### Scenario: Sampling is deterministic for a given seed
- **WHEN** the same split, batch index, and seed are sampled twice
- **THEN** the two sampled batches are identical

#### Scenario: Insufficient task configurations are rejected
- **WHEN** a task configuration cannot produce non-empty train, test, and OOD function pools
- **THEN** construction fails with an error stating that the configuration is too small

### Requirement: Supported fuzzy-logic task configurations
The on-the-fly fuzzy-logic dataset SHALL support the reference task configurations: 3 variables with 2 terms, 4 variables with 2 terms, 4 variables with 3 terms, and 5 variables with 2 terms.

#### Scenario: Every reference configuration constructs
- **WHEN** the on-the-fly dataset is constructed for each reference task configuration
- **THEN** construction succeeds
