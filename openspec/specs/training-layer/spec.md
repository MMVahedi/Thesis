# training-layer Specification

## Purpose

Gives the code package a reusable, task-agnostic training and evaluation scaffold so new task models can be trained reproducibly — with a standard loop, evaluation, checkpoint selection, and run recording — instead of each implementation rewriting and re-instrumenting the loop.

## Requirements

### Requirement: Task-agnostic training engine
The training layer SHALL provide a generic engine that runs a configured number of epochs over caller-supplied training data and a caller-supplied step function that produces a loss (and optional step metrics). It SHALL accept configuration for the optimizer, an optional learning-rate scheduler, the epoch count, the compute device, optional gradient clipping, and a random seed, and SHALL NOT import any task-specific model or dataset code.

#### Scenario: Configured epochs run and produce a history
- **WHEN** the engine is run for a given number of epochs over a training data source
- **THEN** it performs that many epochs and returns a per-epoch history of at least the training loss

#### Scenario: The step function supplies the objective
- **WHEN** a caller-supplied step function returns a loss for a batch
- **THEN** the engine backpropagates that loss and updates the model parameters

#### Scenario: The engine imports without task code
- **WHEN** the training package is imported from the repository root
- **THEN** the import succeeds without importing any task model or dataset module

#### Scenario: A seeded run is reproducible
- **WHEN** two runs are launched with the same seed, data, and configuration
- **THEN** their recorded losses match

### Requirement: Evaluation and best-checkpoint selection
The engine SHALL evaluate on caller-supplied evaluation data at a configurable interval, track a nominated metric, and retain the best-performing checkpoint. It SHALL support optional early stopping when the tracked metric has not improved for a configured number of evaluations.

#### Scenario: Evaluation metrics are reported
- **WHEN** evaluation runs during training
- **THEN** the returned history and the recorded run include the evaluation metrics for the evaluated epochs

#### Scenario: The best checkpoint is retained
- **WHEN** evaluation metrics improve on some epochs and worsen on others
- **THEN** the checkpoint corresponding to the best tracked-metric value is the one kept as best

#### Scenario: Early stopping triggers
- **WHEN** the tracked metric fails to improve for the configured patience
- **THEN** training stops before the full epoch budget and reports that it stopped early

### Requirement: Runs are recorded through the tracking layer
The engine SHALL record each training run through the tracking layer: the run configuration and hyperparameters, the per-epoch training and evaluation metrics, and the best (or final) model artifact. Recording SHALL be disable-able so a quick local run can execute without creating a tracked run.

#### Scenario: Configuration is logged at the start of a run
- **WHEN** a tracked training run starts
- **THEN** its configuration and hyperparameters are recorded before training proceeds

#### Scenario: Per-epoch metrics are logged
- **WHEN** a tracked run completes an epoch
- **THEN** that epoch's training and evaluation metrics are recorded as stepped metrics

#### Scenario: The best model is logged
- **WHEN** a tracked run finishes
- **THEN** the best (or final) model is recorded as a model artifact in the run

#### Scenario: Recording can be disabled
- **WHEN** recording is disabled and the engine runs
- **THEN** no run is created and training still completes

### Requirement: Training documentation and conventions
The training layer SHALL include a minimal usage document at `compgen/training/README.md`, and `AGENTS.md` SHALL state that future training implementations use this engine and record their runs through `compgen.tracking`. Every path these documents reference SHALL exist.

#### Scenario: The usage document exists and its example runs
- **WHEN** the example in `compgen/training/README.md` is executed from the repository root
- **THEN** it completes a short training run without error

#### Scenario: AGENTS states the training convention
- **WHEN** `AGENTS.md` is inspected for the training layer
- **THEN** it names `compgen/training/` and instructs future training implementations to use it and log through `compgen.tracking`

#### Scenario: Documented training paths exist
- **WHEN** every repository-relative path mentioned for the training layer in `AGENTS.md`, `README.md`, and the layer's own README is checked against the filesystem
- **THEN** each path exists in the documented form
