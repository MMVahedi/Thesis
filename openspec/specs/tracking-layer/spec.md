# tracking-layer Specification

## Purpose

Gives the code package a single, task-agnostic way to record experiment runs with MLflow — parameters, metrics, artifacts, and models — defaulting to a local run store so results are captured reproducibly from the first training run without requiring an external service.

## Requirements

### Requirement: Task-agnostic tracking surface
The tracking layer SHALL expose one wrapper over MLflow that records caller-supplied configuration and results — parameter mappings, metric mappings, artifact paths, and PyTorch models — and SHALL NOT import any task-specific model or dataset code. It SHALL be importable from the repository root under the package prefix (`compgen.tracking.*`).

#### Scenario: Records caller-supplied values without knowing the task
- **WHEN** parameters, metrics, or artifacts are logged through the tracking layer
- **THEN** each supplied value is recorded in the run and the layer requires no knowledge of the task that produced it

#### Scenario: The layer imports without a task model
- **WHEN** the tracking package is imported from the repository root
- **THEN** the import succeeds without importing any task model or dataset module

### Requirement: Local-first backend with an override
The tracking layer SHALL default to a local SQLite MLflow store (`sqlite:///mlflow.db` in the working directory, storing run artifacts under `mlartifacts/`) and SHALL allow the backend to be overridden by an explicit tracking URI argument or the `MLFLOW_TRACKING_URI` environment variable. The explicit argument SHALL take precedence over the environment variable, and no tracking server SHALL be required to record a run.

#### Scenario: Runs record to a local store by default
- **WHEN** a run is started with no tracking URI configured
- **THEN** the run is recorded to the local `sqlite:///mlflow.db` store (with artifacts under `mlartifacts/`) and no external service is contacted

#### Scenario: The environment variable overrides the default
- **WHEN** `MLFLOW_TRACKING_URI` is set and no explicit tracking URI is supplied
- **THEN** the run is recorded against the URI from the environment

#### Scenario: An explicit argument overrides the environment
- **WHEN** an explicit tracking URI is supplied while `MLFLOW_TRACKING_URI` is also set
- **THEN** the explicit URI is used

### Requirement: Run lifecycle and logging
The tracking layer SHALL provide a run lifecycle that starts an MLflow run, logs within it, and ends it. It SHALL support logging parameters, metrics with an optional step (so a metric can form a time series), artifacts as individual files or directories, and a PyTorch model. Using the run as a context manager SHALL guarantee the run is ended even if the body raises.

#### Scenario: Parameters and metrics are recorded
- **WHEN** a parameter mapping and a metric mapping are logged inside a run
- **THEN** the run records each parameter and the latest value of each metric

#### Scenario: Stepped metrics form a history
- **WHEN** the same metric is logged at increasing steps
- **THEN** the run stores the full series rather than only the last value

#### Scenario: Artifacts are recorded
- **WHEN** a file or a directory is logged as an artifact
- **THEN** its contents are stored in the run

#### Scenario: A PyTorch model is logged
- **WHEN** a `torch.nn.Module` is logged as a model artifact
- **THEN** the run stores the model and it can be reloaded as a `torch.nn.Module`

### Requirement: Logging does not disturb the model
Logging a model or a metric SHALL NOT mutate the model's parameters, its training/eval state, or any optimizer state, so a model logged mid-training remains usable for continued training.

#### Scenario: Parameters are unchanged after logging
- **WHEN** a model is logged and every parameter is compared before and after
- **THEN** all parameters are unchanged

#### Scenario: The context manager closes the run
- **WHEN** a run context manager exits, normally or by exception
- **THEN** the run is ended and no active run remains

### Requirement: Tracking documentation and conventions
The tracking layer SHALL include a minimal usage document at `compgen/tracking/README.md` that shows how to configure the backend, start a run, and log parameters, metrics, artifacts, and models, and `AGENTS.md` SHALL state the convention that future training and model-building implementations record their runs through `compgen.tracking`. Every path these documents reference SHALL exist.

#### Scenario: The usage document exists and its example runs
- **WHEN** the example in `compgen/tracking/README.md` is executed from the repository root
- **THEN** it completes and records a run without error

#### Scenario: AGENTS states the tracking convention
- **WHEN** `AGENTS.md` is inspected for the tracking layer
- **THEN** it names `compgen/tracking/` and instructs future implementations to log runs through it

#### Scenario: Documented tracking paths exist
- **WHEN** every repository-relative path mentioned for the tracking layer in `AGENTS.md`, `README.md`, and the layer's own README is checked against the filesystem
- **THEN** each path exists in the documented form
