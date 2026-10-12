# Spec Delta

## MODIFIED Requirements

### Requirement: Generated artifacts stay untracked
Generated datasets, training artifacts, experiment-tracking run output, and build outputs SHALL remain untracked: `.gitignore` SHALL cover `data/`, `*.jsonl`, `*.zip`, logs, checkpoints, model binaries, and the MLflow local run output (`mlflow.db` and the `mlartifacts/` directory), and no change SHALL bring any generated artifact under version control.

#### Scenario: Ignore rules cover all artifact locations
- **WHEN** `.gitignore` is inspected
- **THEN** it matches every artifact pattern previously documented (including `data/` and `*.zip`) plus the MLflow local run output `mlflow.db` and `mlartifacts/`, and `git status` reports no generated dataset, checkpoint, archive, or tracking-run files as untracked work
