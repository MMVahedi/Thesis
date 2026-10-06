# fuzzy-logic-contrastive-experiment Specification

## Purpose

Provides a runnable fuzzy-logic experiment that composes the package's model,
dataset, loss, training, and tracking layers to train and evaluate four model
variants — softmax and linear attention, each without and with the proposed
response-token attention-code contrastive term — on the reference task splits.

## Requirements

### Requirement: Runnable fuzzy-logic contrastive experiment

The package SHALL provide a runnable experiment entry point that composes the
existing fuzzy-logic task model, on-the-fly fuzzy-logic dataset, shared-term
contrastive loss, training engine, and experiment tracking. The entry point
SHALL live under the code package, SHALL be runnable from the repository root,
and SHALL NOT reimplement any model, dataset, loss, or training behavior.

#### Scenario: Runs end to end for a reference task
- **WHEN** the experiment is run for a supported fuzzy-logic task configuration
- **THEN** it trains and evaluates its models to completion without error

#### Scenario: Composes the existing package layers
- **WHEN** the experiment's imports and construction of the model, dataset, loss, and training loop are inspected
- **THEN** it obtains each of them from the package's existing layers rather than defining its own

#### Scenario: Deterministic for a fixed seed
- **WHEN** the experiment is run twice with the same seed and configuration
- **THEN** the recorded metrics are identical

### Requirement: Four-model comparison

The experiment SHALL train exactly four models in a single configuration: a
softmax-attention model and a linear-attention model, each with the contrastive
coefficient set to zero and with a single positive coefficient, holding every
other setting equal across the four.

#### Scenario: Four variants are produced
- **WHEN** the experiment is run with its default configuration
- **THEN** it produces results for softmax and linear attention, each with a zero and a positive contrastive coefficient

#### Scenario: Only the attention architecture and coefficient vary
- **WHEN** the configurations of the four models are compared
- **THEN** they are identical except for the attention architecture and the contrastive coefficient

#### Scenario: Zero coefficient reduces to the task loss
- **WHEN** a model is trained with a zero contrastive coefficient
- **THEN** its objective equals the regression task loss and no contrastive term contributes

### Requirement: Response-token attention-code contrastive term

For models with a positive contrastive coefficient, the experiment SHALL compute
the contrastive term over a per-example representation derived from the final
encoder layer's attention, selectable among three forms: the flattened full
attention matrix, the last (response/output) token's attention row over key
positions, and the last token's attention to itself across heads. The
representation SHALL default to the last token's attention to itself across
heads. The experiment SHALL combine the selected representation's contrastive
term with the regression task loss weighted by the contrastive coefficient, and
the term identities used as positives SHALL come from the dataset.

#### Scenario: The code is the final layer's last-token attention across heads
- **WHEN** the experiment runs with its default configuration and a positive contrastive coefficient
- **THEN** the contrastive representation is the last encoder layer's last-token attention to itself, with one feature per attention head

#### Scenario: The full attention matrix can be selected
- **WHEN** the flattened-attention-matrix representation is selected
- **THEN** the contrastive representation is the last encoder layer's attention weights for the batch, flattened over heads, query positions, and key positions, with one vector per example

#### Scenario: The response-token attention row can be selected
- **WHEN** the response-token-row representation is selected
- **THEN** the contrastive representation is the last encoder layer's last-token attention over key positions, with one feature per head and key position

#### Scenario: The contrastive term is combined with the task loss
- **WHEN** a model is trained with a positive contrastive coefficient
- **THEN** its objective is the regression task loss plus the coefficient times the shared-term contrastive loss over the selected representation

#### Scenario: Term identity defines positives
- **WHEN** the contrastive loss is computed for a batch
- **THEN** examples sharing at least one term identifier from the dataset count as positives of each other

### Requirement: Small adjustable defaults

The experiment SHALL default to a small dataset size and a single batch size so
the four-model comparison runs quickly, and SHALL expose its task, dataset size,
batch size, contrastive coefficient, contrastive representation, temperature,
and seed as configuration rather than hard-coding them.

#### Scenario: Defaults are small
- **WHEN** the experiment is run without overrides
- **THEN** it uses a small training/evaluation dataset size and a single batch size

#### Scenario: Configuration is adjustable
- **WHEN** the dataset size, batch size, contrastive coefficient, contrastive representation, temperature, or seed is supplied
- **THEN** the experiment uses the supplied values

#### Scenario: The representation is configurable
- **WHEN** a contrastive representation other than the default is supplied
- **THEN** the experiment computes the contrastive term over that representation for every model with a positive contrastive coefficient

### Requirement: Reported regression and contrastive metrics

The experiment SHALL report, for each model, the mean-squared error, mean
absolute error, the reference per-example R², and the within-±0.05 accuracy, on
the train, test, in-distribution, and out-of-distribution splits, together with
the contrastive loss and the average number of positive partners. It SHALL log
each run through the package's experiment-tracking layer.

#### Scenario: Task metrics are reported per split
- **WHEN** a model finishes training
- **THEN** mean-squared error, mean absolute error, R², and within-±0.05 accuracy are reported for the train, test, id, and ood splits

#### Scenario: Contrastive diagnostics are reported
- **WHEN** a model with a positive contrastive coefficient finishes training
- **THEN** its contrastive loss and average positive-partner count are reported alongside the task metrics

#### Scenario: Runs are tracked
- **WHEN** the experiment runs a model
- **THEN** the run and its metrics are recorded through the package's experiment-tracking layer