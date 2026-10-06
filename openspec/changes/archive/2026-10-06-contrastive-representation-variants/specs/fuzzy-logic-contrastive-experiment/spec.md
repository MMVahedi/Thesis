# Spec Delta

## MODIFIED Requirements

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
the comparison runs quickly, and SHALL expose its task, dataset size, batch
size, contrastive coefficient, contrastive representation, temperature, and seed
as configuration rather than hard-coding them.

#### Scenario: Defaults are small
- **WHEN** the experiment is run without overrides
- **THEN** it uses a small training/evaluation dataset size and a single batch size

#### Scenario: Configuration is adjustable
- **WHEN** the dataset size, batch size, contrastive coefficient, contrastive representation, temperature, or seed is supplied
- **THEN** the experiment uses the supplied values

#### Scenario: The representation is configurable
- **WHEN** a contrastive representation other than the default is supplied
- **THEN** the experiment computes the contrastive term over that representation for every model with a positive contrastive coefficient
