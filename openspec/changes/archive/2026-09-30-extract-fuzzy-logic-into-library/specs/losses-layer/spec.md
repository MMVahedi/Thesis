# Spec Delta

## Purpose

Gives the code package reusable training objectives: a supervised contrastive loss over per-example representations that treats examples sharing a term as positives, its positive-partner accounting, and a model-agnostic loss that combines a task loss with the contrastive term.

## ADDED Requirements

### Requirement: Shared-term supervised contrastive loss
The losses package SHALL provide a supervised contrastive loss over per-example representation vectors in which two examples are positives when their term-identifier sets share at least one term, using cosine similarity scaled by a temperature, excluding each example from its own negatives, and averaging the per-anchor loss over anchors that have at least one positive partner.

#### Scenario: Examples sharing a term are positives
- **WHEN** two examples' term-identifier sets overlap
- **THEN** each counts as a positive partner of the other

#### Scenario: The example itself is excluded from its negatives
- **WHEN** the contrastive loss for an example is computed
- **THEN** the example's similarity to itself is removed from the normalization denominator

#### Scenario: Anchors without positives do not contribute
- **WHEN** every example's term set is pairwise disjoint
- **THEN** the loss is finite and equals zero, and it remains differentiable

#### Scenario: Temperature scales the similarity
- **WHEN** the temperature changes while the representations are held fixed
- **THEN** the computed loss changes accordingly

### Requirement: Positive-partner accounting
The losses package SHALL expose each example's positive-partner count as a separately usable value.

#### Scenario: Counts match the positive mask
- **WHEN** positive partners are computed for a batch of term-identifier sets
- **THEN** each example's count equals the number of other examples sharing at least one term with it

### Requirement: Combined task and contrastive training loss
The losses package SHALL provide a model-agnostic training loss that combines a task loss with the shared-term contrastive loss weighted by a non-negative coefficient, accepting the representation vectors and term identifiers as inputs rather than reading them from a specific model's internals.

#### Scenario: Zero weight reduces to the task loss
- **WHEN** the contrastive coefficient is zero
- **THEN** the combined loss equals the task loss and the contrastive term is zero

#### Scenario: Positive weight adds the weighted contrastive term
- **WHEN** the contrastive coefficient is positive
- **THEN** the combined loss equals the task loss plus the coefficient times the contrastive loss

#### Scenario: The representation source is caller-supplied
- **WHEN** the combined loss is used
- **THEN** it accepts the representations and term identifiers as inputs and does not require a specific model class
