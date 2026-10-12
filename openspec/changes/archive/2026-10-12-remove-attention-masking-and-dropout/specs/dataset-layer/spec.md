# Spec Delta

## ADDED Requirements

### Requirement: Match3 batches carry no padding mask
The Match3 torch collate function SHALL return a batch mapping containing the padded token inputs, position indices, labels, and per-example sequence geometry, and SHALL NOT include a padding-mask entry in that mapping. Batches SHALL be directly consumable by the model layer without any mask.

#### Scenario: No mask entry is produced
- **WHEN** a batch of variable-length Match3 sequences is collated
- **THEN** the returned mapping contains the padded inputs, positions, labels, and geometry, and contains no batch-mask entry

#### Scenario: Collated batch is consumable directly
- **WHEN** a collated Match3 batch is passed to a task model's forward
- **THEN** it is accepted without any mask being read or supplied
