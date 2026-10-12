# Spec Delta

## ADDED Requirements

### Requirement: Encoder layer consumes a constructed feed-forward module
The shared encoder layer SHALL accept an already-constructed feed-forward module and SHALL use that module as its feed-forward sublayer. It SHALL NOT accept a feed-forward variant name, depth, or width configuration, SHALL NOT select or validate a feed-forward variant, and SHALL depend on no feed-forward registry. Constructing the feed-forward module from the selected variant and configuration SHALL be the responsibility of the task model or other caller that composes the encoder layer.

#### Scenario: Constructed module becomes the layer's feed-forward
- **WHEN** an encoder layer is constructed with a constructed feed-forward module and applied to a hidden state
- **THEN** the layer uses that module for its feed-forward sublayer and returns a hidden state of the same shape as its input

#### Scenario: Encoder layer takes no feed-forward variant configuration
- **WHEN** the encoder layer's construction surface is inspected
- **THEN** it accepts no feed-forward variant name, depth, or width parameter and performs no feed-forward variant selection or validation

### Requirement: Feed-forward variants are selectable modules
The model layer SHALL provide the residual ReLU-stack feed-forward and the two-projection GELU feed-forward as constructible modules selectable by stable names, alongside a shared means of constructing a variant from its name. A task model SHALL select its feed-forward variant by name, construct the corresponding module with the model's configuration, and pass the constructed module into each encoder layer.

#### Scenario: Both variants forward at unchanged width
- **WHEN** a feed-forward module is constructed for each supported variant and applied to a hidden state
- **THEN** each returns a tensor of the same shape as its input

#### Scenario: Unknown feed-forward variant is rejected
- **WHEN** a feed-forward variant name that is not supported is requested
- **THEN** construction fails with an error naming the unsupported variant

#### Scenario: Task model constructs the feed-forward it selected
- **WHEN** a task model is constructed from a feed-forward variant name and the resulting encoder stack is inspected
- **THEN** each encoder layer's feed-forward is a constructed module of that variant, and the task model's public construction surface still selects the variant by its name
