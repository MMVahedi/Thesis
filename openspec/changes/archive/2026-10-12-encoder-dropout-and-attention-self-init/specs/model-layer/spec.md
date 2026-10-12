# Spec Delta

## ADDED Requirements

### Requirement: Attention architectures initialize their own weights
The model layer SHALL initialize every registered attention architecture's weights at construction — xavier-normal on its projection weights and zero on its biases — so a constructed attention module is ready to use without any external initialization call. The encoder layer and any architecture-construction helper SHALL NOT initialize or re-initialize a passed attention module.

#### Scenario: A constructed architecture is already initialized
- **WHEN** a registered attention architecture is constructed and its weights are inspected without any external initialization being called
- **THEN** its projection weights follow the xavier-normal scheme and its biases are zero

#### Scenario: Passing a constructed module does not re-initialize it
- **WHEN** an already-constructed attention module with known weights is passed to an encoder layer, and the encoder layer's construction is inspected
- **THEN** the encoder layer leaves those weights unchanged and performs no initialization of the passed module

### Requirement: Encoder layer applies no dropout
The encoder layer SHALL apply no dropout to its residual paths: its attention and feed-forward residual connections SHALL add the sub-module outputs directly. Any dropout SHALL live inside the sub-modules the layer is given (the attention module and the feed-forward block), and the feed-forward block's dropout SHALL remain configurable through its construction.

#### Scenario: Residual connections add without dropout
- **WHEN** an encoder layer with constructed sub-modules processes a hidden state in training mode
- **THEN** its residual connections add the attention and feed-forward outputs without applying any dropout

#### Scenario: Feed-forward dropout lives in the block
- **WHEN** a feed-forward block is constructed with a dropout rate and handed to the encoder layer
- **THEN** that dropout is applied inside the block and the encoder layer holds no dropout module

### Requirement: Feed-forward variants are constructible modules
The model layer SHALL provide the residual ReLU-stack feed-forward and the two-projection GELU feed-forward as constructible modules that a task model instantiates directly and passes into each encoder layer. The model layer SHALL NOT provide a name-keyed feed-forward factory, and there SHALL be no feed-forward variant-name mapping or lookup.

#### Scenario: Both variants forward at unchanged width
- **WHEN** a feed-forward module is constructed for each supported variant and applied to a hidden state
- **THEN** each returns a tensor of the same shape as its input

#### Scenario: Task model constructs the feed-forward directly
- **WHEN** a task model is constructed and its encoder stack is inspected
- **THEN** each encoder layer's feed-forward is a directly instantiated module of the variant the task needs, constructed without a name-to-module factory

#### Scenario: No feed-forward factory is exposed
- **WHEN** the model layer's feed-forward module is searched for a name-to-module construction function
- **THEN** no such factory exists and no feed-forward variant-name lookup is performed

## MODIFIED Requirements

### Requirement: Task models consume an unmasked batch
The fuzzy-logic and Match3 task models SHALL construct their attention without dropout or bias configuration, and their forward paths SHALL NOT accept, read, or forward a batch mask. The Match3 task model SHALL NOT expose an attention-dropout constructor parameter; its feed-forward dropout SHALL remain configurable.

#### Scenario: Fuzzy-logic model forward takes no mask
- **WHEN** the fuzzy-logic task model's forward is called with inputs
- **THEN** it produces predictions and, on request, per-layer attention weights, without any mask being supplied or forwarded to attention

#### Scenario: Match3 model forward takes no mask
- **WHEN** the Match3 task model's forward is called with a collated batch
- **THEN** it produces per-token probabilities and final hidden states without reading a batch-mask entry from the batch

#### Scenario: Match3 attention-dropout parameter is gone
- **WHEN** the Match3 task model is constructed with `use_attention_dropout`
- **THEN** construction fails because the parameter is not accepted, while the feed-forward dropout rate remains configurable

#### Scenario: Attention-returning contract is unchanged
- **WHEN** the fuzzy-logic or Match3 task model is asked for attention weights
- **THEN** each still returns per-layer weights of shape `(batch, num_heads, length, length)`

## REMOVED Requirements

### Requirement: Feed-forward variants are selectable modules
**Reason**: The name-keyed `build_ffn` factory and variant-name selection are removed. Task models now instantiate the feed-forward class directly, so there is no factory and no variant-name lookup to select a variant or reject an unknown one.
**Migration**: Construct `ReluStackFFN` or `GeluFFN` directly and pass the instance into `EncoderLayer`. The replacement requirement "Feed-forward variants are constructible modules" covers the new behavior.
