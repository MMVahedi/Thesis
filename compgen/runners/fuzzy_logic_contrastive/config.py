"""Configuration for the fuzzy-logic four-model contrastive experiment.

Holds the single study configuration and derives the four model variants:
softmax and linear attention, each with a zero and a positive contrastive
coefficient. Model hyperparameters reuse the reference fuzzy-logic transformer
configuration so that only the attention architecture and the contrastive
coefficient vary across variants.
"""

from dataclasses import dataclass
from typing import List, Tuple

from compgen.datasets.torch_datasets.fuzzy_logic_online import TASKS
from compgen.models.attentions import ATTENTION_CLASSES

# Architectures that compose through `EncoderLayer`; `triangular` is pair-level
# and not sequence-composable, so it is excluded from the runner.
_SEQUENCE_COMPOSABLE = tuple(
    name for name in ("softmax", "linear") if name in ATTENTION_CLASSES
)


@dataclass(frozen=True)
class Variant:
    """One model in the comparison: an architecture and a contrastive weight."""

    attention_type: str
    lambda_contrastive: float

    @property
    def name(self) -> str:
        return f"{self.attention_type}_lambda{self.lambda_contrastive:g}"


@dataclass
class ExperimentConfig:
    """Configuration for the fuzzy-logic contrastive comparison."""

    # Task and dataset size (small by default so the comparison runs quickly).
    task: str = "logic_4var_2term"
    num_train: int = 2_000
    num_eval: int = 1_000
    batch_size: int = 64
    epochs: int = 1
    seq_len: int = 16

    # Contrastive objective.
    lambda_contrastive: float = 0.05
    temperature: float = 1.0
    seed: int = 2024
    attention_types: Tuple[str, ...] = ("softmax", "linear")

    # Model (reference fuzzy-logic transformer hyperparameters).
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 8
    qk_head_dim: int = 2
    v_head_dim: int = 2
    ffn_dim: int = 256
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.1
    use_norm: bool = True
    use_relative_bias: bool = True
    relative_bias_num_buckets: int = 16
    relative_bias_max_distance: int = 16
    attention_bias: bool = True

    # Optimization.
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100

    # Runtime.
    device: str = "auto"
    track: bool = True
    output_dir: str = "results/fuzzy_logic_contrastive"

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose from {sorted(TASKS)}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.lambda_contrastive <= 0:
            raise ValueError(
                "lambda_contrastive must be positive so the four variants are distinct"
            )
        if self.num_train <= 0 or self.num_eval <= 0 or self.epochs <= 0:
            raise ValueError("num_train, num_eval, and epochs must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.batch_size < 2:
            raise ValueError("contrastive training requires batch_size >= 2")
        if self.num_train // self.batch_size < 1 or self.num_eval // self.batch_size < 1:
            raise ValueError("batch_size is larger than the training or evaluation set")
        if not self.attention_types:
            raise ValueError("at least one attention type is required")
        for name in self.attention_types:
            if name not in _SEQUENCE_COMPOSABLE:
                raise ValueError(
                    f"Unsupported attention type {name!r}; "
                    f"choose from {list(_SEQUENCE_COMPOSABLE)}"
                )

    @property
    def task_spec(self) -> dict:
        return TASKS[self.task]

    @property
    def num_variables(self) -> int:
        return self.task_spec["num_variables"]

    @property
    def num_terms(self) -> int:
        return self.task_spec["num_terms"]

    @property
    def input_dim(self) -> int:
        return self.num_variables + 1

    def variants(self) -> List[Variant]:
        """The four model variants: each architecture at zero and positive λ."""
        variants: List[Variant] = []
        for attention_type in self.attention_types:
            variants.append(Variant(attention_type, 0.0))
            variants.append(Variant(attention_type, self.lambda_contrastive))
        return variants