"""Configuration for the fuzzy-logic four-model contrastive experiment.

Holds the single study configuration and derives the four model variants:
softmax and linear attention, each with a zero and a positive contrastive
coefficient. Model widths follow the reference fuzzy-logic transformer skeleton
but default to an even split of ``hidden_dim`` across heads for the attention
head dimensions and to ``hidden_dim`` for the feed-forward width, so only the
attention architecture and the contrastive coefficient vary across variants.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from compgen.datasets.torch_datasets.fuzzy_logic_online import TASKS

# Attention labels the runner can build. This is an experiment-local list of the
# architectures this study compares (see run.py, which maps each label to its
# concrete attention class and constructs the object); it is not a model-layer
# registry. `triangular` is pair-level and not sequence-composable, so it is
# excluded.
SUPPORTED_ATTENTION_TYPES = ("softmax", "linear")

# Contrastive representations derived from the final layer's attention. "matrix"
# flattens the full `(B, H, N, N)` attention, "row" keeps the response token's
# attention over key positions `(B, H*N)`, and "code" keeps the response token's
# attention to itself per head `(B, H)`. `code` is the default.
CONTRASTIVE_REPRESENTATIONS = ("matrix", "row", "code")


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
    task: str = "logic_3var_2term"
    num_train: int = 2_000
    num_eval: int = 1_000
    batch_size: int = 64
    epochs: int = 1
    seq_len: int = 16

    # Contrastive objective.
    lambda_contrastive: float = 0.05
    temperature: float = 1.0
    # Which reduction of the final layer's attention forms the contrastive
    # representation: "matrix" (full `(B, H*N*N)`), "row" (response-token row,
    # `(B, H*N)`), or "code" (response-token self-attention per head, `(B, H)`).
    contrastive_representation: str = "code"
    seed: int = 2024
    attention_types: Tuple[str, ...] = ("softmax", "linear")

    # Model architecture.
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 8
    # Per-head query/key dimension. `None` -> hidden_dim // num_heads.
    qk_head_dim: Optional[int] = None
    # Per-head value dimension. `None` -> hidden_dim // num_heads.
    v_head_dim: Optional[int] = None
    # Feed-forward hidden width. `None` -> hidden_dim.
    ffn_dim: Optional[int] = None
    # Dropout on the residual and feed-forward path.
    dropout_rate: float = 0.0
    # Pre-norm LayerNorm on both attention and feed-forward sublayers.
    use_norm: bool = True

    # Optimization.
    learning_rate: float = 1e-3
    # Floor for the cosine schedule; only used when `use_scheduler` is True.
    min_learning_rate: float = 1e-4
    # Cosine LR decay. Disabled by default: training uses a fixed learning rate,
    # set once on the optimizer, with no schedule stepping it down.
    use_scheduler: bool = False
    weight_decay: float = 0.1
    warmup_steps: int = 100

    # Runtime.
    device: str = "auto"
    track: bool = True
    # Persist the per-epoch training history next to each variant's metrics as
    # ``<run_name>.history.jsonl`` (one JSON object per epoch). Disable for
    # metrics-only runs.
    save_history: bool = True
    output_dir: str = "results/fuzzy_logic_contrastive"

    def __post_init__(self) -> None:
        # Resolve architecture widths left as `None` from the other
        # hyperparameters: both attention head dimensions default to an even
        # split of hidden_dim across heads, and the feed-forward width defaults
        # to hidden_dim.
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.qk_head_dim is None:
            self.qk_head_dim = self.hidden_dim // self.num_heads
        if self.v_head_dim is None:
            self.v_head_dim = self.hidden_dim // self.num_heads
        if self.ffn_dim is None:
            self.ffn_dim = self.hidden_dim

        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose from {sorted(TASKS)}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.contrastive_representation not in CONTRASTIVE_REPRESENTATIONS:
            raise ValueError(
                f"Unknown contrastive_representation {self.contrastive_representation!r}; "
                f"choose from {list(CONTRASTIVE_REPRESENTATIONS)}"
            )
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
            if name not in SUPPORTED_ATTENTION_TYPES:
                raise ValueError(
                    f"Unsupported attention type {name!r}; "
                    f"choose from {list(SUPPORTED_ATTENTION_TYPES)}"
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