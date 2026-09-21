"""Experiment configurations matching the fuzzy-logic reference setup."""

from dataclasses import asdict, dataclass


TASKS = {
    "logic_3var_2term": {"num_variables": 3, "num_terms": 2},
    "logic_4var_2term": {"num_variables": 4, "num_terms": 2},
    "logic_4var_3term": {"num_variables": 4, "num_terms": 3},
    "logic_5var_2term": {"num_variables": 5, "num_terms": 2},
}


@dataclass
class ExperimentConfig:
    task: str = "logic_4var_2term"
    num_train: int = 128_000
    num_eval: int = 16_000
    batch_size: int = 128
    epochs: int = 1
    seed: int = 2024
    seq_len: int = 16
    lambda_contrastive: float = 0.0
    temperature: float = 1.0
    learning_rate: float = 1e-3
    warmup_steps: int = 100
    weight_decay: float = 0.1
    emb_dim: int = 128
    qk_dim: int = 16
    v_dim: int = 16
    mlp_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.1
    frac_test: float = 0.5
    frac_ood_conj: float = 0.25
    num_analysis: int = 4_096
    analysis_batch_size: int = 256
    device: str = "auto"

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose from {sorted(TASKS)}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.lambda_contrastive < 0:
            raise ValueError("lambda_contrastive must be non-negative")
        if self.num_train <= 0 or self.num_eval <= 0 or self.epochs <= 0:
            raise ValueError("num_train, num_eval, and epochs must be positive")
        if self.num_analysis <= 0 or self.analysis_batch_size <= 0:
            raise ValueError("num_analysis and analysis_batch_size must be positive")
        if self.batch_size < 2 and self.lambda_contrastive > 0:
            raise ValueError("contrastive training requires batch_size >= 2")
        if self.qk_dim % self.num_heads or self.v_dim % self.num_heads:
            raise ValueError("qk_dim and v_dim must be divisible by num_heads")
        if not 0 <= self.dropout_rate < 1 or not 0 <= self.attention_dropout_rate < 1:
            raise ValueError("dropout rates must be in [0, 1)")

    @property
    def task_spec(self):
        return TASKS[self.task]

    def to_dict(self):
        return {**asdict(self), **self.task_spec}
