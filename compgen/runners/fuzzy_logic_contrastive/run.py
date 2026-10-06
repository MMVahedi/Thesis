"""Orchestration for the fuzzy-logic four-model contrastive experiment.

Builds the on-the-fly dataset and each model variant, trains through
``compgen.training.fit``, evaluates the reference splits, writes a per-variant
metrics JSON, and records each run through ``compgen.tracking``.

The schedule is stepped once per epoch by the training engine (see the change's
design.md): the reference's per-step warmup/cosine schedule is approximated at
epoch granularity, which is immaterial for the small single-epoch default.
"""

import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import List

import numpy as np
import torch

from compgen import tracking
from compgen.datasets.torch_datasets.fuzzy_logic_online import FuzzyLogicOnlineGenerator
from compgen.models.tasks.fuzzy_logic import FuzzyLogicModel
from compgen.runners.fuzzy_logic_contrastive.config import ExperimentConfig, Variant
from compgen.runners.fuzzy_logic_contrastive.step import forward_batch, make_step_fn
from compgen.training import fit

_SPLITS = ("train", "test", "id", "ood")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_generator(config: ExperimentConfig) -> FuzzyLogicOnlineGenerator:
    return FuzzyLogicOnlineGenerator(
        config.num_variables, config.num_terms, seed=config.seed
    )


def sample_batches(generator, config, split, count, device):
    return [
        generator.sample(config.batch_size, config.seq_len, split, index, device)
        for index in range(count)
    ]


def build_model(config: ExperimentConfig, attention_type: str, device) -> FuzzyLogicModel:
    return FuzzyLogicModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        attention_type=attention_type,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        qk_head_dim=config.qk_head_dim,
        v_head_dim=config.v_head_dim,
        ffn_type="gelu",
        ffn_dim=config.ffn_dim,
        dropout_rate=config.dropout_rate,
        attention_dropout_rate=config.attention_dropout_rate,
        use_norm=config.use_norm,
        attention_bias=config.attention_bias,
        dtype=torch.float64,
        device=str(device),
    )


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig):
    # Ideally weight decay is applied selectively: 1-D parameters (biases and
    # LayerNorm gains) are left undecayed, since shrinking a bias or a norm
    # gain toward zero is not regularization we want, while >=2-D weight
    # matrices are decayed. AdamW is used precisely because it decouples the
    # decay from the gradient update, which is what makes a per-group
    # `weight_decay` behave as intended. This split is kept here for reference
    # but is currently NOT used: we decay every parameter uniformly for now.
    # decay, no_decay = [], []
    # for parameter in model.parameters():
    #     (no_decay if parameter.ndim == 1 else decay).append(parameter)

    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer, config: ExperimentConfig):
    """Cosine LR decay; only constructed when ``config.use_scheduler`` is True."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1), eta_min=config.min_learning_rate
    )


def evaluate_split(model, generator, config, split, device, lambda_contrastive) -> dict:
    """Regression + contrastive metrics for one split, mirroring the reference."""
    model.eval()
    steps = config.num_eval // config.batch_size
    totals = {
        "mse": 0.0,
        "mae": 0.0,
        "r2": 0.0,
        "within_005_accuracy": 0.0,
        "contrastive_loss": 0.0,
        "avg_positive_partners": 0.0,
    }
    seen = 0
    with torch.no_grad():
        for index in range(steps):
            batch = generator.sample(config.batch_size, config.seq_len, split, index, device)
            output = forward_batch(
                model,
                batch,
                lambda_contrastive,
                config.temperature,
                config.contrastive_representation,
            )
            error = (output.prediction - output.target).squeeze(-1)
            seen += error.numel()
            totals["mse"] += float(error.square().sum())
            totals["mae"] += float(error.abs().sum())
            totals["r2"] += float(
                (1.0 - error.square() / batch.base_mse.clamp_min(1e-12)).sum()
            )
            totals["within_005_accuracy"] += float((error.abs() <= 0.05).sum())
            totals["contrastive_loss"] += float(output.contrastive_loss)
            totals["avg_positive_partners"] += float(output.average_positives)

    metrics = {
        "mse": totals["mse"] / seen,
        "mae": totals["mae"] / seen,
        "r2": totals["r2"] / seen,
        "within_005_accuracy": totals["within_005_accuracy"] / seen,
        "contrastive_loss": totals["contrastive_loss"] / steps,
        "avg_positive_partners": totals["avg_positive_partners"] / steps,
    }
    return metrics


def _loggable(metrics: dict) -> dict:
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def run_variant(
    config: ExperimentConfig, variant: Variant, device, output_dir: Path
) -> dict:
    """Train and evaluate one variant; write and return its metrics."""
    set_seed(config.seed)
    generator = build_generator(config)
    train_batches = sample_batches(
        generator, config, "train", config.num_train // config.batch_size, device
    )
    model = build_model(config, variant.attention_type, device)
    step_fn = make_step_fn(
        variant.lambda_contrastive,
        config.temperature,
        config.contrastive_representation,
    )
    optimizer = build_optimizer(model, config)
    # Fixed learning rate by default; the cosine schedule is opt-in.
    scheduler = build_scheduler(optimizer, config) if config.use_scheduler else None

    run_name = f"{variant.name}_{config.contrastive_representation}"
    run_context = (
        tracking.start_run(run_name=run_name) if config.track else nullcontext()
    )
    with run_context:
        if config.track:
            tracking.log_params(
                {
                    "attention_type": variant.attention_type,
                    "lambda_contrastive": variant.lambda_contrastive,
                    "temperature": config.temperature,
                    "contrastive_representation": config.contrastive_representation,
                    "task": config.task,
                    "num_train": config.num_train,
                    "num_eval": config.num_eval,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "seq_len": config.seq_len,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "use_scheduler": config.use_scheduler,
                    "seed": config.seed,
                }
            )

        result = fit(
            model,
            train_batches,
            step_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config.epochs,
            device=device,
            seed=config.seed,
            track=False,
        )
        if result.best_state_dict is not None:
            model.load_state_dict(result.best_state_dict)

        metrics = {}
        for split in _SPLITS:
            split_metrics = evaluate_split(
                model, generator, config, split, device, variant.lambda_contrastive
            )
            for key, value in split_metrics.items():
                metrics[f"{split}_{key}"] = value
        metrics["train_loss"] = result.history[-1].train_loss if result.history else float("nan")
        metrics["attention_type"] = variant.attention_type
        metrics["lambda_contrastive"] = variant.lambda_contrastive
        metrics["contrastive_representation"] = config.contrastive_representation
        metrics["run_name"] = run_name

        if config.track:
            tracking.log_metrics(_loggable(metrics))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{run_name}.json").open("w") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    return metrics


def run_experiment(config: ExperimentConfig) -> List[dict]:
    """Run all four variants and return their metrics in variant order."""
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    return [
        run_variant(config, variant, device, output_dir)
        for variant in config.variants()
    ]