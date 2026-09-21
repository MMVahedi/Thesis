"""Train one isolated PyTorch fuzzy-logic attention experiment."""

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from config import ExperimentConfig
from data import FuzzyLogicGenerator
from losses import positive_partner_counts, shared_term_contrastive, training_loss
from model import StandardAttentionTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="logic_4var_2term")
    parser.add_argument("--num-train", type=int, default=128_000)
    parser.add_argument("--num-eval", type=int, default=16_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lambda-contrastive", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--num-analysis", type=int, default=4_096)
    parser.add_argument("--skip-analysis-codes", action="store_true")
    return parser.parse_args()


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA GPU")
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_at(step, total_steps, peak, warmup):
    if step < warmup:
        return peak * step / max(1, warmup)
    # The reference passes decay_steps=(total_steps - warmup_steps) to Optax.
    # Optax counts warmup inside decay_steps, so cosine decay ends one warmup
    # interval before training ends and then remains at end_value.
    reference_decay_steps = total_steps - warmup
    progress = (step - warmup) / max(1, reference_decay_steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 else decay).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


@torch.inference_mode()
def evaluate(model, generator, config, split, device, epoch):
    model.eval()
    totals = {
        "mse": 0.0, "mae": 0.0, "paper_r2": 0.0, "within_005": 0.0,
        "contrastive": 0.0, "positives": 0.0,
    }
    seen = 0
    steps = config.num_eval // config.batch_size
    for batch_index in range(steps):
        batch = generator.sample(
            config.batch_size, config.seq_len, split, batch_index, device
        )
        if config.lambda_contrastive > 0:
            all_predictions, attentions = model(batch.x, return_attention=True)
            attention_code = attentions[-1][:, :, -1, -1]
            contrastive, positives = shared_term_contrastive(
                attention_code, batch.term_ids, config.temperature
            )
        else:
            all_predictions = model(batch.x)
            contrastive = all_predictions.new_zeros(())
            _, positive_counts = positive_partner_counts(batch.term_ids)
            positives = positive_counts.float().mean()
        prediction = all_predictions[:, -1]
        error = prediction - batch.y
        count = len(error)
        totals["mse"] += error.square().sum().item()
        totals["mae"] += error.abs().sum().item()
        totals["paper_r2"] += (1.0 - error.square() / batch.base_mse.clamp_min(1e-12)).sum().item()
        totals["within_005"] += (error.abs() <= 0.05).sum().item()
        totals["contrastive"] += contrastive.item()
        totals["positives"] += positives.item()
        seen += count
    mse = totals["mse"] / seen
    contrastive = totals["contrastive"] / steps
    return {
        f"{split}_mse": mse,
        f"{split}_mae": totals["mae"] / seen,
        f"{split}_r2": totals["paper_r2"] / seen,
        f"{split}_within_005_accuracy": totals["within_005"] / seen,
        f"{split}_contrastive_loss": contrastive,
        f"{split}_total_loss": mse + config.lambda_contrastive * contrastive,
        f"{split}_avg_num_positives": totals["positives"] / steps,
    }


@torch.inference_mode()
def save_attention_codes(model, generator, config, device, workdir):
    model.eval()
    saved = {}
    count = config.num_analysis
    batch_size = min(config.analysis_batch_size, count)
    count = count - count % batch_size
    for split in ("train", "test"):
        code_parts, latent_parts, target_parts = [], [], []
        for batch_index in range(count // batch_size):
            batch = generator.sample(
                batch_size, config.seq_len, split, batch_index, device,
                input_dist="fixed_context",
            )
            prediction, attentions = model(batch.x, return_attention=True)
            # [batch, layer, head], exactly the response-token self-attention code.
            codes = torch.stack([weights[:, :, -1, -1] for weights in attentions], dim=1)
            code_parts.append(codes.cpu().numpy())
            latent_parts.append(batch.latents.cpu().numpy())
            target_parts.append(batch.y.cpu().numpy())
        saved[f"{split}_attention"] = np.concatenate(code_parts)
        saved[f"{split}_latents"] = np.concatenate(latent_parts)
        saved[f"{split}_target"] = np.concatenate(target_parts)
    np.savez_compressed(workdir / "attention_codes.npz", **saved)


def write_history(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = ExperimentConfig(
        task=args.task,
        num_train=args.num_train,
        num_eval=args.num_eval,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lambda_contrastive=args.lambda_contrastive,
        temperature=args.temperature,
        seed=args.seed,
        device=args.device,
        num_analysis=args.num_analysis,
    )
    if config.num_train % config.batch_size or config.num_eval % config.batch_size:
        raise ValueError("num_train and num_eval must be divisible by batch_size, as in the reference loader")

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = choose_device(config.device)
    generator = FuzzyLogicGenerator(
        **config.task_spec,
        frac_test=config.frac_test,
        frac_ood_conj=config.frac_ood_conj,
        seed=config.seed,
    )
    model = StandardAttentionTransformer(config.task_spec["num_variables"] + 1, config).to(device)
    optimizer = make_optimizer(model, config)
    steps_per_epoch = config.num_train // config.batch_size
    total_steps = steps_per_epoch * config.epochs
    if total_steps <= 2 * config.warmup_steps:
        raise ValueError(
            f"Reference schedule requires more than {2 * config.warmup_steps} optimizer "
            f"steps, but this run has {total_steps}. Increase num_train or epochs."
        )

    metadata = {
        **config.to_dict(),
        "device_resolved": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "dataset": generator.describe(),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "standard_softmax_attention": True,
    }
    (workdir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

    history = []
    global_step = 0
    start = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        aggregate = {"task": 0.0, "contrastive": 0.0, "total": 0.0, "positives": 0.0}
        bar = tqdm(
            total=config.num_train,
            unit="sample",
            desc=f"epoch {epoch}/{config.epochs}",
            dynamic_ncols=True,
        )
        for batch_index in range(steps_per_epoch):
            lr = learning_rate_at(
                global_step, total_steps, config.learning_rate, config.warmup_steps
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = generator.sample(
                config.batch_size, config.seq_len, "train", batch_index, device
            )
            optimizer.zero_grad(set_to_none=True)
            total, task, contrastive, positives = training_loss(
                model, batch, config.lambda_contrastive, config.temperature
            )
            total.backward()
            optimizer.step()
            aggregate["task"] += task.item()
            aggregate["contrastive"] += contrastive.item()
            aggregate["total"] += total.item()
            aggregate["positives"] += positives.item()
            global_step += 1
            bar.update(config.batch_size)
        bar.close()

        row = {
            "epoch": epoch,
            "samples_seen": epoch * config.num_train,
            "learning_rate": lr,
            "train_task_mse": aggregate["task"] / steps_per_epoch,
            "train_contrastive_loss": aggregate["contrastive"] / steps_per_epoch,
            "train_total_loss": aggregate["total"] / steps_per_epoch,
            "train_avg_num_positives": aggregate["positives"] / steps_per_epoch,
        }
        for split in ("id", "test", "ood"):
            row.update(evaluate(model, generator, config, split, device, epoch))
        history.append(row)
        write_history(workdir / "epoch_metrics.csv", history)
        print("\nFULL-EPOCH RESULTS\n" + json.dumps(row, indent=2, sort_keys=True), flush=True)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "config": config.to_dict()},
            workdir / "checkpoint.pt",
        )

    if not args.skip_analysis_codes:
        save_attention_codes(model, generator, config, device, workdir)

    final = {
        **metadata,
        **history[-1],
        "elapsed_seconds": time.time() - start,
        "workdir": str(workdir),
        "status": "complete",
    }
    temporary = workdir / "final_metrics.json.tmp"
    temporary.write_text(json.dumps(final, indent=2, sort_keys=True))
    os.replace(temporary, workdir / "final_metrics.json")
    print("\nFINAL RESULTS\n" + json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
