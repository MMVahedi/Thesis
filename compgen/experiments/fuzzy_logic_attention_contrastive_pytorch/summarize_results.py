"""Collect completed runs into tables and comparison charts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    rows = []
    for path in root.rglob("final_metrics.json"):
        try:
            row = json.loads(path.read_text())
            row["result_directory"] = str(path.parent)
            rows.append(row)
        except (json.JSONDecodeError, OSError):
            pass
    if not rows:
        raise FileNotFoundError(f"No completed final_metrics.json files found under {root}")

    frame = pd.DataFrame(rows).sort_values(
        ["task", "num_train", "batch_size", "lambda_contrastive"]
    )
    frame.to_csv(root / "all_results.csv", index=False)
    columns = [
        "task", "num_train", "batch_size", "lambda_contrastive", "temperature",
        "test_r2", "test_mse", "test_mae", "test_within_005_accuracy",
        "test_contrastive_loss", "test_total_loss", "test_avg_num_positives",
        "ood_r2", "ood_mse", "ood_contrastive_loss",
        "attention_mean_macro_f1", "elapsed_seconds",
    ]
    available = [column for column in columns if column in frame]
    frame[available].to_csv(root / "results_compact.csv", index=False)

    for task, task_frame in frame.groupby("task"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for (size, batch), group in task_frame.groupby(["num_train", "batch_size"]):
            group = group.sort_values("lambda_contrastive")
            label = f"N={size:,}, B={batch}"
            axes[0].plot(group.lambda_contrastive, group.test_r2, marker="o", label=label)
            axes[1].plot(group.lambda_contrastive, group.test_mse, marker="o", label=label)
        axes[0].set(ylabel="Held-out test R²", xlabel="Contrastive λ")
        axes[1].set(ylabel="Held-out test MSE", xlabel="Contrastive λ")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
        fig.suptitle(f"{task}: lambda, batch-size, and dataset-size comparison")
        fig.tight_layout()
        fig.savefig(root / f"{task}_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    best = frame.loc[frame.groupby(["task", "num_train"])["test_r2"].idxmax()]
    fig, ax = plt.subplots(figsize=(9, 5))
    for task, group in best.groupby("task"):
        group = group.sort_values("num_train")
        ax.plot(group.num_train, group.test_r2, marker="o", label=task)
    ax.set_xscale("log")
    ax.set(xlabel="Training examples", ylabel="Best held-out test R²")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "dataset_scaling_best_r2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    partner_frame = frame.groupby(["task", "batch_size"], as_index=False)[
        "train_avg_num_positives"
    ].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for task, group in partner_frame.groupby("task"):
        ax.plot(
            group.batch_size, group.train_avg_num_positives,
            marker="o", label=task,
        )
    ax.set(
        xlabel="Batch size",
        ylabel="Mean shared-term positive partners per anchor",
        title="Why batch size matters for the contrastive objective",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "positive_partners_by_batch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(frame[available].to_string(index=False))
    print(f"\nSaved tables and charts to {root}")


if __name__ == "__main__":
    main()
