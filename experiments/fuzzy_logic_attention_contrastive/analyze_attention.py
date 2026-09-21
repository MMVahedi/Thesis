"""Paper-style analysis of fuzzy-logic attention latent codes."""

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def term_ids(latents):
    num_variables = latents.shape[-1]
    weights = 2 ** np.arange(num_variables - 1, -1, -1)
    return np.sum(latents * weights, axis=-1).astype(np.int32)


def analyze(run_dir: Path, max_samples: int, seed: int):
    latent_paths = sorted(run_dir.glob("latent_dataset_step_*.pkl"))
    if not latent_paths:
        raise FileNotFoundError(f"No final latent dataset found under {run_dir}")

    with latent_paths[-1].open("rb") as stream:
        datasets = pickle.load(stream)
    train = datasets["train_fixed_context"]
    test = datasets["test_fixed_context"]

    x_train = np.asarray(train["attn_weight"], dtype=np.float32)
    x_test = np.asarray(test["attn_weight"], dtype=np.float32)
    y_train = term_ids(np.asarray(train["latent"]))
    y_test = term_ids(np.asarray(test["latent"]))
    targets = np.asarray(test["target"]).reshape(len(x_test), -1)[:, -1]
    num_layers, num_heads = x_train.shape[1:]
    num_terms = y_train.shape[1]

    rows = []
    for layer in range(num_layers):
        for term_position in range(num_terms):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced"),
            )
            classifier.fit(x_train[:, layer, :], y_train[:, term_position])
            prediction = classifier.predict(x_test[:, layer, :])
            rows.append({
                "layer": layer,
                "term_position": term_position + 1,
                "macro_f1": float(f1_score(
                    y_test[:, term_position], prediction, average="macro")),
                "accuracy": float(accuracy_score(y_test[:, term_position], prediction)),
                "num_heads": num_heads,
                "train_samples": len(x_train),
                "test_samples": len(x_test),
            })

    f1_path = run_dir / "attention_term_f1.csv"
    with f1_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, num_terms, figsize=(5 * num_terms, 4), squeeze=False)
    for term_position in range(num_terms):
        ax = axes[0, term_position]
        layer_rows = [row for row in rows if row["term_position"] == term_position + 1]
        ax.plot(
            [row["layer"] + 1 for row in layer_rows],
            [row["macro_f1"] for row in layer_rows],
            marker="o",
            label="macro F1",
        )
        ax.plot(
            [row["layer"] + 1 for row in layer_rows],
            [row["accuracy"] for row in layer_rows],
            marker="s",
            label="accuracy",
        )
        ax.set(title=f"Term {term_position + 1}", xlabel="Layer", ylim=(0, 1.02))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.supylabel("Held-out-function decoding score")
    fig.tight_layout()
    f1_plot_path = run_dir / "attention_term_f1.png"
    fig.savefig(f1_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, len(x_test))
    indices = rng.choice(len(x_test), size=sample_count, replace=False)
    color_values = [targets[indices]] + [y_test[indices, i] for i in range(num_terms)]
    color_names = ["Target value"] + [f"Term {i + 1} ID" for i in range(num_terms)]
    fig, axes = plt.subplots(
        num_layers,
        len(color_values),
        figsize=(4.2 * len(color_values), 4 * num_layers),
        squeeze=False,
    )
    for layer in range(num_layers):
        embedding = TSNE(
            n_components=2,
            perplexity=min(30, max(5, sample_count - 1)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x_test[indices, layer, :])
        for column, (colors, color_name) in enumerate(zip(color_values, color_names)):
            ax = axes[layer, column]
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=colors,
                s=8, alpha=0.75, cmap="viridis" if column == 0 else "turbo",
            )
            ax.set_title(f"Layer {layer + 1} - {color_name}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Response-token self-attention codes across heads (held-out functions)",
        y=1.01,
    )
    fig.tight_layout()
    tsne_path = run_dir / "attention_tsne.png"
    fig.savefig(tsne_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "latent_dataset": str(latent_paths[-1]),
        "train_split": "train_fixed_context",
        "test_split": "test_fixed_context",
        "tsne_samples": sample_count,
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "f1_csv": str(f1_path),
        "f1_plot": str(f1_plot_path),
        "tsne_plot": str(tsne_path),
    }
    analysis_path = run_dir / "attention_analysis.json"
    analysis_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    metrics_path = run_dir / "final_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        metrics["attention_mean_macro_f1"] = summary["mean_macro_f1"]
        metrics["attention_mean_accuracy"] = summary["mean_accuracy"]
        temporary_path = metrics_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        temporary_path.replace(metrics_path)

    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    analyze(args.run_dir, args.max_samples, args.seed)


if __name__ == "__main__":
    main()
