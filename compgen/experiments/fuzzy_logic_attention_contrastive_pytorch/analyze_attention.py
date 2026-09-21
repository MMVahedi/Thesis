"""Paper-style clustering/decoding analysis of saved PyTorch attention codes."""

import argparse
import csv
import json
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


def to_term_ids(latents):
    weights = 2 ** np.arange(latents.shape[-1] - 1, -1, -1)
    return np.sum(latents * weights, axis=-1).astype(np.int32)


def analyze(run_dir, max_samples=2_000, seed=0):
    data = np.load(run_dir / "attention_codes.npz")
    x_train, x_test = data["train_attention"], data["test_attention"]
    y_train, y_test = to_term_ids(data["train_latents"]), to_term_ids(data["test_latents"])
    targets = data["test_target"].reshape(len(x_test), -1)[:, -1]
    num_layers, num_heads = x_train.shape[1:]
    num_terms = y_train.shape[1]

    rows = []
    for layer in range(num_layers):
        for term in range(num_terms):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2_000, class_weight="balanced"),
            )
            classifier.fit(x_train[:, layer], y_train[:, term])
            prediction = classifier.predict(x_test[:, layer])
            rows.append({
                "layer": layer + 1,
                "term_position": term + 1,
                "macro_f1": f1_score(y_test[:, term], prediction, average="macro"),
                "accuracy": accuracy_score(y_test[:, term], prediction),
                "num_heads": num_heads,
                "train_samples": len(x_train),
                "test_samples": len(x_test),
            })
    with (run_dir / "attention_term_decoding.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, num_terms, figsize=(5 * num_terms, 4), squeeze=False)
    for term in range(num_terms):
        selected = [row for row in rows if row["term_position"] == term + 1]
        axes[0, term].plot(
            [row["layer"] for row in selected], [row["macro_f1"] for row in selected],
            marker="o", label="macro F1",
        )
        axes[0, term].plot(
            [row["layer"] for row in selected], [row["accuracy"] for row in selected],
            marker="s", label="accuracy",
        )
        axes[0, term].set(title=f"Term {term + 1}", xlabel="Layer", ylim=(0, 1.02))
        axes[0, term].grid(alpha=0.25)
        axes[0, term].legend()
    fig.supylabel("Held-out-function decoding score")
    fig.tight_layout()
    fig.savefig(run_dir / "attention_term_decoding.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.default_rng(seed)
    sample_count = min(max_samples, len(x_test))
    if sample_count < 6:
        raise ValueError("t-SNE attention analysis requires at least 6 samples")
    indices = rng.choice(len(x_test), sample_count, replace=False)
    colors = [("target", targets[indices])] + [
        (f"term {term + 1}", y_test[indices, term]) for term in range(num_terms)
    ]
    fig, axes = plt.subplots(
        num_layers, len(colors), figsize=(4.4 * len(colors), 4 * num_layers), squeeze=False
    )
    for layer in range(num_layers):
        embedding = TSNE(
            n_components=2,
            perplexity=min(30, max(5, sample_count - 1)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(x_test[indices, layer])
        for column, (name, values) in enumerate(colors):
            scatter = axes[layer, column].scatter(
                embedding[:, 0], embedding[:, 1], c=values, s=8, alpha=0.75,
                cmap="viridis" if column == 0 else "turbo",
            )
            axes[layer, column].set_title(f"Layer {layer + 1}: {name}")
            axes[layer, column].set_xticks([])
            axes[layer, column].set_yticks([])
            fig.colorbar(scatter, ax=axes[layer, column], fraction=0.046, pad=0.04)
    fig.suptitle("Response-token self-attention codes across heads", y=1.01)
    fig.tight_layout()
    fig.savefig(run_dir / "attention_tsne.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
        "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "tsne_samples": sample_count,
        "code_definition": "per-layer response-token self-attention weights across heads",
    }
    metrics_path = run_dir / "final_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.update({f"attention_{key}": value for key, value in summary.items()})
    temporary_path = metrics_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    temporary_path.replace(metrics_path)
    # Write the analysis completion marker last so interrupted analysis is retried.
    (run_dir / "attention_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    analyze(args.run_dir.resolve(), args.max_samples, args.seed)
