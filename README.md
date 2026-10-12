# Thesis Workspace

Research on **compositional generalization in Transformers through structured attention**: can changing what the attention score computes — or what it is encouraged to represent — make small Transformers recombine learned constituents into unseen compositions?

Two prongs share one library, `compgen/`:

| Prong | Idea | Where |
|---|---|---|
| **Contrastive regularization** | Tasks sharing a fuzzy-logic constituent term get pulled together at the final-layer attention representation (shared-term InfoNCE on the query token's per-head attention probabilities) | `compgen/losses/`, `compgen/models/tasks/fuzzy_logic.py`, `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` |
| **Structured attention** | Attention mixers beyond plain bilinear scoring — softmax and linear mixers plus retained high-order variants (Strassen, triangular, third-order) — evaluated on the Match3 task | `compgen/models/attentions/`, `notebooks/` |

## Layout

| Path | Contents |
|---|---|
| `compgen/models/` | Attention architectures (`attentions/` — `softmax` and `linear` mixers plus retained high-order variants (`strassen`, `triangular`, `third_order`), all sharing `base.py` with a configurable head count, independently configurable QK/V head dimensions, and projection weights left at PyTorch's default initialization (the task model initializes the whole model); callers may still instantiate the concrete classes directly; a model's architecture is selected by a typed `AttentionArchitecture` enum (never a raw string) through the config-driven factory, with softmax/linear accepting an optional additive `score_bias`), task-agnostic scaffolding (`encoder.py` consuming caller-constructed attention and feed-forward modules, applying no dropout and not re-initializing its inputs; `ffn.py` with the `relu_stack`/`gelu` feed-forward variants, `position.py` with a fixed sinusoidal absolute positional encoding added by the task embeddings, `heads.py` with `TokenClassifier`/`RegressionHead`, centralized weight initialization in `initialization.py` — an abstract `WeightInitializer` base with one subclass per scheme (`XavierInitializer`, `ReferenceInitializer`, `DefaultInitializer`), applied only by the task models through a single initialization method over the whole model, so sub-level modules do not initialize), per-task configuration dataclasses and `AttentionArchitecture`/`FeedForwardVariant` enums (`configs.py`), a `build_model(config)` factory (`factory.py`) that constructs attention/feed-forward from those enums and returns the composed model, and task models as thin glue taking caller-supplied, already-constructed attention and feed-forward modules (`embeddings/<task>.py` + `tasks/<task>.py` — Match3 and fuzzy-logic regression) |
| `compgen/datasets/` | Dataset generators (`generators/`, torch-free, write `.jsonl`) and PyTorch datasets (`torch_datasets/`, including the on-the-fly fuzzy-logic sampler); see `compgen/datasets/README.md` for task specs and usage examples |
| `compgen/losses/` | Reusable objectives: the shared-term supervised contrastive loss and a model-agnostic combined task+contrastive loss |
| `compgen/analytics/` | Reusable, task-agnostic interpretability tools: attention→vector extraction and per-token attention inspection (`attention.py`), PCA projection of representation vectors to (by default) 2D (`projection.py`), scatter/heatmap figure helpers (`plotting.py`), and a model capture helper (`capture.py`) that records per-layer attention and hidden states |
| `compgen/tracking/` | MLflow-backed experiment tracking: run lifecycle and logging of parameters, metrics (with steps), artifacts, and PyTorch models; defaults to the local `sqlite:///mlflow.db` store (artifacts under `mlartifacts/`), overridable via an explicit URI or `MLFLOW_TRACKING_URI` |
| `compgen/training/` | Task-agnostic training/evaluation scaffold: `fit` runs epochs over caller-supplied data and a `step_fn(model, batch) -> (loss, metrics)`, handling optimizer/scheduler, device, seeding, evaluation, best-checkpoint selection, and early stopping, and records runs through `compgen.tracking` |
| `compgen/experiments/` | The retained self-contained PyTorch fuzzy-logic analysis archive with its own README, requirements, and frozen notebook — it imports nothing from the rest of the repo |
| `compgen/runners/` | Runnable experiment entry points that compose the package; currently the four-model fuzzy-logic contrastive comparison (`fuzzy_logic_contrastive/`: softmax/linear attention × without/with the final-layer response-token attention-code contrastive term), run with `python -m compgen.runners.fuzzy_logic_contrastive` |
| `notebooks/` | Match3 notebooks comparing Strassen vs. softmax attention |
| `notes/` | Research notes (Obsidian markdown) |
| `documents/` | Thesis-authored PDFs (proposal, thesis draft) |
| `papers/` | Literature: paper summaries, metadata, and PDFs |
| `presentations/` | Slide decks |
| `openspec/` | OpenSpec planning (active changes + specs) |

## Quickstart

- Library usage and dataset generation: see `compgen/datasets/README.md`.
- Run the contrastive experiment / fuzzy-logic model: see `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/README.md` (analysis archive); the model, on-the-fly data, and loss now live under `compgen/` (`models/tasks/fuzzy_logic.py`, `datasets/torch_datasets/fuzzy_logic_online.py`, `losses/`).
- Run the four-model fuzzy-logic contrastive comparison (softmax/linear × without/with the response-token attention-code contrastive term): `python -m compgen.runners.fuzzy_logic_contrastive` (see `compgen/runners/fuzzy_logic_contrastive/`).
- Match3 model comparisons: open a notebook in `notebooks/` (run with the repo root on `sys.path`).

### Interpretability

`compgen/analytics/` runs on live model outputs. `capture` records per-layer attention, `attention_code` / `attention_row_features` / `attention_matrix_features` turn it into vectors, and `pca_project` projects them to 2D:

```python
import torch
from compgen.analytics import (
    capture, attention_code, attention_row_features, attention_matrix_features,
    token_attention_row, pca_project, plot_projection, plot_attention,
)
from compgen.models import FuzzyLogicConfig, build_model

model = build_model(FuzzyLogicConfig(input_dim=5, hidden_dim=128, num_layers=2, num_heads=8,
                                     qk_head_dim=2, v_head_dim=2, ffn_dim=256))
result = capture(model, torch.rand(32, 16, 5))       # per-layer (B, heads, N, N)
codes = attention_code(result.attentions[-1])         # (B, heads)
row = attention_row_features(result.attentions[-1])   # (B, heads * N)
matrix = attention_matrix_features(result.attentions[-1])  # (B, heads * N * N)
coords = pca_project(codes).coordinates               # (B, 2)
plot_projection(coords)
row = token_attention_row(result.attentions, layer=-1, query_token=-1, average_heads=True)
plot_attention(row)                                   # (N,) attention of the last token
```

`Match3Model.forward(batch, return_attention=True)` returns `(y_hat, hidden_state, attentions)` directly, for callers who prefer the model API over `capture`.

Agent-facing working conventions live in `AGENTS.md`.
