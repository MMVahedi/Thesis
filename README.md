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
| `compgen/models/` | Attention architectures (`attentions/` — `softmax` and `linear` mixers plus retained high-order variants (`strassen`, `triangular`, `third_order`), all sharing `base.py` with a configurable head count and independently configurable QK/V head dimensions — and the `ATTENTION_CLASSES` registry, with softmax/linear accepting an optional additive `score_bias`), task-agnostic scaffolding (`encoder.py` with `relu_stack`/`gelu` feed-forward and optional relative-position bias from `position.py`, `heads.py` with `TokenClassifier`/`RegressionHead`, reference init in `initialization.py`), and task models as thin glue (`embeddings/<task>.py` + `tasks/<task>.py` — Match3 and fuzzy-logic regression) |
| `compgen/datasets/` | Dataset generators (`generators/`, torch-free, write `.jsonl`) and PyTorch datasets (`torch_datasets/`, including the on-the-fly fuzzy-logic sampler); see `compgen/datasets/README.md` for task specs and usage examples |
| `compgen/losses/` | Reusable objectives: the shared-term supervised contrastive loss and a model-agnostic combined task+contrastive loss |
| `compgen/experiments/` | The retained self-contained PyTorch fuzzy-logic analysis archive with its own README, requirements, and frozen notebook — it imports nothing from the rest of the repo |
| `notebooks/` | Match3 notebooks comparing Strassen vs. softmax attention |
| `notes/` | Research notes (Obsidian markdown) |
| `documents/` | Thesis-authored PDFs (proposal, thesis draft) |
| `papers/` | Literature: paper summaries, metadata, and PDFs |
| `presentations/` | Slide decks |
| `openspec/` | OpenSpec planning (active changes + specs) |

## Quickstart

- Library usage and dataset generation: see `compgen/datasets/README.md`.
- Run the contrastive experiment / fuzzy-logic model: see `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/README.md` (analysis archive); the model, on-the-fly data, and loss now live under `compgen/` (`models/tasks/fuzzy_logic.py`, `datasets/torch_datasets/fuzzy_logic_online.py`, `losses/`).
- Match3 model comparisons: open a notebook in `notebooks/` (run with the repo root on `sys.path`).

Agent-facing working conventions live in `AGENTS.md`.
