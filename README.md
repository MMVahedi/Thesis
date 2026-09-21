# Thesis Workspace

Research on **compositional generalization in Transformers through structured attention**: can changing what the attention score computes — or what it is encouraged to represent — make small Transformers recombine learned constituents into unseen compositions?

Two prongs share one library, `compgen/`:

| Prong | Idea | Where |
|---|---|---|
| **Contrastive regularization** | Tasks sharing a fuzzy-logic constituent term get pulled together at the final-layer attention representation (shared-term InfoNCE on the query token's per-head attention probabilities) | `compgen/experiments/fuzzy_logic_attention_contrastive/` (JAX), `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` (PyTorch) |
| **Structured attention** | Attention variants beyond the standard bilinear scorer — Strassen (third-order), triangular, third-order — evaluated on the Match3 task | `compgen/models/attentions/`, `notebooks/` |

## Layout

| Path | Contents |
|---|---|
| `compgen/models/` | Attention architectures (`attentions/` + shared `ATTENTION_CLASSES` registry), task-agnostic scaffolding (`encoder.py`, `heads.py`), and task models as thin glue (`embeddings/<task>.py` + `tasks/<task>.py`) |
| `compgen/datasets/` | Dataset generators (`generators/`, torch-free, write `.jsonl`) and PyTorch datasets (`torch_datasets/`); see `compgen/datasets/README.md` for task specs and usage examples |
| `compgen/experiments/` | Self-contained experiment suites (JAX + PyTorch) with their own READMEs, requirements, and notebooks — they import nothing from the rest of the repo |
| `notebooks/` | Match3 notebooks comparing Strassen vs. standard attention |
| `notes/` | Research notes (Obsidian markdown) |
| `documents/` | Thesis-authored PDFs (proposal, thesis draft) |
| `papers/` | Literature: paper summaries, metadata, and PDFs |
| `presentations/` | Slide decks |
| `openspec/` | OpenSpec planning (active changes + specs) |

## Quickstart

- Library usage and dataset generation: see `compgen/datasets/README.md`.
- Run the contrastive experiment: follow `compgen/experiments/fuzzy_logic_attention_contrastive/README.md` (JAX) or `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/README.md` (PyTorch, incl. the Colab/Kaggle zip workflow).
- Match3 model comparisons: open a notebook in `notebooks/` (run with the repo root on `sys.path`).

Agent-facing working conventions live in `AGENTS.md`.
