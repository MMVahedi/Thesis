# AGENTS.md

Thesis research workspace on attention/contrastive learning for compositional generalization. Not a software product: no build system, CI, linters, or test suite. Also an Obsidian vault (`.obsidian/`, gitignored).

## Frameworks

- All implementation in this project uses PyTorch — do not introduce JAX, TensorFlow, or any other deep-learning framework in library code, notebooks, or experiments.

## Layout

- `compgen/` — the single code package; all runnable code lives under it, imported as `compgen.*`.
  - `compgen/models/` — PyTorch attention variants in `compgen/models/attentions/` (`softmax`, `linear`, `strassen`, `triangular`, `third_order`) with the shared name→class registry in `compgen/models/attentions/__init__.py` (`ATTENTION_CLASSES`; triangular is pair-level `(B, N, N, C)` — not sequence-composable) and a shared `compgen/models/attentions/base.py` (`BaseAttention`) that gives every architecture a configurable head count and independently configurable QK/V head dimensions (`softmax` and `linear` are siblings differing only in the final normalization; the high-order variants are retained for future work), task-agnostic scaffolding in `compgen/models/encoder.py` (`EncoderLayer`, with `relu_stack`/`gelu` feed-forward variants, optional pre-norm, optional relative-position bias from `compgen/models/position.py`, and an optional additive `score_bias`), `compgen/models/heads.py` (`TokenClassifier` and `RegressionHead`), and reference initialization in `compgen/models/initialization.py`; task models as thin glue: a per-task embedding in `compgen/models/embeddings/<task>.py` plus a per-task composition in `compgen/models/tasks/<task>.py` (`match3.py` — Match3 classifier reproducing the strassen-attention-neurips25 config: M=37, hidden_dim=128, 1 layer, 2 heads, dropout 0.4, no LayerNorm; `fuzzy_logic.py` — fuzzy-logic regression reproducing the reference "Attention as a Hypernetwork" transformer). Adding a task = one `embeddings/<task>.py` + one thin `tasks/<task>.py`; adding an architecture = one `attentions/<arch>.py` + one `ATTENTION_CLASSES` entry.
  - `compgen/datasets/` — two-layer split: `compgen/datasets/generators/` (pure, torch-free; writes `.jsonl`) and `compgen/datasets/torch_datasets/` (reads `.jsonl` → `torch.utils.data.Dataset` + collate_fn; also hosts the on-the-fly fuzzy-logic sampler `fuzzy_logic_online.py`). See `compgen/datasets/README.md` for task specs and usage examples.
  - `compgen/losses/` — reusable training objectives: the shared-term supervised contrastive loss (`contrastive.py`) and a model-agnostic combined task+contrastive loss (`combined.py`).
  - `compgen/experiments/` — the retained `fuzzy_logic_attention_contrastive_pytorch` analysis archive, with its own README and requirements file.
- `notebooks/` — Match3 experiments comparing Strassen vs. softmax attention; import `compgen.models`/`compgen.datasets`.
- `notes/` — research notes (Obsidian markdown).
- `documents/` — thesis-authored documents (`proposal.pdf`, thesis draft PDF).
- `papers/`, `presentations/` — literature (summaries + PDFs) and slide decks.

## Running code

- Imports are package-prefixed (`from compgen.models.attentions.strassen import ...`, `from compgen.datasets.generators.match3 import ...`). Run scripts/notebooks with the repo root on `sys.path` (e.g. `python` from the root, or add `sys.path.append("..")` in notebooks). There is no installed package.
- Running from the repo root no longer shadows HuggingFace's `datasets`: there is no top-level `datasets/` directory, so `import datasets` resolves to the installed package if present; local code is only reachable as `compgen.datasets`.
- Root `requirements.txt` pins the local PyTorch-side runtime environment (`torch==2.14.0+cu132`, `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `ipython`). Install or refresh it from the repo root with `pip install -r requirements.txt`. It targets the GPU server (Python 3.12, driver 595.91.07 / CUDA 13.2, RTX 2060 `sm_75`); the retained experiment keeps its own requirements file.
- Development environment: the local machine is the GPU server — NVIDIA GeForce RTX 2060 (Turing, `sm_75`), driver 595.91.07 / CUDA 13.2 — and code runs from the repo-root `venv/` (Python 3.12) with CUDA-enabled PyTorch, using `venv/bin/python`. Colab/Kaggle remain available for notebooks.
- Attention classes default to `torch.float64` and use `opt_einsum.contract` for the higher-order score contractions.
- The fuzzy-logic task model reproduces the reference transformer at 151,009 parameters for a 4-variable/2-term task (`compgen/models/tasks/fuzzy_logic.py`).

## Data flow

- Generate a dataset once to `.jsonl` via a `compgen/datasets/generators/` class, then point a `compgen/datasets/torch_datasets/` class at the file. Generation is seeded/deterministic; regenerate deliberately, not casually.
- `*.jsonl`, `data/`, `logs/`, `wandb/`, `checkpoints/`, `results/`, `outputs/`, `*.pt`, `*.ckpt`, `*.zip` are gitignored — training artifacts and upload archives never go in git.

## compgen/experiments/ rules

- Experiment dirs are intentionally isolated: they live under the package but do NOT import `compgen` (or any root code), and must stay that way (portable for zipping/uploading).
- The retained `fuzzy_logic_attention_contrastive_pytorch` dir is an analysis archive: its model, dataset, loss, training, self-test, and notebook-generation modules moved into `compgen/` (see `compgen/models/tasks/fuzzy_logic.py`, `compgen/datasets/torch_datasets/fuzzy_logic_online.py`, `compgen/losses/`). Its committed Kaggle notebook is frozen — `make_notebook.py` was removed.
- The `hypernetwork-attention` repo lives outside this workspace and must never be modified.

## Notes conventions

- Research notes live in `notes/` as Obsidian markdown with YAML frontmatter (`title`, `aliases`, `tags`, `status`, `created`) — follow that format when adding notes.
- `compgen/models/embeddings/match3.py`'s docstring references its generator: the real path is `compgen/datasets/generators/match3.py`. Keep the two in sync if either moves.
