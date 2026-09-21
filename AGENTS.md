# AGENTS.md

Thesis research workspace on attention/contrastive learning for compositional generalization. Not a software product: no build system, CI, linters, or test suite. Also an Obsidian vault (`.obsidian/`, gitignored).

## Frameworks

- All implementation in this project uses PyTorch — do not introduce JAX, TensorFlow, or any other deep-learning framework in library code, notebooks, or new experiments.
- The existing JAX experiment suite `compgen/experiments/fuzzy_logic_attention_contrastive/` is grandfathered: keep it working, but do not extend it with new JAX code; new experiment work goes through `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` or new PyTorch dirs.

## Layout

- `compgen/` — the single code package; all runnable code lives under it, imported as `compgen.*`.
  - `compgen/models/` — PyTorch attention variants in `compgen/models/attentions/` (`standard`, `strassen`, `triangular`, `third_order`) with the shared name→class registry in `compgen/models/attentions/__init__.py` (`ATTENTION_CLASSES`; triangular is pair-level `(B, N, N, C)` — not sequence-composable), task-agnostic scaffolding in `compgen/models/encoder.py` (`EncoderLayer`) and `compgen/models/heads.py` (`TokenClassifier`), and task models as thin glue: a per-task embedding in `compgen/models/embeddings/<task>.py` plus a per-task composition in `compgen/models/tasks/<task>.py` (currently `match3.py` — Match3 classifier reproducing the strassen-attention-neurips25 config: M=37, hidden_dim=128, 1 layer, 2 heads, dropout 0.4, no LayerNorm). Adding a task = one `embeddings/<task>.py` + one thin `tasks/<task>.py`; adding an architecture = one `attentions/<arch>.py` + one `ATTENTION_CLASSES` entry.
  - `compgen/datasets/` — two-layer split: `compgen/datasets/generators/` (pure, torch-free; writes `.jsonl`) and `compgen/datasets/torch_datasets/` (reads `.jsonl` → `torch.utils.data.Dataset` + collate_fn). See `compgen/datasets/README.md` for task specs and usage examples.
  - `compgen/experiments/` — two self-contained experiment dirs (`fuzzy_logic_attention_contrastive` [JAX], `fuzzy_logic_attention_contrastive_pytorch` [PyTorch]), each with its own README (the executable source of truth for running them) and requirements files.
- `notebooks/` — Match3 experiments comparing Strassen vs. standard attention; import `compgen.models`/`compgen.datasets`.
- `notes/` — research notes (Obsidian markdown).
- `documents/` — thesis-authored documents (`proposal.pdf`, thesis draft PDF).
- `papers/`, `presentations/` — literature (summaries + PDFs) and slide decks.

## Running code

- Imports are package-prefixed (`from compgen.models.attentions.strassen import ...`, `from compgen.datasets.generators.match3 import ...`). Run scripts/notebooks with the repo root on `sys.path` (e.g. `python` from the root, or add `sys.path.append("..")` in notebooks). There is no installed package.
- Running from the repo root no longer shadows HuggingFace's `datasets`: there is no top-level `datasets/` directory, so `import datasets` resolves to the installed package if present; local code is only reachable as `compgen.datasets`.
- No root requirements file; notebooks/experiments assume `torch`, `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib` are available. Python 3.14 is in use (`__pycache__`).
- Development environment: the local machine has **no GPU and no PyTorch install**. Code is written here; torch-dependent code is executed and tested on the GPU server (or Colab/Kaggle for notebooks). Local verification is best-effort CPU-only (e.g. a temporary venv); never block implementation on torch/GPU-dependent tests — run those when on the GPU server.
- Attention classes default to `torch.float64` and use `opt_einsum.contract` for the higher-order score contractions.
- Only self-check: `python self_test.py` inside `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/`.

## Data flow

- Generate a dataset once to `.jsonl` via a `compgen/datasets/generators/` class, then point a `compgen/datasets/torch_datasets/` class at the file. Generation is seeded/deterministic; regenerate deliberately, not casually.
- `*.jsonl`, `data/`, `logs/`, `wandb/`, `checkpoints/`, `results/`, `outputs/`, `*.pt`, `*.ckpt`, `*.zip` are gitignored — training artifacts and upload archives never go in git.

## compgen/experiments/ rules

- Experiment dirs are intentionally isolated: they live under the package but do NOT import `compgen` (or any root code), and must stay that way (portable for zipping/uploading to Colab/Kaggle).
- Upload zips are untracked build artifacts: rebuild on demand with the exact `zip -r` command in each experiment README (run from `compgen/experiments/`, excluding `__pycache__`/`logs`/`wandb`).
- In `compgen/experiments/fuzzy_logic_attention_contrastive/`, the `hyla/` dir is the upstream package name only — it contains plain standard attention too; don't assume `hyla` = HyLA variant.
- `run.py --config.lambda_contrastive=0` reduces to the original MSE baseline; the JAX experiment reads `../../../hypernetwork-attention` as reference only — that repo lives outside this workspace and must never be modified.
- The PyTorch experiment's committed notebook is generated by `make_notebook.py` in that dir.

## Notes conventions

- Research notes live in `notes/` as Obsidian markdown with YAML frontmatter (`title`, `aliases`, `tags`, `status`, `created`) — follow that format when adding notes.
- `compgen/models/embeddings/match3.py`'s docstring references its generator: the real path is `compgen/datasets/generators/match3.py`. Keep the two in sync if either moves.
