# Proposal

## Why

The workspace has grown organically across two research prongs (attention contrastive regularization, structured attention variants) and now mixes runnable code, research notes, thesis documents, and literature at the repo root. The root is cluttered (3 loose research notes, 2 loose PDFs), the local `datasets/` package shadows HuggingFace's `datasets` when running from the root (a documented trap requiring a `sys.path` workaround in notebooks), and import paths (`from models...`, `from datasets...`) expose top-level packages that only make sense in this one repo. Before the thesis moves further into experiments and writing, the layout should be made navigable once, not patched piecemeal.

## What Changes

- **New top-level package `compgen/`** (name adjustable before implementation) containing all runnable code:
  - `compgen/models/` (attentions/, embeddings/, tasks/) and `compgen/datasets/` (generators/, torch_datasets/, README.md), moved via `git mv` to preserve history.
  - **BREAKING**: all imports change from `from models.attentions...` / `from datasets.generators...` to `from compgen.models...` / `from compgen.datasets...` (5 Python files affected, 8 import lines; plus 2 notebooks).
  - Side benefit: no top-level `datasets/` directory remains, so HuggingFace's `datasets` is no longer shadowed and the notebooks' `sys.path.insert(0, ".")` workaround loses its purpose.
- **Root cleanup**:
  - 3 research notes (`attention_contrastive_compositional_generalization.md`, `deep-research-report.md`, `Compositional-Reasoning-with-Transformers-RNNs-and-COT.md`) move to `notes/` (Obsidian vault: frontmatter format unchanged, `.obsidian/` stays at root).
  - `proposal.pdf` and `Trainable Key_Query Decompositions...-1.pdf` move to `documents/`.
- **Doc accuracy**: fix the stale `dataset/generators/match3.py` reference in `models/embeddings/match3.py:11` (real location becomes `compgen/datasets/generators/match3.py`; note AGENTS.md currently misattributes this to `models/tasks/match3.py` — the rewrite fixes both).
- **AGENTS.md rewritten** for the new layout; new root `README.md` giving a one-screen project map (research question, prongs, where things live).
- **`.gitignore`**: add `data/` (AGENTS.md already claims it is ignored; it is not).
- **Experiments join the package (user-directed scope amendment)**:
  - `experiments/fuzzy_logic_attention_contrastive/` and `experiments/fuzzy_logic_attention_contrastive_pytorch/` move to `compgen/experiments/` via `git mv`. Self-containment is preserved: they keep importing nothing from `compgen` or root code (location change only — the zip archives' contents are unchanged, so Kaggle/Colab upload flows are unaffected).
  - One path fix inside the JAX experiment: its README's outside-repo reference `../../hypernetwork-attention` becomes `../../../hypernetwork-attention` (the dir sits one level deeper). No experiment code references that path.
- **Committed upload archives removed**: `experiments/fuzzy_logic_attention_contrastive.zip` and `experiments/fuzzy_logic_attention_contrastive_pytorch.zip` are `git rm`'d and `*.zip` is gitignored — they are build artifacts of the experiment dirs, reproducible on demand with the `zip -r` commands documented in each experiment README (which stay the source of truth).
- **Explicitly unchanged**: `papers/`, `presentations/`, `openspec/`, `.obsidian/`, and all code behavior.

## Capabilities

### New Capabilities

- `repo-structure`: the organizing contract of the workspace — where code, notes, documents, literature, and experiments live (all runnable code, experiments included, under the single package); the `compgen.*` import discipline and experiment isolation; no committed upload archives; docs that must stay accurate to the layout.

### Modified Capabilities

(none — no prior specs exist; this is the first capability)

## Impact

- **Code (paths only, no behavior)**: `models/` and `datasets/` move under `compgen/`, as do both experiment dirs (`compgen/experiments/`); 5 Python files' imports updated; notebooks' import cells and `sys.path` comments updated; `datasets/README.md` moves to `compgen/datasets/README.md` with import examples updated; the JAX experiment README's `../../hypernetwork-attention` reference deepens by one level.
- **Repo artifacts**: the two committed `experiments/*.zip` archives are deleted from git; `*.zip` joins `.gitignore` (alongside the already-planned `data/`).
- **Docs**: `AGENTS.md` rewritten (including its now-outdated "committed `.zip`" and `experiments/` rules); new root `README.md`.
- **Untouched**: `papers/`, `presentations/`, `.obsidian/`, OpenSpec config, and all code behavior.
- **Verification**: no test suite exists; verification is a scripted smoke check — import every public module from the new package, build a `Match3Model` end-to-end from a freshly generated 2-sample dataset in `/tmp`, extract-and-compile all notebook code cells, run the PyTorch experiment's bundled `self_test.py` from its new location, and grep for leftover old-style imports, `compgen` imports inside `compgen/experiments/`, and stale path references.
- **Assumption (recorded)**: the package name `compgen` is a placeholder chosen for brevity and thesis relevance (compositional generalization); any other name is a mechanical find/replace if preferred — confirm before implementation.
