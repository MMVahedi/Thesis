# Tasks

## 1. Preflight

- [x] 1.1 Confirm the package name (`compgen` or the user's replacement) with the user; then verify no clash with an installed module by running `python3 -c "import importlib.util; print(importlib.util.find_spec('compgen'))"` from a neutral cwd (expect `None`). Verification: recorded decision + `None` spec.
- [x] 1.2 Capture the working-tree baseline (`git status --short` and `git stash list`) so post-reorg state is auditable. Verification: baseline text saved to the session log.

## 2. Code package migration

- [x] 2.1 Create `compgen/` with `__init__.py` (plus `compgen/models/__init__.py`, `compgen/datasets/__init__.py` if missing after the moves). Verification: `ls compgen/__init__.py` succeeds.
- [x] 2.2 `git mv models compgen/models` and `git mv datasets compgen/datasets`. Verification: `git status` shows renames (R) for all moved files; no untracked copies left at root.
- [x] 2.3 `git rm experiments/fuzzy_logic_attention_contrastive.zip experiments/fuzzy_logic_attention_contrastive_pytorch.zip` and `git mv experiments/fuzzy_logic_attention_contrastive compgen/experiments/fuzzy_logic_attention_contrastive` + `git mv experiments/fuzzy_logic_attention_contrastive_pytorch compgen/experiments/fuzzy_logic_attention_contrastive_pytorch`. Verification: `git status` shows the two zip deletions (D) and two dir renames (R); `ls experiments/` fails (dir gone); `ls compgen/experiments/` lists exactly the two dirs.
- [x] 2.4 Update the 8 old-style import lines in the 5 moved `.py` files (`compgen/datasets/torch_datasets/fuzzy_logic.py`, `compgen/datasets/torch_datasets/match3.py`, `compgen/datasets/generators/fuzzy_logic.py`, `compgen/datasets/generators/match3.py`, `compgen/models/tasks/match3.py`) from `from models...`/`from datasets...` to `from compgen...`, plus the 4 usage examples in `compgen/datasets/README.md` (import discipline covers docs too). Verification: `rg "from (models|datasets)\." compgen/` returns zero matches.
- [x] 2.5 Fix the stale docstring in `compgen/models/embeddings/match3.py:11` to point at `compgen/datasets/generators/match3.py`. Verification: `rg "dataset/generators" compgen/ --glob '!experiments/**'` returns zero matches.
- [x] 2.6 In `compgen/experiments/fuzzy_logic_attention_contrastive/README.md`, update the outside-repo reference `../../hypernetwork-attention` to `../../../hypernetwork-attention` and the zip build-command context (run from `compgen/experiments/`; archive contents unchanged). Verification: `rg -F` with the backtick-quoted old reference returns zero matches in `compgen/` (note: the regex form `\.\./\.\./hypernetwork` false-positives inside the new path; the fixed-string check discriminates); the documented `zip -r` command, run verbatim from `compgen/experiments/`, produces a zip whose dir name matches the uploaded-archive name used by the experiment's notebooks.

## 3. Notebooks

- [x] 3.1 Update both `notebooks/*.ipynb` import cells to `compgen.models/compgen.datasets` paths and rewrite the `sys.path.insert` comment (shadowing workaround no longer the reason; locating the repo is). Verification: `rg "from (models|datasets)\." notebooks/` is empty; notebook JSON structurally validated via `json.loads` on save (deviation recorded: `nbformat` is not installed in this environment, so its named check is substituted by the JSON parse here plus the code-cell `compile()` step in task 6.1 — same coverage, stdlib-only).

## 4. Root content moves

- [x] 4.1 `git mv` the 3 research notes (`attention_contrastive_compositional_generalization.md`, `deep-research-report.md`, `Compositional-Reasoning-with-Transformers-RNNs-and-COT.md`) into `notes/`. Verification: root `ls *.md` shows only `AGENTS.md` (README.md arrives in task 5.2); `git status` shows the renames.
- [x] 4.2 `git mv` `proposal.pdf` and `Trainable Key_Query Decompositions and Structured Attention for Compositional Generalization-1.pdf` into `documents/`. Verification: `ls documents/` lists exactly these two PDFs.

## 5. Docs and hygiene

- [x] 5.1 Rewrite `AGENTS.md` for the new layout (compgen imports incl. `compgen/experiments/`, `notes/`, `documents/`; experiments' isolation rule restated as location-under-package + no-package-imports; zip rule changed to "untracked, rebuild on demand via each README's command") and correct its docstring-location attribution (`embeddings/match3.py`, not `tasks/`). Verification: every repository-relative path named in AGENTS.md exists (`rg -o "[A-Za-z0-9_/.-]+\.py|[A-Za-z0-9_/.-]+\.md" AGENTS.md` spot-check against filesystem; remaining flags are command-style dir-relative mentions whose sentences name their dir).
- [x] 5.2 Write root `README.md`: one-screen map (research question, two prongs, directory table incl. `notes/`, `papers/`, `documents/`, `presentations/`, `compgen/experiments/`, `openspec/`, quickstart pointers to `compgen/datasets/README.md` and the two experiment READMEs). Verification: file exists; mentioned paths exist.
- [x] 5.3 Add `data/` and `*.zip` to `.gitignore` (aligning with the documented-but-missing rule and the untracked-archives decision). Verification: `git check-ignore data/x some.zip` exits 0 for both; `git ls-files | grep "\.zip$"` is empty.

## 6. End-to-end verification

- [x] 6.1 Smoke script from repo root (throwaway, in `/tmp`): import every public module under `compgen/` (attentions ×4, embeddings/match3, tasks/match3, generators ×3, torch_datasets ×2); generate a 2-sample Match3 dataset to `/tmp` via `compgen.datasets.generators.match3` and build `compgen.models.tasks.match3.Match3Model` over it; extract and `compile()` every code cell of both notebooks. **Deviation recorded (user-approved, option 3):** no torch-bearing Python environment exists on this machine, so the module-import and Match3Model forward checks were skipped. Verified instead: 2-sample Match3 dataset generated through `compgen.datasets.generators.match3` (under `/usr/bin/python3` 3.9 + numpy — generators are torch-free by design), and all 46 notebook code cells compile (IPython `!`/`%` magic lines replaced by `pass` for the check). HF-shadowing scenario vacuous here (HuggingFace `datasets` not installed); structural checks (no top-level `datasets/` dir) passed.
- [x] 6.2 Isolation sweep: `rg "from compgen|from models|from datasets|import models|import datasets" compgen/experiments/` returns zero matches; `git ls-files | grep "\.zip$"` is empty; run the PyTorch experiment's `self_test.py` inside `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/`. **Deviation recorded (user-approved, option 3):** `self_test.py` skipped — requires torch. Isolation rg sweep clean (exit 1 = zero matches); zero tracked zips.
- [x] 6.3 Stale-path sweep: `rg "experiments/" AGENTS.md README.md notes/ 2>/dev/null` shows only `compgen/experiments/`-prefixed paths; old JAX relative reference gone (fixed-string check). Verification: no reference to the removed root `experiments/` dir remains in any doc.
- [x] 6.4 Present final `git status --short` and `git diff --stat` to the user; commit only if the user explicitly asks. Verification: user sees the full change summary; working tree state documented.
