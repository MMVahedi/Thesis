# Design

## Context

Runnable code currently lives in two root packages (`models/`, `datasets/`) imported repo-root-relative by 5 Python files and 2 notebooks; notebooks carry a `sys.path.insert(0, ".")` workaround because the local `datasets/` package shadows HuggingFace's `datasets` when running from the root. Root also holds 3 loose research notes and 2 loose PDFs. `experiments/` holds two self-contained experiment dirs plus two committed upload zips (tracked in git; the dirs never import root code; the JAX dir's only outside-repo reference is the README's `../../hypernetwork-attention` path — verified no code references it). There is no test suite; `datasets/README.md` documents usage examples; `AGENTS.md` is the working contract for agents. See proposal.md — Why for motivation and specs/repo-structure for the target contract. The user directed (scope amendment) that the committed zips be removed and the experiment dirs move under the package too.

## Goals / Non-Goals

**Goals:**
- One mechanical, reviewable move per content kind; `git mv` so history/blame follow files
- Import migration that is a pure prefix change (`models.` → `compgen.models.`, `datasets.` → `compgen.datasets.`), verified by scripted checks
- Root that a reader can parse in one screen

**Non-Goals:**
- Any behavior change in models, generators, notebooks, or experiment code (same objects, same outputs, same seeds; experiments only relocate)
- Regenerating or recommitting upload zips (they become untracked build artifacts, rebuilt on demand)
- Packaging for pip/PyPI (no `pyproject.toml`; sys.path-based use stays the norm)
- Git commits — the apply ends at a clean, verified working tree; committing is the user's call

## Decisions

1. **Package shape: `compgen/` wrapping the existing subpackages** (`compgen/models/`, `compgen/datasets/`), not a flatter `compgen/attentions|data/...`.
   - Rationale: every import becomes a prefix substitution; the `models.tasks.match3` / `datasets.torch_datasets.match3` mental model in `datasets/README.md` and AGENTS.md carries over; the HF-shadowing problem is solved by demotion from top-level, not by renaming inside.
   - Alternatives: (a) flat `compgen/attentions|embeddings|...` — cleaner-looking but a second rename layer to get wrong; (b) `src/compgen/` PEP-621-style src layout — pointless without packaging, and it changes how notebooks find the code; (c) rename only one dir (e.g. `datasets/`→`data/`) — doesn't fix root clutter or shadowing symmetrically.

2. **Package name `compgen`** — recorded assumption in the proposal; single find/replace to change before implementation if the user prefers another name.

3. **Documents directory: `documents/`** for user-authored thesis PDFs, kept separate from `papers/` (literature).
   - Alternative considered: merging into `papers/` — rejected: conflates own writing with cited sources; the proposal PDF and the thesis draft are not literature.

4. **Notes move preserves Obsidian semantics**: vault root stays where it is; files keep frontmatter; no `[[wiki-links]]` exist in any moved note (verified by grep), so path-based breakage risk is nil.

5. **Notebook migration**: update each import cell to `compgen.*`; keep the `sys.path.insert(0, <repo root>)` line (still needed to locate the repo when kernels start elsewhere) but rewrite its stale shadowing comment. Edits target exact code-cell strings; notebooks are then validated with `nbformat` parse + code-cell `compile()`, not full execution (torch-heavy runs belong to experiments, not to a layout check).

6. **Verification is scripted but not committed**: a throwaway smoke script (run from repo root at apply time) that (a) imports every public module under `compgen/`, (b) generates a 2-sample Match3 dataset to `/tmp` and builds `Match3Model` from the new paths, (c) extracts and compiles all notebook code cells, (d) greps `.py`/`.ipynb`/`.md` for old-style imports and stale paths. Rationale: AGENTS.md explicitly avoids building test infrastructure; a committed checker would be maintenance for no recurring value. The smoke script lives in `/tmp` and is never part of the repo.

7. **Experiments relocate to `compgen/experiments/` (user-directed)** — the two dirs move with `git mv`, keeping their names and internal content byte-for-byte except the JAX README's outside-repo reference (`../../hypernetwork-attention` → `../../../hypernetwork-attention`; one level deeper).
   - Rationale: one roof for all runnable code, per the user's direction; experiment *isolation is an import rule, not a location rule* — the spec keeps the "no package imports" contract, so portability for zipping/uploading is unaffected.
   - Alternatives: (a) keep `experiments/` top-level — the original plan, reversed by user direction; (b) rename the dirs for consistency — rejected: dir names are referenced in upload flows and results naming; churn without benefit.
   - Consequence: the zip build commands in the experiment READMEs (and AGENTS.md's rule) change their working directory from `experiments/` to `compgen/experiments/`; the archive *contents* are unchanged, so Kaggle/Colab upload flows and the notebooks' internal instructions stay valid.

8. **Committed upload zips removed; `*.zip` gitignored (user-directed)** — `git rm experiments/*.zip`, add `*.zip` to `.gitignore`.
   - Rationale: zips are build artifacts of the experiment dirs, reproducible via the documented `zip -r` commands; committing them caused the churn the user now wants gone (regenerating after any experiment edit is manual and error-prone).
   - Alternative: keep zips committed but moved — rejected: contradicts the removal request; a repo-wide `*.zip` rule also covers the notebooks' runtime result zips (`match3_own_stack_results.zip`), which are generated on Colab and never belong in git.

9. **`.gitignore` gains `data/`** (to match what AGENTS.md already claims) **and `*.zip`** (per decision 8), plus nothing else.

## Risks / Trade-offs

- [Hidden old-path references (docs, docstrings, notebook markdown cells, experiment-internal upload instructions)] → full-tree grep sweep including `.ipynb` and `papers/*.md`; experiment-internal instructions that mention "the zip" stay valid because archive contents are unchanged; the swept items are the JAX README's relative path and every doc naming `experiments/` as a root dir.
- [JAX experiment's outside-repo `../../hypernetwork-attention` breaks at one level deeper] → single documented line in its README updated to `../../../hypernetwork-attention`; verified no `.py`/config references that path, so nothing else can silently break.
- [Notebook JSON corruption from careless text edits] → string-targeted edits + `nbformat` validation + code-cell compile check.
- [History noise: moves + edits in one commit look large] → apply stages moves and edits as separate steps; recommended single commit keeps revert trivial (`git revert` restores everything atomically).
- [Untracked files in flight (`AGENTS.md`, `papers/*.md`, `openspec/`, `.opencode/`)] → plain `mv` for untracked content; final `git status` is presented to the user before any commit; no generated artifact gets tracked (spec scenario covers this).
- [`compgen` name clash with an installed package] → checked at apply time (`python -c "import compgen"` from a neutral cwd) before the migration proceeds; any clash is a rename decision surfaced to the user.

## Migration Plan

1. Preflight: neutral-cwd `import compgen` clash check; record `git status` baseline.
2. Create `compgen/` (+ `__init__.py`), `notes/`, `documents/`.
3. `git mv models compgen/models`; `git mv datasets compgen/datasets` (includes `datasets/README.md`); `git mv` both experiment dirs to `compgen/experiments/`.
4. `git rm experiments/*.zip`; add `data/` and `*.zip` to `.gitignore`.
5. Prefix-update the 8 import lines in 5 `.py` files; fix `models/embeddings/match3.py:11` docstring path.
6. Update both notebooks' import cells and `sys.path` comments; update the JAX experiment README's `../../hypernetwork-attention` reference and its zip build-command context; update AGENTS.md's zip rule wording.
7. `git mv` the 3 root research notes to `notes/`; the 2 root PDFs to `documents/`.
8. Rewrite `AGENTS.md` layout/imports/data-flow/experiments sections (also correct its `tasks/` vs `embeddings/` docstring attribution); write root `README.md`.
9. Run the smoke script; grep sweep (old imports, `compgen` imports inside `compgen/experiments/`, stale paths, tracked zips); present `git status` to the user for commit.

Rollback: everything lands as staged-but-uncommitted (or one commit if the user asks); `git checkout`/`git revert` returns to the current layout.

## Open Questions

- Package name: `compgen` is the recorded assumption — confirm or replace at the start of apply (one find/replace if changed).
- Root `README.md` tone/length: currently planned as a one-screen map (project question, two prongs, directory table, quickstart pointers to `datasets/README.md` and the two experiment READMEs); adjust freely during apply if the user wants more or less.
