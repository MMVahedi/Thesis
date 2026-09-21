# Spec Delta

## Purpose

Defines the organizing contract of the thesis workspace: one code package, a predictable place for every kind of content, self-contained experiments, and docs that match reality — so future research changes land in a navigable repo instead of accumulating at the root.

## ADDED Requirements

### Requirement: Single top-level code package
All runnable library code (attention models, embeddings, tasks, dataset generators, torch datasets) SHALL live under exactly one top-level Python package, the experiment suites SHALL live under `experiments/` within that package (`compgen/experiments/`), and the repository root SHALL NOT contain top-level `models/`, `datasets/`, or `experiments/` code directories. As a result, running Python from the repository root SHALL NOT shadow any identically named installed third-party package (notably HuggingFace `datasets`).

#### Scenario: Root run resolves third-party datasets
- **WHEN** a process is started from the repository root and `import datasets` executes in an environment where HuggingFace `datasets` is installed
- **THEN** the installed HuggingFace package is imported, not any local code

#### Scenario: All library modules reachable through the package
- **WHEN** every public module formerly under `models/` and `datasets/` is imported via the new package prefix (e.g. `compgen.models.attentions.standard`, `compgen.datasets.generators.match3`) from the repository root
- **THEN** all imports succeed without a top-level `models/` or `datasets/` directory existing at the root

#### Scenario: Experiments relocated under the package
- **WHEN** the repository root is listed after the reorganization
- **THEN** no `experiments/` directory exists at the root and both experiment directories are found at `compgen/experiments/` with unchanged internal content

### Requirement: Import discipline
Python files, notebooks, and usage documentation SHALL import local code exclusively through the top-level package prefix (`compgen.*`); no old-style top-level imports (`from models...`, `from datasets...`) or references describing them SHALL remain anywhere in the repository.

#### Scenario: No stale imports in code
- **WHEN** the repository is searched for `from models.` or `from datasets.` patterns in `.py` and `.ipynb` files
- **THEN** zero matches are found

#### Scenario: Notebooks run against the new package
- **WHEN** each notebook's code cells are extracted and compiled, and their import statements are executed from the repository root
- **THEN** compilation succeeds and imports resolve to modules under the new package

### Requirement: Content placement by kind
Repository content SHALL be organized by kind at fixed locations: runnable code under the single code package, research notes under `notes/`, thesis-authored documents (proposal and thesis drafts) under `documents/`, literature (paper summaries, metadata, and PDFs) under `papers/`, and slide decks under `presentations/`. The repository root SHALL contain only workspace-level files (agent instructions, README, dotfiles/config) plus the fixed content directories.

#### Scenario: Root contains no loose content files
- **WHEN** the repository root is listed
- **THEN** no `.md` research notes or standalone `.pdf` documents appear outside `notes/`, `documents/`, `papers/`, and `presentations/`

#### Scenario: Kind directories hold only their kind
- **WHEN** each of `notes/`, `documents/`, `papers/`, `presentations/` is listed
- **THEN** its contents match its kind (notes are markdown research notes; documents are thesis-authored PDFs; papers are literature; presentations are slides)

### Requirement: Experiments remain self-contained
Each experiment directory under `compgen/experiments/` SHALL stay independently runnable: it MUST NOT import from the enclosing package or any root code, so that a single experiment directory remains portable on its own. Uploadable archives (`.zip`) SHALL NOT be committed; each SHALL be reproducible on demand from its experiment directory via the documented build command, and the archive content SHALL not depend on code outside that directory.

#### Scenario: Experiment isolation preserved
- **WHEN** all `.py` files and notebooks under `compgen/experiments/` are searched for imports of the enclosing package or of root `models`/`datasets`
- **THEN** zero matches are found

#### Scenario: No committed upload archives
- **WHEN** tracked files are listed (`git ls-files`) and searched for `.zip` entries
- **THEN** zero upload archives are tracked, and the documented build command regenerates a byte-comparable archive from its experiment directory alone

#### Scenario: Experiment self-test still passes
- **WHEN** the PyTorch experiment's bundled self-check is run from inside its own directory
- **THEN** it passes without any repository-root code

### Requirement: Documentation matches the layout
`AGENTS.md` and the root `README.md` SHALL accurately describe the actual repository layout, import style, and data flow; every path they document SHALL exist, and no stale references (such as pointing at `dataset/generators/match3.py` or misattributing file locations) SHALL remain.

#### Scenario: Documented paths exist
- **WHEN** every repository-relative path mentioned in `AGENTS.md` and `README.md` is checked against the filesystem
- **THEN** each path exists in the documented form

#### Scenario: Stale module references eliminated
- **WHEN** docstrings and documentation are searched for the outdated `dataset/generators/` path and for imports not matching the new package prefix
- **THEN** zero stale references remain, and the formerly misattributed docstring points at the module's real location

### Requirement: Generated artifacts stay untracked
Generated datasets, training artifacts, and build outputs SHALL remain untracked: `.gitignore` SHALL cover `data/`, `*.jsonl`, `*.zip`, logs, checkpoints, and model binaries, and the reorganization SHALL NOT bring any generated artifact under version control.

#### Scenario: Ignore rules cover all artifact locations
- **WHEN** `.gitignore` is inspected after the reorganization
- **THEN** it matches every artifact pattern previously documented (including `data/`) plus `*.zip`, and `git status` reports no generated dataset, checkpoint, or archive files as untracked work
