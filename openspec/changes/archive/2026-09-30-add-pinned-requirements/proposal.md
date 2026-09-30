# Proposal

## Why

The workspace has no root dependency manifest. The local GPU environment is rebuilt by hand: `venv/` (Python 3.12.3) holds `torch 2.14.0+cu132` against driver 595.91.07 / CUDA 13.2 on an RTX 2060, and the only requirements files live inside the two experiment dirs — the PyTorch one is unpinned and the JAX one targets a different stack entirely. Nothing records which versions the library, notebooks, and PyTorch experiment actually need, so the environment can drift and there is no stated contract that the dependency set is compatible with the installed driver, CUDA runtime, and torch build.

## What Changes

- **Add a root `requirements.txt`** listing the PyTorch-side runtime stack with exact `==` pins: `torch==2.14.0+cu132` plus `numpy`, `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, and `ipython`. The remaining versions are pinned to releases compatible with Python 3.12, the installed `numpy 2.5.2`, and the `cu132` torch build.
- **Add the PyTorch CUDA wheel index** (`--extra-index-url https://download.pytorch.org/whl/cu132`) to the file so the `cu132` torch wheel resolves reproducibly.
- **Verify compatibility for real**: install the pinned set into `venv/` and run a smoke check that imports every package and confirms `torch.cuda.is_available()` and the compiled `sm_75` arch on the RTX 2060.
- **Update `AGENTS.md`** whose "No root requirements file" statement becomes stale, and record the Python/dependency versions the manifest targets.
- **Explicitly unchanged**: the isolated JAX experiment (keeps its own requirements files), all code behavior, and the existing experiment requirements files.

## Capabilities

### New Capabilities

- `runtime-dependencies`: the contract for the project's local runtime environment — a single pinned root dependency manifest, anchored to the host's GPU driver and torch build, that is installable and documented.

### Modified Capabilities

(none — no existing capability covers dependency or environment management)

## Impact

- **New file**: `requirements.txt` at the repository root (a workspace-level config file, consistent with `repo-structure`'s rule that the root holds workspace-level files plus the fixed content directories).
- **Docs**: `AGENTS.md` "Running code" section — remove the "No root requirements file" claim, document the manifest and the pinned Python/torch versions.
- **Environment (verification only)**: `venv/` gains the missing packages (`opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `ipython` and their transitive deps). `torch` and `numpy` are left at their installed versions.
- **Untouched**: both experiment requirements files, the JAX experiment, and all code behavior.
