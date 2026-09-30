# Design

## Context

See `proposal.md` — Why. The constraints that shape the approach:

- Host: NVIDIA GeForce RTX 2060 (Turing, compute capability 7.5 / `sm_75`), driver 595.91.07, CUDA 13.2.
- Local environment: `venv/` at the repository root, Python 3.12.3, already containing `torch 2.14.0+cu132`, `torchvision 0.29.0+cu132`, `numpy 2.5.2`, and the CUDA runtime wheels. `torch.cuda.is_available()` is `True` and `torch.cuda.get_arch_list()` includes `sm_75`.
- Imported by the code package / notebooks / PyTorch experiment but absent from `venv/`: `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `ipython`.
- A read-only resolution (`pip install --dry-run --report` against the existing venv) showed all six missing packages resolve without changing `torch` or `numpy`, yielding the versions below.
- Existing requirements files exist only inside the experiment dirs and are not touched: `compgen/experiments/fuzzy_logic_attention_contrastive/requirements*.txt` (JAX stack) and `..._pytorch/requirements-kaggle.txt` (unpinned, assumes Kaggle's torch).

## Goals / Non-Goals

**Goals:**
- A single root `requirements.txt` that pins the direct PyTorch-side runtime dependencies exactly.
- Pin `torch` to the installed `cu132` build and make the CUDA wheel index resolvable.
- Prove the file installs cleanly into `venv/` without disturbing the anchored `torch`/`numpy`.
- Keep `AGENTS.md` truthful about the manifest and the Python/torch versions.

**Non-Goals:**
- Locking transitive dependencies (no `pip freeze` lockfile).
- Managing the JAX experiment's environment or modifying its requirements files.
- Adding CI, a packaging build (`pyproject.toml`), or a cross-platform matrix.
- Runtime code changes.

## Decisions

### Pin only direct dependencies, not the full transitive closure
Use exact `==` pins for the direct dependencies and let pip resolve transitives. A `pip freeze`-style lock would bake in platform-specific CUDA wheels (`nvidia-*`, `triton`) and every plotting/IPython transitive, producing a large, brittle file that fights the torch pin. **Alternative considered:** full `pip freeze` lock — rejected for noise and poor portability. If a future need for bit-exact reproducibility appears, add a separate lockfile rather than expanding this manifest.

### Torch pin plus `--extra-index-url`, not `--index-url`
Put `--extra-index-url https://download.pytorch.org/whl/cu132` at the top of the file and pin `torch==2.14.0+cu132`. `--extra-index-url` keeps PyPI as the primary source for the other packages while letting the `+cu132` local build resolve. **Alternative considered:** `--index-url` pointing at the PyTorch index — rejected because it would route every dependency through that index, which does not mirror all of PyPI.

### Expected pinned set (to be re-confirmed at implementation)
| Package | Pin | Rationale |
| --- | --- | --- |
| torch | `==2.14.0+cu132` | Installed build; matches driver 595.91.07 / CUDA 13.2 and `sm_75`. |
| numpy | `==2.5.2` | Installed; already compatible with torch 2.14. |
| opt_einsum | `==3.4.0` | Required by all four attention modules. |
| pandas | `==3.0.6` | Used by `summarize_results.py` and notebooks. |
| scikit-learn | `==1.9.1` | Used by `analyze_attention.py` and notebooks. |
| matplotlib | `==3.11.2` | Used by `summarize_results.py` / `analyze_attention.py`. |
| tqdm | `==4.70.1` | Used by `train.py` and notebooks. |
| ipython | `==9.17.1` | Used by `make_notebook.py` and notebook display. |

These come from resolving against the current venv; implementation re-derives them with `pip install --dry-run --report` so the file reflects reality rather than this snapshot.

### Exclude `torchvision`
`torchvision 0.29.0+cu132` is installed but not imported anywhere in the code package, notebooks, or the PyTorch experiment, so it is not a direct dependency. Leaving it out keeps the manifest to what the project actually imports; it can be added later if it gains real use.

### File location and shape
`requirements.txt` at the repository root. This is a workspace-level config file, which the `repo-structure` capability already permits at the root. Shape: the `--extra-index-url` directive first, then a short comment that the file pins the local PyTorch-side environment, then one `package==version` per line.

### Verification installs into the venv
Because the user runs code from `venv/`, verification installs the manifest there and runs `pip check` plus an import/CUDA smoke check. `torch` and `numpy` are asserted unchanged before and after. **Alternative considered:** `--dry-run` only — rejected because the user explicitly wants the environment actually working, not just resolvable.

### Record Python 3.12 in `AGENTS.md`
`AGENTS.md` currently says "Python 3.14 is in use (`__pycache__`)" and "No root requirements file". Both are corrected: the venv is Python 3.12.3 and the manifest now exists. The "Running code" section gains the install command (`pip install -r requirements.txt`).

## Risks / Trade-offs

- **A transitive upgrade breaks torch or numpy** → install is additive (the six packages + their deps); assert `torch` and `numpy` versions are unchanged after install and run `pip check`.
- **pandas 3.0 behavior changes break `summarize_results.py` / `analyze_attention.py` at runtime** → smoke-test those scripts (import and a tiny synthetic run); if they break, fall back to the latest 2.x pandas and pin that instead.
- **Pinning `+cu132` makes the file host-specific (not Colab/Kaggle portable)** → acceptable and intended: it targets the local GPU server. The experiment dirs keep their own portable requirements for hosted notebooks.
- **`scikit-learn==1.9.1` pulls `scipy==1.18.1`** → resolver already confirmed compatibility with numpy 2.5.2; `pip check` covers it.
- **PyTorch index unavailable at install time** → `torch==2.14.0+cu132` is already installed, so pip treats the requirement as satisfied and does not need to download it.

## Migration Plan

1. Add `requirements.txt` at the repository root.
2. Install into `venv/`; run `pip check` and the import/CUDA smoke check; confirm `torch`/`numpy` unchanged.
3. Update `AGENTS.md` to document the manifest and Python 3.12.
4. Rollback: delete `requirements.txt`, revert the `AGENTS.md` edits, and `pip uninstall` the newly added packages if a clean venv is wanted; `torch`/`numpy` are never modified.

## Open Questions

None blocking. A future decision — whether to add a fully-locked transitive lockfile or CI-enforced install test — is out of scope here and does not affect this manifest.
