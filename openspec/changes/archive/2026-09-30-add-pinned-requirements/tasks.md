# Tasks

## 1. Author the pinned manifest

- [x] 1.1 Re-resolve current compatible versions: run `venv/bin/python -m pip install --dry-run --report /tmp/resolve.json opt_einsum pandas scikit-learn matplotlib tqdm ipython` from the repo root and read the report. Verify the report leaves `torch` and `numpy` out of the install set and lists versions for all six packages.
- [x] 1.2 Confirm the direct-dependency set against the code: grep imports across `compgen/models`, `compgen/datasets`, `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch`, and `notebooks/` and check every third-party import maps to an entry in the manifest. Verify no third-party import is left uncovered and no JAX-only dependency appears.
- [x] 1.3 Create `requirements.txt` at the repository root: put `--extra-index-url https://download.pytorch.org/whl/cu132` at the top, pin `torch==2.14.0+cu132` and `numpy==2.5.2`, and pin `opt_einsum`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `ipython` to the versions from 1.1. Verify with `grep -E '^[a-zA-Z]' requirements.txt` that every package line ends in an exact `==version` (torch carrying the `+cu132` local tag) and that `jax|flax|optax|tensorflow|wandb|einops` are absent.

## 2. Install and verify in the venv

- [x] 2.1 Record the baseline versions: run `venv/bin/python -c "import torch,numpy;print(torch.__version__,numpy.__version__)"`. Verify it prints `2.14.0+cu132 2.5.2` before any installation.
- [x] 2.2 Install the manifest: run `venv/bin/python -m pip install -r requirements.txt`. Verify the command exits 0.
- [x] 2.3 Assert the anchored packages did not move: re-run the 2.1 command and verify it still prints `2.14.0+cu132 2.5.2`; then run `venv/bin/python -m pip check` and verify it reports no broken requirements.
- [x] 2.4 Run the GPU smoke check: with the venv interpreter, import `torch`, `numpy`, `opt_einsum`, `pandas`, `sklearn`, `matplotlib`, `tqdm`, `IPython` and print `torch.cuda.is_available()` and `torch.cuda.get_arch_list()`. Verify all imports succeed, CUDA is `True`, and `'sm_75'` is in the arch list.
- [x] 2.5 Exercise the pandas/matplotlib/sklearn consumers: import `summarize_results` and `analyze_attention` from `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` and run their top-level code paths on a tiny synthetic input. Verify they run without errors; if pandas 3.0 breaks them, pin the latest 2.x pandas in `requirements.txt` and repeat 2.2–2.4.

## 3. Document the manifest

- [x] 3.1 Update `AGENTS.md` "Running code": remove the "No root requirements file" sentence, document the root `requirements.txt` and its install command (`pip install -r requirements.txt`), and correct the Python version claim to the venv's Python 3.12 (torch `2.14.0+cu132`). Verify the documented install command and versions match `requirements.txt` and `venv/bin/python --version`.
- [x] 3.2 Search `AGENTS.md` for the removed claims (`No root requirements file`, `Python 3.14`). Verify zero stale matches remain.

## 4. Integration validation

- [x] 4.1 Run `openspec validate add-pinned-requirements --strict` and verify the change validates with no errors.
- [x] 4.2 Confirm experiment isolation: run `git status --short` and `git diff --exit-code -- compgen/experiments/fuzzy_logic_attention_contrastive/requirements.txt compgen/experiments/fuzzy_logic_attention_contrastive/requirements-colab.txt compgen/experiments/fuzzy_logic_attention_contrastive/requirements-kaggle.txt compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/requirements-kaggle.txt`. Verify the experiment requirements files are unchanged.
- [x] 4.3 End-to-end environment check: run `python self_test.py` inside `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/` with the venv interpreter. Verify the self-test passes.
