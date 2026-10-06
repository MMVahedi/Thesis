# Tasks

## 1. Fixed positional encoding block

- [x] 1.1 Replace the contents of `compgen/models/position.py` with a parameter-free fixed sinusoidal positional encoding (Vaswani formulation) returning `(length, hidden)` for a requested length and hidden width; verify with `venv/bin/python` that the output has the expected shape, exposes no `nn.Parameter`s, honors the requested dtype/device, and matches the sin/cos formula for a few positions
- [x] 1.2 Confirm the new block is importable as `compgen.models.position` from the repository root and that `import compgen.models.position` succeeds without the old relative-bias symbols

## 2. Wire the encoding into the fuzzy-logic embedding

- [x] 2.1 Add the fixed positional encoding to `FuzzyLogicEmbedding.forward` before dropout (`dropout(input_proj(x) + pe[:length])`); verify the output shape is unchanged `(batch, length, hidden)` and that adding the encoding changes the output versus the same embedding with the encoding subtracted
- [x] 2.2 Verify the fuzzy-logic embedding still runs on both CPU and, where available, the default device, and that the encoding is cast to the module dtype

## 3. Remove relative bias from the model layer

- [x] 3.1 Remove the `RelativePositionBias` import, the `use_relative_bias`/`relative_bias_num_buckets`/`relative_bias_max_distance` parameters, the `relative_bias` submodule, and the automatic `score_bias` fallback from `compgen/models/encoder.py`, keeping the optional `score_bias` forward argument; verify an `EncoderLayer` has no `relative_bias` attribute and a forward pass with `score_bias=None` returns a hidden state of the input shape
- [x] 3.2 Remove the relative-bias constructor parameters and their pass-through from `compgen/models/tasks/fuzzy_logic.py`; verify the model constructs with the reference architecture settings and `forward(..., return_attention=True)` returns per-token predictions plus one attention matrix per layer
- [x] 3.3 Verify no `relative_bias`, `RelativePositionBias`, or `relative_position_buckets` references remain under `compgen/models/`

## 4. Clean up the experiment configuration and runner

- [x] 4.1 Remove `use_relative_bias`, `relative_bias_num_buckets`, and `relative_bias_max_distance` from `ExperimentConfig` in `compgen/runners/fuzzy_logic_contrastive/config.py`; verify `ExperimentConfig()` constructs, exposes no relative-bias attributes, and its existing validation still passes
- [x] 4.2 Stop passing the removed fields in `build_model` in `compgen/runners/fuzzy_logic_contrastive/run.py`; verify `build_model(ExperimentConfig(), "softmax", torch.device("cpu"))` constructs and the experiment's argparse still builds a config
- [x] 4.3 Verify no `relative_bias` references remain under `compgen/runners/`

## 5. Documentation

- [x] 5.1 Update `AGENTS.md` (and any root `README.md` model-layer text) to describe absolute positional encoding and remove the relative-position-bias wording; verify every documented `compgen/...` path still exists and that `grep -ri "relative.position" AGENTS.md` finds nothing
- [x] 5.2 Update the attention docstrings in `compgen/models/attentions/softmax.py` and `compgen/models/attentions/linear.py` that cite a relative-position bias as the `score_bias` example; verify `grep -ri "relative.position" compgen` returns no matches outside the results note

## 6. Preserve the relative-bias idea

- [x] 6.1 Create `results/relative_position_bias.md` describing what the removed block did (T5-style bucketed per-head attention bias), why it was replaced by fixed absolute positional encoding, and how it could be re-tested later; verify the file exists and reads as a self-contained note
- [x] 6.2 Add a `.gitignore` negation so `results/relative_position_bias.md` is tracked despite `results/` being ignored; verify with `git check-ignore -v results/relative_position_bias.md` (should report the negation or nothing) and `git status` listing the note as untracked

## 7. Integration verification

- [x] 7.1 Smoke-run the experiment end to end: `venv/bin/python -m compgen.runners.fuzzy_logic_contrastive --num-train 128 --num-eval 128 --epochs 1 --no-track`, and verify it completes all four variants, prints metrics, and writes result JSONs
- [x] 7.2 Run `openspec validate add-absolute-positional-encoding --strict` and verify it passes
