# Design

## Context

See `proposal.md` — Why. The relevant current state:

- The model layer is a modular stack: `compgen/models/attentions/` (registry `ATTENTION_CLASSES`, shared `BaseAttention`), `compgen/models/encoder.py` (`EncoderLayer`), `compgen/models/heads.py` (`TokenClassifier`), and thin task models (`compgen/models/tasks/match3.py`) built on per-task embeddings (`compgen/models/embeddings/match3.py`).
- The dataset layer is a two-layer convention: torch-free generators writing JSON Lines (`compgen/datasets/generators/`) and torch readers (`compgen/datasets/torch_datasets/`). Fuzzy-logic already has an offline generator and reader, but no on-the-fly variant and no train/test/ood partitioning.
- There is no losses module anywhere in the package.
- The PyTorch experiment (`compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/`) is a standalone transformer: `model.py` (`StandardAttentionTransformer`, `TransformerBlock`, `StandardAttention`, `RelativePositionBias`), `data.py` (on-the-fly `FuzzyLogicGenerator` + `LogicBatch`), `losses.py` (shared-term contrastive loss + combined `training_loss`), plus training/analysis harness. Its reference self-check pins the model at **151,009 parameters** for `logic_4var_2term`.
- The JAX experiment `compgen/experiments/fuzzy_logic_attention_contrastive/` is grandfathered and, per `AGENTS.md` and the `repo-structure`/`runtime-dependencies` specs, currently expected to exist.

Constraints: PyTorch only; experiment directories must not import the package; attention default dtype is `torch.float64`; the existing Match3 path and its parameterization must not change.

## Goals / Non-Goals

**Goals:**
- Re-express the experiment's fuzzy-logic model faithfully using reusable shared blocks, so it reproduces the reference architecture and its 151,009-parameter count.
- Add the on-the-fly fuzzy-logic dataset and the losses package as first-class package modules.
- Remove the JAX experiment and reduce the PyTorch experiment to a self-contained analysis archive.
- Keep the existing Match3/offline-dataset behavior byte-for-byte intact.

**Non-Goals:**
- No new attention architectures; only an optional additive score bias on the sequence-composable mixers.
- No changes to the offline fuzzy-logic generator/reader beyond documentation.
- No migration of the experiment's training loop, notebook generator, or self-test into the package.
- No attempt to make high-order variants (strassen/third_order/triangular) support a relative-position bias.

## Decisions

### D1. Extraction destinations

| Source | Destination |
|---|---|
| `model.py::StandardAttentionTransformer` + `TransformerBlock` | `compgen/models/tasks/fuzzy_logic.py` (thin task model) + `compgen/models/embeddings/fuzzy_logic.py` |
| `model.py::RelativePositionBias` + `_relative_position_bucket` | `compgen/models/position.py` |
| `model.py` GELU MLP + final norm/output | `compgen/models/encoder.py` (FFN variant) + `compgen/models/heads.py::RegressionHead` |
| `model.py` reference init | `compgen/models/initialization.py` |
| `data.py::FuzzyLogicGenerator` + `LogicBatch` + `TASKS` | `compgen/datasets/torch_datasets/fuzzy_logic_online.py` |
| `losses.py` contrastive parts | `compgen/losses/contrastive.py` |
| `losses.py::training_loss` | `compgen/losses/combined.py` |

Rationale: mirrors the existing "one embedding + one thin task file" and model/head/position split, keeps `datasets/generators/` torch-free, and gives losses its own top-level package. Alternative considered: a single `compgen/models/fuzzy_logic.py` monolith — rejected because it duplicates the encoder/head scaffolding the package already owns.

New task model/embedding are placed under `compgen/models/tasks/` and `compgen/models/embeddings/` with the same conventions as `match3.py`; the registration point in `compgen/models/tasks/__init__.py` (if any) and docs are updated.

### D2. Faithful model via configurable shared blocks

`EncoderLayer` is extended (additively, defaults unchanged) with:
- `ffn_type`: `"relu_stack"` (current behavior) or `"gelu"` (two-projection `Linear(ffn_dim) -> GELU(approximate="tanh") -> Linear(hidden)`, dropout before each projection); `ffn_dim` configurable.
- `use_relative_bias`: when true the layer owns a `RelativePositionBias(num_heads, num_buckets, max_distance)` and computes its bias from the runtime sequence length, passing it to attention as `score_bias`.

The fuzzy-logic task model configures: two layers, `hidden_dim=128`, `ffn_type="gelu"`, `ffn_dim=256`, `num_heads=8`, per-head query/key and value dimensions of 2 (the reference's `qk_dim=v_dim=16` split across `num_heads=8`), pre-norm on, relative bias on with `num_buckets=max_distance=seq_len=16`, input dropout, final normalization, and the reference init helper.

Rationale: faithful behavior is achieved by *configuration* of shared blocks rather than a duplicated transformer, satisfying "faithful as new shared blocks". Alternative: port `TransformerBlock` verbatim as a new task model — rejected as it re-duplicates encoder/head code.

Acceptance: the model built this way exposes 151,009 parameters for the reference config, matching `self_test.py` today.

### D3. Attention additive score bias

`BaseAttention`-derived `SoftmaxAttention` and `LinearAttention` gain an optional `score_bias` keyword on `forward` (default `None`), added to the scaled scores before masking/normalization. `EncoderLayer.forward` gains an optional `score_bias` and forwards it, and every registered attention accepts the keyword for signature uniformity; the high-order variants accept but ignore it and remain out of scope for bias semantics.

Alternative: give `BaseAttention` a `bias_provider` callback — rejected as more coupling than a plain tensor argument.

### D4. On-the-fly dataset placement

Add `compgen/datasets/torch_datasets/fuzzy_logic_online.py` exposing a torch `Dataset`-style on-the-fly sampler (the reference samples by `(split, batch_index)` rather than by index, so this is a sampler, not a map-style `Dataset`), its batch bundle, the reference `TASKS` configurations, and the same deterministic seeded sampling, split offsets, device caching, and Zadeh target computation as the experiment. `generators/` stays torch-free; the offline fuzzy-logic generator/reader are untouched.

Rationale: an on-the-fly sampler can't obey the JSON Lines convention, so it belongs in the torch-facing layer, not `generators/`. Alternative: fold partitioning into the offline generator — rejected because the reference's batched tensor bundle and per-batch seeded sampling are the point.

### D5. Losses package shape

`compgen/losses/`:
- `contrastive.py`: `positive_partner_counts(term_ids)` and a `SharedTermContrastiveLoss` (`nn.Module`, holds `temperature`) whose `forward(representations, term_ids)` returns `(loss, average_positive_partners)`, matching the reference formula exactly (cosine similarity, self excluded from negatives, anchors without positives skipped).
- `combined.py`: `combined_loss(task_loss, representations, term_ids, lambda_contrastive, temperature)` returning total/task/contrastive/positive diagnostics; when `lambda_contrastive == 0` it skips the contrastive computation and returns a zero contrastive term, preserving the reference's zero-weight branch.
- `__init__.py` re-exports the public names.

Rationale: "package + combined training loss", model-agnostic. The extraction of the response-token attention code (`attentions[-1][:, :, -1, -1]`) stays with the task model (a small helper), because it encodes attention layout, not loss semantics.

### D6. Experiment removal and docs

- Delete the whole `compgen/experiments/fuzzy_logic_attention_contrastive/` directory.
- Delete `model.py`, `data.py`, `losses.py`, `train.py`, `self_test.py`, `make_notebook.py` from the PyTorch experiment. Keep `config.py`, `analyze_attention.py`, `summarize_results.py`, the committed notebook, `README.md`, `LICENSE`, `requirements-kaggle.txt`.
- Update `AGENTS.md`, root `README.md`, and the PyTorch experiment `README.md` to drop the JAX experiment and the removed harness/self-test/notebook-generation references.

Rationale: the removed files are superseded by the package modules; the retained files are self-contained analysis (they import only third-party libraries and read run outputs). The JAX experiment is removed by explicit user decision, overriding the previous grandfathering.

## Risks / Trade-offs

- **[Parameter-count drift]** shared blocks may not reproduce 151,009 exactly → build the fuzzy-logic model from the extracted blocks and assert the count *before* deleting the experiment sources; if a mismatch appears, adjust block defaults (FFN dims, norm placement, bias presence) rather than weaken the check, and record any irreducible divergence.
- **[Notebook can no longer be regenerated]** deleting `make_notebook.py` freezes the committed notebook → acceptable; note it in the experiment `README.md` so future edits are manual.
- **[Experiment no longer runnable end-to-end]** train/self-test removed → intended by user decision; `repo-structure`'s self-containment requirement is restated for the analysis-only experiment.
- **[Spec/AGENTS conflict]** removing the JAX experiment contradicts the current `AGENTS.md` "keep it working" rule and two spec requirements → this change updates all three in the same work.
- **[High-order bias gap]** strassen/third_order/triangular do not accept a score bias → documented non-goal; the model-layer spec scopes the bias requirement to softmax/linear.
- **[Match3 regression]** extending `EncoderLayer`/attention could alter default behavior → additive kwargs with defaults equal to current behavior; re-verify the modularize-attention equivalence checks for softmax/linear and the Match3 forward.

## Migration Plan

1. Add the new shared blocks and the optional `score_bias` plumbing (no behavior change for existing callers).
2. Add the losses package and the on-the-fly dataset.
3. Add the fuzzy-logic embedding + task model; verify the 151,009-parameter count and a forward/eval/contrastive-backward smoke run.
4. Re-verify Match3 and the softmax/linear equivalence from the previous change.
5. Delete the JAX experiment and the superseded PyTorch experiment files; update the experiment `README.md`.
6. Update `AGENTS.md` and root `README.md`.
7. Validate specs and run the package-level checks from the configured venv.

Rollback is a single `git revert`; nothing is published.

## Open Questions

None — the material choices (source, fidelity approach, dataset form, losses shape, removal scope) are settled.
