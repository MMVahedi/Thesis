# Tasks

## 1. Shared reusable model blocks

- [x] 1.1 Add `compgen/models/position.py` with the relative-position bias block and its T5-style bucketing helper; verify the returned bias is broadcastable over batch with one value per head and query/key position.
- [x] 1.2 Add a `RegressionHead` to `compgen/models/heads.py` (final normalization + scalar projection); verify it maps `(batch, length, hidden)` to `(batch, length, 1)`.
- [x] 1.3 Extend `compgen/models/encoder.py::EncoderLayer` with a `ffn_type` selection (`"relu_stack"` preserving the current behavior and `"gelu"` two-projection feed-forward with configurable `ffn_dim`) and a `use_relative_bias` option that computes and supplies the bias at runtime; verify both FFN variants forward a batch unchanged in width and that existing Match3 defaults are unaffected.
- [x] 1.4 Add an optional `score_bias` argument to the `forward` of `SoftmaxAttention` and `LinearAttention` and thread it through `EncoderLayer.forward`; verify attention weights change when a non-zero bias is supplied and keep their shape, and that a `Match3Model` still constructs and forwards.
- [x] 1.5 Add `compgen/models/initialization.py` with the reference truncated-normal weight/bias/embedding initialization helper; verify applying it sets linear biases to zero and weights to the reference standard deviation.

## 2. Losses package

- [x] 2.1 Create `compgen/losses/contrastive.py` with `positive_partner_counts` and a `SharedTermContrastiveLoss` module; verify examples sharing a term are positives, an example is excluded from its own negatives, pairwise-disjoint term sets give a zero but differentiable loss, and changing temperature changes the loss.
- [x] 2.2 Create `compgen/losses/combined.py` with a model-agnostic combined task+contrastive loss; verify a zero coefficient reduces to the task loss, a positive coefficient adds the weighted contrastive term, and it takes caller-supplied representations/term ids without a model class.
- [x] 2.3 Create `compgen/losses/__init__.py` re-exporting the public names; verify each import resolves from the repository root.

## 3. On-the-fly fuzzy-logic dataset

- [x] 3.1 Add `compgen/datasets/torch_datasets/fuzzy_logic_online.py` porting the reference on-the-fly generator (train/test/id/ood pools, deterministic seeded sampling, device caching, Zadeh targets) with its batch bundle and the reference task configurations; verify all four pools are non-empty for every reference configuration, batch shapes and fields are correct, repeated sampling is identical, and too-small configurations raise an error.
- [x] 3.2 Document the on-the-fly dataset in `compgen/datasets/README.md`, keeping the offline generator/reader sections accurate; verify the documented example runs and that `compgen/datasets/generators/` still imports no torch.

## 4. Fuzzy-logic task model

- [x] 4.1 Add `compgen/models/embeddings/fuzzy_logic.py` producing `(batch, length, hidden)` inputs from the dataset's `x`; verify the embedding shape and that input dropout is applied.
- [x] 4.2 Add `compgen/models/tasks/fuzzy_logic.py` composing the embedding, the configurable encoder stack (softmax attention, GELU feed-forward, pre-norm, relative bias), the regression head, and the reference initialization; verify the reference configuration reports exactly 151,009 parameters and returns per-token predictions plus one attention matrix per layer.
- [x] 4.3 Verify the fuzzy-logic task model accepts every sequence-composable registered architecture and rejects an unknown name with an error naming it.
- [x] 4.4 Add a helper on the task model that extracts the response-token self-attention code across heads and wire it to the combined loss; verify a forward pass plus `loss.backward()` on a sampled batch yields finite gradients.
- [x] 4.5 Update the model-layer description in `AGENTS.md` and the root `README.md` to include the fuzzy-logic task and the new shared blocks; verify every documented model-layer path exists.

## 5. Remove experiments and reconcile docs

- [x] 5.1 Delete the JAX experiment directory `compgen/experiments/fuzzy_logic_attention_contrastive/`; verify the directory is gone and no repository file or documentation still references it.
- [x] 5.2 Delete the superseded PyTorch experiment files `model.py`, `data.py`, `losses.py`, `train.py`, `self_test.py`, and `make_notebook.py`; verify the retained `.py` files contain only self-contained analysis code and import neither `compgen` nor root `models`/`datasets`/`losses`.
- [x] 5.3 Update `compgen/experiments/fuzzy_logic_attention_contrastive_pytorch/README.md` to drop the removed harness, self-test, and notebook-generation references and note the frozen notebook; verify every path it documents exists.
- [x] 5.4 Update `AGENTS.md` and the root `README.md` to remove the JAX experiment and its rules and to describe the retained experiment; verify documented paths exist and no stale references remain.

## 6. Integration verification

- [x] 6.1 From the repository root with the configured venv, run a smoke script that constructs the fuzzy-logic model and each sequence-composable attention through both task models, forwards a sampled fuzzy-logic batch and a Match3 batch on CUDA, and runs a contrastive backward; verify finite outputs/gradients and the 151,009-parameter count.
- [x] 6.2 Run `openspec validate "extract-fuzzy-logic-into-library" --strict` and a repository-wide search for the removed JAX experiment and deleted experiment modules; verify the change validates and zero stale references remain.
