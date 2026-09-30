# Fuzzy-logic attention contrastive experiment — PyTorch (analysis archive)

This directory retains the **analysis tooling** for the PyTorch fuzzy-logic
attention contrastive experiment. The model, on-the-fly dataset, and the
shared-term contrastive loss have been extracted into the `compgen/` package and
now live there:

- model: `compgen/models/tasks/fuzzy_logic.py` (+ `compgen/models/embeddings/fuzzy_logic.py`)
- on-the-fly data: `compgen/datasets/torch_datasets/fuzzy_logic_online.py`
- loss: `compgen/losses/` (shared-term contrastive + combined task/contrastive)
- reference hyperparameters/task definitions: `config.py` (still here)

The original standalone `model.py`, `data.py`, `losses.py`, `train.py`,
`self_test.py`, and `make_notebook.py` were removed once the package modules
superseded them. This directory therefore only holds self-contained analysis
code and the committed notebook.

## What is preserved

- Zadeh fuzzy logic (`min` for AND, `max` for OR), task construction, 50/50 held-out
  in-distribution function split, and 25% held-out conjunctions for OOD.
- Reference hyperparameters in `config.py`: sequence length 16; 2 layers; embedding 128;
  MLP 256; 8 heads; total Q/K and V dimensions 16; relative-position bias; standard
  attention; AdamW; learning rate 0.001; 100 warmup steps; cosine decay to 0.0001;
  weight decay 0.1.
- MSE regression and the reference per-example R² definition.
- Reference random seed 2024. PyTorch's random stream differs from JAX's, so the sampled
  functions follow the same split procedure but are not byte-identical.
- Shared-term supervised contrastive loss on the final layer's response-token
  self-attention scores across heads. Temperature is fixed to 1.0 by the notebook.

The extracted package model is a faithful reimplementation of the reference
standard softmax-attention transformer, not HyLA.

## Contrastive formula

For attention code `z_i`, temperature `T=1`, and positives `P(i)` (other batch
functions that share at least one term):

```text
L_con = mean_i [ -1/|P(i)| sum_{p in P(i)}
          log( exp(cos(z_i,z_p)/T) / sum_{a != i} exp(cos(z_i,z_a)/T) ) ]
L_total = L_MSE + lambda * L_con
```

Anchors with no positive partner are excluded. Batch size therefore changes both the
number of negatives and the probability/number of shared-term positive partners.

## Retained files

- `config.py` — reference `ExperimentConfig` and the `TASKS` task definition table.
- `analyze_attention.py` — paper-style clustering/decoding analysis of a run's saved
  `attention_codes.npz`, writing decoding tables and t-SNE plots into the run directory:

  ```bash
  python analyze_attention.py --run-dir <run-dir>
  ```

- `summarize_results.py` — collects completed `final_metrics.json` files under a results
  root into comparison tables and charts:

  ```bash
  python summarize_results.py --results-root <results-root>
  ```

- `kaggle_fuzzy_logic_attention_contrastive_pytorch.ipynb` — the original run's Kaggle/Colab
  notebook, now **frozen**: its generator (`make_notebook.py`) was removed, so it can no
  longer be regenerated. It expects an uploaded experiment zip containing the (now removed)
  `train.py`/`self_test.py`, so it no longer runs end-to-end; it is retained as a record of
  the original workflow. Edit it by hand to revive it against the `compgen/` package.

## Notes

- `analyze_attention.py` and `summarize_results.py` read run outputs
  (`attention_codes.npz`, `final_metrics.json`) produced previously; there is no bundled
  training entry point anymore — train with the `compgen/` package modules.
- `LICENSE` and `requirements-kaggle.txt` are unchanged.
