# Fuzzy-logic attention contrastive experiment — PyTorch

This is an isolated, PyTorch-only reimplementation of the fuzzy-logic experiment from
`hypernetwork-attention`. It does not import or modify the original JAX repository or the
earlier JAX experiment. The model is the reference **standard softmax-attention** Transformer,
not HyLA.

## What is preserved

- Zadeh fuzzy logic (`min` for AND, `max` for OR), task construction, 50/50 held-out
  in-distribution function split, and 25% held-out conjunctions for OOD.
- Sequence length 16; 2 layers; embedding 128; MLP 256; 8 heads; total Q/K and V dimensions
  16; relative-position bias; standard attention; AdamW; learning rate 0.001; 100 warmup
  steps; cosine decay to 0.0001; weight decay 0.1.
- MSE regression and the reference per-example R² definition.
- Reference random seed 2024. PyTorch's random stream differs from JAX's, so the sampled
  functions follow the same split procedure but are not byte-identical.
- Shared-term supervised contrastive loss on the final layer's response-token
  self-attention scores across heads. Temperature is fixed to 1.0 by the notebook.

This is a faithful reimplementation, not a bit-for-bit reproduction: PyTorch and JAX use
different random-number generators and backend kernels.

## Kaggle

1. Upload the provided zip as a Kaggle Dataset (or add it as notebook input).
2. Enable **GPU T4 x2** or **GPU P100** in Notebook options (the code uses one GPU).
3. Upload/open `kaggle_fuzzy_logic_attention_contrastive_pytorch.ipynb`.
4. Add the zip Dataset as notebook input and run the setup cells. The notebook locates the
   zip anywhere below `/kaggle/input` and extracts a working copy into `/kaggle/working`.
   It then runs `self_test.py` before any long experiment.
5. Run only the individual sweep cells you want. Completed models are skipped when a cell is
   rerun, so you can continue after an interruption.

Kaggle already contains a CUDA-enabled PyTorch. The notebook deliberately never installs,
upgrades, or imports JAX, Flax, Optax, TensorFlow, or `toolz`.

## One command

```bash
python train.py \
  --task logic_4var_2term --num-train 128000 --num-eval 16000 \
  --batch-size 128 --epochs 1 --lambda-contrastive 0.05 \
  --temperature 1.0 --workdir results/example
python analyze_attention.py --run-dir results/example
```

Each epoch shows exact processed samples via `tqdm`. Full ID/test/OOD evaluation happens
only after the epoch, never after individual batches. Every run saves a checkpoint,
`epoch_metrics.csv`, `final_metrics.json`, attention codes, decoding metrics, and t-SNE plot.

## Built-in validation

`python self_test.py` checks all four dataset combinatorics, tensor shapes, Zadeh targets,
fixed contexts, the reference model's parameter count, normalized attention, contrastive
forward/backward gradients, and the no-positive contrastive edge case.

## Contrastive formula

For attention code `z_i`, temperature `T=1`, and positives `P(i)` (other batch functions that
share at least one term):

```text
L_con = mean_i [ -1/|P(i)| sum_{p in P(i)}
          log( exp(cos(z_i,z_p)/T) / sum_{a != i} exp(cos(z_i,z_a)/T) ) ]
L_total = L_MSE + lambda * L_con
```

Anchors with no positive partner are excluded. Batch size therefore changes both the number
of negatives and the probability/number of shared-term positive partners.
