# Fuzzy-Logic Attention Contrastive Experiment

This directory is a standalone copy of the code needed to compare the original
standard-attention fuzzy-logic baseline from *Attention as a Hypernetwork* with
a shared-term contrastive loss on final-layer attention representations. The
Colab notebook sweeps task complexity, dataset size, batch size, and
contrastive-loss weight with a configurable epoch count.

The source repository at `../../../hypernetwork-attention` is not modified.

Both the baseline and proposed experiment use the repository's standard
softmax-attention Transformer:

```text
sequence_mixer = "softmax_attention"
attention_norm = "softmax"
target_network = "default"
```

They do not use `linear_hypatt`, `softmax_hypatt`, or another HyLA attention
variant. The copied directory named `hyla/` is only the official repository's
Python package name; it contains both standard attention and the paper's new
attention variants.

## Setup

Install JAX for the target platform, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Colab and Kaggle ZIP workflow

Use `colab_fuzzy_logic_attention_contrastive.ipynb` on either Colab or Kaggle.
It detects the platform and selects the correct writable paths. Create the ZIP
from the parent `compgen/experiments` directory:

```bash
zip -r fuzzy_logic_attention_contrastive.zip \
  fuzzy_logic_attention_contrastive \
  -x '*/__pycache__/*' '*/logs/*' '*/wandb/*'
```

For Colab, upload the ZIP into `/content` through the Files sidebar and run the
notebook sequentially. If it is absent, the extraction cell opens an upload
dialog. Results are stored under:

```text
/content/fuzzy_logic_attention_results
```

Nothing is written to Google Drive. Download the result archive made by the
last notebook cell before the Colab runtime ends.

For Kaggle:

1. Upload the notebook and select a GPU under the notebook Session options.
2. In the Input pane, choose Add Input or Upload, create/attach a private input
   containing `fuzzy_logic_attention_contrastive.zip`, and start the session.
3. Run the notebook from the beginning. The extraction cell discovers the ZIP
   below `/kaggle/input` and extracts it to
   `/kaggle/working/fuzzy_logic_attention_contrastive`.
4. Results go to `/kaggle/working/fuzzy_logic_attention_results` and the final
   archive goes to `/kaggle/working/fuzzy_logic_attention_results_export.zip`.
5. Use Save Version before the session ends to preserve the contents of
   `/kaggle/working` as notebook output. The final cell also displays a link to
   the result archive.

Kaggle inputs are treated as read-only; code and outputs are never written back
under `/kaggle/input`.

Neither accelerator requirements file installs JAX or `toolz`: Colab or Kaggle
supplies its internally matched GPU-compatible JAX stack. Colab pins
`flax==0.12.9`; Kaggle pins `flax==0.11.2`, which declares `jax>=0.6.0` and does
not force Kaggle's JAX 0.7.x packages to upgrade. The notebook checks JAX,
jaxlib, the CUDA plugin, Flax, the backend, visible devices, and an actual JIT
GPU operation in a short-lived subprocess before training.

## Baseline

```bash
python run.py \
  --config='configs/logic.py:logic_4var_2term;transformer' \
  --config.lambda_contrastive=0.0 \
  --logger=standard
```

With `lambda_contrastive=0`, the loss and model forward path reduce to the
original MSE baseline.

## Shared-term attention contrastive experiment

```bash
python run.py \
  --config='configs/logic.py:logic_4var_2term;transformer' \
  --config.lambda_contrastive=0.05 \
  --config.temperature=1.0 \
  --logger=standard
```

The added loss uses the final query token's self-attention probability from all
eight heads in the last Transformer layer, giving `Z.shape == [batch_size, 8]`.
Two batch elements are positives when their unordered pairs of fuzzy terms share
at least one term. Term IDs are training metadata and are never model inputs.

## Metrics

Metrics are printed and written under `logs/<run>/event.log`:

- `train_task_loss`, `train_contrastive_loss`, and `train_total_loss`
- `train_avg_num_positives`
- `id_r2`: training-combination distribution
- `test_r2`: held-out combinations of individually seen terms (compositional OOD)
- `ood_r2`: combinations of completely held-out constituent terms

During training, tqdm shows processed training samples over the complete
dataset size for the current epoch. It does not print per-batch metrics. At the
end of every complete epoch, the runner reports the epoch-average training
metrics and evaluates all 16,000 examples in each of the ID,
compositional-test, and unseen-term sets. R² is the appropriate "accuracy"
measure for this regression task. `config.num_epochs` controls the number of
complete passes. The generator is deterministic for a fixed seed, so later
epochs revisit the same generated dataset rather than drawing a new stream.

## Notebook sweep and exports

The default notebook matrix contains four tasks:

- `logic_3var_2term`
- `logic_4var_2term` (the paper's reference task)
- `logic_4var_3term`
- `logic_5var_2term`

It evaluates dataset sizes 128,000, 640,000, and 6,400,000 samples per epoch,
batch sizes 32, 64, and 128, and lambda values 0, 0.01, 0.05, and 0.1. The first
two sizes are 2% and 10% of the paper-scale 6,400,000-sample configuration.
Lambda 0 is the baseline. Contrastive temperature is fixed at 1.0 for all runs,
so lambda is the only swept parameter controlling the balance between MSE and
contrastive loss. Dataset size is held fixed when comparing batch sizes; this
means smaller batches perform more optimizer updates per epoch. The default
single-seed matrix contains 144 runs.

The notebook divides execution into 12 cells, one per task and dataset-size
variation. Each cell runs its batch/lambda combinations sequentially and skips
completed combinations when rerun, allowing an interrupted variation to
continue without restarting the entire matrix.

The fuzzy-logic generator forms unordered combinations of distinct
conjunctions. For `logic_5var_2term`, 32 conjunctions are possible. Eight are
reserved for OOD, leaving C(24, 2) = 276 in-distribution functions split evenly
into 138 train and 138 test functions; the OOD pool contains C(8, 2) = 28
functions.

## Attention-code analysis

Each completed model also runs the analysis used in the paper: it collects the
response token's self-attention score across heads for every layer while
holding the in-context inputs fixed. It then:

- produces t-SNE plots colored by the target and each constituent term;
- trains logistic-regression decoders on training-function attention codes and
  reports macro F1 and accuracy on held-out function combinations for each
  layer and term position.

Per-run outputs are `attention_tsne.png`, `attention_term_f1.csv`,
`attention_term_f1.png`, and `attention_analysis.json`.

The final notebook cell creates per-run and aggregated CSV files, epoch history,
R² and MSE charts per dataset size, a direct dataset-size versus compositional
test R² chart, a chart of average positive partners per anchor by batch size,
and includes every model's attention analysis in the downloaded archive. Its
path is `/content/fuzzy_logic_attention_results_export.zip` on Colab and
`/kaggle/working/fuzzy_logic_attention_results_export.zip` on Kaggle.
