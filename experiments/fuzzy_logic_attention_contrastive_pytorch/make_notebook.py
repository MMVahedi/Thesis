"""Build the checked-in Kaggle/Colab notebook with one resumable cell per task/size."""

import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(True),
    }


cells = [
    markdown("""# PyTorch fuzzy-logic attention experiment (Kaggle/Colab)

This notebook uses the reference **standard softmax-attention** Transformer—not HyLA—and
adds the shared-term contrastive objective. It contains no JAX/Flax/Optax code. Every exact
task × dataset-size × batch-size × lambda model has its own resumable cell. A model runs in a child process, so its CUDA
memory is fully released when the model finishes or errors.

Fuzzy logic is regression. The full-split metrics reported once per epoch are MSE, MAE, the
paper's R², and the clearly defined fraction of predictions within ±0.05."""),
    markdown("## 1. Locate and extract your uploaded zip"),
    code("""from pathlib import Path
import gc, json, os, shutil, subprocess, sys, zipfile

IS_KAGGLE = Path('/kaggle/input').exists()
IS_COLAB = 'google.colab' in sys.modules
PLATFORM_ROOT = Path('/kaggle/working' if IS_KAGGLE else '/content')
RESULTS_ROOT = PLATFORM_ROOT / 'fuzzy_logic_attention_pytorch_results'
EXTRACT_ROOT = PLATFORM_ROOT / 'fuzzy_logic_attention_pytorch_code'
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

if IS_KAGGLE:
    candidates = sorted(Path('/kaggle/input').rglob('fuzzy_logic_attention_contrastive_pytorch*.zip'))
    if not candidates:
        candidates = sorted(Path('/kaggle/input').rglob('*.zip'))
    if not candidates:
        raise FileNotFoundError('Add the experiment zip as a Kaggle notebook Input, then rerun.')
    ZIP_PATH = candidates[0]
else:
    from google.colab import files
    uploaded = files.upload()
    ZIP_PATH = Path('/content') / next(name for name in uploaded if name.endswith('.zip'))

if EXTRACT_ROOT.exists():
    shutil.rmtree(EXTRACT_ROOT)
EXTRACT_ROOT.mkdir(parents=True)
with zipfile.ZipFile(ZIP_PATH) as archive:
    archive.extractall(EXTRACT_ROOT)
matches = list(EXTRACT_ROOT.rglob('train.py'))
if len(matches) != 1:
    raise RuntimeError(f'Expected one train.py in the zip, found {len(matches)}: {matches}')
EXPERIMENT_ROOT = matches[0].parent
print('Platform:', 'Kaggle' if IS_KAGGLE else 'Colab')
print('Zip:', ZIP_PATH)
print('Code:', EXPERIMENT_ROOT)
print('Results:', RESULTS_ROOT)
"""),
    markdown("## 2. Verify dependencies and GPU\n\nKaggle/Colab already provides CUDA PyTorch. This cell never upgrades it and never installs JAX."),
    code("""import importlib.util

required = {'numpy':'numpy', 'pandas':'pandas', 'matplotlib':'matplotlib',
            'sklearn':'scikit-learn', 'tqdm':'tqdm'}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)

subprocess.run([sys.executable, '-c',
    "import torch; print('PyTorch:', torch.__version__); "
    "print('CUDA available:', torch.cuda.is_available()); "
    "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"], check=True)
subprocess.run(['nvidia-smi'], check=False)
subprocess.run([sys.executable, '-u', 'self_test.py'], cwd=EXPERIMENT_ROOT, check=True)
"""),
    markdown("## 3. Experiment settings\n\nThe lambda and batch-size models are explicit cells below. Temperature stays fixed at 1.0. "
             "You may change `EPOCHS`, `NUM_EVAL`, `NUM_ANALYSIS`, or `SEED` before running model cells."),
    code("""LAMBDAS = [0.0, 0.01, 0.05, 0.1]
BATCH_SIZES = [32, 64, 128]
TEMPERATURE = 1.0
EPOCHS = 1
NUM_EVAL = 16_000
NUM_ANALYSIS = 4_096
SEED = 2024

TASKS = ['logic_3var_2term', 'logic_4var_2term', 'logic_4var_3term', 'logic_5var_2term']
DATASET_SIZES = [128_000, 640_000, 6_400_000]
print(f'{len(TASKS) * len(DATASET_SIZES) * len(BATCH_SIZES) * len(LAMBDAS)} total possible models')
"""),
    markdown("## 4. Resumable runner\n\nEach completed model is skipped on rerun. Progress bars count exact training "
             "samples. Evaluation prints only after a full epoch."),
    code("""def release_after_model():
    reclaimed = gc.collect()
    print(f'Python garbage collector reclaimed {reclaimed} objects.')
    subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader'], check=False)

def run_model(task, num_train, batch_size, lambda_contrastive):
    label = (f'{task}__n{num_train}__b{batch_size}__lambda{lambda_contrastive:g}'
             f'__T{TEMPERATURE:g}__e{EPOCHS}__seed{SEED}')
    run_dir = (RESULTS_ROOT / task / f'n{num_train}' /
               f'b{batch_size}_lambda{lambda_contrastive:g}_T{TEMPERATURE:g}_e{EPOCHS}_seed{SEED}')
    final_path = run_dir / 'final_metrics.json'
    training_complete = False
    if final_path.exists():
        try:
            if json.loads(final_path.read_text()).get('status') == 'complete':
                training_complete = True
                print('SKIP completed training:', label)
        except json.JSONDecodeError:
            pass
    command = [
        sys.executable, '-u', 'train.py', '--task', task,
        '--num-train', str(num_train), '--num-eval', str(NUM_EVAL),
        '--batch-size', str(batch_size), '--epochs', str(EPOCHS),
        '--lambda-contrastive', str(lambda_contrastive),
        '--temperature', str(TEMPERATURE), '--seed', str(SEED),
        '--num-analysis', str(NUM_ANALYSIS), '--workdir', str(run_dir),
    ]
    print('\\nSTART', label)
    print(' '.join(command))
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    try:
        if not training_complete:
            subprocess.run(command, cwd=EXPERIMENT_ROOT, env=environment, check=True)
        analysis_path = run_dir / 'attention_analysis.json'
        analysis_complete = False
        if analysis_path.exists() and final_path.exists():
            try:
                analysis_complete = 'attention_mean_macro_f1' in json.loads(final_path.read_text())
            except json.JSONDecodeError:
                pass
        if not analysis_complete:
            subprocess.run([sys.executable, '-u', 'analyze_attention.py',
                            '--run-dir', str(run_dir), '--seed', str(SEED)],
                           cwd=EXPERIMENT_ROOT, env=environment, check=True)
        else:
            print('SKIP completed attention analysis:', label)
    finally:
        release_after_model()
"""),
    markdown("## 5. Individual model cells\n\nRun only what you need. Each cell trains exactly one model; a completed "
             "model is skipped if its cell is rerun."),
]

for task in ("logic_3var_2term", "logic_4var_2term", "logic_4var_3term", "logic_5var_2term"):
    for size in (128_000, 640_000, 6_400_000):
        cells.append(markdown(f"### {task} — {size:,} training examples"))
        for batch_size in (32, 64, 128):
            for lambda_contrastive in (0.0, 0.01, 0.05, 0.1):
                cells.append(markdown(
                    f"B={batch_size}, λ={lambda_contrastive:g}, T=1.0"
                ))
                cells.append(code(
                    f"run_model('{task}', {size}, {batch_size}, {lambda_contrastive})\n"
                ))

cells.extend([
    markdown("## 6. Aggregate results and plot charts\n\nThis can be rerun at any time and includes every completed model."),
    code("""subprocess.run([sys.executable, 'summarize_results.py',
                '--results-root', str(RESULTS_ROOT)],
               cwd=EXPERIMENT_ROOT, check=True)

from IPython.display import display, Image
import pandas as pd
display(pd.read_csv(RESULTS_ROOT / 'results_compact.csv'))
for chart in sorted(RESULTS_ROOT.glob('*.png')):
    print(chart.name)
    display(Image(filename=str(chart)))
"""),
    markdown("## 7. Download all current results\n\nKaggle results live in `/kaggle/working`; this creates one downloadable archive there."),
    code("""archive = shutil.make_archive(str(PLATFORM_ROOT / 'fuzzy_logic_attention_pytorch_results'),
                              'zip', root_dir=RESULTS_ROOT)
print('Created:', archive)
if IS_COLAB:
    from google.colab import files
    files.download(archive)
"""),
])

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": False},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = Path(__file__).with_name("kaggle_fuzzy_logic_attention_contrastive_pytorch.ipynb")
path.write_text(json.dumps(notebook, indent=1))
print(path)
