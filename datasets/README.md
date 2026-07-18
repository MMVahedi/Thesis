# dataset

Dataset generation and loading, split into two independent layers:

- **`generators/`** — pure, torch-free data generation. Each task has a
  `<Task>Generator` (subclassing `generators/base.py:DatasetGenerator`) that
  takes a task-specific config, builds examples in memory, and serializes
  them to a `.jsonl` file (one JSON object per line). Nothing in this layer
  knows about PyTorch.
- **`torch_datasets/`** — reads a `.jsonl` file produced by a generator and
  exposes it as a `torch.utils.data.Dataset`, plus a `collate_fn` for
  batching/padding. Nothing in this layer knows how the data was generated.

This split means a dataset only needs to be generated once and saved to disk;
after that, training code just points a `torch_datasets` class at the file.

## Match3

**Task**: each example is a sequence of integers drawn from `{1, ..., M-1}`.
A position is labeled `1` if it takes part in at least one triple of
positions (indices may repeat, so a value can pair with itself) whose values
sum to `0 (mod M)`, and `0` otherwise. Generated sequences are bucketed by
their fraction of positive labels so a dataset covers a spread of
difficulties instead of skewing toward mostly-negative sequences.

Example record in the output `.jsonl`:

```json
{"seq": [29, 25, 7, 8, 18], "labels": [0, 0, 1, 0, 1]}
```

### Parameters (`generators/match3.py:Match3Config`)

| Parameter | Default | Description |
|---|---|---|
| `num_instances` | required | Total number of examples the generated dataset should contain. |
| `seq_len` | required | Maximum sequence length (inclusive). |
| `M` | `5` | Modulus; tokens are drawn from `{1, ..., M-1}`. |
| `min_len` | `3` | Minimum sequence length (inclusive). |
| `num_bins` | `5` | Number of positive-label-percentage buckets to balance the dataset across. |
| `max_percent_positive_target` | `40` | Generation never *targets* a denser sequence than this (actual achieved percentage can still drift above it). |
| `oversample_pool` | `5000` | Number of sequences actually generated; the rest of `num_instances` is filled by permuting these, so generation cost stays bounded even for large `num_instances`. |

### Generating a dataset file

```python
import random
import numpy as np

from dataset.generators.match3 import Match3Config, Match3Generator

random.seed(0)
np.random.seed(0)

config = Match3Config(
    num_instances=50_000,
    seq_len=35,
    M=37,
    min_len=30,
    num_bins=4,
)
Match3Generator(config).save("data/match3/train.jsonl")
```

### Loading it for training

```python
from torch.utils.data import DataLoader

from dataset.torch_datasets.match3 import Match3Dataset, match3_collate_fn

dataset = Match3Dataset("data/match3/train.jsonl")
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=match3_collate_fn)
```

## Fuzzy Logic

**Task** (from "Attention as a Hypernetwork"): an in-context-learning (ICL)
task, not a classification task like Match3. Each example is one "concept" —
a fuzzy DNF formula over `num_variables` boolean variables, built by OR-ing
together `num_terms` distinct *minterms* (maximal conjunctions using every
variable, e.g. `x1 AND NOT x2 AND x3`) — shown as a sequence of `seq_len`
`(input, output)` pairs, with the final output hidden. The model has to infer
the concept from the visible pairs and predict the hidden output for the
last (query) input. Inputs are continuous, in `[0, 1]^num_variables`; the
formula is evaluated with Zadeh fuzzy logic (`NOT(x) = 1 - x`, `AND = min`,
`OR = max`), so the target is a continuous truth value in `[0, 1]`, not a
hard boolean.

To test generalization, the `2**num_variables` minterms are split into an
in-distribution pool and an out-of-distribution pool (`frac_ood_conj`).
"train"/"test" formulas are disjoint combinations of `num_terms` minterms
drawn only from the in-distribution pool (`frac_test` controls that split);
"ood" formulas are drawn only from the out-of-distribution pool, so the
model never saw those minterms during training at all. This partition is a
deterministic function of `(num_variables, num_terms, frac_test,
frac_ood_conj, seed)`, so constructing a `train`, `test`, and `ood` generator
with those five fields matching always agrees on the same partition —
that's what makes it safe to save each split to its own file independently.

Example record in the output `.jsonl` (`num_variables=5`, `num_terms=3`,
`seq_len=16`):

```json
{"inputs": [[0.1, 0.9, ...], ...], "targets": [0.4, 0.8, ...], "latent": [[1, 0, 1, 0, 1], ...]}
```

### Parameters (`generators/fuzzy_logic.py:FuzzyLogicConfig`)

| Parameter | Default | Description |
|---|---|---|
| `num_instances` | required | Total number of examples the generated dataset should contain. |
| `seq_len` | required | Number of `(input, output)` pairs per example: context pairs plus the 1 hidden query. |
| `num_variables` | required | Number of boolean variables a formula is defined over. |
| `num_terms` | required | Number of minterms OR'd together to form one formula. |
| `split` | `"train"` | Which formula pool to draw from: `"train"`, `"test"`, or `"ood"`. |
| `frac_test` | `0.5` | Fraction of in-distribution formulas held out for `"test"`. |
| `frac_ood_conj` | `0.25` | Fraction of minterms held out entirely for `"ood"`. |
| `seed` | `0` | Fixes the train/test/ood partition (must match across split generators, see above). |

### Generating dataset files

```python
import random
import numpy as np

from dataset.generators.fuzzy_logic import FuzzyLogicConfig, FuzzyLogicGenerator

# Shared across splits so they all agree on the same train/test/ood partition.
common = dict(num_variables=5, num_terms=3, frac_test=0.5, frac_ood_conj=0.25, seed=0)

for split, num_instances, path in [
    ("train", 50_000, "data/fuzzy_logic/train.jsonl"),
    ("test", 5_000, "data/fuzzy_logic/test.jsonl"),
    ("ood", 5_000, "data/fuzzy_logic/ood.jsonl"),
]:
    random.seed(0)
    np.random.seed(0)
    config = FuzzyLogicConfig(num_instances=num_instances, seq_len=16, split=split, **common)
    FuzzyLogicGenerator(config).save(path)
```

### Loading it for training

```python
from torch.utils.data import DataLoader

from dataset.torch_datasets.fuzzy_logic import FuzzyLogicDataset

# Every example has the same seq_len/num_variables, so no custom collate_fn
# is needed (unlike Match3's variable-length sequences).
dataset = FuzzyLogicDataset("data/fuzzy_logic/train.jsonl")
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```
