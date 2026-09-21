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

from datasets.generators.match3 import Match3Config, Match3Generator

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

from datasets.torch_datasets.match3 import Match3Dataset, match3_collate_fn

dataset = Match3Dataset("data/match3/train.jsonl")
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=match3_collate_fn)
```

## Fuzzy Logic

**Task** (adapted from "Attention as a Hypernetwork"): a plain supervised
regression task, not a classification task like Match3. Each example is a
single `(input, target)` pair for one "concept" — a fuzzy DNF formula over
`num_variables` boolean variables, built by OR-ing together `num_terms`
distinct *minterms* (maximal conjunctions using every variable, e.g. `x1 AND
NOT x2 AND x3`). The input is a continuous point in `[0, 1]^num_variables`;
the formula is evaluated at that point with Zadeh fuzzy logic (`NOT(x) = 1 -
x`, `AND = min`, `OR = max`), so the target is a continuous truth value in
`[0, 1]`, not a hard boolean. Each example draws its own formula (uniformly
at random from all `2**num_variables` possible minterms) and its own input
point independently — there's no shared context between examples, and the
generator does no train/test/ood partitioning; that's left to whoever
consumes the generated data (e.g. by generating separate files with
different `num_instances`, or splitting a generated file afterward).

Each record also carries a `meta` field with a human-readable rendering of
every term in the formula (e.g. `"x_1^not(x_2)^x_3"` for a term that's
`x1 AND NOT x2 AND x3`) — handy for debugging/inspection, not used in
training.

Example record in the output `.jsonl` (`num_variables=5`, `num_terms=3`):

```json
{
  "input": [0.1, 0.9, 0.4, 0.2, 0.7],
  "target": 0.4,
  "latent": [[1, 0, 1, 0, 1], ...],
  "meta": {"term1": "x_1^not(x_2)^x_3^not(x_4)^x_5", "term2": "...", "term3": "..."}
}
```

### Parameters (`generators/fuzzy_logic.py:FuzzyLogicConfig`)

| Parameter | Default | Description |
|---|---|---|
| `num_instances` | required | Total number of examples the generated dataset should contain. |
| `num_variables` | required | Number of boolean variables a formula is defined over. |
| `num_terms` | required | Number of minterms OR'd together to form one formula. |

### Generating dataset files

```python
from datasets.generators.fuzzy_logic import FuzzyLogicConfig, FuzzyLogicGenerator

config = FuzzyLogicConfig(num_instances=50_000, num_variables=5, num_terms=3)
FuzzyLogicGenerator(config).save("data/fuzzy_logic/train.jsonl")
```

### Loading it for training

```python
from torch.utils.data import DataLoader

from datasets.torch_datasets.fuzzy_logic import FuzzyLogicDataset

# Every example has the same num_variables, so no custom collate_fn is
# needed (unlike Match3's variable-length sequences).
dataset = FuzzyLogicDataset("data/fuzzy_logic/train.jsonl")
loader = DataLoader(dataset, batch_size=64, shuffle=True)
```
