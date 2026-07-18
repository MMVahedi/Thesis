"""
Torch-facing fuzzy logic dataset.

Reads the JSON Lines file produced by `generators/fuzzy_logic.py` (one example
per line: {"inputs": ..., "targets": ..., "latent": ...}) and exposes it in
the masked in-context-learning format used by the paper: the model sees a
sequence of (input, target) pairs with the LAST target hidden (set to 0), and
must predict it.

Every example has the same seq_len/num_variables (fixed by the generator's
config), so — unlike Match3 — there is no padding to do, and no custom
collate_fn: the default DataLoader collation stacks examples directly.
"""

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset


class FuzzyLogicDataset(Dataset):
    def __init__(self, path: str):
        self.examples: List[dict] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        record = self.examples[idx]
        inputs = torch.tensor(record["inputs"], dtype=torch.float32)  # (seq_len, num_variables)
        targets = torch.tensor(record["targets"], dtype=torch.float32).unsqueeze(-1)  # (seq_len, 1)
        latent = torch.tensor(record["latent"], dtype=torch.float32)  # (num_terms, num_variables)

        # One ICL "token" per step is (input, target-so-far); hide the final
        # (query) target since that's what the model has to predict.
        x = torch.cat([inputs, targets], dim=-1)
        x[-1, -1] = 0.0

        return {
            "x": x,
            "y": targets[-1],
            "latent": latent,
        }
