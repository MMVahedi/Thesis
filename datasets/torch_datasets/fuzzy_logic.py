"""
Torch-facing fuzzy logic dataset.

Reads the JSON Lines file produced by `generators/fuzzy_logic.py` (one
example per line: {"input": ..., "target": ..., "latent": ...}, optionally
preceded by a {"__meta__": ...} header line with the generator's config —
see `generators/base.py`) and exposes each as a single (input, target) pair
— a plain supervised example, not an in-context sequence.
"""

import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from datasets.generators.base import META_KEY


class FuzzyLogicDataset(Dataset):
    def __init__(self, path: str):
        self.meta: dict = {}
        self.examples: List[dict] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if META_KEY in record:
                    self.meta = record[META_KEY]
                else:
                    self.examples.append(record)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        record = self.examples[idx]
        return {
            "x": torch.tensor(record["input"], dtype=torch.float32),  # (num_variables,)
            "y": torch.tensor(record["target"], dtype=torch.float32),  # scalar
            "latent": torch.tensor(record["latent"], dtype=torch.float32),  # (num_terms, num_variables)
        }
