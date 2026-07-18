"""
Torch-facing Match3 dataset.

Reads the JSON Lines file produced by `generators/match3.py` (one example per
line: {"seq": [...], "labels": [...]}) and exposes it as a `torch.utils.data.
Dataset`, plus a `collate_fn` that pads variable-length sequences to the
batch's max length. Contains no generation logic — that lives entirely in the
generator so datasets can be built once and reused across runs.
"""

import json
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class Match3Dataset(Dataset):
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
        seq = record["seq"]
        return {
            "seq_input": torch.tensor(seq, dtype=torch.int),
            "labels": torch.tensor(record["labels"], dtype=torch.int),
            "shape": len(seq),
        }


def match3_collate_fn(batch: list) -> dict:
    """Pad a batch of variable-length Match3 sequences to the batch's max length."""
    max_len = max(item["shape"] for item in batch)

    seq = torch.stack([
        F.pad(item["seq_input"], (0, max_len - item["shape"]), value=0)
        for item in batch
    ])
    labels = torch.stack([
        F.pad(item["labels"], (0, max_len - item["shape"]), value=-100)
        for item in batch
    ])
    attention_mask = torch.stack([
        F.pad(torch.ones(item["shape"], dtype=torch.int64), (0, max_len - item["shape"]), value=0)
        for item in batch
    ])
    position = torch.arange(max_len).unsqueeze(0).expand(len(batch), -1)

    return {
        "seq": seq.unsqueeze(-1),
        "position": position.unsqueeze(-1),
        "label": labels,
        "shape": position,
        "batch_mask": {
            "attention_mask": attention_mask == 0,
            "mask": None,
        },
    }
