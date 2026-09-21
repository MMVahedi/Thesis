"""
Base class for dataset generators.

A generator's only job is to produce a list of in-memory examples and
serialize them to disk as JSON Lines — it has no knowledge of how those
examples will later be loaded into a training pipeline (that's the job of
`torch_datasets/`). Keeping this shared save/serialize logic in one place
means every task-specific generator (Match3, or future tasks) gets the same
`.jsonl` output format and `save()` behavior for free.
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, List, TypeVar

# The in-memory representation of a single example, as produced by `generate()`
# and consumed by `to_record()`. Each subclass fixes this to whatever is most
# convenient for its task (e.g. a (sequence, labels) tuple of numpy arrays for
# Match3) — the base class only needs it to be *some* consistent type.
Example = TypeVar("Example")

# Key wrapping the dataset-level metadata header line `save()` writes at the
# top of the output file (see `dataset_meta`). Readers (torch_datasets/*)
# check for this key to tell the header apart from an example record.
META_KEY = "__meta__"


class DatasetGenerator(ABC, Generic[Example]):
    """
    Subclasses must implement:
      - `generate()`: build and return the in-memory examples for this
        dataset (task-specific sampling logic lives here).
      - `to_record(example)`: convert a single example into a plain
        JSON-serializable dict (e.g. numpy arrays -> lists).

    Subclasses may also override `meta(example)` to attach optional
    per-example metadata (e.g. a human-readable description of what was
    generated) under the record's "meta" key, without cluttering
    `to_record`'s core fields, and `dataset_meta()` to control the
    dataset-level metadata (defaults to this generator's config) written as
    a header line.

    `save(path)` ties these together: generate, write an optional
    `{"__meta__": ...}` header line, then one JSON object per example.
    """

    @abstractmethod
    def generate(self) -> List[Example]:
        """Build and return the in-memory examples for this dataset."""
        raise NotImplementedError

    @abstractmethod
    def to_record(self, example: Example) -> dict:
        """Convert one example (as returned by `generate`) into a JSON-serializable dict."""
        raise NotImplementedError

    def meta(self, example: Example) -> dict:
        """Optional per-example metadata to attach under the record's "meta" key. Empty by default."""
        return {}

    def dataset_meta(self) -> dict:
        """
        Optional dataset-level metadata saved as the output's header line
        (see `save`). Defaults to this generator's `self.config`, if it has
        one and it's a dataclass, so every generator's config is recorded
        for free; override to customize or suppress it.
        """
        config = getattr(self, "config", None)
        if config is not None and dataclasses.is_dataclass(config):
            return dataclasses.asdict(config)
        return {}

    def to_records(self, examples: List[Example]) -> Iterator[dict]:
        for example in examples:
            record = self.to_record(example)
            meta = self.meta(example)
            if meta:
                record["meta"] = meta
            yield record

    def save(self, path: str) -> None:
        """
        Generate examples and write them to `path` as JSON Lines: an
        optional `{"__meta__": ...}` header line (see `dataset_meta`)
        followed by one JSON object per example.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self.generate()
        with out_path.open("w") as f:
            meta = self.dataset_meta()
            if meta:
                f.write(json.dumps({META_KEY: meta}) + "\n")
            for record in self.to_records(examples):
                f.write(json.dumps(record) + "\n")
