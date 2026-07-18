"""
Base class for dataset generators.

A generator's only job is to produce a list of in-memory examples and
serialize them to disk as JSON Lines — it has no knowledge of how those
examples will later be loaded into a training pipeline (that's the job of
`torch_datasets/`). Keeping this shared save/serialize logic in one place
means every task-specific generator (Match3, or future tasks) gets the same
`.jsonl` output format and `save()` behavior for free.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, Iterator, List, TypeVar

# The in-memory representation of a single example, as produced by `generate()`
# and consumed by `to_record()`. Each subclass fixes this to whatever is most
# convenient for its task (e.g. a (sequence, labels) tuple of numpy arrays for
# Match3) — the base class only needs it to be *some* consistent type.
Example = TypeVar("Example")


class DatasetGenerator(ABC, Generic[Example]):
    """
    Subclasses must implement:
      - `generate()`: build and return the in-memory examples for this
        dataset (task-specific sampling logic lives here).
      - `to_record(example)`: convert a single example into a plain
        JSON-serializable dict (e.g. numpy arrays -> lists).

    `save(path)` ties the two together: generate, convert every example to a
    record, and write one JSON object per line to `path`.
    """

    @abstractmethod
    def generate(self) -> List[Example]:
        """Build and return the in-memory examples for this dataset."""
        raise NotImplementedError

    @abstractmethod
    def to_record(self, example: Example) -> dict:
        """Convert one example (as returned by `generate`) into a JSON-serializable dict."""
        raise NotImplementedError

    def to_records(self, examples: List[Example]) -> Iterator[dict]:
        for example in examples:
            yield self.to_record(example)

    def save(self, path: str) -> None:
        """Generate examples and write them to `path` as JSON Lines."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        examples = self.generate()
        with out_path.open("w") as f:
            for record in self.to_records(examples):
                f.write(json.dumps(record) + "\n")
