import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from compgen.datasets.generators.base import DatasetGenerator

# A fuzzy-logic example: (point, target, latent).
#   point:  (num_variables,) float array in [0, 1] — the input point.
#   target: float in [0, 1] — the formula's truth value at that point.
#   latent: (num_terms, num_variables) {0, 1} array — the formula itself (which
#           minterms it OR's together), kept around for analysis/debugging.
FuzzyLogicExample = Tuple[np.ndarray, float, np.ndarray]


@dataclass
class FuzzyLogicConfig:
    num_instances: int  # total number of examples the generated dataset should contain
    num_variables: int  # number of boolean variables a formula is defined over
    num_terms: int  # number of minterms OR'd together to form one formula


class FuzzyLogicGenerator(DatasetGenerator[FuzzyLogicExample]):
    """
    Generates single (point, target) fuzzy-logic examples. Each example
    draws its own formula uniformly at random (`num_terms` distinct minterms
    OR'd together, out of all `2**num_variables` possible minterms) and its
    own input point independently — there's no train/test/ood partitioning
    here; that's left to whoever consumes the generated data.
    """

    def __init__(self, config: FuzzyLogicConfig):
        self.config = config
        # minterms[m] is the m-th {0, 1}^num_variables polarity pattern.
        self.minterms = np.array(list(itertools.product([0, 1], repeat=config.num_variables)))

    @staticmethod
    def evaluate_fuzzy_dnf(latent: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate a fuzzy DNF formula `latent` (num_terms, num_variables) on a
        batch of continuous inputs (num_points, num_variables) using Zadeh fuzzy
        logic, returning (num_points,) truth values in [0, 1].

        NOT(x) = 1 - x, AND = min over the term's variables, OR = max over terms.
        """
        # literal_truth[p, t, v] = truth of variable v's literal (as used in term
        # t) at input point p — i.e. x if the term uses the variable un-negated
        # (latent == 1), or (1 - x) if negated (latent == 0).
        literal_truth = np.where(latent[None, :, :], inputs[:, None, :], 1.0 - inputs[:, None, :])
        term_truth = literal_truth.min(axis=2)  # fuzzy AND across each term's variables
        return term_truth.max(axis=1)  # fuzzy OR across terms

    @staticmethod
    def _term_to_str(term: np.ndarray) -> str:
        """Render one minterm's {0, 1} polarity pattern as a readable conjunction, e.g. "x_1^not(x_2)^x_3"."""
        literals = [f"x_{v + 1}" if bit else f"not(x_{v + 1})" for v, bit in enumerate(term)]
        return "^".join(literals)

    def generate(self) -> List[FuzzyLogicExample]:
        cfg = self.config
        num_minterms = len(self.minterms)

        examples = []
        for _ in range(cfg.num_instances):
            # Sample num_terms distinct minterms uniformly at random to form
            # this example's formula.
            minterm_ids = random.sample(range(num_minterms), cfg.num_terms)
            latent = self.minterms[minterm_ids]

            point = np.random.uniform(size=cfg.num_variables)
            target = float(self.evaluate_fuzzy_dnf(latent, point[None, :])[0])
            examples.append((point, target, latent))
        return examples

    def to_record(self, example: FuzzyLogicExample) -> dict:
        """Convert one (point, target, latent) triple into a JSON-serializable dict."""
        point, target, latent = example
        return {
            "input": point.tolist(),
            "target": target,
            "latent": latent.tolist(),
        }

    def meta(self, example: FuzzyLogicExample) -> dict:
        """Human-readable formula for each term, e.g. {"term1": "x_1^not(x_2)", ...}."""
        _, _, latent = example
        return {f"term{i + 1}": self._term_to_str(term) for i, term in enumerate(latent)}
