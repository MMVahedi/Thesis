"""
Fuzzy logic dataset generator, from "Attention as a Hypernetwork".

This is an in-context-learning (ICL) task: each example is one "concept" (a
fuzzy DNF formula over `num_variables` boolean variables) shown as a sequence
of (input, output) pairs, with the final output hidden — the model has to
infer the concept from the context pairs and predict the hidden output for
the last (query) input.

Concept space
-------------
A *minterm* is a maximal conjunction over all `num_variables` variables, e.g.
for 3 variables: (x1 AND NOT x2 AND x3). There are exactly 2**num_variables
of them (every possible pattern of polarities), and we represent one as a
{0, 1}^num_variables vector (1 = the variable itself, 0 = its negation).

A *formula* (a "concept") is the OR of `num_terms` distinct minterms — i.e. a
fuzzy DNF formula. Evaluated on real-valued inputs in [0, 1]^num_variables
using Zadeh fuzzy logic (NOT = 1 - x, AND = min, OR = max), it gives a
continuous target in [0, 1] instead of a hard boolean.

Generalization splits
----------------------
To test both compositional generalization (new combinations of known
minterms) and out-of-distribution generalization (entirely unseen minterms),
the 2**num_variables minterms are first split into an "in-distribution" pool
and an "out-of-distribution" pool (`frac_ood_conj` controls the split). Then:
  - "train"/"test": disjoint sets of formulas (combinations of `num_terms`
    minterms), both built only from in-distribution minterms. `frac_test`
    controls the train/test split.
  - "ood": formulas built only from out-of-distribution minterms — the model
    never saw these minterms at all during training.
This partition must be identical across the train/test/ood generators for a
given configuration, so it is derived deterministically from
(num_variables, num_terms, frac_test, frac_ood_conj, seed) alone, using a rng
local to `__init__` — never the ambient `random`/`np.random` global state
(unlike the per-instance sampling in `generate()`, see below). That keeps the
split reproducible and consistent no matter which split you instantiate
first, or how much global random state calling code has already consumed.
"""

import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from datasets.generators.base import DatasetGenerator

# A fuzzy-logic example: (inputs, targets, latent).
#   inputs:  (seq_len, num_variables) float array in [0, 1] — the context/query points.
#   targets: (seq_len,) float array in [0, 1] — the formula's truth value at each input.
#   latent:  (num_terms, num_variables) {0, 1} array — the formula itself (which
#            minterms it OR's together), kept around for analysis/debugging.
FuzzyLogicExample = Tuple[np.ndarray, np.ndarray, np.ndarray]


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


@dataclass
class FuzzyLogicConfig:
    num_instances: int  # total number of examples the generated dataset should contain
    seq_len: int  # number of (input, output) pairs per example, context + 1 query
    num_variables: int  # number of boolean variables a formula is defined over
    num_terms: int  # number of minterms OR'd together to form one formula
    split: str = "train"  # which formula pool to draw from: "train", "test", or "ood"
    frac_test: float = 0.5  # fraction of in-distribution formulas held out for "test"
    frac_ood_conj: float = 0.25  # fraction of minterms held out entirely for "ood"
    seed: int = 0  # fixes the train/test/ood partition (see module docstring)


class FuzzyLogicGenerator(DatasetGenerator[FuzzyLogicExample]):
    """
    Generates fuzzy-logic ICL examples for one split ("train"/"test"/"ood") of
    the task. To build a full dataset, construct one `FuzzyLogicGenerator` per
    split, sharing the same (num_variables, num_terms, frac_test,
    frac_ood_conj, seed) so they agree on the same underlying partition, and
    `save()` each to its own file — see `dataset/README.md` for an example.
    """

    def __init__(self, config: FuzzyLogicConfig):
        self.config = config

        num_minterms = 2 ** config.num_variables
        # minterms[m] is the m-th {0, 1}^num_variables polarity pattern.
        self.minterms = np.array(list(itertools.product([0, 1], repeat=config.num_variables)))

        num_ood_minterms = int(num_minterms * config.frac_ood_conj)
        in_dist_minterm_ids = range(num_minterms - num_ood_minterms)
        ood_minterm_ids = range(num_minterms - num_ood_minterms, num_minterms)

        # A "formula" is identified by which `num_terms` minterms it OR's
        # together, so every size-num_terms combination of minterm ids from a
        # pool is one candidate formula for that pool.
        in_dist_formulas = list(itertools.combinations(in_dist_minterm_ids, config.num_terms))
        ood_formulas = list(itertools.combinations(ood_minterm_ids, config.num_terms))

        # Shuffle with a rng seeded (and used) only here, so the train/test
        # split is a pure function of the config and identical across
        # separately-constructed generator instances (see module docstring).
        random.Random(config.seed).shuffle(in_dist_formulas)
        num_test_formulas = int(len(in_dist_formulas) * config.frac_test)
        test_formulas = in_dist_formulas[:num_test_formulas]
        train_formulas = in_dist_formulas[num_test_formulas:]

        assert train_formulas, "train formula pool is empty"
        assert test_formulas, "test formula pool is empty"
        if config.frac_ood_conj > 0:
            assert len(list(ood_minterm_ids)) >= config.num_terms, "not enough ood minterms"
            assert ood_formulas, "ood formula pool is empty"

        self._formula_pools = {
            "train": train_formulas,
            "test": test_formulas,
            "ood": ood_formulas,
            "ind": in_dist_formulas,
            "ind+ood": in_dist_formulas + ood_formulas,
        }

    def generate(self) -> List[FuzzyLogicExample]:
        cfg = self.config
        pool = self._formula_pools[cfg.split]

        examples = []
        for _ in range(cfg.num_instances):
            # Sample one formula (with replacement) for this example, and
            # look up the actual {0, 1} minterm patterns it's built from.
            minterm_ids = random.choice(pool)
            latent = self.minterms[list(minterm_ids)]

            inputs = np.random.uniform(size=(cfg.seq_len, cfg.num_variables))
            targets = evaluate_fuzzy_dnf(latent, inputs)
            examples.append((inputs, targets, latent))
        return examples

    def to_record(self, example: FuzzyLogicExample) -> dict:
        """Convert one (inputs, targets, latent) triple into a JSON-serializable dict."""
        inputs, targets, latent = example
        return {
            "inputs": inputs.tolist(),
            "targets": targets.tolist(),
            "latent": latent.tolist(),
        }
