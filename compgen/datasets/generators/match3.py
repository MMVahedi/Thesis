import random
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from compgen.datasets.generators.base import DatasetGenerator

# A Match3 example is the (sequence, labels) pair produced by
# `Match3Generator.generate_sequence` and `Match3Generator.compute_labels`.
Match3Example = Tuple[np.ndarray, np.ndarray]


@dataclass
class Match3Config:
    num_instances: int  # total number of examples the generated dataset should contain
    seq_len: int  # maximum sequence length (inclusive)
    M: int = 5  # modulus; tokens are drawn from {1, ..., M-1}
    min_len: int = 3  # minimum sequence length (inclusive)
    num_bins: int = 5  # number of positive-label-percentage buckets to balance across
    max_percent_positive_target: int = 40  # generation never *targets* denser sequences than this
    oversample_pool: int = 5000  # sequences actually generated; the rest of num_instances
    # is filled by permuting these (see Match3Generator.generate), so generation cost
    # stays bounded even when num_instances is very large.


class Match3Generator(DatasetGenerator[Match3Example]):

    def __init__(self, config: Match3Config):
        self.config = config
        self.alphabet = np.arange(1, config.M)

    def _bucket_of(self, percent_positive: float) -> int:
        """Map a percentage in [0, 100] to a bucket index in [0, num_bins - 1]."""
        bucket = int(percent_positive // (100 / self.config.num_bins))
        return min(bucket, self.config.num_bins - 1)  # clamp the 100% edge case into the last bucket

    @staticmethod
    def _pairwise_sums_mod(values: np.ndarray, modulus: int) -> np.ndarray:
        """
        All sums a + b (mod `modulus`) for a, b drawn from `values`, repeats
        allowed (this also covers a paired with itself, via the diagonal a + a).

        `values[:, None] + values[None, :]` broadcasts to the full L x L grid of
        sums, i.e. every ordered pair (a, b) including a == b — exactly the set of
        "two other tokens" a triple-sum can draw its remaining two slots from.
        """
        return (values[:, None] + values[None, :]) % modulus

    @staticmethod
    def compute_labels(seq: np.ndarray, modulus: int) -> np.ndarray:
        """
        Label position i with 1 iff some triple of indices (i, j, k) — repeats
        allowed — satisfies seq[i] + seq[j] + seq[k] ≡ 0 (mod `modulus`).

        Naively this is an O(L^3) triple loop over all (i, j, k). We instead
        exploit that j and k range independently over the *whole* sequence
        (including j == k, and either equal to i): the condition only depends on
        seq[i] and the set of pairwise sums seq[j] + seq[k] achievable from the
        full sequence. So we precompute that achievable-sums set once (O(L^2))
        and then, for every position i, just look up whether the complement
        (-seq[i]) mod modulus is in it (O(L)) — an O(L^2) replacement for O(L^3).
        """
        achievable = np.zeros(modulus, dtype=bool)
        achievable[np.unique(Match3Generator._pairwise_sums_mod(seq, modulus))] = True
        needed = (-seq) % modulus  # the pair-sum that would complete seq[i]'s triple to 0
        return achievable[needed].astype(np.int64)

    @staticmethod
    def _split_candidates(
        placed: np.ndarray, alphabet: np.ndarray, modulus: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split `alphabet` into values that, if appended next, would immediately be
        labeled 1 (paired with two already-placed tokens, or with itself thrice)
        versus the rest.

        Used only during generation, to bias sampling toward hitting a target
        number of positives — it is a cheap local heuristic, not the ground truth.
        (It doesn't account for a new token pairing with just *one* placed token
        plus itself again; that's fine, since `compute_labels` recomputes exact
        labels on the finished sequence regardless of how it was built.)
        """
        achievable = np.zeros(modulus, dtype=bool)
        if placed.size:
            # Sums achievable from two already-placed tokens (self-pairs included).
            achievable[np.unique(Match3Generator._pairwise_sums_mod(placed, modulus))] = True

        # A candidate value v would be positive if either:
        #  - some pair of placed tokens sums to (-v) mod modulus, so v completes it, or
        #  - v is a "self-triple": v + v + v ≡ 0 (mod modulus), so v is positive on its own.
        would_be_positive = achievable[(-alphabet) % modulus] | ((3 * alphabet) % modulus == 0)
        return alphabet[would_be_positive], alphabet[~would_be_positive]

    @staticmethod
    def generate_sequence(
        length: int, alphabet: np.ndarray, modulus: int, num_positive: int
    ) -> np.ndarray:
        """
        Build a length-`length` sequence over `alphabet`, steering it toward
        roughly `num_positive` positions ending up labeled 1 (see `compute_labels`).

        Steering is heuristic: at each step the next token is biased toward a
        value that would be immediately positive/negative; once the (recomputed)
        positive count reaches the target, the rest of the sequence is filled with
        non-positive-inducing values where possible. The exact labels used
        downstream are always recomputed from scratch with `compute_labels`.
        """
        # Positions we'll *try* to make positive. Positions 0 and 1 are excluded:
        # with fewer than 2 tokens already placed there isn't enough history for
        # the "paired with two placed tokens" heuristic in `_split_candidates` to
        # bite, so nudging them would mostly be a no-op. Sampling with replacement
        # (random.choices) means fewer than num_positive distinct positions may
        # end up targeted, which is fine — this is a bias, not a guarantee.
        target_positions = (
            set(random.choices(range(2, length), k=num_positive))
            if length > 2 and num_positive > 0
            else set()
        )

        seq = []
        for i in range(length):
            placed = np.array(seq, dtype=alphabet.dtype)
            positive_cand, negative_cand = Match3Generator._split_candidates(placed, alphabet, modulus)

            if i in target_positions and positive_cand.size:
                # This position was chosen to be nudged positive, and a
                # positive-inducing value exists — use it.
                seq.append(random.choice(positive_cand.tolist()))
            elif negative_cand.size:
                # Default case: keep the sequence negative where possible.
                seq.append(random.choice(negative_cand.tolist()))
            else:
                # Every remaining value would be positive (can happen with a
                # small modulus/alphabet) — no negative option left, so fall
                # back to a positive one.
                seq.append(random.choice(positive_cand.tolist()))

            # Recompute exact labels on the sequence built so far. Once we've
            # already hit (or passed) the positive quota, stop trying to steer
            # and just fill out the rest with non-positive-inducing values (the
            # actual final count can still drift, since this heuristic is only
            # approximate — see `_split_candidates`).
            if Match3Generator.compute_labels(np.array(seq), modulus).sum() >= num_positive:
                while len(seq) < length:
                    placed = np.array(seq, dtype=alphabet.dtype)
                    positive_cand, negative_cand = Match3Generator._split_candidates(placed, alphabet, modulus)
                    seq.append(random.choice((negative_cand if negative_cand.size else positive_cand).tolist()))
                break

        # Shuffle so the positions we deliberately targeted above aren't
        # concentrated in a suspicious pattern (e.g. always index >= 2).
        seq = np.array(seq, dtype=alphabet.dtype)
        np.random.shuffle(seq)
        return seq

    def generate(self) -> List[Match3Example]:
        cfg = self.config

        # Split num_instances as evenly as possible across the buckets,
        # handing the remainder to the first few buckets.
        target_per_bucket = [cfg.num_instances // cfg.num_bins] * cfg.num_bins
        for i in range(cfg.num_instances % cfg.num_bins):
            target_per_bucket[i] += 1

        buckets: List[List[Match3Example]] = [[] for _ in range(cfg.num_bins)]

        # Cap how many sequences we actually generate: generation is the
        # expensive part, so beyond this pool we top up under-filled buckets
        # cheaply by permuting sequences we already have (see below).
        pool_size = min(cfg.num_instances, cfg.oversample_pool)

        for _ in range(pool_size):
            if all(len(b) >= t for b, t in zip(buckets, target_per_bucket)):
                break  # every bucket already has enough natural samples

            length = random.randint(cfg.min_len, cfg.seq_len)
            percent_target = random.randint(1, cfg.max_percent_positive_target - 1)
            num_positive = int(length * percent_target / 100)

            seq = self.generate_sequence(length, self.alphabet, cfg.M, num_positive)
            labels = self.compute_labels(seq, cfg.M)  # ground-truth labels, independent of how seq was steered

            bucket = self._bucket_of(100 * labels.sum() / length)
            if len(buckets[bucket]) < target_per_bucket[bucket]:
                buckets[bucket].append((seq, labels))
            # else: this bucket is already full, discard the sample and keep going

        # Some buckets (e.g. very high positive-percentage ones) may rarely
        # or never occur naturally, since generation only *targets* up to
        # max_percent_positive_target. Top those up by permuting an existing
        # example from the same bucket — a permutation doesn't change which
        # positions are labeled 1 (see compute_labels), just their order, so
        # it's a valid (if less diverse) additional sample for that bucket.
        for bucket, target in zip(buckets, target_per_bucket):
            while len(bucket) < target and bucket:
                seq, labels = random.choice(bucket)
                perm = np.random.permutation(len(seq))
                bucket.append((seq[perm], labels[perm]))

        examples: List[Match3Example] = []
        for bucket_idx, bucket in enumerate(buckets):
            if len(bucket) < target_per_bucket[bucket_idx]:
                # Bucket started with zero natural samples, so there was
                # nothing to permute-and-top-up from either — the dataset
                # will end up smaller than num_instances requested.
                warnings.warn(
                    f"Match3Generator: bucket {bucket_idx} only has {len(bucket)}/"
                    f"{target_per_bucket[bucket_idx]} examples (no natural samples "
                    "landed in it to resample from)."
                )
            examples.extend(bucket)
        return examples

    def to_record(self, example: Match3Example) -> dict:
        """Convert one (seq, labels) numpy pair into a JSON-serializable dict."""
        seq, labels = example
        return {"seq": seq.tolist(), "labels": labels.tolist()}
