"""On-the-fly PyTorch generator for the paper's fuzzy-logic regression task."""

import itertools
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LogicBatch:
    x: torch.Tensor
    y: torch.Tensor
    base_mse: torch.Tensor
    latents: torch.Tensor
    term_ids: torch.Tensor


class FuzzyLogicGenerator:
    def __init__(self, num_variables, num_terms, frac_test=0.5, frac_ood_conj=0.25, seed=0):
        self.num_variables = num_variables
        self.num_terms = num_terms
        self.seed = seed

        conjunctions = np.asarray(
            list(itertools.product([0, 1], repeat=num_variables)), dtype=np.int64
        )
        num_ood = int(len(conjunctions) * frac_ood_conj)
        in_ids = range(len(conjunctions) - num_ood)
        ood_ids = range(len(conjunctions) - num_ood, len(conjunctions))
        in_functions = np.asarray(list(itertools.combinations(in_ids, num_terms)), dtype=np.int64)
        ood_functions = np.asarray(list(itertools.combinations(ood_ids, num_terms)), dtype=np.int64)

        rng = np.random.default_rng(seed)
        in_functions = in_functions[rng.permutation(len(in_functions))]
        num_test = int(len(in_functions) * frac_test)
        self.conjunctions = torch.from_numpy(conjunctions)
        self.function_ids = {
            "train": torch.from_numpy(in_functions[num_test:]),
            "test": torch.from_numpy(in_functions[:num_test]),
            # The reference's "id" validation loader samples training functions.
            "id": torch.from_numpy(in_functions[num_test:]),
            "ood": torch.from_numpy(ood_functions),
        }
        if any(len(self.function_ids[name]) == 0 for name in ("train", "test", "ood")):
            raise ValueError("Requested task does not contain enough train/test/OOD functions")
        self._device_cache = {}

    def describe(self):
        return {
            "conjunctions": len(self.conjunctions),
            "train_functions": len(self.function_ids["train"]),
            "test_functions": len(self.function_ids["test"]),
            "ood_functions": len(self.function_ids["ood"]),
        }

    def _rng(self, split, batch_index, device):
        split_offset = {"train": 0, "test": 10_000_000, "id": 20_000_000, "ood": 30_000_000}[split]
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed + split_offset + batch_index)
        return generator

    def _tensors_on(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = (
                self.conjunctions.to(device),
                {name: values.to(device) for name, values in self.function_ids.items()},
            )
        return self._device_cache[key]

    @torch.no_grad()
    def sample(self, batch_size, seq_len, split, batch_index, device, input_dist="uniform"):
        generator = self._rng(split, batch_index, device)
        conjunctions, function_ids = self._tensors_on(device)
        function_pool = function_ids[split]
        selected = torch.randint(
            len(function_pool), (batch_size,), generator=generator, device=device
        )
        term_ids = function_pool[selected]
        latents = conjunctions[term_ids]

        if input_dist == "uniform":
            inputs = torch.rand(
                batch_size, seq_len, self.num_variables, generator=generator, device=device
            )
        elif input_dist == "fixed_context":
            fixed = torch.Generator(device=device)
            fixed.manual_seed(self.seed + 77_777)
            context = torch.rand(
                1, seq_len - 1, self.num_variables, generator=fixed, device=device
            ).expand(batch_size, -1, -1)
            query = torch.rand(
                batch_size, 1, self.num_variables, generator=generator, device=device
            )
            inputs = torch.cat((context, query), dim=1)
        else:
            raise ValueError(f"Unsupported input distribution: {input_dist}")

        literals = torch.where(
            latents[:, None].bool(), inputs[:, :, None, :], 1.0 - inputs[:, :, None, :]
        )
        targets = literals.amin(dim=-1).amax(dim=-1, keepdim=True)
        x = torch.cat((inputs, targets), dim=-1)
        x[:, -1, -1] = 0.0
        y = targets[:, -1]
        base_mse = (targets - targets.mean(dim=1, keepdim=True)).square().mean(dim=1)
        return LogicBatch(x=x, y=y, base_mse=base_mse, latents=latents, term_ids=term_ids)
