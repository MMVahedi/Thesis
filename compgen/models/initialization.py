"""
Reference weight initialization for the fuzzy-logic transformer.

Ports the experiment's ``self.apply(_init_reference_style)``: linear projections
get truncated-normal weights with a T5-style standard deviation derived from
their fan-in, linear biases are zeroed, and embedding tables get normal weights
scaled by ``1 / sqrt(num_embeddings)``. Task-agnostic and importable so any task
model can opt into the reference initialization.
"""

import math

import torch.nn as nn


def _init_reference_style(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        fan_in = module.weight.shape[1]
        std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
        nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=math.sqrt(1.0 / module.weight.shape[0]))


def init_reference_weights(module: nn.Module) -> nn.Module:
    """Apply the reference initialization to every Linear/Embedding in `module`."""
    module.apply(_init_reference_style)
    return module
