"""
Centralized weight initialization for the model layer.

Weight initialization is controlled from one place. Each initialization scheme
is a class: the abstract :class:`WeightInitializer` base plus one concrete
subclass per scheme, and applying a policy to a module initializes that module
and its submodules. Initialization is owned at the **model level**: a task model
accepts the policy class it should apply and runs one initialization method over
the whole model it composes. Sub-level modules — attention, feed-forward,
prediction heads, embeddings, and the encoder layer — perform no initialization
of their own and keep PyTorch's default parameter initialization.

Schemes (one class each, no string identifier):

- :class:`XavierInitializer`: xavier-normal on ``nn.Linear``/``nn.Embedding``
  weights and zero on their biases — the scheme the attention architectures and
  the classification head use.
- :class:`ReferenceInitializer`: the fuzzy-logic transformer's T5-style scheme —
  truncated-normal Linear weights with a fan-in-derived standard deviation,
  zero linear biases, and embedding weights scaled by ``1 / sqrt(num_embeddings)``.
- :class:`DefaultInitializer`: a no-op, i.e. keep PyTorch's default parameter
  initialization.
"""

import math
from abc import ABC, abstractmethod

import torch.nn as nn


class WeightInitializer(ABC):
    """Abstract base for a weight-initialization policy applied to a module.

    Calling an initializer (or ``apply``) runs its scheme over the module and
    all of its submodules via ``module.apply`` and returns the module, so it
    composes with ``module.apply``-style usage. A subclass implements
    :meth:`_init` for a single module; the policy class itself names the scheme
    — there is no string identifier.
    """

    def __call__(self, module: nn.Module) -> nn.Module:
        module.apply(self._init)
        return module

    def apply(self, module: nn.Module) -> nn.Module:
        """Apply this policy to `module` (and its submodules) in place."""
        return self(module)

    @abstractmethod
    def _init(self, module: nn.Module) -> None:
        """Initialize any parameters of one module this policy handles."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class XavierInitializer(WeightInitializer):
    """Xavier-normal weights and zero biases for Linear/Embedding modules."""

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


class ReferenceInitializer(WeightInitializer):
    """The reference T5-style initialization for Linear/Embedding modules."""

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            fan_in = module.weight.shape[1]
            std = math.sqrt(1.0 / fan_in) / 0.87962566103423978
            nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=math.sqrt(1.0 / module.weight.shape[0]))


class DefaultInitializer(WeightInitializer):
    """No-op: keep PyTorch's default initialization for every parameter."""

    def _init(self, module: nn.Module) -> None:
        """Leave the module's parameters untouched."""
