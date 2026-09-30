"""
Capture a model's per-layer attention and hidden states.

`capture(model, batch)` runs a model over a batch and returns a
`CaptureResult` holding, in layer order:

- ``attentions``: each attention block's weights, shape ``(B, H, N, N)``;
- ``hidden_states``: the hidden state entering each attention block,
  shape ``(B, N, D)``.

The helper is task-agnostic: it finds every shared ``BaseAttention`` submodule
in the model (by registration order) and attaches hooks, so it works for the
fuzzy-logic model, the Match3 model, and any future model composed of the
shared encoder layers — without importing task-specific code or requiring an
attention-returning `forward`.

Assumptions and guarantees:

- The model's attention blocks are ``BaseAttention`` subclasses (one per
  encoder layer). A model with none raises a clear error.
- Capture runs in inference mode (dropout disabled) so the weights are the
  clean evaluation-time distributions, and the model's previous training flag
  is restored afterward. Parameters are never modified.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from compgen.models.attentions.base import BaseAttention


@dataclass
class CaptureResult:
    """Per-layer attention weights and the hidden states entering them."""

    attentions: List[torch.Tensor]
    hidden_states: List[torch.Tensor]


def _attention_modules(model: nn.Module) -> List[BaseAttention]:
    return [module for module in model.modules() if isinstance(module, BaseAttention)]


def capture(model: nn.Module, batch, device: Optional[str] = None) -> CaptureResult:
    """Run ``model`` over ``batch`` and capture per-layer attention and inputs.

    Args:
        model: a model composed of shared ``BaseAttention`` blocks (for example
            ``FuzzyLogicModel`` or ``Match3Model``).
        batch: whatever the model's ``forward`` accepts (a dict for Match3, a
            tensor or object exposing ``.x`` for fuzzy logic).
        device: optional device to move the model (and a tensor batch) to
            before running.

    Returns:
        `CaptureResult` with one attention tensor and one hidden state per
        attention block, in layer order.
    """
    modules = _attention_modules(model)
    if not modules:
        raise ValueError(
            "capture found no shared attention modules (BaseAttention subclasses) "
            "in the model; only models composed of the shared encoder layers are supported"
        )

    if device is not None:
        model.to(device)
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)

    was_training = model.training
    model.eval()

    attentions: List[torch.Tensor] = []
    hidden_states: List[torch.Tensor] = []
    handles = []

    def _save_attention(module, inputs, output):
        weights = output[1] if isinstance(output, tuple) else output
        attentions.append(weights.detach())

    def _save_input(module, args, kwargs):
        hidden_state = kwargs.get("hidden_state", args[0] if args else None)
        if hidden_state is not None:
            hidden_states.append(hidden_state.detach())

    for module in modules:
        handles.append(module.register_forward_hook(_save_attention))
        handles.append(module.register_forward_pre_hook(_save_input, with_kwargs=True))

    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    return CaptureResult(attentions=attentions, hidden_states=hidden_states)
