"""
Attention-weight interpretation helpers.

Task-agnostic utilities that turn attention weights into numbers you can
analyze. They operate on plain tensors and import no task-specific code:

- `attention_code(attention, query_token, key_token)`: one scalar per head for
  a chosen (query, key) pair — the classic "response-token code" when both
  default to the last position.
- `attention_row_features(attention, query_token)`: the full attention row of a
  query token, flattened over heads and keys.
- `stack_layers(attentions, reducer)`: concatenate per-layer features so layer
  identity is preserved in the resulting vector.
- `token_attention_row(attentions, layer, query_token, batch_index, ...)`: the
  raw attention distribution of a specific token over the key positions.

Attention tensors are expected in ``(batch, heads, query, key)`` layout, the
layout every architecture returns. Out-of-range indices raise an error naming
the invalid value rather than returning a silently wrong slice.
"""

from typing import List, Sequence, Union

import torch

Attention = torch.Tensor
LayerSet = Union[torch.Tensor, Sequence[torch.Tensor]]


def as_layers(attentions: LayerSet) -> List[torch.Tensor]:
    """Normalize an attention argument into a list of ``(B, H, N, N)`` tensors.

    Accepts a single ``(B, H, N, N)`` tensor, a stacked ``(L, B, H, N, N)``
    tensor, or any sequence of ``(B, H, N, N)`` tensors.
    """
    if isinstance(attentions, torch.Tensor):
        if attentions.dim() == 4:
            return [attentions]
        if attentions.dim() == 5:
            return list(attentions.unbind(0))
        raise ValueError(
            "attention tensor must have shape (B, H, N, N) or (L, B, H, N, N), "
            f"got {tuple(attentions.shape)}"
        )
    layers = list(attentions)
    if not layers:
        raise ValueError("attentions must contain at least one layer")
    for layer in layers:
        if layer.dim() != 4:
            raise ValueError(
                f"each attention layer must have shape (B, H, N, N), got {tuple(layer.shape)}"
            )
    return layers


def _check_attention(attention: torch.Tensor) -> None:
    if attention.dim() != 4:
        raise ValueError(
            f"attention must have shape (B, H, N, N), got {tuple(attention.shape)}"
        )


def attention_code(
    attention: torch.Tensor, query_token: int = -1, key_token: int = -1
) -> torch.Tensor:
    """Per-example, per-head attention value at a chosen (query, key) pair.

    Returns a ``(B, H)`` tensor. The defaults (last query, last key) reproduce
    the response-token self-attention code used by the fuzzy-logic model.
    """
    _check_attention(attention)
    return attention[:, :, query_token, key_token]


def attention_row_features(
    attention: torch.Tensor, query_token: int = -1
) -> torch.Tensor:
    """Full attention row of ``query_token``, flattened over heads and keys.

    Returns a ``(B, H * N)`` tensor.
    """
    _check_attention(attention)
    row = attention[:, :, query_token, :]  # (B, H, N)
    batch, heads, length = row.shape
    return row.reshape(batch, heads * length)


def stack_layers(
    attentions: LayerSet,
    reducer: str = "code",
    query_token: int = -1,
    key_token: int = -1,
) -> torch.Tensor:
    """Concatenate per-layer features into one vector per example.

    ``reducer="code"`` uses :func:`attention_code` and returns ``(B, L * H)``;
    ``reducer="row"`` uses :func:`attention_row_features` and returns
    ``(B, L * H * N)``. Layer order is preserved along the feature dimension.
    """
    if reducer not in ("code", "row"):
        raise ValueError(f"reducer must be 'code' or 'row', got {reducer!r}")
    layers = as_layers(attentions)
    if reducer == "code":
        features = [attention_code(layer, query_token, key_token) for layer in layers]
    else:
        features = [attention_row_features(layer, query_token) for layer in layers]
    return torch.cat(features, dim=1)


def token_attention_row(
    attentions: LayerSet,
    layer: int,
    query_token: int,
    batch_index: int = 0,
    average_heads: bool = False,
) -> torch.Tensor:
    """Attention of one query token over the key positions.

    Selects ``batch_index`` and ``layer``, then returns that token's raw
    attention over keys: ``(H, N)`` by default, or ``(N,)`` when
    ``average_heads=True``.
    """
    layers = as_layers(attentions)
    if not -len(layers) <= layer < len(layers):
        raise IndexError(
            f"layer index {layer} is out of range for {len(layers)} layer(s)"
        )
    attention = layers[layer]  # (B, H, N, N)
    batch, heads, length, _ = attention.shape
    if not -batch <= batch_index < batch:
        raise IndexError(
            f"batch index {batch_index} is out of range for batch size {batch}"
        )
    if not -length <= query_token < length:
        raise IndexError(
            f"query token index {query_token} is out of range for sequence length {length}"
        )
    row = attention[batch_index, :, query_token, :]  # (H, N)
    if average_heads:
        row = row.mean(dim=0)
    return row
