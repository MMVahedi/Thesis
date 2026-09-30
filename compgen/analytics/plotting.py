"""
Plotting helpers for interpretability output.

Thin, task-agnostic wrappers that turn analytics results into figures:

- `plot_projection(coordinates, labels=None, ...)` scatters 2D coordinates,
  optionally coloring each point by a supplied label/value array.
- `plot_attention(attention, ...)` renders a token's attention row or a
  per-head attention matrix as a heatmap.

Both return ``matplotlib.figure.Figure`` objects and never write to disk; the
caller chooses the backend and decides whether to save. Inputs may be
`torch.Tensor` or `numpy.ndarray`.
"""

from typing import Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ArrayLike = Union[torch.Tensor, np.ndarray]


def _to_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _figure_and_axes(ax: Optional[Axes]):
    if ax is None:
        return plt.subplots()
    return ax.figure, ax


def plot_projection(
    coordinates: ArrayLike,
    labels: Optional[ArrayLike] = None,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Figure:
    """Scatter 2D coordinates, optionally coloring points by ``labels``.

    ``coordinates`` must be ``(B, 2)``; ``labels`` (if given) must have ``B``
    entries and is passed to matplotlib's ``c=`` for per-point coloring.
    """
    coords = _to_numpy(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"coordinates must have shape (B, 2), got {coords.shape}"
        )
    figure, axes = _figure_and_axes(ax)
    if labels is not None:
        labels = _to_numpy(labels)
        if labels.shape[0] != coords.shape[0]:
            raise ValueError(
                f"labels length {labels.shape[0]} does not match {coords.shape[0]} points"
            )
        scatter = axes.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=12)
        figure.colorbar(scatter, ax=axes)
    else:
        axes.scatter(coords[:, 0], coords[:, 1], s=12)
    axes.set(xlabel="PC 1", ylabel="PC 2")
    if title is not None:
        axes.set_title(title)
    return figure


def plot_attention(
    attention: ArrayLike,
    ax: Optional[Axes] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
) -> Figure:
    """Render attention values as a heatmap.

    Accepts a single token's attention row ``(N,)`` (shown as one row), a
    per-head set of rows ``(H, N)``, or a full query-by-key matrix ``(N, N)``.
    """
    values = _to_numpy(attention)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(
            f"attention must be 1-D or 2-D, got shape {values.shape}"
        )
    figure, axes = _figure_and_axes(ax)
    image = axes.imshow(values, aspect="auto", cmap=cmap)
    figure.colorbar(image, ax=axes)
    axes.set(xlabel="Key position", ylabel="Head / query")
    if title is not None:
        axes.set_title(title)
    return figure
