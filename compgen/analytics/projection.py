"""
Principal-component projection of representation vectors.

Task-agnostic dimensionality reduction for interpretation: fit a PCA on a set
of per-example representation vectors (attention-derived codes, hidden states,
anything shaped ``(B, D)``) and project them to a small number of components,
two by default, for a scatter plot.

`PCAProjector` fits and can transform new vectors; `pca_project` is a
one-call convenience returning a `ProjectionResult`. The projection is made
deterministic: each component is sign-fixed so that its largest-absolute
loading is positive, removing PCA's inherent sign ambiguity so the same input
always yields the same coordinates.

Inputs may be `torch.Tensor` or `numpy.ndarray`; coordinates are returned as
`numpy.ndarray`.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
from sklearn.decomposition import PCA

Vectors = Union[torch.Tensor, np.ndarray]


def _to_numpy(vectors: Vectors) -> np.ndarray:
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    return np.asarray(vectors, dtype=np.float64)


def _fix_component_signs(pca: PCA) -> None:
    """Flip each component so its largest-absolute loading is positive."""
    components = pca.components_
    if components.size == 0:
        return
    leading = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), leading])
    signs[signs == 0] = 1.0
    pca.components_ = components * signs[:, None]


@dataclass
class ProjectionResult:
    """Coordinates and explanatory metadata from a PCA projection.

    - ``coordinates``: ``(B, n_components)`` projected vectors.
    - ``explained_variance_ratio``: ``(n_components,)`` fraction of variance.
    - ``projector``: the fitted `PCAProjector`, reusable via ``.transform``.
    """

    coordinates: np.ndarray
    explained_variance_ratio: np.ndarray
    projector: "PCAProjector"


class PCAProjector:
    """Fit a PCA on representation vectors and project them.

    Unlike `sklearn.decomposition.PCA`, this fixes the component signs so
    projections are reproducible, and validates the requested component count
    against the input shape with an error that names the mismatch.
    """

    def __init__(self, n_components: int = 2):
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        self.n_components = n_components
        self._pca: PCA = None

    def fit(self, vectors: Vectors) -> "PCAProjector":
        x = _to_numpy(vectors)
        if x.ndim != 2:
            raise ValueError(
                f"vectors must be two-dimensional (B, D), got shape {x.shape}"
            )
        n_samples, n_features = x.shape
        limit = min(n_samples, n_features)
        if self.n_components > limit:
            raise ValueError(
                f"n_components={self.n_components} exceeds the available "
                f"dimensionality min(samples={n_samples}, features={n_features})={limit}"
            )
        pca = PCA(n_components=self.n_components)
        pca.fit(x)
        _fix_component_signs(pca)
        self._pca = pca
        return self

    def transform(self, vectors: Vectors) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("PCAProjector must be fitted before calling transform")
        return self._pca.transform(_to_numpy(vectors))

    def fit_transform(self, vectors: Vectors) -> np.ndarray:
        return self.fit(vectors).transform(vectors)

    @property
    def components_(self) -> np.ndarray:
        self._require_fitted()
        return self._pca.components_

    @property
    def mean_(self) -> np.ndarray:
        self._require_fitted()
        return self._pca.mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        self._require_fitted()
        return self._pca.explained_variance_ratio_

    def _require_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("PCAProjector must be fitted first")


def pca_project(vectors: Vectors, n_components: int = 2) -> ProjectionResult:
    """Fit a PCA on ``vectors`` and return its projection and metadata."""
    projector = PCAProjector(n_components=n_components).fit(vectors)
    return ProjectionResult(
        coordinates=projector.transform(vectors),
        explained_variance_ratio=projector.explained_variance_ratio_,
        projector=projector,
    )
