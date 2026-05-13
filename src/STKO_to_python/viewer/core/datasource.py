"""``DataSource`` protocol — adapter between :class:`MPCODataSet` and layers.

The :class:`DataSource` is the **one** place where the viewer's
backend-agnostic Layer/Scene code touches anything STKO-specific.
Defining the protocol here lets every layer be written against the
protocol — a future viewer source (e.g. a streaming OpenSeesPy
adapter) only has to implement the protocol; the layers don't change.

The concrete :class:`MPCODataSourceAdapter` lives in
``viewer/core/_mpco_adapter.py`` (added in Phase 2.3); it is the
default implementation that the layers landing in Phase 2.4 onward
consume.

See ``docs/viewer/01-architecture.md`` §5.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .selection import SelectionSpec
from .types import BBox

if TYPE_CHECKING:
    from ...core.dataset import MPCODataSet


@runtime_checkable
class DataSource(Protocol):
    """Adapter that exposes :class:`MPCODataSet` queries as layer inputs.

    Methods return numpy arrays in shapes the layers expect, hiding the
    DataFrame / MultiIndex assembly that lives inside the existing
    query engines.
    """

    @property
    def dataset(self) -> "MPCODataSet":
        """Underlying dataset, exposed for layers that need richer access."""

    # ----- Geometry -------------------------------------------------- #

    def node_coords(self, ids: np.ndarray | None = None) -> np.ndarray:
        """``(N, 3)`` node coordinates. ``ids=None`` returns every node."""

    def element_centroids(self, ids: np.ndarray | None = None) -> np.ndarray:
        """``(E, 3)`` element centroids in element-id order."""

    def model_bbox(self) -> BBox:
        """Axis-aligned bounding box over every node in the model."""

    # ----- Time axis ------------------------------------------------- #

    def n_steps(self, stage: str | None = None) -> int:
        """Number of time steps in the requested stage (or current)."""

    def time(self, stage: str | None = None) -> np.ndarray:
        """``(T,)`` time values for the requested stage (or current)."""

    # ----- Selection resolution -------------------------------------- #

    def resolve_node_ids(self, spec: SelectionSpec) -> np.ndarray:
        """Return the node IDs matching ``spec`` as a ``(K,)`` int64 array.

        Empty input ``SelectionSpec.empty()`` returns every node.
        """

    def resolve_element_ids(self, spec: SelectionSpec) -> np.ndarray:
        """Return the element IDs matching ``spec`` as a ``(K,)`` int64 array.

        Empty input ``SelectionSpec.empty()`` returns every element.
        """
