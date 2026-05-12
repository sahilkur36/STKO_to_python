"""Picklable, dataset-free specifications for section cuts.

Two specs live here:

- :class:`SectionCutSpec` — the geometric + filter definition of a cut.
  Pure data: plane, element filter, side, optional label. No dataset
  binding, no I/O. Picklable.
- :class:`DriftSpec` — a node-pair drift definition (top node, bottom
  node, displacement component, optional height for drift ratio).
  Carries an ``apply(dataset, model_stage=...)`` that wraps the
  existing ``NodalResults.drift()``. Picklable.

Both are frozen dataclasses so they hash by value and travel cleanly
through pickle, ``MPCOResults`` apply loops, and on-disk caches.
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import numpy as np
import pandas as pd

from .plane import Plane

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


Side = Literal["positive", "negative"]


# ---------------------------------------------------------------------- #
# Pickle I/O helpers — shared by both Specs and matched to the rest of
# the library (gzip when the path ends in ``.gz``).
# ---------------------------------------------------------------------- #
def _save_pickle(
    obj: object,
    path: str | Path,
    *,
    compress: bool | None = None,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    p = Path(path)
    if compress is None:
        compress = p.suffix.lower() == ".gz"
    opener = gzip.open if compress else open
    with opener(p, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
    return p


def _load_pickle(path: str | Path) -> object:
    p = Path(path)
    opener = gzip.open if p.suffix.lower() == ".gz" else open
    with opener(p, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------- #
# SectionCutSpec
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class SectionCutSpec:
    """Geometric + filter definition of a section cut.

    Parameters
    ----------
    plane:
        The cut plane (point + outward unit normal).
    selection_set_name, selection_set_id, element_ids:
        Element filter — at least one must be provided. Multiple sources
        union per :class:`SelectionSetResolver` semantics. ``element_ids``
        is stored as a tuple so the spec remains hashable.
    side:
        ``"positive"`` (default) treats the side along the plane normal
        as the "kept" side; ``"negative"`` flips that. The resultant
        returned by a cut is the force the kept side exerts on the
        discarded side (action–reaction; classic internal-force sign).
    label:
        Optional display label used by plotters.
    name:
        Optional human-readable identifier for logs and on-disk caches.
    bounding_polygon:
        Optional convex polygon on ``plane`` restricting the cut to
        elements whose intersection falls inside it. Each vertex is a
        ``(x, y, z)`` tuple lying on ``plane`` (within tolerance). At
        least three vertices required. Useful when the recorded
        selection sets don't pre-filter to the region of interest —
        e.g. cut just the left half of a wall by passing a polygon on
        the cut plane covering only that half. Non-convex polygons are
        not supported in v1.6 (validated away at construction).
    """

    plane: Plane
    selection_set_name: str | None = None
    selection_set_id: int | None = None
    element_ids: tuple[int, ...] | None = None
    side: Side = "positive"
    label: str | None = None
    name: str | None = None
    bounding_polygon: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plane, Plane):
            raise TypeError(
                f"plane must be a Plane, got {type(self.plane).__name__}."
            )
        if self.side not in ("positive", "negative"):
            raise ValueError(
                f"side must be 'positive' or 'negative', got {self.side!r}."
            )
        if (
            self.selection_set_name is None
            and self.selection_set_id is None
            and self.element_ids is None
        ):
            raise ValueError(
                "SectionCutSpec requires at least one of: "
                "selection_set_name, selection_set_id, element_ids."
            )
        if self.element_ids is not None:
            tup = _coerce_int_tuple(self.element_ids, label="element_ids")
            if not tup:
                raise ValueError("element_ids must be non-empty when provided.")
            object.__setattr__(self, "element_ids", tup)
        if self.selection_set_id is not None:
            object.__setattr__(self, "selection_set_id", int(self.selection_set_id))
        if self.bounding_polygon is not None:
            poly = _coerce_polygon(self.bounding_polygon)
            _validate_bounding_polygon(poly, self.plane)
            object.__setattr__(self, "bounding_polygon", poly)

    @property
    def signed_normal(self) -> np.ndarray:
        """Outward normal of the kept side.

        When ``side == "negative"`` the plane normal is flipped so cut
        kernels can treat "positive" as the kept side without branching.
        """
        n = self.plane.normal_arr
        return n if self.side == "positive" else -n

    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        return _save_pickle(self, path, compress=compress, protocol=protocol)

    @classmethod
    def load_pickle(cls, path: str | Path) -> "SectionCutSpec":
        obj = _load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj


# ---------------------------------------------------------------------- #
# DriftSpec
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class DriftSpec:
    """Node-pair drift definition.

    Parameters
    ----------
    top_node, bottom_node:
        Node IDs to compute drift between.
    component:
        Displacement component. Accepts whatever
        :meth:`NodalResults.drift` accepts (int 1/2/3 or string ``"X"``,
        ``"Y"``, ``"Z"``).
    normalize_by:
        If set, divides drift by this value at apply time — e.g. story
        height to get drift ratio, or building height to get global
        drift ratio.
    label:
        Optional display label; if set, used as the returned Series'
        ``name`` (handy for legends).
    """

    top_node: int
    bottom_node: int
    component: int | str
    normalize_by: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        try:
            top = int(self.top_node)
            bot = int(self.bottom_node)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node IDs must be integers: {exc}") from None
        if top == bot:
            raise ValueError(
                f"top_node and bottom_node must differ, both got {top}."
            )
        object.__setattr__(self, "top_node", top)
        object.__setattr__(self, "bottom_node", bot)
        if self.normalize_by is not None:
            nb = float(self.normalize_by)
            if nb == 0.0:
                raise ValueError("normalize_by must be nonzero when provided.")
            object.__setattr__(self, "normalize_by", nb)

    def apply(self, dataset: "MPCODataSet", *, model_stage: str) -> pd.Series:
        """Compute the drift time-history against a dataset.

        Returns a ``pandas.Series`` indexed by time (or drift ratio if
        ``normalize_by`` is set). The Series ``name`` is ``self.label``
        when provided.
        """
        nr = dataset.nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage=model_stage,
            node_ids=[self.top_node, self.bottom_node],
        )
        s = nr.drift(
            top=self.top_node,
            bottom=self.bottom_node,
            component=self.component,
        )
        if not isinstance(s, pd.Series):
            # nr.drift can return a float when reduce != "series"; we
            # always request the default reduce="series", so this branch
            # only triggers if the upstream API changes shape.
            raise TypeError(
                f"NodalResults.drift returned {type(s).__name__}, expected Series."
            )
        if self.normalize_by is not None:
            s = s / self.normalize_by
        if self.label:
            s.name = self.label
        return s

    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        return _save_pickle(self, path, compress=compress, protocol=protocol)

    @classmethod
    def load_pickle(cls, path: str | Path) -> "DriftSpec":
        obj = _load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickled object is {type(obj).__name__}, expected {cls.__name__}."
            )
        return obj


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _coerce_int_tuple(
    values: Iterable[int] | np.ndarray, *, label: str
) -> tuple[int, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{label} must be 1-D, got shape {values.shape}.")
        return tuple(int(x) for x in values)
    try:
        return tuple(int(x) for x in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain integers: {exc}") from None


def _coerce_polygon(
    values: Iterable[Iterable[float]] | np.ndarray,
) -> tuple[tuple[float, float, float], ...]:
    """Normalise a polygon to a hashable ``tuple[tuple[float, float, float], ...]``.

    Accepts any iterable of length-3 iterables or an ``(M, 3)`` ndarray.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"bounding_polygon must be a sequence of (x, y, z) triples or "
            f"shape (M, 3); got shape {arr.shape}."
        )
    return tuple((float(v[0]), float(v[1]), float(v[2])) for v in arr)


def _validate_bounding_polygon(
    polygon: tuple[tuple[float, float, float], ...],
    plane: Plane,
    *,
    tol: float = 1e-6,
) -> None:
    """Enforce the v1.6 contract on a ``bounding_polygon``.

    - At least 3 vertices.
    - All vertices must lie on ``plane`` (signed distance ≤ ``tol``).
    - Non-degenerate planar area (after projection to the plane basis).
    - Convex.

    Non-convex polygons / off-plane vertices / degenerate polygons all
    raise ``ValueError`` here so the misuse surfaces at construction
    rather than producing silent zero-area cuts downstream.
    """
    if len(polygon) < 3:
        raise ValueError(
            f"bounding_polygon must have at least 3 vertices; got {len(polygon)}."
        )
    arr = np.asarray(polygon, dtype=float)
    d = plane.signed_distance(arr)
    if np.any(np.abs(d) > tol):
        worst = float(np.max(np.abs(d)))
        raise ValueError(
            f"bounding_polygon must lie on the cut plane within tol={tol}; "
            f"worst off-plane distance is {worst}."
        )
    # Lazy import — geometry depends on Plane, and Plane depends on
    # nothing in geometry, so we keep them in different modules and
    # import the helpers only where needed.
    from .geometry import (
        _plane_basis,
        _polygon_signed_area_2d,
        _project_to_plane_basis,
        is_convex_2d,
    )
    e1, e2 = _plane_basis(plane)
    poly_2d = _project_to_plane_basis(arr, plane, basis=(e1, e2))
    area = abs(_polygon_signed_area_2d(poly_2d))
    if area < tol:
        raise ValueError(
            f"bounding_polygon is degenerate (planar area {area} ≈ 0). "
            "All vertices may be collinear or coincident."
        )
    if not is_convex_2d(poly_2d):
        raise ValueError(
            "bounding_polygon must be convex (Cyrus-Beck clipping in "
            "v1.6 assumes convexity). For non-convex regions, decompose "
            "into convex sub-polygons and call section_cut() per part."
        )
