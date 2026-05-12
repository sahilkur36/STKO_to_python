"""Gauss-point → nodal extrapolation with cross-element averaging.

Adapted from apeGmsh ``results/_gauss_extrapolation.py``.

Used by the future ``ContourLayer`` (Phase 3 of the viewer roadmap) and
by any other path that needs to render continuous GP-valued fields as
nodal contours. Today the module is consumable from any caller that can
supply the inputs in pure numpy — there is no viewer-runtime dependency.

Pipeline
--------

For each element with ``n_gp`` Gauss points:

1. Look up the element's shape functions ``N`` (from
   :mod:`STKO_to_python.format.shape_functions`) and the linear-corner
   node IDs.
2. Evaluate ``N`` at the GP natural coordinates → matrix ``A`` of shape
   ``(n_gp, n_corner)``.
3. Compute the Moore–Penrose pseudo-inverse ``M = pinv(A)`` of shape
   ``(n_corner, n_gp)``.

   * ``n_gp == n_corner`` (e.g. Brick + 2×2×2 GPs, ASDShellQ4 + 2×2):
     ``M`` is the exact inverse; constant and linear fields are
     reproduced exactly at the corners.
   * ``n_gp < n_corner`` (e.g. Tet4 / Tri3 with one GP): ``M`` is the
     least-squares fit and reduces to "assign the GP value to every
     corner."
   * ``n_gp > n_corner`` (over-determined integration rule): ``M`` is
     the least-squares projection.

4. Per timestep ``t``: ``corner[t, k] = M_k · gp_values[t, :]``.
5. Accumulate each per-element corner contribution into a global
   per-node sum + count; the final nodal value is the mean over
   neighbouring elements.

Smoothing across element boundaries via nodal averaging is the standard
post-processing approach used by STKO, ParaView, and most academic
viewers. Sharp discontinuities at material interfaces are smeared — a
known trade and the price of a single-mesh nodal contour. The
discrete-contour path (no averaging) is :func:`extrapolate_per_element`.

Higher-order elements
---------------------

For future higher-order element classes (e.g. an 8-node serendipity
shell), the GP values must be projected onto the **linear counterpart's**
corner shape functions — not the full higher-order ``N``. Reasons:

* The viewer substrate is built with linear cells; mid-side and centre
  nodes are dropped at scene-build time.
* Using the full higher-order ``N`` yields a ``pinv`` that produces
  non-constant nodal fields for a truly constant GP input
  (minimum-norm regularization of the under-determined system).

The :data:`LINEAR_COUNTERPART` mapping captures this redirection. It is
empty today because STKO's shape-function catalog has no higher-order
entries; add one entry per future higher-order class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...format.shape_functions import get_shape_functions


# Higher-order STKO class -> linear counterpart class. Used by
# ``build_extrapolation_matrix`` to project GP values onto the linear
# corner shape functions instead of the full higher-order ones. Empty
# today; populate as quadratic elements are added to
# :mod:`STKO_to_python.format.shape_functions`.
#
# Example future entries::
#
#     LINEAR_COUNTERPART = {
#         "204-ASDShellQ8": "203-ASDShellQ4",
#         "...-TwentyEightNodeBrick": "56-Brick",
#     }
LINEAR_COUNTERPART: dict[str, str] = {}


__all__ = [
    "LINEAR_COUNTERPART",
    "PerElementCornerValues",
    "per_element_max_gp_count",
    "build_extrapolation_matrix",
    "extrapolate_per_element",
    "extrapolate_to_nodes_averaged",
]


def per_element_max_gp_count(element_index: np.ndarray) -> int:
    """Largest number of GPs found in any single element.

    Used as a quick discriminator: ``1`` means cell-constant rendering
    is sufficient; anything else needs the extrapolation pipeline.

    Args:
        element_index: ``(n_total_gp,)`` integer array. Each row is the
            element ID that the corresponding GP row belongs to. Multiple
            rows with the same ID indicate multiple GPs in that element.

    Returns:
        ``0`` when the input is empty; otherwise the largest count of
        rows that share a single element ID.
    """
    eidx = np.asarray(element_index, dtype=np.int64)
    if eidx.size == 0:
        return 0
    _, counts = np.unique(eidx, return_counts=True)
    return int(counts.max())


def build_extrapolation_matrix(
    natural_coords: np.ndarray,
    element_class: str,
) -> Optional[np.ndarray]:
    """Build the pinv-based GP→corner projection matrix for one element.

    Args:
        natural_coords: ``(n_gp, parent_dim)`` GP positions in the
            parent domain. A ``(n_gp,)`` 1-D array is treated as
            ``(n_gp, 1)``.
        element_class: STKO element class string (e.g.
            ``"56-Brick"``). Higher-order classes are redirected to
            their linear counterpart via :data:`LINEAR_COUNTERPART`.

    Returns:
        ``M`` of shape ``(n_corner, n_gp)`` such that
        ``corner_values = gp_values @ M.T``, or ``None`` when the class
        is not in :mod:`STKO_to_python.format.shape_functions`. Returning
        ``None`` is intentional — callers should fall back to a
        per-element mean rather than guessing a shape function.
    """
    effective_class = LINEAR_COUNTERPART.get(element_class, element_class)
    catalog = get_shape_functions(effective_class)
    if catalog is None:
        return None
    N_fn, _, _ = catalog
    nat = np.asarray(natural_coords, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    A = N_fn(nat)             # (n_gp, n_corner)
    return np.linalg.pinv(A)  # (n_corner, n_gp)


@dataclass(frozen=True)
class PerElementCornerValues:
    """Per-element extrapolated corner values, no cross-element averaging.

    Each element keeps its own corner values; elements that share a node
    in the mesh will generally hold *different* values for that node.
    Pass this object to :func:`extrapolate_to_nodes_averaged` to collapse
    the duplicates into a smooth mesh-wide nodal field, or consume it
    directly to render discontinuous per-element contours.

    Attributes:
        element_ids: ``(E,)`` int64 — FEM element IDs, ascending order.
        corner_node_ids: Length-``E`` list of ``(n_corner_e,)`` int64
            arrays — corner node IDs in the same order as the element's
            shape-function evaluation. Matches the corner ordering used
            elsewhere in the package (CCW in 2-D, OpenSees convention in
            3-D — see :mod:`STKO_to_python.format.shape_functions`).
        values: Length-``E`` list of ``(T, n_corner_e)`` float64
            arrays — extrapolated values at each corner of each element.
        time_count: Number of timesteps ``T``.
    """

    element_ids: np.ndarray
    corner_node_ids: list
    values: list
    time_count: int


def extrapolate_per_element(
    element_index: np.ndarray,
    natural_coords: np.ndarray,
    gp_values: np.ndarray,
    element_lookup: dict,
) -> PerElementCornerValues:
    """Project GP values onto each element's corner nodes.

    No averaging across shared corners — each element keeps its own
    corner values. Use :func:`extrapolate_to_nodes_averaged` to get a
    mesh-wide averaged nodal field.

    Args:
        element_index: ``(n_total_gp,)`` int — element ID per GP row.
            Rows for the same element do not need to be contiguous;
            they are grouped internally.
        natural_coords: ``(n_total_gp, dim)`` float — GP natural coords,
            same row order as ``element_index``. ``(n_total_gp,)``
            shape is also accepted and treated as ``(n_total_gp, 1)``.
        gp_values: ``(T, n_total_gp)`` or ``(n_total_gp,)`` float — GP
            values to project. A 1-D input is treated as a single
            timestep.
        element_lookup: ``{element_id: (element_class, corner_node_ids)}``
            map. ``element_class`` is a STKO class key
            (e.g. ``"56-Brick"``); ``corner_node_ids`` is a
            ``(n_corner,)`` int array. For future higher-order classes
            the caller is responsible for truncating
            ``corner_node_ids`` to the linear counterpart's corner
            count.

    Returns:
        :class:`PerElementCornerValues`. Elements not in
        ``element_lookup`` (or whose class is not in the shape-function
        catalog) are silently skipped. Single-GP elements broadcast the
        value to every corner without invoking the shape-function path.
    """
    eidx = np.asarray(element_index, dtype=np.int64)
    nat = np.asarray(natural_coords, dtype=np.float64)
    if nat.ndim == 1:
        nat = nat[:, None]
    values = np.asarray(gp_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    T = values.shape[0]

    if eidx.size == 0:
        return PerElementCornerValues(
            element_ids=np.zeros(0, dtype=np.int64),
            corner_node_ids=[],
            values=[],
            time_count=T,
        )

    # Group rows by element id, ascending order. Argsort + diff splits
    # is faster than groupby for the sizes we care about, and gives a
    # stable ordering downstream consumers can rely on.
    order = np.argsort(eidx, kind="stable")
    eidx_sorted = eidx[order]
    splits = np.where(np.diff(eidx_sorted) != 0)[0] + 1
    groups = np.split(order, splits)

    out_eids: list[int] = []
    out_corner_nids: list = []
    out_values: list = []

    for rows in groups:
        if rows.size == 0:
            continue
        eid = int(eidx[rows[0]])
        info = element_lookup.get(eid)
        if info is None:
            continue
        element_class, corner_nids = info
        corner_nids = np.asarray(corner_nids, dtype=np.int64)

        n_gp_e = rows.size
        gp_vals = values[:, rows]        # (T, n_gp_e)
        nat_e = nat[rows]                # (n_gp_e, dim)

        if n_gp_e == 1:
            # Single GP: broadcast to every corner. Avoids invoking the
            # shape-function path (which would yield a degenerate pinv
            # on a 1xN matrix anyway).
            per_corner = np.broadcast_to(
                gp_vals, (T, corner_nids.size),
            ).astype(np.float64, copy=True)
        else:
            M = build_extrapolation_matrix(nat_e, element_class)
            if M is None or M.shape[0] != corner_nids.size:
                # Class not in the catalog, or corner count mismatch.
                # Fall back to per-element mean so the caller still gets
                # *some* sensible value at every corner.
                mean_vals = gp_vals.mean(axis=1, keepdims=True)
                per_corner = np.broadcast_to(
                    mean_vals, (T, corner_nids.size),
                ).astype(np.float64, copy=True)
            else:
                per_corner = gp_vals @ M.T  # (T, n_corner)
                per_corner = np.ascontiguousarray(per_corner, dtype=np.float64)

        out_eids.append(eid)
        out_corner_nids.append(corner_nids)
        out_values.append(per_corner)

    return PerElementCornerValues(
        element_ids=np.asarray(out_eids, dtype=np.int64),
        corner_node_ids=out_corner_nids,
        values=out_values,
        time_count=T,
    )


def extrapolate_to_nodes_averaged(
    per_elem: PerElementCornerValues,
) -> tuple[np.ndarray, np.ndarray]:
    """Average per-element corner values into per-node mesh values.

    Args:
        per_elem: Output of :func:`extrapolate_per_element`.

    Returns:
        Tuple ``(node_ids, nodal_values)``:

        * ``node_ids`` — ``(N,)`` int64, sorted ascending. Each entry is
          a node that received at least one corner contribution.
        * ``nodal_values`` — ``(T, N)`` float64, column-aligned with
          ``node_ids``. The value at column ``j`` is the arithmetic mean
          of every per-element corner value that touched
          ``node_ids[j]``.

    Notes:
        Sharp discontinuities at element boundaries are smeared by the
        averaging — that's the standard post-processing convention. If
        discontinuities must be preserved (different materials meeting
        at a shared corner), consume :class:`PerElementCornerValues`
        directly and render each element's corners separately.
    """
    T = per_elem.time_count
    if per_elem.element_ids.size == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for corner_nids, per_corner in zip(
        per_elem.corner_node_ids, per_elem.values,
    ):
        for c, nid in enumerate(corner_nids):
            nid_i = int(nid)
            col = per_corner[:, c]
            existing = sums.get(nid_i)
            if existing is None:
                sums[nid_i] = col.astype(np.float64).copy()
                counts[nid_i] = 1
            else:
                existing += col
                counts[nid_i] += 1

    if not sums:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros((T, 0), dtype=np.float64),
        )

    node_ids = np.fromiter(sums.keys(), dtype=np.int64, count=len(sums))
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    nodal = np.zeros((T, node_ids.size), dtype=np.float64)
    for j, nid in enumerate(node_ids):
        nodal[:, j] = sums[int(nid)] / counts[int(nid)]
    return node_ids, nodal
