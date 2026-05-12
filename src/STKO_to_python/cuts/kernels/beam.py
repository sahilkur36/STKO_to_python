"""Beam section-cut kernel — geometry phase.

This module locates where a cut plane crosses each beam element in a
spec's filter. The output is purely geometric: element id, natural
coordinate on the beam axis, intersection point in global coordinates,
end-node ids and coordinates. The result-reading + resultant
aggregation step lives in a sibling module (added in the next phase).

Why a separate phase? The geometry is testable end-to-end against any
fixture that has beams in its model — even ones without
``section.force`` recorded (e.g. ``5-ElasticBeam3d`` in the
elasticFrame example). Decoupling lets us land the geometry with
deterministic unit coverage before plugging in the heavier
result-reading machinery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..specs import SectionCutSpec
    from ...core.dataset import MPCODataSet


# Class names this kernel knows how to handle. The element index stores
# decorated types like ``"5-ElasticBeam3d"``; we strip the ``<tag>-``
# prefix before matching against this set.
BEAM_ELEMENT_CLASSES: tuple[str, ...] = (
    "ElasticBeam3d",
    "DispBeamColumn3d",
    "ForceBeamColumn3d",
    "MixedBeamColumn3d",
)


_TAG_PREFIX_RE = re.compile(r"^\d+-")


@dataclass(frozen=True)
class BeamIntersection:
    """A single beam crossing the cut plane.

    All coordinates are global. ``xi`` is the natural coordinate on the
    beam axis (``[-1, +1]``); ``t`` is the segment parameter from
    ``node1 -> node2`` (``[0, 1]``). They satisfy ``xi == 2*t - 1``.

    Attributes
    ----------
    element_id : int
    element_type : str
        Decorated type string (e.g. ``"5-ElasticBeam3d"``).
    xi : float
        Natural coordinate on the beam axis.
    t : float
        Segment parameter from node 1 to node 2.
    point_global : tuple[float, float, float]
        Intersection point in global coordinates.
    end_node_ids : tuple[int, int]
    end_coords : tuple[tuple[float, float, float], tuple[float, float, float]]
        End-node global coordinates ``((x1, y1, z1), (x2, y2, z2))``.
    """

    element_id: int
    element_type: str
    xi: float
    t: float
    point_global: tuple[float, float, float]
    end_node_ids: tuple[int, int]
    end_coords: tuple[tuple[float, float, float], tuple[float, float, float]]

    @property
    def point_arr(self) -> np.ndarray:
        return np.asarray(self.point_global, dtype=float)

    @property
    def end_coords_arr(self) -> np.ndarray:
        return np.asarray(self.end_coords, dtype=float)

    @property
    def axis_length(self) -> float:
        a, b = self.end_coords_arr
        return float(np.linalg.norm(b - a))


def _strip_class_tag(decorated: str) -> str:
    """``'5-ElasticBeam3d' -> 'ElasticBeam3d'``; idempotent on plain names."""
    return _TAG_PREFIX_RE.sub("", str(decorated))


def find_beam_intersections(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
) -> list[BeamIntersection]:
    """Find intersections between ``spec.plane`` and every beam in the filter.

    The spec's filter (``selection_set_name``, ``selection_set_id``,
    ``element_ids`` — any combination) is resolved via the dataset's
    :class:`SelectionSetResolver`. Only beam-type elements
    (:data:`BEAM_ELEMENT_CLASSES`) are inspected — other element types
    are silently ignored.

    A beam is skipped without warning when:
    - It does not cross the plane (segment fully on one side).
    - It lies parallel within numerical tolerance (e.g. a horizontal
      beam exactly on a horizontal cut).
    - Its connectivity has anything other than two nodes.

    Returns
    -------
    list[BeamIntersection]
        One entry per crossing beam. Empty list if no beam crosses.
    """
    # 1. Resolve the spec's filter into a candidate id set.
    candidate_ids = dataset._selection_resolver.resolve_elements(
        names=spec.selection_set_name,
        ids=spec.selection_set_id,
        explicit_ids=spec.element_ids,
    )
    candidate_set = {int(x) for x in candidate_ids}

    # 2. Pull the element index + a node-id -> coord map once.
    df_elems = dataset.elements_info["dataframe"]
    df_nodes = dataset.nodes_info["dataframe"]
    node_coord: dict[int, tuple[float, float, float]] = {
        int(r.node_id): (float(r.x), float(r.y), float(r.z))
        for r in df_nodes.itertuples(index=False)
    }

    # 3. Restrict to beam-type elements in the candidate set.
    beam_set = set(BEAM_ELEMENT_CLASSES)
    is_beam = df_elems["element_type"].map(
        lambda s: _strip_class_tag(s) in beam_set
    )
    in_filter = df_elems["element_id"].isin(candidate_set)
    beam_rows = df_elems[is_beam & in_filter]

    # 4. Walk the beams; intersect each with the plane.
    plane = spec.plane
    out: list[BeamIntersection] = []
    for row in beam_rows.itertuples(index=False):
        node_list = row.node_list
        if len(node_list) != 2:
            continue
        n1, n2 = int(node_list[0]), int(node_list[1])
        c1 = node_coord.get(n1)
        c2 = node_coord.get(n2)
        if c1 is None or c2 is None:
            continue
        hit = plane.intersect_segment(c1, c2)
        if hit is None:
            continue
        point, t = hit
        xi = 2.0 * t - 1.0
        out.append(
            BeamIntersection(
                element_id=int(row.element_id),
                element_type=str(row.element_type),
                xi=float(xi),
                t=float(t),
                point_global=(float(point[0]), float(point[1]), float(point[2])),
                end_node_ids=(n1, n2),
                end_coords=(c1, c2),
            )
        )
    out.sort(key=lambda b: b.element_id)
    return out
