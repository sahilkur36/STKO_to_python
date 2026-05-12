"""Geometry helpers and renderer for beam elements as 3D extruded solids.

:func:`extrude_beam_geometry` is the pure-numpy geometry kernel — it
sweeps a ``BeamProfile`` between two beam endpoints and returns
``(vertices, faces)`` for one element.

:func:`plot_beam_solids` is the dataset-level renderer (wired on the
:class:`~STKO_to_python.plotting.plot.Plot` facade as
``ds.plot.beam_solids``). It walks the cdata's
``beam_profile_assignments``, builds the per-element extrusion using
the geometry kernel, accumulates one combined triangle batch, and
renders it as a :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.
Optional structural edges (section outlines at both ends + per-sweep
longitudinals) overlay as a :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`.

PR scope
--------
Variable cross-section (multiple ``(profile_id, weight)`` entries in
``beam_profile_assignments``) is not yet supported: the renderer picks
the **first** profile assignment per element. The geometry kernel
itself only ever takes one profile per call, so a future per-segment
variant can layer on top without changing the public API here.

Coordinate convention
---------------------
- The profile lives in the section's local ``(y, z)`` plane; column 0
  of ``profile.points`` is local-y, column 1 is local-z.
- The beam's local x-axis runs along the element from ``axis_start`` to
  ``axis_end``. ``R`` is the local→global rotation produced by
  :func:`STKO_to_python.quaternion_to_rotation_matrix` (``v_global =
  R @ v_local``).
- ``section_offset = (yOff, zOff)`` is applied in the section's local
  frame before the rotation; it shifts the geometric centroid relative
  to the integration axis (matches the ``*SECTION_OFFSET`` cdata block).
"""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..model.cdata_reader import BeamProfile

logger = logging.getLogger(__name__)


def extrude_beam_geometry(
    profile: "BeamProfile",
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    R: np.ndarray,
    section_offset: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude a 2D section between two beam endpoints to a triangle mesh.

    Two copies of the profile's points are placed in 3D — one at
    ``axis_start``, one at ``axis_end`` — both lifted from local
    ``(y, z)`` to global via ``R`` and shifted by ``section_offset``
    in the local frame. The returned mesh has three face groups:

    1. **End-1 cap** — ``profile.triangles`` with the winding reversed
       so the cap's outward normal points away from ``axis_end``.
    2. **End-2 cap** — ``profile.triangles`` with vertex indices
       offset by ``n_pts`` so the cap's outward normal points away
       from ``axis_start``.
    3. **Side surface** — two triangles per segment of the
       ``profile.sweeps`` polyline, treated as a closed loop (the
       last sweep point wraps to the first). For a profile with no
       sweeps (or only one), no side surface is emitted.

    Args:
        profile: A :class:`BeamProfile`. Used read-only.
        axis_start: ``(3,)`` global coordinates of the beam's first
            end node.
        axis_end: ``(3,)`` global coordinates of the beam's second
            end node.
        R: ``(3, 3)`` rotation matrix from element-local to global
            (``v_global = R @ v_local``).
        section_offset: Optional ``(2,)`` ``(yOff, zOff)`` offset in
            the section's local frame. Defaults to zero.

    Returns:
        Tuple ``(vertices, faces)``:

        - ``vertices``: ``(2 * n_pts, 3)`` ``float64`` array. Rows
          ``0..n_pts`` are the end-1 cap; rows ``n_pts..2*n_pts``
          are the end-2 cap, aligned row-for-row.
        - ``faces``: ``(n_faces, 3)`` ``int64`` array of vertex
          indices into ``vertices``.
    """
    points = np.asarray(profile.points, dtype=float)
    triangles = np.asarray(profile.triangles, dtype=np.int64)
    sweeps = np.asarray(profile.sweeps, dtype=np.int64)

    n_pts = points.shape[0]
    if section_offset is None:
        yoff, zoff = 0.0, 0.0
    else:
        yoff, zoff = float(section_offset[0]), float(section_offset[1])

    # Lift each (y, z) profile point into local 3D as (0, y, z) so the
    # extrusion runs along the element's local x-axis.
    local = np.zeros((n_pts, 3), dtype=float)
    local[:, 1] = points[:, 0] + yoff
    local[:, 2] = points[:, 1] + zoff

    # Rotate every local row to global in one matmul. For per-row
    # `v_global = R @ v_local`, the equivalent batched form is
    # `local @ R.T`.
    R_arr = np.asarray(R, dtype=float)
    global_rel = local @ R_arr.T

    end1 = global_rel + np.asarray(axis_start, dtype=float)
    end2 = global_rel + np.asarray(axis_end, dtype=float)
    vertices = np.vstack([end1, end2])

    # End-1 cap: reverse winding so the outward normal points along
    # local -x (away from axis_end). End-2 cap keeps the original
    # winding so its outward normal points along local +x.
    if triangles.size:
        cap_end1 = triangles[:, [0, 2, 1]]
        cap_end2 = triangles + n_pts
    else:
        cap_end1 = np.empty((0, 3), dtype=np.int64)
        cap_end2 = np.empty((0, 3), dtype=np.int64)

    # Side surface: each sweep segment becomes a quad split into two
    # triangles. With sweeps wound CCW when viewed from local +x, the
    # outward normal points radially outward.
    if sweeps.size >= 2:
        i0 = sweeps
        i1 = np.roll(sweeps, -1)
        a = i0
        b = i1
        c = i1 + n_pts
        d = i0 + n_pts
        side_tri_1 = np.stack([a, b, c], axis=1)
        side_tri_2 = np.stack([a, c, d], axis=1)
        side = np.vstack([side_tri_1, side_tri_2])
    else:
        side = np.empty((0, 3), dtype=np.int64)

    faces = np.vstack([cap_end1, cap_end2, side]).astype(np.int64, copy=False)
    return vertices, faces


def _structural_edge_segments(
    profile: "BeamProfile",
    end1_vertices: np.ndarray,
    end2_vertices: np.ndarray,
) -> np.ndarray:
    """Collect the "visually structural" edges of one extruded element.

    Returns a ``(n_segments, 2, 3)`` array of line endpoints in global
    coordinates, suitable for feeding to a
    :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`. Three groups
    of edges are emitted:

    1. The section outline at ``axis_start`` (one polyline per entry
       of ``profile.edges``).
    2. The section outline at ``axis_end`` (same polylines, end-2 copy).
    3. The sweep longitudinals — one straight line per ``profile.sweeps``
       point, connecting the end-1 and end-2 copies of that point.

    Interior triangulation edges are intentionally not emitted. The
    overlay is the difference between "this looks like a triangulated
    surface" and "this looks like a beam".
    """
    segs: List[np.ndarray] = []

    # Section outlines: each `edge` is a polyline of point indices.
    for end_vertices in (end1_vertices, end2_vertices):
        for edge_indices in profile.edges:
            if edge_indices.size < 2:
                continue
            pts = end_vertices[edge_indices]
            # Consecutive endpoint pairs: (p0, p1), (p1, p2), ...
            for i in range(pts.shape[0] - 1):
                segs.append(np.stack([pts[i], pts[i + 1]]))

    # Sweep longitudinals: one straight line per sweep point.
    sweeps = np.asarray(profile.sweeps, dtype=np.int64)
    for s in sweeps:
        segs.append(np.stack([end1_vertices[int(s)], end2_vertices[int(s)]]))

    if not segs:
        return np.empty((0, 2, 3), dtype=float)
    return np.asarray(segs, dtype=float)


def plot_beam_solids(
    dataset: "MPCODataSet",
    *,
    element_ids: Union[int, Sequence[int], np.ndarray, None] = None,
    selection_set_id: Union[int, Sequence[int], None] = None,
    selection_set_name: Union[str, Sequence[str], None] = None,
    ax: Any = None,
    face_color: Any = "C0",
    edge_color: Optional[Any] = "0.25",
    linewidth: float = 0.6,
    alpha: float = 0.85,
    title: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Render beam elements as 3D extruded section solids.

    Walks the cdata sidecar's ``beam_profile_assignments`` to find every
    element that has a section profile, builds one
    :func:`extrude_beam_geometry` triangle batch per element, and renders
    the combined mesh as a single
    :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`. When
    ``edge_color`` is set, the **structural** outline of each element
    (section perimeter at both ends + sweep longitudinals; no interior
    triangulation edges) is overlaid via
    :class:`~mpl_toolkits.mplot3d.art3d.Line3DCollection`.

    Args:
        dataset: Source :class:`MPCODataSet`.
        element_ids: Optional explicit element IDs. May combine with the
            two ``selection_set_*`` parameters; their union is intersected
            with the set of elements that actually have profile
            assignments.
        selection_set_id: Optional selection-set ID (or sequence).
        selection_set_name: Optional selection-set name (or sequence).
        ax: Existing 3D matplotlib axes. ``None`` creates a new figure
            with a ``projection="3d"`` subplot. If you supply ``ax``
            yourself, it must be 3D — the renderer does not downscale to
            2D because the extrusion is inherently three-dimensional.
        face_color: Polygon fill color (matplotlib color spec).
        edge_color: Structural-edge color, or ``None`` to disable the
            edge overlay. Defaults to dark gray so the section outlines
            and sweep longitudinals are visible against the fill.
        linewidth: Width of structural edges (no effect when
            ``edge_color`` is ``None``).
        alpha: Polygon alpha. Transparency lets overlapping beams stay
            readable; the default ``0.85`` is opaque enough to read
            shape but still hint at occluded geometry.
        title: Optional axes title.

    Returns:
        ``(ax, meta)``. ``meta`` carries:

        * ``element_count`` — number of beams rendered.
        * ``triangle_count`` — total triangles in the Poly3DCollection.
        * ``skipped_elements`` — list of ids skipped because of missing
          ``*LOCAL_AXES``, missing nodes, or a profile id not present
          in ``cdata.beam_profiles``. Skipped reason logged at INFO.
        * ``profile_ids`` — sorted list of unique profile ids that
          actually contributed to the rendered mesh.
        * ``is_3d`` — always ``True``; kept for parity with the other
          plot helpers' meta blocks.

    Raises:
        ValueError: if the cdata sidecar carries no beam profile
            assignments at all, or if the filter resolves to zero
            elements with assignments.
    """
    assignments = dataset.cdata.beam_profile_assignments
    profiles = dataset.cdata.beam_profiles
    if not assignments:
        raise ValueError(
            "Dataset has no *BEAM_PROFILE_ASSIGNMENT entries in its "
            ".cdata sidecar — nothing to render."
        )
    if not profiles:
        raise ValueError(
            "Dataset has no *BEAM_PROFILE entries in its .cdata sidecar — "
            "assignments reference profile ids that aren't defined."
        )

    assignment_ids = {int(eid) for eid in assignments.keys()}

    user_filtered = (
        element_ids is not None
        or selection_set_id is not None
        or selection_set_name is not None
    )
    if user_filtered:
        resolved = dataset._selection_resolver.resolve_elements(
            names=selection_set_name,
            ids=selection_set_id,
            explicit_ids=element_ids,
        )
        target = sorted(assignment_ids & {int(e) for e in resolved})
    else:
        target = sorted(assignment_ids)

    if not target:
        raise ValueError(
            "No beam elements with profile assignments matched the filter."
        )

    # Pull every element's end-node coords from the cached dataframe in
    # one pass; build an eid → (n0_xyz, n1_xyz) lookup so we don't pay
    # per-element dataframe filtering inside the geometry loop.
    df_elements = dataset.elements_info["dataframe"]
    df_nodes = dataset.nodes_info["dataframe"]
    node_coords = {
        int(row.node_id): np.array([row.x, row.y, row.z], dtype=float)
        for row in df_nodes.itertuples(index=False)
    }
    elements_by_id = df_elements.set_index("element_id", drop=False)

    # Vectorized rotation lookup. Elements without *LOCAL_AXES are
    # filtered out before this call so rotation_matrices doesn't raise.
    local_axes = dataset.cdata.local_axes
    section_offsets = dataset.cdata.section_offsets

    skipped: List[int] = []
    eligible_ids: List[int] = []
    for eid in target:
        if eid not in local_axes:
            skipped.append(eid)
            logger.info(
                "plot_beam_solids: skipping element %d — no *LOCAL_AXES entry",
                eid,
            )
            continue
        eligible_ids.append(eid)

    if not eligible_ids:
        raise ValueError(
            "All matched elements lacked *LOCAL_AXES entries; nothing to draw."
        )

    ids_arr, R_batch = dataset.cdata.rotation_matrices(eligible_ids)
    # ids_arr should align with eligible_ids after the rotation_matrices
    # call sorts. Re-key R by element id so the geometry loop is clear.
    R_by_id = {int(eid): R for eid, R in zip(ids_arr.tolist(), R_batch)}

    # Per-element accumulators. Concatenate vertices/faces with running
    # offsets so the final Poly3DCollection sees one big batch.
    all_vertices: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    all_edge_segs: List[np.ndarray] = []
    profile_ids_used: set = set()
    vertex_offset = 0

    for eid in eligible_ids:
        # Resolve profile (first assignment for v1 — variable cross-section
        # is a future extension).
        pid, _weight = assignments[eid][0]
        profile = profiles.get(int(pid))
        if profile is None:
            skipped.append(eid)
            logger.info(
                "plot_beam_solids: skipping element %d — profile id %d not in beam_profiles",
                eid,
                pid,
            )
            continue

        try:
            elem_row = elements_by_id.loc[eid]
        except KeyError:
            skipped.append(eid)
            logger.info(
                "plot_beam_solids: skipping element %d — not in elements_info",
                eid,
            )
            continue

        node_list = elem_row["node_list"]
        if len(node_list) != 2:
            # Beam profile assignments shouldn't exist for non-line
            # elements, but guard anyway.
            skipped.append(eid)
            logger.info(
                "plot_beam_solids: skipping element %d — expected 2-node line element, got %d nodes",
                eid,
                len(node_list),
            )
            continue
        try:
            n0 = node_coords[int(node_list[0])]
            n1 = node_coords[int(node_list[1])]
        except KeyError:
            skipped.append(eid)
            logger.info(
                "plot_beam_solids: skipping element %d — node not in nodes_info",
                eid,
            )
            continue

        R = R_by_id[eid]
        offset_xy = section_offsets.get(eid)  # may be None
        vertices, faces = extrude_beam_geometry(
            profile,
            axis_start=n0,
            axis_end=n1,
            R=R,
            section_offset=offset_xy,
        )
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        if edge_color is not None:
            n_pts = profile.points.shape[0]
            end1 = vertices[:n_pts]
            end2 = vertices[n_pts:]
            all_edge_segs.append(_structural_edge_segments(profile, end1, end2))
        vertex_offset += vertices.shape[0]
        profile_ids_used.add(int(pid))

    if not all_vertices:
        raise ValueError(
            "Every matched element was skipped; check the logger output "
            "for individual reasons (no profile, no local axes, etc.)."
        )

    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    polygons = combined_vertices[combined_faces]  # (n_faces, 3, 3)

    # Axes creation. The output is inherently 3D — caller's `ax` must be
    # a 3D axes if supplied, otherwise we make one.
    if ax is None:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    poly = Poly3DCollection(
        polygons,
        facecolors=face_color,
        edgecolors="none",
        linewidths=0.0,
        alpha=alpha,
    )
    ax.add_collection3d(poly)

    if edge_color is not None and all_edge_segs:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        edge_array = np.vstack(all_edge_segs)
        line_coll = Line3DCollection(
            edge_array,
            colors=edge_color,
            linewidths=linewidth,
            alpha=1.0,
        )
        ax.add_collection3d(line_coll)

    # Framing from the combined vertex cloud.
    from .deformed_shape import _autoscale_axes

    _autoscale_axes(ax, combined_vertices, is_3d=True)

    # Equal-aspect 3D box. matplotlib's default 3D axes use a unit cube
    # regardless of data extents, which makes beams look squashed when
    # one dimension dominates. ``set_box_aspect`` accepts the post-pad
    # span tuple so the visual proportions match the data.
    spans = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ],
        dtype=float,
    )
    span_lengths = spans[:, 1] - spans[:, 0]
    if np.all(span_lengths > 0):
        ax.set_box_aspect(tuple(span_lengths))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)

    if skipped:
        warnings.warn(
            f"[plot_beam_solids] Skipped {len(skipped)} elements "
            f"(see INFO log for details).",
            RuntimeWarning,
        )

    meta: Dict[str, Any] = {
        "element_count": len(all_vertices),
        "triangle_count": int(combined_faces.shape[0]),
        "skipped_elements": skipped,
        "profile_ids": sorted(profile_ids_used),
        "is_3d": True,
    }
    return ax, meta


__all__ = ["extrude_beam_geometry", "plot_beam_solids"]
