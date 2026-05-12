"""Geometry helpers for rendering beam elements as 3D extruded solids.

Builds a triangle mesh from a 2D cross-section (``BeamProfile``) swept
along a 1D axis between two endpoints. The output is a pair of plain
numpy arrays — vertices in global coordinates and triangular faces as
integer indices — that downstream callers can feed to
``mpl_toolkits.mplot3d.art3d.Poly3DCollection`` or any other triangle
renderer.

PR scope
--------
This module is intentionally matplotlib-free. The dataset-level plot
wrapper (``ds.plot.beam_solids``) that consumes :func:`extrude_beam_geometry`
lands in a follow-up PR. The function takes a single profile per call,
so variable cross-section (multiple ``(profile_id, weight)`` entries in
``beam_profile_assignments``) is handled by the caller — typically by
picking the first entry until the v2 follow-up.

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

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..model.cdata_reader import BeamProfile


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


__all__ = ["extrude_beam_geometry"]
