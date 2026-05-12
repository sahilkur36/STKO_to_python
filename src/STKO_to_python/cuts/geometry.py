"""Plane-coplanar 2D geometry helpers for bounding-polygon clipping.

A :class:`~.specs.SectionCutSpec` may carry an optional
``bounding_polygon`` — a convex polygon lying on the cut plane that
restricts the cut to the region inside the polygon. The clipping math
is most cleanly expressed in 2D, so this module:

- builds an orthonormal basis on the plane,
- projects coplanar 3-D points to 2-D plane coordinates,
- clips a segment against a convex polygon (Cyrus-Beck), and
- tests whether a point lies inside a convex polygon.

The helpers are internal — they're called by the kernels through
:func:`prepare_clipper`, which packages the plane basis + 2-D polygon
into a small struct so we don't recompute the projection for every
beam / shell intersection.

Convex-polygon assumption
-------------------------
``bounding_polygon`` is validated as convex at :class:`SectionCutSpec`
construction. The clipper here exploits that: each polygon edge defines
a half-plane and the polygon is the intersection of all of them, so a
segment-vs-polygon clip reduces to two scalar parameter bounds (Cyrus-
Beck). Non-convex polygons would need general line-clipping (e.g.
Sutherland-Hodgman) and are out of scope for v1.6.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    from .plane import Plane


_TOL_DEFAULT = 1e-9


# ---------------------------------------------------------------------- #
# Plane basis + projection
# ---------------------------------------------------------------------- #
def _plane_basis(plane: "Plane") -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal in-plane basis ``(e1, e2)`` for ``plane``.

    ``e1`` is picked deterministically from the global axis least
    parallel to the plane normal (so the basis is repeatable for a
    given plane). ``e2 = normal × e1`` so ``(e1, e2, normal)`` forms a
    right-handed frame.
    """
    n = plane.normal_arr
    # Pick the global axis with smallest |component along n|; that axis
    # is the "most perpendicular" to the normal and yields a numerically
    # stable cross product.
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    # Normalise defensively; n is already unit but cross with a unit
    # vector might pick up roundoff.
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _project_to_plane_basis(
    points_3d: np.ndarray,
    plane: "Plane",
    *,
    basis: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Project ``points_3d`` onto the plane's 2-D basis.

    Parameters
    ----------
    points_3d : ``(N, 3)`` or ``(3,)`` array
        Points assumed to lie on (or very near) ``plane``.
    plane : :class:`Plane`
        The reference plane.
    basis : optional pre-computed ``(e1, e2)``
        Pass the result of :func:`_plane_basis` when projecting many
        batches against the same plane to avoid recomputation.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` or ``(2,)``. The plane's anchor point projects
        to the origin of the 2-D frame.
    """
    pts = np.asarray(points_3d, dtype=float)
    scalar = pts.ndim == 1
    if scalar:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"points must be shape (3,) or (N, 3), got {pts.shape}."
        )
    e1, e2 = basis if basis is not None else _plane_basis(plane)
    rel = pts - plane.point_arr
    u = rel @ e1
    v = rel @ e2
    out = np.stack([u, v], axis=1)
    return out[0] if scalar else out


# ---------------------------------------------------------------------- #
# Polygon utilities
# ---------------------------------------------------------------------- #
def _polygon_signed_area_2d(polygon_2d: np.ndarray) -> float:
    """Signed area of a 2-D polygon via the shoelace formula.

    Positive when vertices are CCW, negative when CW. Used to:

    - detect degenerate polygons (zero area),
    - normalise edge orientation in :func:`_polygon_edge_normals` so
      every "inward" normal points consistently inward.
    """
    x = polygon_2d[:, 0]
    y = polygon_2d[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_edge_normals(polygon_2d: np.ndarray) -> np.ndarray:
    """Return the inward unit normal of each edge of a 2-D convex polygon.

    For a CCW polygon (signed area > 0), the inside lies to the LEFT of
    each edge's walking direction; left-perpendicular ``(-dy, dx)`` of
    edge ``v_i → v_{i+1}`` is the inward normal. For CW (signed area
    < 0) it's the opposite (right-perpendicular). Edge ``i`` connects
    ``polygon_2d[i]`` to ``polygon_2d[(i+1) % n]``; the returned normal
    is at the same index.
    """
    edges = np.roll(polygon_2d, -1, axis=0) - polygon_2d  # (n, 2)
    # Left-perpendicular = (-dy, dx); rotates the walking direction 90°
    # counter-clockwise. For a CCW polygon, that's the inward direction.
    left_perp = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    norms = np.linalg.norm(left_perp, axis=1, keepdims=True)
    # Avoid divide-by-zero for zero-length edges; the polygon validation
    # already rejects those, so this is just defensive.
    norms = np.where(norms < 1e-300, 1.0, norms)
    inward = left_perp / norms
    # If the polygon is CW (negative signed area), the left-perpendicular
    # points outward — flip.
    if _polygon_signed_area_2d(polygon_2d) < 0:
        inward = -inward
    return inward


def _clip_point_inside_2d(
    point_2d: np.ndarray,
    polygon_2d: np.ndarray,
    *,
    tol: float = _TOL_DEFAULT,
) -> bool:
    """Is ``point_2d`` inside the convex polygon (within ``tol``)?

    Uses the half-plane test: the point is inside iff its signed
    distance from every inward-oriented edge is ``>= -tol``. A point
    exactly on an edge counts as inside.
    """
    normals = _polygon_edge_normals(polygon_2d)  # (n, 2)
    rel = point_2d[None, :] - polygon_2d        # (n, 2)
    # Signed distance from each edge: rel · inward_normal.
    d = np.einsum("ij,ij->i", rel, normals)
    return bool(np.all(d >= -tol))


def _clip_segment_against_convex_polygon_2d(
    p0: np.ndarray,
    p1: np.ndarray,
    polygon_2d: np.ndarray,
    *,
    tol: float = _TOL_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Cyrus-Beck clip of segment ``p0 → p1`` against a convex polygon.

    Parameters
    ----------
    p0, p1 : ``(2,)`` arrays
        Segment endpoints in plane-2D coords.
    polygon_2d : ``(M, 2)`` array
        Convex polygon vertices in plane-2D coords.

    Returns
    -------
    (q0, q1, t0, t1) or None
        Clipped endpoints and the parameter interval
        ``q = p0 + t * (p1 - p0)``. ``None`` if the segment misses
        the polygon entirely.

    Algorithm
    ---------
    For each polygon edge with inward normal ``n_e`` and a point
    ``v_e`` on it, define ``f(t) = n_e · (p_t - v_e)`` where
    ``p_t = p0 + t * (p1 - p0)``. ``f(t) >= 0`` means ``p_t`` is on
    the inside half-plane of that edge. The segment is inside the
    polygon iff every edge's ``f(t) >= 0``; intersect those intervals
    in ``t`` to find the clipped sub-segment.
    """
    direction = p1 - p0
    normals = _polygon_edge_normals(polygon_2d)
    t_enter = 0.0
    t_exit = 1.0
    n_edges = polygon_2d.shape[0]
    for i in range(n_edges):
        n_e = normals[i]
        v_e = polygon_2d[i]
        # f(t) = n · (p0 - v) + t * (n · direction)
        c = float(np.dot(n_e, p0 - v_e))
        d = float(np.dot(n_e, direction))
        if abs(d) < tol:
            # Segment is parallel to this edge. If currently outside
            # this half-plane, the segment never enters.
            if c < -tol:
                return None
            continue
        t_cross = -c / d
        if d > 0:
            # f increases with t → entering this half-plane at t_cross.
            if t_cross > t_enter:
                t_enter = t_cross
        else:
            # f decreases with t → leaving this half-plane at t_cross.
            if t_cross < t_exit:
                t_exit = t_cross
        if t_enter > t_exit + tol:
            return None
    if t_exit - t_enter <= tol:
        return None
    q0 = p0 + t_enter * direction
    q1 = p0 + t_exit * direction
    return q0, q1, t_enter, t_exit


# ---------------------------------------------------------------------- #
# Public packaging used by kernels
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolygonClipper:
    """Pre-computed plane basis + 2-D polygon, reused across many cuts.

    Built once per :class:`SectionCutSpec` by :func:`prepare_clipper`
    and consumed by the beam and shell kernels.
    """
    plane: "Plane"
    e1: np.ndarray  # (3,) in-plane axis 1
    e2: np.ndarray  # (3,) in-plane axis 2
    polygon_2d: np.ndarray  # (M, 2)

    def point_inside(self, point_3d: np.ndarray, *, tol: float = _TOL_DEFAULT) -> bool:
        """Whether a coplanar 3-D point lies inside the polygon."""
        p2 = _project_to_plane_basis(point_3d, self.plane, basis=(self.e1, self.e2))
        return _clip_point_inside_2d(p2, self.polygon_2d, tol=tol)

    def clip_segment(
        self,
        p0_3d: np.ndarray,
        p1_3d: np.ndarray,
        *,
        tol: float = _TOL_DEFAULT,
    ) -> tuple[np.ndarray, np.ndarray, float, float] | None:
        """Clip a coplanar 3-D segment against the polygon.

        Returns
        -------
        (q0_3d, q1_3d, t_enter, t_exit) or None
            Clipped endpoints in 3-D (re-embedded into the plane) and
            the parameter interval ``q = p0 + t * (p1 - p0)``. ``None``
            if the segment misses the polygon.
        """
        p0_2d = _project_to_plane_basis(p0_3d, self.plane, basis=(self.e1, self.e2))
        p1_2d = _project_to_plane_basis(p1_3d, self.plane, basis=(self.e1, self.e2))
        clipped = _clip_segment_against_convex_polygon_2d(
            p0_2d, p1_2d, self.polygon_2d, tol=tol
        )
        if clipped is None:
            return None
        _, _, t_enter, t_exit = clipped
        # Re-embed into 3-D from the original 3-D endpoints (don't
        # round-trip through 2-D coords — that would accumulate
        # projection error).
        direction = np.asarray(p1_3d, dtype=float) - np.asarray(p0_3d, dtype=float)
        q0_3d = np.asarray(p0_3d, dtype=float) + t_enter * direction
        q1_3d = np.asarray(p0_3d, dtype=float) + t_exit * direction
        return q0_3d, q1_3d, t_enter, t_exit


def prepare_clipper(plane: "Plane", polygon_3d: Sequence[Sequence[float]]) -> PolygonClipper:
    """Build a :class:`PolygonClipper` for a (plane, polygon) pair.

    The polygon's vertices must already lie on the plane — validated
    on :class:`SectionCutSpec` construction. We compute the plane basis
    once and project the polygon to 2-D once; per-intersection clipping
    then operates in cached 2-D coords.
    """
    e1, e2 = _plane_basis(plane)
    polygon_2d = _project_to_plane_basis(
        np.asarray(polygon_3d, dtype=float), plane, basis=(e1, e2),
    )
    return PolygonClipper(
        plane=plane, e1=e1, e2=e2, polygon_2d=polygon_2d,
    )


def is_convex_2d(polygon_2d: np.ndarray, *, tol: float = 1e-9) -> bool:
    """Check whether a 2-D polygon is convex.

    A polygon is convex iff every consecutive cross product of edge
    vectors has the same sign (all ``>= 0`` for CCW, all ``<= 0`` for
    CW). Treats vertices that are exactly collinear (cross within
    ``tol``) as non-degenerate so a polygon with three collinear
    consecutive vertices passes — useful when STKO geometry exports
    redundant midpoints.
    """
    n = polygon_2d.shape[0]
    if n < 3:
        return False
    sign = 0
    for i in range(n):
        a = polygon_2d[i]
        b = polygon_2d[(i + 1) % n]
        c = polygon_2d[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross > tol:
            if sign < 0:
                return False
            sign = 1
        elif cross < -tol:
            if sign > 0:
                return False
            sign = -1
    return True
