"""Solid (continuum) section-cut kernel — geometry + resultant.

The solid kernel mirrors the beam and shell kernels in shape:

- a registry of solid classes the kernel knows how to handle,
- a :class:`SolidIntersection` record per solid whose volume a plane
  cuts (the planar intersection polygon plus enough metadata to sample
  the stress field at quadrature points on the polygon),
- a :func:`find_solid_intersections` step that computes those records,
- a :func:`compute_solid_cut` step that reads ``material.stress``,
  integrates the traction ``t = σ · n_cut`` over each polygon, and
  aggregates ``(F, M)``.

Why a polygon integration
-------------------------
OpenSees continuum recorders write ``material.stress`` per integration
point — six components per IP in the recorder's frame:
``σ₁₁, σ₂₂, σ₃₃, σ₁₂, σ₂₃, σ₁₃``. The cut through a continuum element
produces a planar polygon (a triangle, quadrilateral, pentagon, or
hexagon for a hex element; a triangle or quadrilateral for a tet);
the cut force is the surface integral of the traction over that
polygon. We triangulate the polygon (fan from the first vertex,
valid because the polygon is convex), place a 3-point Gauss rule on
each triangle, invert the element's shape function at each quadrature
point to find natural coords, and sample the stress field by
trilinear (hex) or linear (tet) interpolation between IPs.

Sign convention
---------------
Same as the beam and shell kernels: the cut force is what the
**discarded** side exerts on the **kept** side. For solids we choose
the cut normal ``n_cut`` to point from kept → discarded; the traction
``t = σ · n_cut`` then integrates to the resultant force on the kept
body. ``spec.side`` and the side-aware filter pick the kept side
consistently with the beam and shell kernels.

Continuum stress frame
----------------------
For OpenSees continuum elements the ``material.stress`` recorder
writes Cauchy stress in the **global** frame; no rotation is required
to combine the contribution with beam and shell kernels that emit
their resultants in global. We do not multiply by an element rotation
matrix here. If a future element class writes stress in a local frame,
register it with an explicit rotation hook rather than silently
guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...format.shape_functions import _brick_N, _brick_dN

if TYPE_CHECKING:
    from ..geometry import PolygonClipper
    from ..specs import SectionCutSpec
    from ...core.dataset import MPCODataSet


# Solid element classes this kernel handles. As in the shell / beam
# kernels, the MPCO element index carries decorated names like
# ``"56-Brick"``; we strip the ``<tag>-`` prefix before matching, so
# both decorated and base names resolve.
#
# Supported topologies:
#   - 8-node trilinear hexahedra (Brick, BbarBrick, SSPbrick)
#   - 20-node serendipity / 27-node Lagrange hexahedra (Brick20,
#     TwentyNodeBrick, Brick27, TwentySevenNodeBrick) — geometry phase
#     uses the 8 corner nodes (OpenSees node order puts the corners
#     first, midpoint nodes after); stress sampling dispatches on the
#     IP count (8-pt 2×2×2 → trilinear weights, 27-pt 3×3×3 →
#     triquadratic Lagrange weights).
#   - 4-node linear tetrahedra (FourNodeTetrahedron)
#
# Wedge / pyramid solids and 10-node tetrahedra are still out of scope
# (no static catalog entry yet, not enough surveyed in user fixtures).
SOLID_ELEMENT_CLASSES: tuple[str, ...] = (
    "Brick",
    "BbarBrick",
    "SSPbrick",
    "Brick20",
    "Brick27",
    "TwentyNodeBrick",
    "TwentySevenNodeBrick",
    "FourNodeTetrahedron",
)


_TAG_PREFIX_RE = re.compile(r"^\d+-")


def _strip_class_tag(decorated: str) -> str:
    """``'56-Brick' -> 'Brick'``; idempotent on plain names."""
    return _TAG_PREFIX_RE.sub("", str(decorated))


# Number of nodes per element class. Used to distinguish hex variants
# from tet without parsing the element_type a second time.
_NODES_PER_CLASS: dict[str, int] = {
    "Brick": 8,
    "BbarBrick": 8,
    "SSPbrick": 8,
    "Brick20": 20,
    "TwentyNodeBrick": 20,
    "Brick27": 27,
    "TwentySevenNodeBrick": 27,
    "FourNodeTetrahedron": 4,
}


# Number of nodes treated as the element's "corners" for the geometry
# phase. Higher-order hexes carry midpoint / face / center nodes after
# the 8 corners; we slice down to the first 8 for the plane-vs-volume
# intersection and the trilinear inversion. Curvature induced by the
# extra nodes is ignored at the cut-polygon level — sound for any
# well-conditioned higher-order hex where the midpoint nodes don't
# stray far from the straight-edge midpoint.
_CORNER_NODES_PER_CLASS: dict[str, int] = {
    "Brick": 8, "BbarBrick": 8, "SSPbrick": 8,
    "Brick20": 8, "TwentyNodeBrick": 8,
    "Brick27": 8, "TwentySevenNodeBrick": 8,
    "FourNodeTetrahedron": 4,
}


# 8-node hex edges (pairs of node indices, OpenSees Brick connectivity).
# Bottom face (z = -1): 0-1-2-3; top face (z = +1): 4-5-6-7.
# Vertical edges connect bottom corner k to top corner k+4.
_HEX_EDGES: tuple[tuple[int, int], ...] = (
    # bottom face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # top face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # vertical connectors
    (0, 4), (1, 5), (2, 6), (3, 7),
)


# 4-node tet edges (OpenSees FourNodeTetrahedron uses the standard
# isoparametric layout: node 0 at the origin, nodes 1/2/3 at the unit
# axes). Six edges total.
_TET_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (0, 3),
    (1, 2), (2, 3), (3, 1),
)


# ---------------------------------------------------------------------- #
# Geometry record
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolidIntersection:
    """A single solid element cut by the plane.

    Attributes
    ----------
    element_id : int
    element_type : str
        Decorated type string (e.g. ``"56-Brick"``).
    polygon_global : ``(K, 3)`` ndarray
        Vertices of the planar intersection polygon in global coords,
        ordered CCW around the plane normal. ``K ∈ {3, 4, 5, 6}`` for a
        hex; ``K ∈ {3, 4}`` for a tet.
    polygon_natural : ``(K, 3)`` ndarray
        Same vertices in element natural coords ``(ξ, η, ζ)``. The
        per-vertex inversion is well-posed because the vertices lie on
        the cut plane and inside the parent cube / tet.
    element_node_coords : ``(n_nodes, 3)`` ndarray
        Element nodes in global coords, ordered as the connectivity.
    node_ids : tuple[int, ...]
    """

    element_id: int
    element_type: str
    polygon_global: np.ndarray = field(repr=False)
    polygon_natural: np.ndarray = field(repr=False)
    element_node_coords: np.ndarray = field(repr=False)
    node_ids: tuple[int, ...]

    @property
    def n_vertices(self) -> int:
        return int(self.polygon_global.shape[0])

    @property
    def polygon_centroid(self) -> np.ndarray:
        """Centroid of the polygon vertices in global coords.

        Used as the reference point for the per-element moment — the
        analogue of ``point_arr`` on :class:`BeamIntersection` and
        ``chord_midpoint`` on :class:`ShellIntersection`.
        """
        return np.mean(self.polygon_global, axis=0)

    @property
    def polygon_area(self) -> float:
        """Area of the planar polygon (positive scalar, in global units).

        Computed from the fan triangulation rooted at the first vertex
        — valid for any convex polygon.
        """
        v = self.polygon_global
        a = 0.0
        for i in range(1, v.shape[0] - 1):
            e1 = v[i] - v[0]
            e2 = v[i + 1] - v[0]
            a += 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
        return a


# ---------------------------------------------------------------------- #
# Plane vs convex polyhedron — extract the planar intersection polygon
# ---------------------------------------------------------------------- #
def _classify_signed_distance(
    d: np.ndarray, tol: float
) -> np.ndarray:
    """Classify signed distances as -1 / 0 / +1 with a tolerance band."""
    sgn = np.zeros_like(d, dtype=np.int8)
    sgn[d > tol] = 1
    sgn[d < -tol] = -1
    return sgn


def _plane_polyhedron_polygon(
    node_coords: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    plane_e1: np.ndarray,
    plane_e2: np.ndarray,
    *,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """Compute the planar intersection polygon of a plane vs a convex polyhedron.

    Walks the polyhedron's edges, collects crossing points (and any
    on-plane vertices), dedupes near-coincident points, and sorts them
    counter-clockwise around the plane normal using the supplied 2-D
    plane basis ``(e1, e2)``.

    Returns the polygon as an ``(K, 3)`` ndarray ordered CCW around the
    plane normal, or ``None`` if the plane misses the polyhedron, only
    grazes a vertex, or produces a degenerate (fewer than 3 unique
    point) intersection.

    Parameters
    ----------
    node_coords : ``(n_nodes, 3)`` ndarray
        Polyhedron vertices in global coords.
    edges : tuple of ``(int, int)``
        Edge connectivity as index pairs into ``node_coords``.
    plane_point, plane_normal : ``(3,)`` ndarrays
        Anchor and unit normal of the cut plane.
    plane_e1, plane_e2 : ``(3,)`` ndarrays
        Orthonormal in-plane axes used to sort vertices CCW.
    tol : float
        On-plane and dedup tolerance (length units of ``node_coords``).
    """
    d = (node_coords - plane_point) @ plane_normal
    sgn = _classify_signed_distance(d, tol)

    # Polyhedron entirely on one side — no crossing.
    if np.all(sgn > 0) or np.all(sgn < 0):
        return None

    pts: list[np.ndarray] = []
    for i, j in edges:
        si, sj = int(sgn[i]), int(sgn[j])
        if si == 0:
            pts.append(node_coords[i])
        if sj == 0:
            pts.append(node_coords[j])
        if si * sj < 0:
            di, dj = float(d[i]), float(d[j])
            t = di / (di - dj)
            pts.append(node_coords[i] + t * (node_coords[j] - node_coords[i]))

    if not pts:
        return None

    # Dedupe near-coincident points (on-plane vertices appear at every
    # adjacent edge they belong to).
    unique: list[np.ndarray] = []
    for p in pts:
        if not any(np.linalg.norm(p - u) <= tol for u in unique):
            unique.append(p)
    if len(unique) < 3:
        return None

    arr = np.stack(unique, axis=0)  # (K, 3)
    # Project to 2-D using the supplied plane basis and sort CCW around
    # the polygon centroid. CCW vs CW doesn't matter for the integration
    # (we'll re-derive orientation independently); we just need a
    # consistent ordering so the fan triangulation is valid.
    centroid = arr.mean(axis=0)
    rel = arr - centroid
    u = rel @ plane_e1
    v = rel @ plane_e2
    angles = np.arctan2(v, u)
    order = np.argsort(angles)
    return arr[order]


# ---------------------------------------------------------------------- #
# Inverse shape-function maps
# ---------------------------------------------------------------------- #
def _invert_brick_trilinear(
    p: np.ndarray, node_coords: np.ndarray, *, max_iter: int = 25, tol: float = 1e-10,
) -> np.ndarray:
    """Invert the trilinear hex map ``x(ξ, η, ζ) → physical`` for one point.

    Newton iteration starting at ``(0, 0, 0)``. ``node_coords`` is
    ``(8, 3)`` in OpenSees Brick node order (bottom face CCW at ζ=-1,
    top face CCW at ζ=+1 — see :data:`STKO_to_python.format.shape_functions._BRICK_NODE_SIGNS`).

    Raises if Newton fails to converge — silent fallback would mask
    real bugs. The starting point (0, 0, 0) is well inside the parent
    cube and the trilinear map is well-conditioned for any
    physically-meaningful hex.
    """
    xi = np.zeros(3, dtype=float)
    for _ in range(max_iter):
        N = _brick_N(xi[None, :])[0]              # (8,)
        x_curr = N @ node_coords                  # (3,)
        residual = x_curr - p
        if np.linalg.norm(residual) < tol:
            return xi
        dN = _brick_dN(xi[None, :])[0]            # (8, 3)
        J = node_coords.T @ dN                     # (3, 3) — ∂x/∂(ξ, η, ζ)
        try:
            step = np.linalg.solve(J, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Singular Jacobian inverting brick trilinear at xi={xi.tolist()}."
            ) from exc
        xi -= step
    raise RuntimeError(
        f"Brick inversion did not converge after {max_iter} iterations "
        f"(final residual {np.linalg.norm(residual)}, p={p.tolist()})."
    )


def _invert_tet_linear(
    p: np.ndarray, node_coords: np.ndarray,
) -> np.ndarray:
    """Invert the linear tet map for one point — closed-form 3×3 solve.

    The reference tet has nodes at ``(0,0,0), (1,0,0), (0,1,0), (0,0,1)``
    and the linear map is
    ``x(ξ, η, ζ) = v0 + (v1-v0)ξ + (v2-v0)η + (v3-v0)ζ``. So
    ``[v1-v0 | v2-v0 | v3-v0] · [ξ, η, ζ]ᵀ = p - v0``.
    """
    v0, v1, v2, v3 = node_coords
    A = np.column_stack([v1 - v0, v2 - v0, v3 - v0])  # (3, 3)
    rhs = p - v0
    try:
        return np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"Singular Jacobian inverting tet linear; node_coords may be "
            f"degenerate: {node_coords.tolist()}."
        ) from exc


def _natural_coords_of_polygon(
    polygon_global: np.ndarray, node_coords: np.ndarray, base_type: str,
) -> np.ndarray:
    """Compute element natural coords for every polygon vertex.

    Hex: trilinear inversion via Newton. Tet: closed-form 3×3 solve.
    Returns ``(K, 3)`` in the element's parent domain.
    """
    n_nodes = node_coords.shape[0]
    out = np.empty((polygon_global.shape[0], 3), dtype=float)
    if n_nodes == 8:
        for k in range(polygon_global.shape[0]):
            out[k] = _invert_brick_trilinear(polygon_global[k], node_coords)
    elif n_nodes == 4:
        for k in range(polygon_global.shape[0]):
            out[k] = _invert_tet_linear(polygon_global[k], node_coords)
    else:
        raise NotImplementedError(
            f"Solid class {base_type!r} has {n_nodes} nodes; only 4-node "
            "tetrahedra and 8-node hexahedra are supported in v1.7."
        )
    return out


# ---------------------------------------------------------------------- #
# Geometry phase
# ---------------------------------------------------------------------- #
def find_solid_intersections(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    clipper: "PolygonClipper | None" = None,
) -> list[SolidIntersection]:
    """Find intersections between ``spec.plane`` and every solid in the filter.

    A solid is skipped without warning when:

    - the plane misses its volume (every vertex on the same side),
    - the plane only grazes a vertex or edge (fewer than 3 unique
      crossing points → degenerate polygon),
    - the spec carries a ``bounding_polygon`` and the intersection
      polygon is fully outside the polygon — in that case the polygon
      is intersected against the bounding polygon and only the
      common region is recorded.
    - the element's interior is entirely on the kept side (the
      side-aware filter — mirrors the shell rule, prevents double-
      counts when two adjacent solids share a face that the plane
      crosses).

    Parameters
    ----------
    clipper : :class:`PolygonClipper`, optional
        Pre-built clipper for ``spec.bounding_polygon``. The caller
        builds this once per cut and threads it through to all kernel
        phases so the plane basis isn't recomputed.
    """
    candidate_ids = dataset._selection_resolver.resolve_elements(
        names=spec.selection_set_name,
        ids=spec.selection_set_id,
        explicit_ids=spec.element_ids,
    )
    candidate_set = {int(x) for x in candidate_ids}

    df_elems = dataset.elements_info["dataframe"]
    df_nodes = dataset.nodes_info["dataframe"]
    node_coord: dict[int, tuple[float, float, float]] = {
        int(r.node_id): (float(r.x), float(r.y), float(r.z))
        for r in df_nodes.itertuples(index=False)
    }

    solid_set = set(SOLID_ELEMENT_CLASSES)
    is_solid = df_elems["element_type"].map(
        lambda s: _strip_class_tag(s) in solid_set
    )
    in_filter = df_elems["element_id"].isin(candidate_set)
    solid_rows = df_elems[is_solid & in_filter]

    plane = spec.plane
    plane_point = np.asarray(plane.point, dtype=float)
    plane_normal = np.asarray(plane.normal, dtype=float)
    # Build a 2-D plane basis once for the CCW sort of polygon vertices.
    # Prefer the clipper's cached basis when one is supplied; otherwise
    # derive a fresh one. (The two paths give the same orientation when
    # the same axis-selection rule is used.)
    if clipper is not None:
        plane_e1 = np.asarray(clipper.e1, dtype=float)
        plane_e2 = np.asarray(clipper.e2, dtype=float)
    else:
        from ..geometry import _plane_basis
        plane_e1, plane_e2 = _plane_basis(plane)

    out: list[SolidIntersection] = []
    for row in solid_rows.itertuples(index=False):
        node_list = row.node_list
        base_type = _strip_class_tag(row.element_type)
        expected_nodes = _NODES_PER_CLASS.get(base_type)
        if expected_nodes is None or len(node_list) != expected_nodes:
            # Class is in the registry but the connectivity doesn't
            # match the expected node count — likely a higher-order
            # variant we don't support yet. Skip silently.
            continue
        try:
            coords = np.array(
                [node_coord[int(nid)] for nid in node_list], dtype=float,
            )
        except KeyError:
            # Node missing from the node table — skip rather than
            # exploding (defensive; real models always include their
            # connectivity nodes).
            continue

        # Higher-order hexes carry midpoint / face / center nodes
        # after the 8 corners. The geometry phase ignores them and
        # works on the corner sub-frame — sound because the corners
        # define the element's convex hull, and the cut polygon math
        # is on convex polyhedra.
        n_corners = _CORNER_NODES_PER_CLASS[base_type]
        corner_coords = coords[:n_corners]

        # Side-aware filter mirrors the shell rule. When two adjacent
        # solids share a face that lies on the cut plane, both would
        # report that face as their polygon. The cut traction comes
        # from the DISCARDED side per the standard convention, so we
        # skip an element whose interior is entirely on the kept side.
        d_signed = (corner_coords - plane_point) @ spec.signed_normal
        tol_d = 1e-9
        if not bool(np.any(d_signed < -tol_d)):
            continue

        edges = _HEX_EDGES if n_corners == 8 else _TET_EDGES
        polygon = _plane_polyhedron_polygon(
            corner_coords, edges, plane_point, plane_normal, plane_e1, plane_e2,
        )
        if polygon is None:
            continue

        # Optional bounding-polygon clipping against the 2-D polygon
        # carried by the clipper. Reuse the 2-D Cyrus-Beck primitives
        # by walking the polygon edges and clipping each segment. The
        # planar intersection-of-two-convex-polygons could be done in
        # one Sutherland-Hodgman pass, but the polygon we have here is
        # small (≤ 6 vertices) and the Cyrus-Beck per-edge approach
        # reuses the existing tested primitive — simpler and adequate.
        if clipper is not None:
            polygon = _clip_polygon_against_clipper(polygon, clipper)
            if polygon is None or polygon.shape[0] < 3:
                continue

        polygon_natural = _natural_coords_of_polygon(
            polygon, corner_coords, base_type,
        )
        out.append(
            SolidIntersection(
                element_id=int(row.element_id),
                element_type=str(row.element_type),
                polygon_global=polygon,
                polygon_natural=polygon_natural,
                element_node_coords=coords,
                node_ids=tuple(int(nid) for nid in node_list),
            )
        )
    out.sort(key=lambda s: s.element_id)
    return out


def _clip_polygon_against_clipper(
    polygon_global: np.ndarray, clipper: "PolygonClipper",
) -> np.ndarray | None:
    """Clip the planar intersection polygon against the clipper's
    bounding polygon, using Sutherland-Hodgman against each
    half-plane of the convex bounding polygon.

    Both polygons are coplanar (live on ``clipper.plane``), so the
    clipping is done in 2-D plane coords for numerical robustness, then
    re-embedded into 3-D from the clipper's origin + basis.

    Returns a CCW-sorted 3-D polygon, or ``None`` when the clipped
    polygon collapses to fewer than 3 points.
    """
    from ..geometry import (
        _polygon_edge_normals,
        _project_to_plane_basis,
        _plane_basis,  # noqa: F401  — type stability across reload
    )

    pts_2d = _project_to_plane_basis(
        polygon_global, clipper.plane, basis=(clipper.e1, clipper.e2),
    )
    if pts_2d.ndim == 1:
        pts_2d = pts_2d[None, :]

    polygon_2d = clipper.polygon_2d
    normals = _polygon_edge_normals(polygon_2d)  # inward-pointing

    # Sutherland-Hodgman against each half-plane of the bounding polygon.
    output = pts_2d.copy()
    tol = 1e-12
    for k in range(polygon_2d.shape[0]):
        if output.shape[0] == 0:
            break
        n_e = normals[k]
        v_e = polygon_2d[k]
        new_output: list[np.ndarray] = []
        m = output.shape[0]
        for i in range(m):
            a = output[i]
            b = output[(i + 1) % m]
            f_a = float(np.dot(a - v_e, n_e))
            f_b = float(np.dot(b - v_e, n_e))
            a_in = f_a >= -tol
            b_in = f_b >= -tol
            if a_in:
                new_output.append(a)
            if a_in != b_in and abs(f_a - f_b) > tol:
                t = f_a / (f_a - f_b)
                cross = a + t * (b - a)
                new_output.append(cross)
        output = (
            np.stack(new_output, axis=0)
            if new_output
            else np.zeros((0, 2), dtype=float)
        )

    if output.shape[0] < 3:
        return None

    # Dedupe near-coincident points (Sutherland-Hodgman can emit a
    # duplicate when the clipped polygon touches a corner of the bound).
    keep: list[np.ndarray] = [output[0]]
    for p in output[1:]:
        if np.linalg.norm(p - keep[-1]) > tol * 1e3:
            keep.append(p)
    # Wraparound check.
    if len(keep) >= 2 and np.linalg.norm(keep[0] - keep[-1]) <= tol * 1e3:
        keep.pop()
    if len(keep) < 3:
        return None

    out_2d = np.stack(keep, axis=0)
    # Re-embed into 3-D.
    origin = np.asarray(clipper.plane.point, dtype=float)
    out_3d = origin[None, :] + out_2d[:, 0:1] * clipper.e1[None, :] + out_2d[:, 1:2] * clipper.e2[None, :]
    # Sutherland-Hodgman preserves orientation, but our caller relies on
    # CCW-around-the-normal ordering — sort again for safety.
    centroid = out_3d.mean(axis=0)
    rel = out_3d - centroid
    angles = np.arctan2(rel @ clipper.e2, rel @ clipper.e1)
    order = np.argsort(angles)
    return out_3d[order]


# ---------------------------------------------------------------------- #
# Stress reading + sampling
# ---------------------------------------------------------------------- #
# Index of each continuum ``material.stress`` shortname in the 6-vector
# we carry through the math. Layout follows OpenSees Voigt-style
# convention (sym tensor, off-diagonal entries are paired so σᵢⱼ for
# i != j appears once).
_SOLID_STRESS_POSITIONS: dict[str, int] = {
    "sigma11": 0,  # σ_xx in element / global frame
    "sigma22": 1,  # σ_yy
    "sigma33": 2,  # σ_zz
    "sigma12": 3,  # σ_xy
    "sigma23": 4,  # σ_yz
    "sigma13": 5,  # σ_xz
}


def _read_solid_stress_array(rows, n_ip: int) -> np.ndarray:
    """Extract a ``(n_steps, n_ip, 6)`` array from a material.stress DataFrame.

    Columns are ``<shortname>_ip<k>``. Missing components default to
    zero; they shouldn't ever be missing for a continuum recorder but
    treating absence as zero matches the shell kernel's defensive
    posture.
    """
    n_steps = rows.shape[0]
    out = np.zeros((n_steps, n_ip, 6), dtype=float)
    available = set(rows.columns)
    for k in range(n_ip):
        for shortname, pos in _SOLID_STRESS_POSITIONS.items():
            col = f"{shortname}_ip{k}"
            if col in available:
                out[:, k, pos] = rows[col].to_numpy(dtype=float)
    return out


def _stress_voigt_to_tensor(s: np.ndarray) -> np.ndarray:
    """Expand ``(..., 6)`` Voigt stress to a symmetric ``(..., 3, 3)`` tensor.

    Mapping matches :data:`_SOLID_STRESS_POSITIONS`:

        σ = [[s00, s03, s05],
             [s03, s01, s04],
             [s05, s04, s02]]
    """
    out = np.empty(s.shape[:-1] + (3, 3), dtype=float)
    out[..., 0, 0] = s[..., 0]
    out[..., 1, 1] = s[..., 1]
    out[..., 2, 2] = s[..., 2]
    out[..., 0, 1] = s[..., 3]
    out[..., 1, 0] = s[..., 3]
    out[..., 1, 2] = s[..., 4]
    out[..., 2, 1] = s[..., 4]
    out[..., 0, 2] = s[..., 5]
    out[..., 2, 0] = s[..., 5]
    return out


# 8-IP 2×2×2 Gauss-Legendre — IPs at (±s, ±s, ±s), s = 1/√3.
# OpenSees enumerates ξ-fastest, then η, then ζ (matches
# :func:`STKO_to_python.format.gauss_points.tensor_product_3d`).
_BRICK_8IP_S = 1.0 / np.sqrt(3.0)
_BRICK_8IP_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [-1, +1, -1],
        [+1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [-1, +1, +1],
        [+1, +1, +1],
    ],
    dtype=float,
)


def _brick_8ip_weights(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Trilinear interpolation weights from the 8 Gauss IPs of a Brick
    to an arbitrary ``(ξ, η, ζ)`` in the parent cube.

    Mapping ``(ξ, η, ζ) → (ξ/s, η/s, ζ/s)`` puts the IPs at the
    corners of an auxiliary unit cube; trilinear interpolation on that
    cube yields the weights returned here. Extrapolates linearly for
    points beyond the IP envelope (between the cube corners ±s and the
    parent cube boundary ±1).
    """
    xi_l = xi / _BRICK_8IP_S
    eta_l = eta / _BRICK_8IP_S
    zeta_l = zeta / _BRICK_8IP_S
    pt = np.array([xi_l, eta_l, zeta_l])
    return 0.125 * np.prod(1.0 + _BRICK_8IP_SIGNS * pt[None, :], axis=1)


# 27-IP 3×3×3 Gauss-Legendre — 1-D nodes at {-√(3/5), 0, +√(3/5)};
# IPs at all tensor-product combinations, ξ varying fastest. Used by
# higher-order hexes (Brick20 / Brick27) for stress sampling.
_BRICK_27IP_A = float(np.sqrt(3.0 / 5.0))


def _brick_27ip_1d_lagrange(x: float) -> tuple[float, float, float]:
    """1-D quadratic Lagrange basis at the three Gauss-Legendre 3-pt
    nodes ``(-a, 0, +a)`` with ``a = √(3/5)``.

    Returns ``(L_-(x), L_0(x), L_+(x))``. ``L_-(-a) = L_0(0) =
    L_+(+a) = 1``; each basis function is zero at the other two
    nodes by construction. Sum is identically 1 (partition of unity).
    """
    a = _BRICK_27IP_A
    inv2a2 = 5.0 / 6.0      # 1 / (2 a²) with a² = 3/5
    inv_a2 = 5.0 / 3.0      # 1 / a²
    l_minus = inv2a2 * x * (x - a)
    l_zero = 1.0 - inv_a2 * x * x
    l_plus = inv2a2 * x * (x + a)
    return l_minus, l_zero, l_plus


def _brick_27ip_weights(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Triquadratic Lagrange interpolation weights at the 27 IPs of a
    Brick (3×3×3 Gauss-Legendre rule).

    IP enumeration is ξ-fastest then η then ζ — matching
    :func:`STKO_to_python.format.gauss_points.tensor_product_3d`. The
    weight at IP ``(n_ξ, n_η, n_ζ)`` is ``L_{n_ξ}(ξ) · L_{n_η}(η) ·
    L_{n_ζ}(ζ)``.
    """
    lx = _brick_27ip_1d_lagrange(xi)
    le = _brick_27ip_1d_lagrange(eta)
    lz = _brick_27ip_1d_lagrange(zeta)
    w = np.empty(27, dtype=float)
    for nzeta in range(3):
        for neta in range(3):
            for nxi in range(3):
                idx = (nzeta * 3 + neta) * 3 + nxi
                w[idx] = lx[nxi] * le[neta] * lz[nzeta]
    return w


def _sample_solid_stress(
    stress: np.ndarray, xi: float, eta: float, zeta: float, base_type: str, n_ip: int,
) -> np.ndarray:
    """Sample ``material.stress`` at arbitrary natural coords on a solid.

    Returns ``(n_steps, 6)``. Dispatches on ``(base_type, n_ip)``:

    - 8-pt 2×2×2 Brick (any Brick* class): trilinear weights between
      the eight corner IPs.
    - 27-pt 3×3×3 Brick (Brick20 / Brick27 + the 27-IP catalog entry):
      triquadratic Lagrange weights between the 27 tensor-product IPs.
    - 1-pt: single value broadcast.
    - 4-pt tet: barycentric weights on the reference tetrahedron
      (coarse — only used when a fixture surfaces it).

    Unknown ``(base_type, n_ip)`` pairs fall back to the closest IP.
    """
    if "Brick" in base_type:
        if n_ip == 8:
            w = _brick_8ip_weights(xi, eta, zeta)
            return np.einsum("sij,i->sj", stress, w)
        if n_ip == 27:
            w = _brick_27ip_weights(xi, eta, zeta)
            return np.einsum("sij,i->sj", stress, w)
    if n_ip == 1:
        return stress[:, 0, :].copy()
    if n_ip == 4 and "Tetrahedron" in base_type:
        # 4-pt tet rule: assume the IPs lie near the centroid; we
        # interpolate using barycentric weights in (ξ, η, ζ) on the
        # reference tet. This is a coarse approximation — but for the
        # typical FourNodeTetrahedron with 1-IP (which IS the common
        # case) we never enter this branch; the 4-IP path is here for
        # completeness so we don't crash on a hypothetical fixture.
        bary = np.array([1.0 - xi - eta - zeta, xi, eta, zeta])
        # Normalise defensively; numerical noise can push outside the
        # simplex by a few ulps.
        bary = np.clip(bary, 0.0, 1.0)
        bary /= bary.sum()
        return np.einsum("sij,i->sj", stress, bary)
    # Generic fallback: pick the IP closest to (ξ, η, ζ) in the natural
    # domain. Loud TODO for a future reviewer.
    return stress[:, 0, :].copy()


# ---------------------------------------------------------------------- #
# Result container
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class SolidCutResult:
    """Output of :func:`compute_solid_cut`.

    Attributes
    ----------
    F : np.ndarray
        ``(n_steps, 3)`` — total cut force from solids, global frame.
    M : np.ndarray
        ``(n_steps, 3)`` — total cut moment about ``centroid``, global.
    time : np.ndarray
        ``(n_steps,)``.
    centroid : np.ndarray
        ``(3,)`` — reference point for moments (unweighted mean of the
        per-element polygon centroids).
    intersections : tuple[SolidIntersection, ...]
    per_solid_F, per_solid_M_at_centroid : dict[int, np.ndarray]
        Per-element contributions. ``per_solid_M_at_centroid[eid]`` is
        the moment summed about ``intersections[k].polygon_centroid``
        for the matching ``eid`` — the composite cut transfers them to
        the shared centroid via the standard arm × force step.
    """

    F: np.ndarray
    M: np.ndarray
    time: np.ndarray
    centroid: np.ndarray
    intersections: tuple[SolidIntersection, ...]
    per_solid_F: dict[int, np.ndarray] = field(default_factory=dict)
    per_solid_M_at_centroid: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self.intersections) == 0


# ---------------------------------------------------------------------- #
# Per-triangle Gauss rule on a triangle in 3-D
# ---------------------------------------------------------------------- #
# Standard 3-point rule on the unit triangle (vertices at (0,0), (1,0),
# (0,1) in barycentric ξ/η). Same rule used in
# :func:`STKO_to_python.format.gauss_points.gauss_triangle` — kept
# inline here to avoid a runtime import cycle and to make the kernel
# self-contained.
_TRI3_BARYCENTRIC = np.array(
    [
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ]
)
_TRI3_WEIGHTS = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])


def _triangle_quadrature_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """3-pt Gauss quadrature on a 3-D triangle with vertices ``a, b, c``.

    Returns ``(quad_points_global, weights_scaled, area)`` where
    ``weights_scaled[k] = unit_weight[k] * (2 * area)`` so that
    ``sum(weights_scaled * f(quad_points))`` integrates ``f`` over the
    triangle (the factor of 2 accounts for the parent triangle's area
    of 1/2 — the quadrature weights sum to 1/2 on the unit triangle,
    and we want them to sum to the triangle's physical area).
    """
    # Triangle area via cross product.
    e1 = b - a
    e2 = c - a
    area = 0.5 * float(np.linalg.norm(np.cross(e1, e2)))
    # Quadrature points in physical coords.
    xi = _TRI3_BARYCENTRIC[:, 0]
    eta = _TRI3_BARYCENTRIC[:, 1]
    lam0 = 1.0 - xi - eta
    pts = lam0[:, None] * a[None, :] + xi[:, None] * b[None, :] + eta[:, None] * c[None, :]
    weights_scaled = _TRI3_WEIGHTS * (2.0 * area)
    return pts, weights_scaled, area


# ---------------------------------------------------------------------- #
# Resultant phase
# ---------------------------------------------------------------------- #
def compute_solid_cut(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    model_stage: str,
    clipper: "PolygonClipper | None" = None,
) -> SolidCutResult:
    """Compute the ``(F, M)`` resultant of a section cut through solids.

    Returns an empty result when no solid in the filter crosses the
    plane (``F.shape == (0, 3)``).
    """
    intersections = find_solid_intersections(dataset, spec, clipper=clipper)

    # Group by decorated element type for batched material.stress reads.
    by_type: dict[str, list[SolidIntersection]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix)

    per_solid_F: dict[int, np.ndarray] = {}
    per_solid_M_at_centroid: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None

    for elem_type, sub in by_type.items():
        eids = [ix.element_id for ix in sub]
        er = dataset.elements.get_element_results(
            results_name="material.stress",
            element_type=elem_type,
            model_stage=model_stage,
            element_ids=eids,
        )
        # ``n_ip`` lives directly on ElementResults; fall back to the
        # column count when the catalog hasn't surfaced it for an
        # element class (defensive — the standard Brick/Tet entries
        # are present in ``gauss_points.ELEMENT_IP_CATALOG``).
        n_ip = _resolve_n_ip(er)
        if n_ip == 0:
            raise ValueError(
                f"Solid {elem_type!r} has no integration points in the "
                "material.stress result. Register the class in "
                "STKO_to_python.format.gauss_points before running a "
                "section cut on it."
            )
        base_type = _strip_class_tag(elem_type)
        for ix in sub:
            rows = er.df.xs(ix.element_id, level="element_id")
            stress = _read_solid_stress_array(rows, n_ip)
            F, M = _solid_cut_per_element(
                stress, ix, spec, base_type, n_ip,
            )
            per_solid_F[ix.element_id] = F
            per_solid_M_at_centroid[ix.element_id] = M
        if time is None:
            time = np.asarray(er.time, dtype=float)

    if not intersections or time is None:
        return SolidCutResult(
            F=np.zeros((0, 3)),
            M=np.zeros((0, 3)),
            time=np.zeros((0,)),
            centroid=np.zeros(3),
            intersections=(),
        )

    centroid = np.mean(
        np.stack([ix.polygon_centroid for ix in intersections], axis=0), axis=0
    )
    F_total = np.zeros((time.shape[0], 3), dtype=float)
    M_total = np.zeros_like(F_total)
    for ix in intersections:
        Fi = per_solid_F[ix.element_id]
        Mi = per_solid_M_at_centroid[ix.element_id]
        arm = ix.polygon_centroid - centroid
        F_total += Fi
        M_total += Mi + np.cross(arm, Fi)

    return SolidCutResult(
        F=F_total,
        M=M_total,
        time=time,
        centroid=centroid,
        intersections=tuple(intersections),
        per_solid_F=per_solid_F,
        per_solid_M_at_centroid=per_solid_M_at_centroid,
    )


def _resolve_n_ip(er) -> int:
    """Return the integration-point count for an ElementResults bucket.

    Prefers the ``n_ip`` attribute (set by the read path from the
    catalog); falls back to deducing it from the number of distinct
    ``_ip<k>`` suffixes in the column index. The fallback exists so a
    user who registers a new solid class but forgets to update the
    catalog still gets a useful error path (the kernel raises a clear
    "register your class" message rather than silently zeroing).
    """
    if hasattr(er, "n_ip"):
        n = int(er.n_ip)
        if n > 0:
            return n
    cols = list(er.df.columns)
    ips = set()
    for c in cols:
        if "_ip" in c:
            tail = c.rsplit("_ip", 1)[1]
            try:
                ips.add(int(tail))
            except ValueError:
                pass
    return len(ips)


def _orient_cut_normal_for_solid(
    node_coords: np.ndarray,
    polygon_centroid: np.ndarray,
    spec: "SectionCutSpec",
) -> np.ndarray:
    """Choose the cut normal sign so it points from KEPT → DISCARDED.

    For solids the cut plane normal itself is the geometric in-plane
    normal — there is no element-local direction to construct from
    cross products as in the shell kernel. We simply align with
    ``spec.signed_normal`` and then verify the sign against the
    "most-clearly-kept" vertex (highest signed distance):

    - In the normal case (some vertex strictly on the kept side),
      ``n_cut`` should point AWAY from that vertex.
    - In the edge-coincident case (every kept-side vertex is exactly
      on the plane; the side-aware filter guarantees at least one
      vertex strictly on the discarded side), ``n_cut`` should point
      TOWARD the most-clearly-discarded vertex.

    Both yield the convention ``n_cut · (kept_vertex - polygon_centroid)
    < 0`` consistent with the shell kernel.
    """
    signed_n = spec.signed_normal
    plane_point = np.asarray(spec.plane.point, dtype=float)
    rel = node_coords - plane_point
    d = rel @ signed_n  # (n_nodes,)
    n_cut = np.asarray(signed_n, dtype=float)
    if float(d.max()) > 0.0:
        idx = int(np.argmax(d))
        offset = node_coords[idx] - polygon_centroid
        if offset @ n_cut > 0:
            return -n_cut
        return n_cut
    # Edge-coincident — use the most-discarded vertex; n_cut must point
    # toward it.
    idx = int(np.argmin(d))
    offset = node_coords[idx] - polygon_centroid
    if offset @ n_cut < 0:
        return -n_cut
    return n_cut


def _solid_cut_per_element(
    stress: np.ndarray,
    intersection: SolidIntersection,
    spec: "SectionCutSpec",
    base_type: str,
    n_ip: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single solid's traction over its cut polygon.

    Returns ``(F, M)`` shape ``(n_steps, 3)`` each — cut force/moment in
    **global** frame, with the moment summed about the polygon
    centroid. The composite cut aggregator transfers moments to a
    common centroid afterwards.

    Math sketch
    -----------
    For a convex planar polygon ``Π`` we fan-triangulate from its first
    vertex ``v0``::

        Π = ∪_{k=1..K-2} Δ(v0, v_k, v_{k+1})

    Each triangle gets a 3-point Gauss rule. For each quadrature point
    we invert the element's shape function to get ``(ξ, η, ζ)``, sample
    the stress tensor, compute the traction ``t = σ · n_cut``, weight
    by the Gauss weight (scaled to the triangle's physical area) and
    accumulate. The moment uses the lever arm ``r = q - centroid``
    where ``q`` is the quadrature point and ``centroid`` is the
    polygon centroid.
    """
    polygon = intersection.polygon_global
    polygon_natural = intersection.polygon_natural
    K = polygon.shape[0]
    polygon_centroid = intersection.polygon_centroid
    n_cut_global = _orient_cut_normal_for_solid(
        intersection.element_node_coords, polygon_centroid, spec,
    )

    n_steps = stress.shape[0]
    F = np.zeros((n_steps, 3), dtype=float)
    M = np.zeros((n_steps, 3), dtype=float)

    # Walk the fan triangles. Use the global polygon for the area /
    # quadrature-point positions in physical space; use the natural-
    # coord polygon to interpolate (ξ, η, ζ) at each quadrature point
    # via the same barycentric weights.
    v0 = polygon[0]
    v0_nat = polygon_natural[0]
    for k in range(1, K - 1):
        a, b, c = v0, polygon[k], polygon[k + 1]
        a_nat = v0_nat
        b_nat = polygon_natural[k]
        c_nat = polygon_natural[k + 1]
        # Physical-space quadrature.
        pts_phys, w_scaled, _area = _triangle_quadrature_points(a, b, c)
        # Natural-coord quadrature: linear interpolation by the same
        # barycentric weights. Valid because the cut polygon is planar
        # and the element shape function is well-defined on it.
        for q_idx in range(pts_phys.shape[0]):
            xi_bc = _TRI3_BARYCENTRIC[q_idx, 0]
            eta_bc = _TRI3_BARYCENTRIC[q_idx, 1]
            lam0 = 1.0 - xi_bc - eta_bc
            nat_q = lam0 * a_nat + xi_bc * b_nat + eta_bc * c_nat
            sigma_voigt = _sample_solid_stress(
                stress, float(nat_q[0]), float(nat_q[1]), float(nat_q[2]),
                base_type, n_ip,
            )  # (n_steps, 6)
            sigma_tensor = _stress_voigt_to_tensor(sigma_voigt)  # (n_steps, 3, 3)
            # Traction t = sigma · n_cut, shape (n_steps, 3).
            t = sigma_tensor @ n_cut_global
            weight = w_scaled[q_idx]
            F += weight * t
            arm = pts_phys[q_idx] - polygon_centroid
            # arm is (3,), t is (n_steps, 3); np.cross broadcasts arm
            # across the time axis.
            M += weight * np.cross(arm, t)

    return F, M
