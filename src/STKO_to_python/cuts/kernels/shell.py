"""Shell section-cut kernel — geometry + resultant.

The shell kernel mirrors the beam kernel in shape:

- a registry of shell classes the kernel knows how to handle,
- a ``ShellIntersection`` record per shell whose midsurface a plane
  crosses (the chord plus enough metadata to interpolate),
- a ``find_shell_intersections`` step that computes those records,
- a ``compute_shell_cut`` step that reads ``section.force``, integrates
  the line traction over each chord, and aggregates ``(F, M)``.

Why the chord integration matters
---------------------------------
OpenSees shells record ``section.force`` per integration point in
element-local axes — eight components per IP:
``Fxx, Fyy, Fxy, Mxx, Myy, Mxy, Vxz, Vyz`` — and each component is a
**force / moment per unit length of the midsurface line**. The cut
through one shell produces a chord along that line; the cut resultant
contribution is therefore a true ``∫`` over the chord, not the
discrete IP-point evaluation used by beams. We use Gauss-Legendre
quadrature along the chord; section forces are usually linear (for
elastic / bilinear sections) or low-order over a single shell element,
so 2-point quadrature is sufficient in practice.

Layered shells are transparent
------------------------------
``section.force`` is already through-thickness-integrated by the shell
formulation regardless of layer count (the section is the same
abstraction whether it's an ``ElasticMembranePlateSection`` or a
``LayeredShell`` with many fibers). The per-layer breakdown via
``material.fiber.stress`` is a v2 feature; this kernel only reads
``section.force``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...model.layered_section_reader import LayerInfo

if TYPE_CHECKING:
    from ..specs import SectionCutSpec
    from ..geometry import PolygonClipper
    from ...core.dataset import MPCODataSet


# Shell element classes this kernel can handle. The MPCO element index
# carries decorated names like ``"203-ASDShellQ4"``; we strip the
# ``<tag>-`` prefix before matching, so both decorated and base names
# resolve. ``LayeredShell``-equipped variants (``ASDShellQ4`` /
# ``ASDShellT3`` with a ``LayeredShell`` section) are not distinct
# element classes — they're the same elements with a richer section,
# and ``section.force`` is identical in shape.
SHELL_ELEMENT_CLASSES: tuple[str, ...] = (
    "ASDShellQ4",
    "ASDShellT3",
    "ShellMITC4",
    "ShellNLDKGQ",
    "ShellNLDKGT",
    "ShellDKGQ",
    "ShellDKGT",
)


_TAG_PREFIX_RE = re.compile(r"^\d+-")


def _strip_class_tag(decorated: str) -> str:
    """``'203-ASDShellQ4' -> 'ASDShellQ4'``; idempotent on plain names."""
    return _TAG_PREFIX_RE.sub("", str(decorated))


# ---------------------------------------------------------------------- #
# Geometry record
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShellIntersection:
    """A single shell midsurface crossing the cut plane.

    Attributes
    ----------
    element_id : int
    element_type : str
        Decorated type string (e.g. ``"203-ASDShellQ4"``).
    chord_endpoints_global : tuple of two ``(3,)`` tuples
        The two physical-space endpoints of the chord, in global coords.
    chord_param_natural : tuple of two ``(2,)`` tuples
        The chord endpoints in element natural coords ``(ξ, η)`` — these
        feed the shape-function-based sampling along the chord.
    midsurface_polygon_global : ``(n_nodes, 3)`` ndarray
        Element nodes in global coords, ordered as the connectivity.
    node_ids : tuple of int
    """

    element_id: int
    element_type: str
    chord_endpoints_global: tuple[tuple[float, float, float], tuple[float, float, float]]
    chord_param_natural: tuple[tuple[float, float], tuple[float, float]]
    midsurface_polygon_global: np.ndarray = field(repr=False)
    node_ids: tuple[int, ...]

    @property
    def chord_endpoints_arr(self) -> np.ndarray:
        return np.asarray(self.chord_endpoints_global, dtype=float)

    @property
    def chord_midpoint(self) -> np.ndarray:
        """Physical midpoint of the chord — used as the reference point
        for the per-element moment (the analogue of ``point_arr`` on a
        :class:`BeamIntersection`)."""
        p = self.chord_endpoints_arr
        return 0.5 * (p[0] + p[1])

    @property
    def chord_length(self) -> float:
        p = self.chord_endpoints_arr
        return float(np.linalg.norm(p[1] - p[0]))


# ---------------------------------------------------------------------- #
# Inverse shape-function maps — (chord 3-D endpoint) → (ξ, η)
# ---------------------------------------------------------------------- #
def _invert_quad_bilinear(
    p: np.ndarray, verts: np.ndarray, *, max_iter: int = 25, tol: float = 1e-10,
) -> np.ndarray:
    """Invert the bilinear quad map ``x(ξ, η) → physical`` for one point.

    Newton iteration starting at ``(ξ, η) = (0, 0)``. ``verts`` is
    ``(4, 3)`` in OpenSees ASDShellQ4 node order:

        node 1 ↔ (ξ, η) = (-1, -1)
        node 2 ↔ (+1, -1)
        node 3 ↔ (+1, +1)
        node 4 ↔ (-1, +1)

    For a planar quad the Newton step converges in 1-2 iterations; for
    a warped quad we tolerate up to ``max_iter``. The chord endpoint
    must lie on the midsurface; if Newton fails to converge we raise
    rather than return a meaningless point so the kernel can be
    debugged loudly.
    """
    # Quad shape functions and derivatives in ξ, η (see
    # STKO_to_python.format.shape_functions)
    def shape(xi: float, eta: float) -> np.ndarray:
        return 0.25 * np.array(
            [(1 - xi) * (1 - eta),
             (1 + xi) * (1 - eta),
             (1 + xi) * (1 + eta),
             (1 - xi) * (1 + eta)]
        )

    def shape_grad(xi: float, eta: float) -> np.ndarray:
        # 4 nodes × 2 parametric directions
        return 0.25 * np.array([
            [-(1 - eta), -(1 - xi)],
            [ (1 - eta), -(1 + xi)],
            [ (1 + eta),  (1 + xi)],
            [-(1 + eta),  (1 - xi)],
        ])

    # Use the shell midsurface plane's 2-D coords for the Newton solve.
    # That keeps the residual a clean 2-vector (and the Jacobian 2×2)
    # rather than fighting an under-determined 3-vector residual on a
    # 2-DoF unknown.
    centroid = verts.mean(axis=0)
    rel = verts - centroid
    # Two largest singular vectors of the vertex offsets span the
    # midsurface plane.
    u_svd, _, _ = np.linalg.svd(rel, full_matrices=False)
    # ``u_svd`` shape ``(4, k)`` where k = min(4, 3) = 3; the first two
    # left singular vectors are the in-plane basis weights, but we want
    # the in-plane physical directions — take the right singular
    # vectors instead.
    _, _, vh_svd = np.linalg.svd(rel, full_matrices=False)
    e1 = vh_svd[0]
    e2 = vh_svd[1]
    verts_2d = np.column_stack([rel @ e1, rel @ e2])
    p_2d = np.array([(p - centroid) @ e1, (p - centroid) @ e2])

    xi, eta = 0.0, 0.0
    for _ in range(max_iter):
        N = shape(xi, eta)
        # Position in 2D: x(ξ, η) = sum N_i * verts_2d_i
        x_curr = N @ verts_2d  # (2,)
        residual = x_curr - p_2d
        if np.linalg.norm(residual) < tol:
            return np.array([xi, eta])
        dN = shape_grad(xi, eta)  # (4, 2)
        J = verts_2d.T @ dN       # (2, 2) — ∂x/∂(ξ, η)
        try:
            step = np.linalg.solve(J, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Singular Jacobian inverting quad bilinear at "
                f"(ξ, η) = ({xi}, {eta})."
            ) from exc
        xi -= float(step[0])
        eta -= float(step[1])
    raise RuntimeError(
        f"Bilinear inversion did not converge after {max_iter} iterations "
        f"(final residual {np.linalg.norm(residual)}, p={p.tolist()})."
    )


def _invert_tri_linear(p: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Invert the linear-triangle map for one point.

    Returns ``(ξ, η)`` on the unit triangle (vertices at (0, 0), (1, 0),
    (0, 1)). For a planar tri the inversion is exact:

        x(ξ, η) = v0 * (1 - ξ - η) + v1 * ξ + v2 * η
                = v0 + (v1 - v0) * ξ + (v2 - v0) * η

    Solve the 2-D linear system in the triangle's own plane.
    """
    v0, v1, v2 = verts
    e1 = v1 - v0
    e2 = v2 - v0
    rel = p - v0
    # Build a 2-D basis on the triangle plane and project.
    n = np.cross(e1, e2)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-30:
        raise RuntimeError("Degenerate (collinear) triangle vertices.")
    e_a = e1 / np.linalg.norm(e1)
    e_b = np.cross(n / n_norm, e_a)
    rel_2d = np.array([rel @ e_a, rel @ e_b])
    A = np.column_stack([
        np.array([e1 @ e_a, e1 @ e_b]),
        np.array([e2 @ e_a, e2 @ e_b]),
    ])
    sol = np.linalg.solve(A, rel_2d)
    return sol  # (ξ, η)


def _natural_coords_of_chord(
    chord_global: np.ndarray, midsurface_verts: np.ndarray, base_type: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute element natural coords ``(ξ, η)`` for the two chord
    endpoints, dispatching on element class.

    ASDShellQ4: bilinear inversion (Newton). ASDShellT3 / generic
    triangular shell: linear inversion (closed form). Other classes
    that ship 4-node quads / 3-node tris reuse the same inversions.
    """
    a = chord_global[0]
    b = chord_global[1]
    if midsurface_verts.shape[0] == 4:
        xi_a = _invert_quad_bilinear(a, midsurface_verts)
        xi_b = _invert_quad_bilinear(b, midsurface_verts)
    elif midsurface_verts.shape[0] == 3:
        xi_a = _invert_tri_linear(a, midsurface_verts)
        xi_b = _invert_tri_linear(b, midsurface_verts)
    else:
        raise NotImplementedError(
            f"Shell class {base_type!r} has {midsurface_verts.shape[0]} "
            "nodes; only 3- and 4-node midsurfaces are supported in v1.6."
        )
    return (
        (float(xi_a[0]), float(xi_a[1])),
        (float(xi_b[0]), float(xi_b[1])),
    )


# ---------------------------------------------------------------------- #
# Geometry phase
# ---------------------------------------------------------------------- #
def find_shell_intersections(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    clipper: "PolygonClipper | None" = None,
) -> list[ShellIntersection]:
    """Find intersections between ``spec.plane`` and every shell in the filter.

    A shell is skipped without warning when:

    - its midsurface lies parallel to the cut plane within numerical
      tolerance,
    - the intersection chord has zero length (the plane grazes a
      vertex only),
    - the spec carries a ``bounding_polygon`` and the chord is fully
      outside the polygon — in that case the chord is clipped against
      the polygon and only the in-polygon portion is recorded.

    Parameters
    ----------
    clipper : :class:`PolygonClipper`, optional
        Pre-built clipper for ``spec.bounding_polygon``. The caller
        builds this once per cut and threads it through to both the
        beam and shell phases so the plane basis isn't recomputed.
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

    shell_set = set(SHELL_ELEMENT_CLASSES)
    is_shell = df_elems["element_type"].map(
        lambda s: _strip_class_tag(s) in shell_set
    )
    in_filter = df_elems["element_id"].isin(candidate_set)
    shell_rows = df_elems[is_shell & in_filter]

    plane = spec.plane
    out: list[ShellIntersection] = []
    for row in shell_rows.itertuples(index=False):
        node_list = row.node_list
        n_nodes = len(node_list)
        if n_nodes not in (3, 4):
            # v1.6 supports tri3 + quad4 only; higher-order shells skip.
            continue
        coords = np.array(
            [node_coord[int(nid)] for nid in node_list if int(nid) in node_coord],
            dtype=float,
        )
        if coords.shape[0] != n_nodes:
            continue
        # Side-aware shared-edge rule: when two adjacent shells share
        # an edge that lies on the cut plane, both report that edge as
        # a chord. Keeping both double-counts. Per the standard cut
        # convention, the cut traction comes from the DISCARDED side —
        # so for each shell skip it when its interior is entirely on
        # the kept side (no vertex strictly on the discarded side).
        d_signed = (coords - np.asarray(plane.point, dtype=float)) @ spec.signed_normal
        tol_d = 1e-9
        has_discarded = bool(np.any(d_signed < -tol_d))
        if not has_discarded:
            # Interior on the kept side or entirely on the plane —
            # the neighbouring shell on the discarded side will report
            # the chord. Skipping here avoids the double-count.
            continue
        chord = plane.intersect_polygon(coords)
        if chord is None:
            continue
        p0, p1 = chord
        if np.linalg.norm(p1 - p0) < 1e-12:
            continue
        chord_arr = np.stack([p0, p1])

        # Apply bounding-polygon clipping in 2-D plane coords.
        if clipper is not None:
            clipped = clipper.clip_segment(p0, p1)
            if clipped is None:
                continue
            q0, q1, _, _ = clipped
            chord_arr = np.stack([q0, q1])

        base_type = _strip_class_tag(row.element_type)
        xi_a, xi_b = _natural_coords_of_chord(chord_arr, coords, base_type)
        out.append(
            ShellIntersection(
                element_id=int(row.element_id),
                element_type=str(row.element_type),
                chord_endpoints_global=(
                    (float(chord_arr[0, 0]), float(chord_arr[0, 1]), float(chord_arr[0, 2])),
                    (float(chord_arr[1, 0]), float(chord_arr[1, 1]), float(chord_arr[1, 2])),
                ),
                chord_param_natural=(xi_a, xi_b),
                midsurface_polygon_global=coords,
                node_ids=tuple(int(nid) for nid in node_list),
            )
        )
    out.sort(key=lambda s: s.element_id)
    return out


# ---------------------------------------------------------------------- #
# Section-force layout
# ---------------------------------------------------------------------- #
# Index of each shell ``section.force`` shortname in the 8-vector we
# carry through the math. Eight components per IP — STKO records all of
# them for the standard ASDShell sections. Missing components default
# to zero (e.g., a membrane-only formulation that omits Mxx).
_SHELL_SECTION_POSITIONS: dict[str, int] = {
    "Fxx": 0,  # in-plane normal force per unit length, local x
    "Fyy": 1,  # in-plane normal force per unit length, local y
    "Fxy": 2,  # in-plane shear per unit length
    "Mxx": 3,  # bending moment per unit length, about local x
    "Myy": 4,  # bending moment per unit length, about local y
    "Mxy": 5,  # twisting moment per unit length
    "Vxz": 6,  # transverse (through-thickness) shear, local x face
    "Vyz": 7,  # transverse shear, local y face
}


def _read_shell_section_force_array(rows, n_ip: int) -> np.ndarray:
    """Extract a ``(n_steps, n_ip, 8)`` array from a shell section.force
    DataFrame.

    Columns are ``<shortname>_ip<k>``. Missing components default to
    zero — useful for partial recorders or non-standard shells that
    omit bending / shear terms.
    """
    n_steps = rows.shape[0]
    out = np.zeros((n_steps, n_ip, 8), dtype=float)
    available = set(rows.columns)
    for k in range(n_ip):
        for shortname, pos in _SHELL_SECTION_POSITIONS.items():
            col = f"{shortname}_ip{k}"
            if col in available:
                out[:, k, pos] = rows[col].to_numpy(dtype=float)
    return out


# ---------------------------------------------------------------------- #
# Quad / tri sampling weights from IP values to an arbitrary (ξ, η)
# ---------------------------------------------------------------------- #
_QUAD_GAUSS_S = 1.0 / np.sqrt(3.0)  # IP coord magnitude for 2×2 Gauss


def _quad_ip_weights(xi: float, eta: float) -> np.ndarray:
    """Bilinear interpolation weights from the 4 Gauss-Legendre 2×2 IPs
    of ASDShellQ4 to an arbitrary ``(ξ, η)``.

    The IPs sit at ``(±s, ±s)`` with ``s = 1/√3``, ordered ξ-fastest:

        IP 0 ↔ (-s, -s)
        IP 1 ↔ (+s, -s)
        IP 2 ↔ (-s, +s)
        IP 3 ↔ (+s, +s)

    Mapping ``ξ → ξ_local = ξ / s`` puts the IPs at the corners of an
    auxiliary unit square; bilinear interpolation on that square gives
    the weights below. Extrapolates linearly for chord points beyond
    the IP envelope (typical when a quad has IPs at ``±1/√3`` but the
    cut passes near the element edge at ``ξ = ±1``).
    """
    xi_local = xi / _QUAD_GAUSS_S
    eta_local = eta / _QUAD_GAUSS_S
    return 0.25 * np.array([
        (1 - xi_local) * (1 - eta_local),
        (1 + xi_local) * (1 - eta_local),
        (1 - xi_local) * (1 + eta_local),
        (1 + xi_local) * (1 + eta_local),
    ])


def _tri_ip_weights(xi: float, eta: float) -> np.ndarray:
    """Linear interpolation weights from the 3 Gauss IPs of ASDShellT3
    to an arbitrary ``(ξ, η)`` on the unit triangle.

    The IPs sit at ``(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)``. Solving the
    inverse-linear system gives the weight set used below — verified
    by partition-of-unity at the centroid and unit response at each IP.
    """
    return np.array([
        5.0 / 3.0 - 2.0 * xi - 2.0 * eta,
        -1.0 / 3.0 + 2.0 * xi,
        -1.0 / 3.0 + 2.0 * eta,
    ])


def _sample_shell_section_force(
    section_force: np.ndarray, xi: float, eta: float, base_type: str,
) -> np.ndarray:
    """Sample ``section.force`` at arbitrary natural coords on a shell.

    Returns ``(n_steps, 8)``. Dispatches to bilinear (quads) or linear
    (tris) interpolation depending on the element class. The choice
    follows the IP layout that
    :mod:`STKO_to_python.format.gauss_points` declares for each class.
    """
    n_ip = section_force.shape[1]
    if n_ip == 4 and "Q4" in base_type:
        w = _quad_ip_weights(xi, eta)
    elif n_ip == 3 and "T3" in base_type:
        w = _tri_ip_weights(xi, eta)
    elif n_ip == 1:
        # 1-IP shell: just broadcast the single IP value.
        return section_force[:, 0, :].copy()
    elif n_ip == 4:
        w = _quad_ip_weights(xi, eta)
    elif n_ip == 3:
        w = _tri_ip_weights(xi, eta)
    else:
        raise NotImplementedError(
            f"Shell {base_type!r} has {n_ip} IPs; only 1/3/4 IPs are "
            "supported in v1.6."
        )
    # (n_steps, n_ip, 8) · (n_ip,) → (n_steps, 8)
    return np.einsum("sij,i->sj", section_force, w)


# ---------------------------------------------------------------------- #
# Result container
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ShellCutResult:
    """Output of :func:`compute_shell_cut`.

    Attributes
    ----------
    F : np.ndarray
        ``(n_steps, 3)`` — total cut force from shells, global frame.
    M : np.ndarray
        ``(n_steps, 3)`` — total cut moment about ``centroid``, global.
    time : np.ndarray
        ``(n_steps,)``.
    centroid : np.ndarray
        ``(3,)`` — reference point for moments (unweighted mean of the
        chord midpoints). The composite ``SectionCut`` may reaggregate
        about a different centroid that also includes beam intersection
        points.
    intersections : tuple[ShellIntersection, ...]
    per_shell_F, per_shell_M_at_midpoint : dict[int, np.ndarray]
    """

    F: np.ndarray
    M: np.ndarray
    time: np.ndarray
    centroid: np.ndarray
    intersections: tuple[ShellIntersection, ...]
    per_shell_F: dict[int, np.ndarray] = field(default_factory=dict)
    per_shell_M_at_midpoint: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self.intersections) == 0


# ---------------------------------------------------------------------- #
# Resultant phase
# ---------------------------------------------------------------------- #
_GAUSS_LEGENDRE_2 = (
    np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
    np.array([1.0, 1.0]),
)


def compute_shell_cut(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    model_stage: str,
    clipper: "PolygonClipper | None" = None,
) -> ShellCutResult:
    """Compute the ``(F, M)`` resultant of a section cut through shells.

    Returns an empty result when no shell in the filter crosses the
    plane (``F.shape == (0, 3)``).
    """
    intersections = find_shell_intersections(dataset, spec, clipper=clipper)

    # Group by decorated element type for batched section.force reads.
    by_type: dict[str, list[ShellIntersection]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix)

    per_shell_F: dict[int, np.ndarray] = {}
    per_shell_M_at_mid: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None

    for elem_type, sub in by_type.items():
        eids = [ix.element_id for ix in sub]
        er = dataset.elements.get_element_results(
            results_name="section.force",
            element_type=elem_type,
            model_stage=model_stage,
            element_ids=eids,
        )
        n_ip = int(er.n_ip) if er.n_ip > 0 else 0
        if n_ip == 0:
            raise ValueError(
                f"Shell {elem_type!r} has no integration points in the "
                "section.force result. This means the shell class is not "
                "in the Gauss-point catalog at "
                "STKO_to_python.format.gauss_points — register it before "
                "running a section cut on it."
            )
        base_type = _strip_class_tag(elem_type)
        for ix in sub:
            rows = er.df.xs(ix.element_id, level="element_id")
            section_force = _read_shell_section_force_array(rows, n_ip)
            F, M = _shell_cut_per_element(
                section_force, ix, dataset, spec, base_type,
            )
            per_shell_F[ix.element_id] = F
            per_shell_M_at_mid[ix.element_id] = M
        if time is None:
            time = np.asarray(er.time, dtype=float)

    if not intersections or time is None:
        return ShellCutResult(
            F=np.zeros((0, 3)),
            M=np.zeros((0, 3)),
            time=np.zeros((0,)),
            centroid=np.zeros(3),
            intersections=(),
        )

    centroid = np.mean(
        np.stack([ix.chord_midpoint for ix in intersections], axis=0), axis=0
    )
    F_total = np.zeros((time.shape[0], 3), dtype=float)
    M_total = np.zeros_like(F_total)
    for ix in intersections:
        Fi = per_shell_F[ix.element_id]
        Mi = per_shell_M_at_mid[ix.element_id]
        arm = ix.chord_midpoint - centroid
        F_total += Fi
        M_total += Mi + np.cross(arm, Fi)

    return ShellCutResult(
        F=F_total,
        M=M_total,
        time=time,
        centroid=centroid,
        intersections=tuple(intersections),
        per_shell_F=per_shell_F,
        per_shell_M_at_midpoint=per_shell_M_at_mid,
    )


# ---------------------------------------------------------------------- #
# Per-element kernel
# ---------------------------------------------------------------------- #
def _shell_cut_per_element(
    section_force: np.ndarray,
    intersection: ShellIntersection,
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    base_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a single shell's traction along its chord.

    Returns ``(F, M)`` shape ``(n_steps, 3)`` each — cut force/moment in
    **global** frame, with the moment summed about the chord midpoint.
    The composite cut aggregator transfers moments to a common
    centroid afterwards.
    """
    chord = intersection.chord_endpoints_arr  # (2, 3)
    L = intersection.chord_length
    p_mid = intersection.chord_midpoint

    # Cut in-plane normal in global frame.
    # Construct the shell normal from the midsurface polygon (first
    # non-collinear cross product), then the chord-perpendicular within
    # the shell plane is direction × shell_normal.
    n_shell = _midsurface_normal(intersection.midsurface_polygon_global)
    chord_dir = chord[1] - chord[0]
    chord_dir /= np.linalg.norm(chord_dir)
    n_cut_global = np.cross(chord_dir, n_shell)
    n_cut_norm = float(np.linalg.norm(n_cut_global))
    if n_cut_norm < 1e-12:
        # Shouldn't happen for a real cut (chord direction is by
        # construction perpendicular to the cut plane normal, which is
        # in the shell plane only when the shell normal is colinear
        # with the chord — i.e. the cut plane is tangent to the shell
        # midsurface, which we treat as a no-op above).
        return np.zeros((section_force.shape[0], 3)), np.zeros((section_force.shape[0], 3))
    n_cut_global /= n_cut_norm
    # Pick the sign so that n_cut_global points from the KEPT side to
    # the DISCARDED side. Then the traction t = N · n_cut is the force
    # the discarded side exerts on the kept side — the cut convention.
    n_cut_global = _orient_cut_normal(
        n_cut_global, intersection.midsurface_polygon_global, p_mid, spec,
    )

    # Express the cut normal in element-local coords (the frame the
    # section forces are recorded in).
    R = dataset.cdata.rotation_matrix(intersection.element_id)  # local→global
    n_cut_local = R.T @ n_cut_global  # global→local
    n_x = float(n_cut_local[0])
    n_y = float(n_cut_local[1])
    # Element-local z is the shell normal direction — should be ~0.
    # We take the in-plane components only; the through-thickness shear
    # uses (n_x, n_y) directly.

    # Gauss-Legendre 2-pt along the chord.
    gs, gw = _GAUSS_LEGENDRE_2
    xi_a, eta_a = intersection.chord_param_natural[0]
    xi_b, eta_b = intersection.chord_param_natural[1]
    half_L = 0.5 * L

    n_steps = section_force.shape[0]
    F_local = np.zeros((n_steps, 3), dtype=float)
    M_local = np.zeros((n_steps, 3), dtype=float)

    for s, w in zip(gs, gw):
        alpha = 0.5 * (s + 1.0)  # ∈ [0, 1]
        xi_g = (1 - alpha) * xi_a + alpha * xi_b
        eta_g = (1 - alpha) * eta_a + alpha * eta_b
        sf = _sample_shell_section_force(section_force, xi_g, eta_g, base_type)
        # sf: (n_steps, 8) in element-local axes, ordered per
        # _SHELL_SECTION_POSITIONS: Fxx, Fyy, Fxy, Mxx, Myy, Mxy, Vxz, Vyz.
        Fxx, Fyy, Fxy = sf[:, 0], sf[:, 1], sf[:, 2]
        Mxx, Myy, Mxy = sf[:, 3], sf[:, 4], sf[:, 5]
        Vxz, Vyz = sf[:, 6], sf[:, 7]
        # Traction per unit length on the cut line, element-local:
        #   force x = Fxx * n_x + Fxy * n_y           (membrane)
        #   force y = Fxy * n_x + Fyy * n_y           (membrane)
        #   force z = Vxz * n_x + Vyz * n_y           (through-thickness)
        # Moment per unit length about element-local axes lying in the
        # midsurface (no z-axis contribution from section forces):
        #   moment x = Mxx * n_x + Mxy * n_y
        #   moment y = Mxy * n_x + Myy * n_y
        #   moment z = 0
        f_per_unit_x = Fxx * n_x + Fxy * n_y
        f_per_unit_y = Fxy * n_x + Fyy * n_y
        f_per_unit_z = Vxz * n_x + Vyz * n_y
        m_per_unit_x = Mxx * n_x + Mxy * n_y
        m_per_unit_y = Mxy * n_x + Myy * n_y
        weight = w * half_L
        F_local[:, 0] += weight * f_per_unit_x
        F_local[:, 1] += weight * f_per_unit_y
        F_local[:, 2] += weight * f_per_unit_z
        M_local[:, 0] += weight * m_per_unit_x
        M_local[:, 1] += weight * m_per_unit_y
        # m_per_unit_z is identically zero — skip.

    F_global = F_local @ R.T
    M_global = M_local @ R.T
    return F_global, M_global


# ---------------------------------------------------------------------- #
# Per-layer breakdown (v1.7)
# ---------------------------------------------------------------------- #
#
# A layered shell's ``section.fiber.stress`` recorder writes one block
# per ``(gauss_point × thickness_layer)`` pair. The library's column-name
# parser flattens those blocks into columns named
# ``sigma11_l<L>_ip<K>`` (or ``sigma11_f<F>_l<L>_ip<K>`` when there are
# fibers within a layer). The layer 0 is conventionally the bottom; the
# top layer is index ``n_layers - 1``.
#
# Per-layer math
# --------------
# The shell's standard ``section.force`` 8-vector is the through-
# thickness integral of the layer-wise stress against simple geometry
# weights:
#
#     Fxx = ∫ σ_11 dz  ≈  Σ_k σ_11^(k) · t_k
#     Mxx = ∫ σ_11 · z dz  ≈  Σ_k σ_11^(k) · t_k · z_offset_k
#
# Replacing the standard 8-vector with the kth-layer-only version gives
# the per-layer contribution to the cut. The rest of the math (chord
# integration, rotation to global) is identical to
# :func:`_shell_cut_per_element`.

# Voigt column order used internally by the layered-shell kernel.
#     0: sigma11   1: sigma22   2: sigma33
#     3: sigma12   4: sigma23   5: sigma13
_LAYER_STRESS_SHORTNAMES = ("sigma11", "sigma22", "sigma33", "sigma12", "sigma23", "sigma13")


# Layered-shell nDMaterial response — most OpenSees nDMaterials don't
# tag their stress components with standard codes, so MPCO falls back
# to ``UnknownStress(n)`` placeholders. The five-component PlateFiber
# layout is consistent across PlaneStress-wrapped and full PlateFiber
# materials in the OpenSees source:
#
#     UnknownStress     -> sigma11 (in-plane normal x)
#     UnknownStress(1)  -> sigma22 (in-plane normal y)
#     UnknownStress(2)  -> sigma12 (in-plane shear)
#     UnknownStress(3)  -> sigma13 (transverse shear x-z) — often ~0
#     UnknownStress(4)  -> sigma23 (transverse shear y-z) — often ~0
#
# Each tuple entry is ``(column_basename, voigt_position)``.
_UNKNOWN_STRESS_TO_VOIGT: tuple[tuple[str, int], ...] = (
    ("UnknownStress", 0),     # sigma11
    ("UnknownStress(1)", 1),  # sigma22
    ("UnknownStress(2)", 3),  # sigma12
    ("UnknownStress(3)", 5),  # sigma13
    ("UnknownStress(4)", 4),  # sigma23
)


def _layer_column_candidates(
    layer_idx: int, ip: int, available: set[str],
) -> list[tuple[str, int]]:
    """Return ``[(column_name, voigt_position), ...]`` for one
    (layer, IP) pair.

    Probes the column index in priority order:

    1. Explicit ``sigma11_l<L>_ip<K>`` (per the docs / format
       conventions §17).
    2. Explicit ``sigma11_f<L>_ip<K>`` (alternate when MPCO writes the
       layer axis as a fiber index).
    3. ``UnknownStress(n)_f<L>_ip<K>`` (fallback for layered shells
       whose nDMaterial doesn't register response codes — the standard
       Test_NLShell case).

    Returns only columns that exist in ``available``.
    """
    candidates: list[tuple[str, int]] = []
    # Path 1: explicit sigma<ij>_l<L>_ip<K>.
    for pos, name in enumerate(_LAYER_STRESS_SHORTNAMES):
        col = f"{name}_l{layer_idx}_ip{ip}"
        if col in available:
            candidates.append((col, pos))
    if candidates:
        return candidates
    # Path 2: explicit sigma<ij>_f<L>_ip<K>.
    for pos, name in enumerate(_LAYER_STRESS_SHORTNAMES):
        col = f"{name}_f{layer_idx}_ip{ip}"
        if col in available:
            candidates.append((col, pos))
    if candidates:
        return candidates
    # Path 3: UnknownStress(n)_f<L>_ip<K> (the nDMaterial fallback —
    # most common for Test_NLShell-style layered plate sections).
    for base, pos in _UNKNOWN_STRESS_TO_VOIGT:
        col = f"{base}_f{layer_idx}_ip{ip}"
        if col in available:
            candidates.append((col, pos))
    return candidates


def _fiber_in_layer_column_candidates(
    fiber_idx: int, layer_idx: int, ip: int, available: set[str],
) -> list[tuple[str, int]]:
    """Return ``[(column_name, voigt_position), ...]`` for one
    (fiber, layer, IP) triple.

    Probes the column index in priority order:

    1. Explicit ``sigma11_f<F>_l<L>_ip<K>``.
    2. ``UnknownStress(n)_f<F>_l<L>_ip<K>`` (nDMaterial fallback).
    """
    candidates: list[tuple[str, int]] = []
    # Path 1: explicit sigma<ij>_f<F>_l<L>_ip<K>.
    for pos, name in enumerate(_LAYER_STRESS_SHORTNAMES):
        col = f"{name}_f{fiber_idx}_l{layer_idx}_ip{ip}"
        if col in available:
            candidates.append((col, pos))
    if candidates:
        return candidates
    # Path 2: UnknownStress(n)_f<F>_l<L>_ip<K>.
    for base, pos in _UNKNOWN_STRESS_TO_VOIGT:
        col = f"{base}_f{fiber_idx}_l{layer_idx}_ip{ip}"
        if col in available:
            candidates.append((col, pos))
    return candidates


def _read_fiber_in_layer_stress_array(
    rows, n_ip: int, layer_idx: int, fiber_idx: int,
) -> np.ndarray:
    """Extract a ``(n_steps, n_ip, 6)`` Voigt stress array for one
    fiber within one layer.

    Recognises ``sigma11_f<F>_l<L>_ip<K>`` and
    ``UnknownStress(n)_f<F>_l<L>_ip<K>``. Missing components default
    to zero.
    """
    n_steps = rows.shape[0]
    out = np.zeros((n_steps, n_ip, 6), dtype=float)
    available = set(rows.columns)
    for k in range(n_ip):
        for col, pos in _fiber_in_layer_column_candidates(
            fiber_idx, layer_idx, k, available,
        ):
            out[:, k, pos] = rows[col].to_numpy(dtype=float)
    return out


_FIBER_IN_LAYER_RE = re.compile(
    r"_f(\d+)_l(\d+)_ip(\d+)$"
)


def _discover_fiber_count_in_layer(columns, layer_idx: int) -> int:
    """Return the number of distinct fibers in ``layer_idx`` by scanning
    the available column names for ``_f<F>_l<layer_idx>_ip<K>``
    patterns.

    Returns 0 when no fiber-in-layer columns exist for the requested
    layer — the caller treats that as "this layer has no fibers" and
    raises a clear error.
    """
    seen: set[int] = set()
    for c in columns:
        m = _FIBER_IN_LAYER_RE.search(str(c))
        if m is None:
            continue
        f_idx = int(m.group(1))
        l_idx = int(m.group(2))
        if l_idx == int(layer_idx):
            seen.add(f_idx)
    return len(seen)


def _read_layer_stress_array(
    rows, n_ip: int, layer_idx: int,
) -> np.ndarray:
    """Extract a ``(n_steps, n_ip, 6)`` Voigt stress array for one layer.

    Recognises three column conventions (see
    :func:`_layer_column_candidates`):

    - ``sigma11_l<L>_ip<K>`` — explicit ``_l<L>_`` layer index with
      named stress shortnames (per the format-conventions doc).
    - ``sigma11_f<L>_ip<K>`` — explicit ``_f<L>_`` index when MPCO
      treats the layer axis as a fiber axis.
    - ``UnknownStress(n)_f<L>_ip<K>`` — the ``nDMaterial`` fallback
      with indexed component names. The mapping to Voigt positions
      follows the PlateFiber convention (σ11, σ22, σ12, σ13, σ23).

    Missing components default to zero — many materials (e.g. rebar
    uniaxial wrapped into a layer) only carry σ11.
    """
    n_steps = rows.shape[0]
    out = np.zeros((n_steps, n_ip, 6), dtype=float)
    available = set(rows.columns)
    for k in range(n_ip):
        for col, pos in _layer_column_candidates(layer_idx, k, available):
            out[:, k, pos] = rows[col].to_numpy(dtype=float)
    return out


def _sample_layer_stress(
    stress: np.ndarray, xi: float, eta: float, base_type: str,
) -> np.ndarray:
    """Sample layer Voigt stress at arbitrary ``(ξ, η)`` on the shell.

    Reuses the standard quad/tri interpolation weights. ``stress``
    shape ``(n_steps, n_ip, 6)`` → returns ``(n_steps, 6)``.
    """
    n_ip = stress.shape[1]
    if n_ip == 4 and "Q4" in base_type:
        w = _quad_ip_weights(xi, eta)
    elif n_ip == 3 and "T3" in base_type:
        w = _tri_ip_weights(xi, eta)
    elif n_ip == 1:
        return stress[:, 0, :].copy()
    elif n_ip == 4:
        w = _quad_ip_weights(xi, eta)
    elif n_ip == 3:
        w = _tri_ip_weights(xi, eta)
    else:
        raise NotImplementedError(
            f"Shell {base_type!r} has {n_ip} layered IPs; only 1/3/4 IPs "
            "are supported in v1.7."
        )
    return np.einsum("sij,i->sj", stress, w)


def _resolve_layer_table(
    dataset: "MPCODataSet", element_id: int,
):
    """Look up the layer table for one element.

    Chain: ``cdata.element_info[eid].physical_property_id ->
    dataset.layered_sections[section_id]``.

    The two side tables are populated from different files
    (``.cdata`` for the element-info index, ``sections.tcl`` for the
    layer geometry); a mismatch surfaces here as ``KeyError`` rather
    than silently producing zero per-layer forces.
    """
    info = dataset.cdata.element_info.get(int(element_id))
    if info is None:
        raise KeyError(
            f"Element {element_id} has no *ELEMENT_INFO record in the "
            f".cdata sidecar. Per-layer cuts require it to resolve the "
            f"element's physical_property_id."
        )
    section_id = int(info.physical_property_id)
    layers = dataset.layered_sections.get(section_id)
    if layers is None:
        raise KeyError(
            f"Element {element_id}: physical_property_id={section_id} "
            "does not match any LayeredShell section parsed from "
            "sections.tcl. Verify that sections.tcl lives beside the "
            ".mpco recorder output and that the section id matches the "
            "STKO physical_property_id."
        )
    return layers


def compute_shell_cut_per_layer(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    model_stage: str,
    layer_idx: int,
    clipper: "PolygonClipper | None" = None,
) -> "ShellCutResult":
    """Compute a per-layer slice of the shell section cut.

    The geometry phase is identical to :func:`compute_shell_cut` — same
    shells, same chords. The resultant phase replaces
    ``section.force`` with the layer's own contribution to the section
    force 8-vector::

        Fxx^(k) = σ_11^(k) · t_k
        Fyy^(k) = σ_22^(k) · t_k
        Fxy^(k) = σ_12^(k) · t_k
        Mxx^(k) = σ_11^(k) · t_k · z_offset_k
        Myy^(k) = σ_22^(k) · t_k · z_offset_k
        Mxy^(k) = σ_12^(k) · t_k · z_offset_k
        Vxz^(k) = σ_13^(k) · t_k
        Vyz^(k) = σ_23^(k) · t_k

    Summing all per-layer cuts must recover the standard cut to
    numerical tolerance (verified by
    :meth:`SectionCut.per_layer_force` and its test).

    Parameters
    ----------
    layer_idx : int
        Layer index from 0 (bottom) to ``n_layers - 1`` (top), matching
        the ordering in ``sections.tcl``.

    Raises
    ------
    KeyError
        If a contributing shell's section can't be resolved (missing
        ``*ELEMENT_INFO`` or no ``LayeredShell`` definition for its
        ``physical_property_id``).
    """
    if not isinstance(layer_idx, (int, np.integer)) or int(layer_idx) < 0:
        raise ValueError(f"layer_idx must be a non-negative int, got {layer_idx!r}.")
    layer_idx = int(layer_idx)

    intersections = find_shell_intersections(dataset, spec, clipper=clipper)

    by_type: dict[str, list[ShellIntersection]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix)

    per_shell_F: dict[int, np.ndarray] = {}
    per_shell_M_at_mid: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None

    for elem_type, sub in by_type.items():
        eids = [ix.element_id for ix in sub]
        er = dataset.elements.get_element_results(
            results_name="section.fiber.stress",
            element_type=elem_type,
            model_stage=model_stage,
            element_ids=eids,
        )
        n_ip = int(er.n_ip) if er.n_ip > 0 else 0
        if n_ip == 0:
            raise ValueError(
                f"Shell {elem_type!r} has no integration points in the "
                "section.fiber.stress result. The per-layer cut needs "
                "stress recorded at each in-plane Gauss point; verify "
                "the recorder writes section.fiber.stress on this class."
            )
        base_type = _strip_class_tag(elem_type)
        for ix in sub:
            layers = _resolve_layer_table(dataset, ix.element_id)
            if layer_idx >= len(layers):
                raise IndexError(
                    f"layer_idx={layer_idx} out of range for element "
                    f"{ix.element_id}: section has {len(layers)} layers."
                )
            layer = layers[layer_idx]
            rows = er.df.xs(ix.element_id, level="element_id")
            layer_stress = _read_layer_stress_array(rows, n_ip, layer_idx)
            F, M = _shell_cut_per_element_for_layer(
                layer_stress, layer, ix, dataset, spec, base_type,
            )
            per_shell_F[ix.element_id] = F
            per_shell_M_at_mid[ix.element_id] = M
        if time is None:
            time = np.asarray(er.time, dtype=float)

    if not intersections or time is None:
        return ShellCutResult(
            F=np.zeros((0, 3)),
            M=np.zeros((0, 3)),
            time=np.zeros((0,)),
            centroid=np.zeros(3),
            intersections=(),
        )

    centroid = np.mean(
        np.stack([ix.chord_midpoint for ix in intersections], axis=0), axis=0
    )
    F_total = np.zeros((time.shape[0], 3), dtype=float)
    M_total = np.zeros_like(F_total)
    for ix in intersections:
        Fi = per_shell_F[ix.element_id]
        Mi = per_shell_M_at_mid[ix.element_id]
        arm = ix.chord_midpoint - centroid
        F_total += Fi
        M_total += Mi + np.cross(arm, Fi)

    return ShellCutResult(
        F=F_total,
        M=M_total,
        time=time,
        centroid=centroid,
        intersections=tuple(intersections),
        per_shell_F=per_shell_F,
        per_shell_M_at_midpoint=per_shell_M_at_mid,
    )


def compute_shell_cut_per_fiber(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    model_stage: str,
    layer_idx: int,
    fiber_idx: int,
    clipper: "PolygonClipper | None" = None,
) -> "ShellCutResult":
    """Compute a per-fiber slice of the shell section cut.

    A layered shell whose layer L is itself a Fiber section produces
    ``section.fiber.stress`` columns named
    ``<comp>_f<F>_l<L>_ip<K>``. This kernel reads the F-th fiber's
    stress within the L-th layer and integrates its contribution to
    the cut, using the fiber's tributary thickness (defaulted to
    ``t_layer / n_fibers_in_layer`` for uniform distribution) and its
    z_offset within the layer.

    Sum-of-fibers identity: summing every fiber's per-fiber cut within
    a layer recovers that layer's standard per-layer cut, when the
    fiber distribution is uniform. Layers without fiber decomposition
    raise — use :func:`compute_shell_cut_per_layer` for those.

    Parameters
    ----------
    layer_idx : int
        Layer index from 0 (bottom) to ``n_layers - 1`` (top).
    fiber_idx : int
        Fiber index within the layer, 0 to ``n_fibers_in_layer - 1``.

    Raises
    ------
    KeyError
        If a contributing shell's section can't be resolved.
    ValueError
        If the requested layer carries no fiber columns (use
        :func:`compute_shell_cut_per_layer` instead).
    IndexError
        If ``layer_idx`` or ``fiber_idx`` is out of range.
    """
    if not isinstance(layer_idx, (int, np.integer)) or int(layer_idx) < 0:
        raise ValueError(f"layer_idx must be a non-negative int, got {layer_idx!r}.")
    if not isinstance(fiber_idx, (int, np.integer)) or int(fiber_idx) < 0:
        raise ValueError(f"fiber_idx must be a non-negative int, got {fiber_idx!r}.")
    layer_idx = int(layer_idx)
    fiber_idx = int(fiber_idx)

    intersections = find_shell_intersections(dataset, spec, clipper=clipper)

    by_type: dict[str, list[ShellIntersection]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix)

    per_shell_F: dict[int, np.ndarray] = {}
    per_shell_M_at_mid: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None

    for elem_type, sub in by_type.items():
        eids = [ix.element_id for ix in sub]
        er = dataset.elements.get_element_results(
            results_name="section.fiber.stress",
            element_type=elem_type,
            model_stage=model_stage,
            element_ids=eids,
        )
        n_ip = int(er.n_ip) if er.n_ip > 0 else 0
        if n_ip == 0:
            raise ValueError(
                f"Shell {elem_type!r} has no integration points in the "
                "section.fiber.stress result."
            )
        base_type = _strip_class_tag(elem_type)
        for ix in sub:
            layers = _resolve_layer_table(dataset, ix.element_id)
            if layer_idx >= len(layers):
                raise IndexError(
                    f"layer_idx={layer_idx} out of range for element "
                    f"{ix.element_id}: section has {len(layers)} layers."
                )
            layer = layers[layer_idx]
            rows = er.df.xs(ix.element_id, level="element_id")
            n_fibers = _discover_fiber_count_in_layer(rows.columns, layer_idx)
            if n_fibers == 0:
                raise ValueError(
                    f"Element {ix.element_id}, layer {layer_idx}: no "
                    "fiber-in-layer columns (`_f<F>_l<L>_ip<K>`) "
                    "available. Use compute_shell_cut_per_layer for "
                    "non-fibered layers."
                )
            if fiber_idx >= n_fibers:
                raise IndexError(
                    f"fiber_idx={fiber_idx} out of range for element "
                    f"{ix.element_id} layer {layer_idx}: layer has "
                    f"{n_fibers} fibers."
                )
            fiber_stress = _read_fiber_in_layer_stress_array(
                rows, n_ip, layer_idx, fiber_idx,
            )
            # Uniform fiber distribution within the layer's thickness.
            # Fiber k of N gets thickness t_layer / N, centred at
            # z_layer + (k - (N-1)/2) * (t_layer / N).
            t_fiber = float(layer.thickness) / float(n_fibers)
            z_layer = float(layer.z_offset)
            z_fiber = z_layer + (fiber_idx - 0.5 * (n_fibers - 1)) * t_fiber
            fiber_layer_info = LayerInfo(
                material_id=int(layer.material_id),
                thickness=t_fiber,
                z_offset=z_fiber,
            )
            F, M = _shell_cut_per_element_for_layer(
                fiber_stress, fiber_layer_info, ix, dataset, spec, base_type,
            )
            per_shell_F[ix.element_id] = F
            per_shell_M_at_mid[ix.element_id] = M
        if time is None:
            time = np.asarray(er.time, dtype=float)

    if not intersections or time is None:
        return ShellCutResult(
            F=np.zeros((0, 3)),
            M=np.zeros((0, 3)),
            time=np.zeros((0,)),
            centroid=np.zeros(3),
            intersections=(),
        )

    centroid = np.mean(
        np.stack([ix.chord_midpoint for ix in intersections], axis=0), axis=0
    )
    F_total = np.zeros((time.shape[0], 3), dtype=float)
    M_total = np.zeros_like(F_total)
    for ix in intersections:
        Fi = per_shell_F[ix.element_id]
        Mi = per_shell_M_at_mid[ix.element_id]
        arm = ix.chord_midpoint - centroid
        F_total += Fi
        M_total += Mi + np.cross(arm, Fi)

    return ShellCutResult(
        F=F_total,
        M=M_total,
        time=time,
        centroid=centroid,
        intersections=tuple(intersections),
        per_shell_F=per_shell_F,
        per_shell_M_at_midpoint=per_shell_M_at_mid,
    )


def _shell_cut_per_element_for_layer(
    layer_stress: np.ndarray,
    layer,
    intersection: ShellIntersection,
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    base_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate one layer's traction along the chord of one shell.

    Parallel to :func:`_shell_cut_per_element` — same chord, same
    Gauss rule, same cut-normal orientation — but the 8-vector input
    is built from the layer's Voigt stress, its thickness, and its
    through-thickness offset rather than the recorded
    ``section.force``.
    """
    chord = intersection.chord_endpoints_arr
    L = intersection.chord_length
    p_mid = intersection.chord_midpoint

    n_shell = _midsurface_normal(intersection.midsurface_polygon_global)
    chord_dir = chord[1] - chord[0]
    chord_dir /= np.linalg.norm(chord_dir)
    n_cut_global = np.cross(chord_dir, n_shell)
    n_cut_norm = float(np.linalg.norm(n_cut_global))
    if n_cut_norm < 1e-12:
        return (
            np.zeros((layer_stress.shape[0], 3)),
            np.zeros((layer_stress.shape[0], 3)),
        )
    n_cut_global /= n_cut_norm
    n_cut_global = _orient_cut_normal(
        n_cut_global, intersection.midsurface_polygon_global, p_mid, spec,
    )

    R = dataset.cdata.rotation_matrix(intersection.element_id)
    n_cut_local = R.T @ n_cut_global
    n_x = float(n_cut_local[0])
    n_y = float(n_cut_local[1])

    gs, gw = _GAUSS_LEGENDRE_2
    xi_a, eta_a = intersection.chord_param_natural[0]
    xi_b, eta_b = intersection.chord_param_natural[1]
    half_L = 0.5 * L

    n_steps = layer_stress.shape[0]
    F_local = np.zeros((n_steps, 3), dtype=float)
    M_local = np.zeros((n_steps, 3), dtype=float)

    t_k = float(layer.thickness)
    z_k = float(layer.z_offset)

    for s, w in zip(gs, gw):
        alpha = 0.5 * (s + 1.0)
        xi_g = (1 - alpha) * xi_a + alpha * xi_b
        eta_g = (1 - alpha) * eta_a + alpha * eta_b
        sigma = _sample_layer_stress(layer_stress, xi_g, eta_g, base_type)
        # Voigt order matches _LAYER_STRESS_SHORTNAMES:
        # (sigma11, sigma22, sigma33, sigma12, sigma23, sigma13)
        s11 = sigma[:, 0]
        s22 = sigma[:, 1]
        # s33 is the through-thickness normal stress — not used in the
        # shell section-force 8-vector (vanishes in classical plate
        # theory and isn't part of (Fxx, Fyy, Fxy, ...)).
        s12 = sigma[:, 3]
        s23 = sigma[:, 4]
        s13 = sigma[:, 5]
        # Build the layer's contribution to the standard 8-vector and
        # apply the same traction formulas as the full-section kernel:
        #     Fxx^(k) = s11 * t_k
        #     Mxx^(k) = s11 * t_k * z_k
        # ... and so on. Substitute into the per-unit-length traction
        # formulas.
        Fxx = s11 * t_k
        Fyy = s22 * t_k
        Fxy = s12 * t_k
        Mxx = s11 * t_k * z_k
        Myy = s22 * t_k * z_k
        Mxy = s12 * t_k * z_k
        Vxz = s13 * t_k
        Vyz = s23 * t_k
        f_per_unit_x = Fxx * n_x + Fxy * n_y
        f_per_unit_y = Fxy * n_x + Fyy * n_y
        f_per_unit_z = Vxz * n_x + Vyz * n_y
        m_per_unit_x = Mxx * n_x + Mxy * n_y
        m_per_unit_y = Mxy * n_x + Myy * n_y
        weight = w * half_L
        F_local[:, 0] += weight * f_per_unit_x
        F_local[:, 1] += weight * f_per_unit_y
        F_local[:, 2] += weight * f_per_unit_z
        M_local[:, 0] += weight * m_per_unit_x
        M_local[:, 1] += weight * m_per_unit_y

    F_global = F_local @ R.T
    M_global = M_local @ R.T
    return F_global, M_global


def _midsurface_normal(verts: np.ndarray) -> np.ndarray:
    """Unit normal of a 3- or 4-node midsurface polygon.

    Cross-product of the first two non-degenerate edges. The polygon
    is assumed to be ordered (CCW or CW); a flipped normal is fine —
    the side-orientation step :func:`_orient_cut_normal` independently
    fixes the cut normal's sign.
    """
    e1 = verts[1] - verts[0]
    e2 = verts[2] - verts[0]
    n = np.cross(e1, e2)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-12:
        # Try the next edge pair — degenerate triangle at this vertex.
        if verts.shape[0] >= 4:
            e3 = verts[3] - verts[0]
            n = np.cross(e1, e3)
            n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-12:
            raise RuntimeError(
                f"Cannot compute shell midsurface normal — degenerate "
                f"vertex layout: {verts.tolist()}"
            )
    return n / n_norm


def _orient_cut_normal(
    n_cut_global: np.ndarray,
    midsurface_verts: np.ndarray,
    p_mid: np.ndarray,
    spec: "SectionCutSpec",
) -> np.ndarray:
    """Choose the sign of ``n_cut_global`` so it points away from the
    kept side (i.e., toward the discarded side).

    For a normal cut where the polygon crosses the plane interior, the
    "most-clearly-kept" vertex has ``d > 0`` and ``n_cut`` is flipped
    if it points toward that vertex. For an edge-coincident cut where
    the polygon's interior lies entirely on the discarded side (only
    case admitted by the side-aware filter in
    :func:`find_shell_intersections`), the polygon's farthest-from-the-
    plane vertex sits on the discarded side; ``n_cut`` is flipped if
    it points AWAY from that vertex.

    Both branches yield the same convention: ``n_cut · (kept_side -
    p_mid) < 0``.
    """
    signed_n = spec.signed_normal
    rel = midsurface_verts - np.asarray(spec.plane.point, dtype=float)
    d = rel @ signed_n  # (n_nodes,)
    idx = int(np.argmax(d))
    if d[idx] > 0.0:
        # Normal case: there is a strictly-kept vertex. n_cut should
        # point AWAY from it.
        offset = midsurface_verts[idx] - p_mid
        if offset @ n_cut_global > 0:
            return -n_cut_global
        return n_cut_global
    # Edge-coincident case: polygon interior is on the discarded side
    # (the side-aware filter guarantees at least one vertex has d<0).
    # Use the most-clearly-discarded vertex; n_cut must point TOWARD
    # it. Equivalent statement: flip when n_cut points AWAY.
    idx = int(np.argmin(d))
    offset = midsurface_verts[idx] - p_mid  # has component on discarded side
    if offset @ n_cut_global < 0:
        return -n_cut_global
    return n_cut_global
