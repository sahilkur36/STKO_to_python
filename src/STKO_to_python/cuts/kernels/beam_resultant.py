"""Beam section-cut resultant — read internal forces at intersections and sum.

Companion to ``cuts/kernels/beam.py``. The geometry phase finds *where*
the plane crosses each beam; this module reads the internal force at
each intersection, transforms to global, and aggregates the total
``(F, M)`` resultant about the cut centroid.

Two element families are supported in v1:

- **Closed-form elastic beams** (``ElasticBeam3d``). Recorded as
  ``force`` — 12 components per element, already in **global frame**
  (verified to satisfy ``force == R @ localForce`` on the elasticFrame
  fixture). With no element load between ends, the internal force at
  any station is computed from statics of the end forces.
- **Line-station beams** (``DispBeamColumn3d``, ``ForceBeamColumn3d``,
  ``MixedBeamColumn3d``). Recorded as ``section.force`` — 6 components
  per integration point in **section/element local frame**. We linearly
  interpolate between bracketing IPs to the cut's natural coordinate,
  then rotate to global via ``cdata.rotation_matrix(eid)``.

Sign convention
---------------
The cut resultant is **the force the discarded side exerts on the kept
side** (action-reaction; standard structural free-body convention).
``spec.side`` chooses which side is "kept":

- ``"positive"`` — the side the plane normal points into.
- ``"negative"`` — the opposite side; ``spec.signed_normal`` flips the
  normal so kernels can treat positive as kept without branching.

For each beam, the kept side is identified by the signed distance of
the first node (node 1, i-node) from the plane. If node 1 lies on the
kept side, the cut force/moment is read directly from the section's
natural convention; otherwise both are negated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .beam import BEAM_ELEMENT_CLASSES, BeamIntersection, _strip_class_tag, find_beam_intersections

if TYPE_CHECKING:
    from ..specs import SectionCutSpec
    from ...core.dataset import MPCODataSet


# Element types using the closed-form ``force`` recorder (no integration
# points). All other entries in BEAM_ELEMENT_CLASSES route to
# ``section.force``.
_CLOSED_FORM_BEAMS: frozenset[str] = frozenset({"ElasticBeam3d"})


@dataclass(frozen=True)
class BeamCutResult:
    """Output of :func:`compute_beam_cut`.

    Attributes
    ----------
    F : np.ndarray
        ``(n_steps, 3)`` — total force the discarded side exerts on the
        kept side, summed across all beams crossing the cut. Global frame.
    M : np.ndarray
        ``(n_steps, 3)`` — total moment about ``centroid``, in global frame.
    time : np.ndarray
        ``(n_steps,)`` — time axis aligned with the rows of ``F`` / ``M``.
    centroid : np.ndarray
        ``(3,)`` — reference point for the moment summation. Set to the
        unweighted average of the per-beam intersection points so that a
        single-beam cut reports zero arm contribution by construction.
    intersections : tuple[BeamIntersection, ...]
        Per-beam intersection records (sorted by element_id).
    per_beam_F, per_beam_M_at_intersection : dict[int, np.ndarray]
        Each maps ``element_id`` to a ``(n_steps, 3)`` array. ``per_beam_F``
        is the cut force contribution; ``per_beam_M_at_intersection`` is
        the moment contribution evaluated **at the intersection point**
        (without the moment-arm transfer to the centroid).
    """

    F: np.ndarray
    M: np.ndarray
    time: np.ndarray
    centroid: np.ndarray
    intersections: tuple[BeamIntersection, ...]
    per_beam_F: dict[int, np.ndarray] = field(default_factory=dict)
    per_beam_M_at_intersection: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def is_empty(self) -> bool:
        return len(self.intersections) == 0


def compute_beam_cut(
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
    *,
    model_stage: str,
) -> BeamCutResult:
    """Compute the (F, M) resultant of a section cut through beams.

    Parameters
    ----------
    dataset:
        Source data.
    spec:
        Cut specification (plane + filter + side).
    model_stage:
        Model stage to read from (e.g. ``"MODEL_STAGE[1]"``).

    Returns
    -------
    BeamCutResult
        Empty result (``F``/``M`` shape ``(0, 3)``) when no beam in the
        filter crosses the plane.
    """
    intersections = _deduplicate_at_node_intersections(
        find_beam_intersections(dataset, spec)
    )

    # 2. Group by element type so we can batch a single result fetch per type.
    by_type: dict[str, list[BeamIntersection]] = {}
    for ix in intersections:
        by_type.setdefault(ix.element_type, []).append(ix)

    per_beam_F: dict[int, np.ndarray] = {}
    per_beam_M_at_int: dict[int, np.ndarray] = {}
    time: np.ndarray | None = None

    for elem_type, sub in by_type.items():
        base = _strip_class_tag(elem_type)
        eids = [ix.element_id for ix in sub]
        if base in _CLOSED_FORM_BEAMS:
            er = dataset.elements.get_element_results(
                results_name="force",
                element_type=elem_type,
                model_stage=model_stage,
                element_ids=eids,
            )
            for ix in sub:
                F, M = _elastic_beam_cut(er, ix, dataset, spec)
                per_beam_F[ix.element_id] = F
                per_beam_M_at_int[ix.element_id] = M
        else:
            er = dataset.elements.get_element_results(
                results_name="section.force",
                element_type=elem_type,
                model_stage=model_stage,
                element_ids=eids,
            )
            for ix in sub:
                F, M = _section_force_cut(er, ix, dataset, spec)
                per_beam_F[ix.element_id] = F
                per_beam_M_at_int[ix.element_id] = M
        if time is None:
            time = np.asarray(er.time, dtype=float)

    if not intersections or time is None:
        return BeamCutResult(
            F=np.zeros((0, 3)),
            M=np.zeros((0, 3)),
            time=np.zeros((0,)),
            centroid=np.zeros(3),
            intersections=(),
        )

    # 3. Aggregate. Centroid is the unweighted average of intersection points.
    centroid = np.mean(
        np.stack([ix.point_arr for ix in intersections], axis=0), axis=0
    )

    F_total = np.zeros((time.shape[0], 3), dtype=float)
    M_total = np.zeros_like(F_total)
    for ix in intersections:
        Fi = per_beam_F[ix.element_id]
        Mi = per_beam_M_at_int[ix.element_id]
        arm = ix.point_arr - centroid  # (3,)
        # np.cross broadcasts (3,) against (n_steps, 3) row-wise.
        M_arm = np.cross(arm, Fi)
        F_total += Fi
        M_total += Mi + M_arm

    return BeamCutResult(
        F=F_total,
        M=M_total,
        time=time,
        centroid=centroid,
        intersections=tuple(intersections),
        per_beam_F=per_beam_F,
        per_beam_M_at_intersection=per_beam_M_at_int,
    )


# ---------------------------------------------------------------------- #
# Intersection deduplication
# ---------------------------------------------------------------------- #
def _deduplicate_at_node_intersections(
    intersections: list[BeamIntersection], *, tol: float = 1e-6,
) -> list[BeamIntersection]:
    """Drop intersections that share a global point with another.

    When a cut plane lands exactly on a node shared by two beams (e.g.
    the boundary between meshed column segments at z=2000), both
    adjacent elements report an intersection at the same global point.
    Without deduplication the kernel sums both contributions and
    double-counts the cut force.

    For each cluster of intersections within ``tol * (1 + |x|)`` of one
    another, keep the one with the smallest ``element_id`` — both
    elements share the same node by continuity, so either's section
    force is correct.
    """
    if len(intersections) <= 1:
        return list(intersections)
    keep: list[BeamIntersection] = []
    for cand in sorted(intersections, key=lambda ix: ix.element_id):
        is_duplicate = False
        for kept in keep:
            scale = 1.0 + float(np.linalg.norm(kept.point_arr))
            if np.linalg.norm(cand.point_arr - kept.point_arr) < tol * scale:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(cand)
    return keep


# ---------------------------------------------------------------------- #
# Per-beam kernels
# ---------------------------------------------------------------------- #
def _kept_side_is_node1(
    intersection: BeamIntersection, spec: "SectionCutSpec"
) -> bool:
    """Is the i-node of the beam on the kept side of the cut?

    The signed distance of node 1 from the plane, projected onto
    ``spec.signed_normal`` (which already accounts for ``side``), tells
    us which side of the cut each end of the beam lives on. The cut's
    sign convention follows: if node 1 is on the kept side, the section
    force from the section's natural ``+x → -x`` convention already
    points the right way; otherwise we negate.
    """
    x_node1 = np.asarray(intersection.end_coords[0], dtype=float)
    x_int = intersection.point_arr
    d1 = float(np.dot(x_node1 - x_int, spec.signed_normal))
    return d1 > 0.0


def _elastic_beam_cut(
    er,
    intersection: BeamIntersection,
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form cut force/moment for an ``ElasticBeam3d``.

    Uses the global-frame ``force`` recorder. The 12 columns are
    ``(Px_1, Py_1, Pz_1, Mx_1, My_1, Mz_1, Px_2, ..., Mz_2)``.

    For an elastic beam with no element load between the ends, the
    internal force at any station is constant along the beam, and the
    internal moment varies linearly with the distance from the i-node.
    Returns ``(F, M)`` both in global frame at the intersection point.
    """
    eid = intersection.element_id
    rows = er.df.xs(eid, level="element_id")
    P_1 = rows[["Px_1", "Py_1", "Pz_1"]].to_numpy(dtype=float)
    Mcc_1 = rows[["Mx_1", "My_1", "Mz_1"]].to_numpy(dtype=float)

    x_node1 = np.asarray(intersection.end_coords[0], dtype=float)
    x_int = intersection.point_arr
    offset = x_int - x_node1  # (3,)

    # F_sec_+_to_- (force +x portion exerts on -x portion) = -P_1.
    # M_sec_+_to_- = -Mcc_1 + (x_int - x_node1) × P_1, evaluated at the
    # cut point (couple is point-of-application invariant; force arm is
    # measured from the cut).
    M_sec_plus_to_minus = -Mcc_1 + np.cross(offset, P_1)

    if _kept_side_is_node1(intersection, spec):
        # kept = -x part; cut force/moment is the natural +→- value.
        F_cut = -P_1
        M_cut_at_int = M_sec_plus_to_minus
    else:
        F_cut = +P_1
        M_cut_at_int = -M_sec_plus_to_minus
    return F_cut, M_cut_at_int


def _section_force_cut(
    er,
    intersection: BeamIntersection,
    dataset: "MPCODataSet",
    spec: "SectionCutSpec",
) -> tuple[np.ndarray, np.ndarray]:
    """Cut force/moment from ``section.force`` at the intersection xi.

    Thin glue: pulls section forces and rotation matrix from the broker,
    augments shear from moment gradients when the section doesn't emit
    it explicitly, then delegates to the pure-math interpolation +
    rotation helpers.
    """
    eid = intersection.element_id
    gp_xi = np.asarray(er.gp_xi, dtype=float)
    n_ip = int(er.n_ip)
    if n_ip == 0 or gp_xi.size == 0:
        raise ValueError(
            f"Element {eid} ({intersection.element_type}) has no integration "
            "points but was routed through the section.force path. This "
            "likely indicates a missing section.force recorder."
        )

    rows = er.df.xs(eid, level="element_id")
    section_force, has_vy, has_vz = _read_section_force_array(rows, n_ip)
    _augment_shear_from_moment_gradients(
        section_force, gp_xi, intersection.axis_length,
        has_vy=has_vy, has_vz=has_vz,
    )

    f_local = _interpolate_section_force_at_xi(
        section_force, gp_xi, intersection.xi
    )

    R = dataset.cdata.rotation_matrix(eid)
    return _rotate_section_force_and_apply_side(
        f_local, R, _kept_side_is_node1(intersection, spec)
    )


# ---------------------------------------------------------------------- #
# Pure-math helpers (testable without I/O)
# ---------------------------------------------------------------------- #
# OpenSees section.force may omit any subset of these components — most
# fiber sections report only (P, Mz, My, T) since shear is computed from
# moment equilibrium and not stored as a separate section state. The
# kernel slots each available shortname into the 6-vector position below;
# missing components default to zero.
_SECTION_FORCE_POSITIONS: dict[str, int] = {
    "P": 0,    # axial
    "Vy": 1,   # shear along local y
    "Vz": 2,   # shear along local z
    "T": 3,    # torsion about local x
    "My": 4,   # bending about local y
    "Mz": 5,   # bending about local z
}


def _read_section_force_array(rows, n_ip: int) -> tuple[np.ndarray, bool, bool]:
    """Extract a ``(n_steps, n_ip, 6)`` array from a section.force DataFrame.

    ``rows`` is a DataFrame slice for one element (index by step). The
    target 6-vector is ``(N, Vy, Vz, T, My, Mz)`` in element-local frame.

    OpenSees sections vary in which components they report under
    ``section.force``: ``ElasticSection3d`` / ``FiberSection3d`` emit
    ``(P, Mz, My, T)``, while sections with explicit shear deformation
    emit all six. Missing columns default to zero; the caller is
    expected to augment shear from moment gradients when the section
    didn't record it explicitly.

    Returns
    -------
    (section_force, has_vy, has_vz)
        section_force shape ``(n_steps, n_ip, 6)``; ``has_vy`` and
        ``has_vz`` indicate whether the corresponding shear shortnames
        were present in the DataFrame columns (so the caller can decide
        whether to augment).
    """
    n_steps = rows.shape[0]
    out = np.zeros((n_steps, n_ip, 6), dtype=float)
    available = set(rows.columns)
    has_vy = any(f"Vy_ip{k}" in available for k in range(n_ip))
    has_vz = any(f"Vz_ip{k}" in available for k in range(n_ip))
    for k in range(n_ip):
        for shortname, pos in _SECTION_FORCE_POSITIONS.items():
            col = f"{shortname}_ip{k}"
            if col in available:
                out[:, k, pos] = rows[col].to_numpy(dtype=float)
    return out, has_vy, has_vz


def _augment_shear_from_moment_gradients(
    section_force: np.ndarray,
    gp_xi: np.ndarray,
    L: float,
    *,
    has_vy: bool,
    has_vz: bool,
) -> None:
    """Fill in shear components from the moment gradient when absent.

    OpenSees fiber/elastic 3D sections don't carry shear as a section
    state — they record ``(P, Mz, My, T)`` only. The shear is recovered
    from beam equilibrium of a differential slice (no distributed load):

        dM/dx_local + e_x_local × V_local = 0

    Component-wise that gives ``V_z = dM_y/dx_local`` and
    ``V_y = -dM_z/dx_local``. Mapping ``x_local`` to the natural
    coordinate ``ξ ∈ [-1, +1]`` via ``x_local = (ξ + 1) L / 2`` gives the
    scaling factor ``dξ/dx_local = 2/L``.

    Modifies ``section_force`` in place. No-op if both shears are
    already present (``has_vy`` and ``has_vz`` both True).

    Notes
    -----
    The gradient is computed via :func:`numpy.gradient` which uses
    second-order central differences for interior IPs and first-order
    forward/backward at the endpoints. For piecewise-linear section
    moments (typical of displacement-based beams with elastic sections),
    this recovers the exact constant shear within each element.
    """
    if has_vy and has_vz:
        return
    if L <= 0.0:
        # Defensive: shouldn't happen for a real intersection (the kernel
        # only constructs intersections from real elements), but if we
        # ever get a zero-length beam there's no meaningful gradient.
        return
    # Layout: section_force[:, :, pos] with pos from _SECTION_FORCE_POSITIONS.
    My = section_force[:, :, _SECTION_FORCE_POSITIONS["My"]]
    Mz = section_force[:, :, _SECTION_FORCE_POSITIONS["Mz"]]
    scale = 2.0 / L
    if not has_vz:
        dMy_dxi = np.gradient(My, gp_xi, axis=1)
        section_force[:, :, _SECTION_FORCE_POSITIONS["Vz"]] = scale * dMy_dxi
    if not has_vy:
        dMz_dxi = np.gradient(Mz, gp_xi, axis=1)
        section_force[:, :, _SECTION_FORCE_POSITIONS["Vy"]] = -scale * dMz_dxi


def _interpolate_section_force_at_xi(
    section_force: np.ndarray,
    gp_xi: np.ndarray,
    xi: float,
) -> np.ndarray:
    """Linear interpolation between bracketing IPs.

    Parameters
    ----------
    section_force : (n_steps, n_ip, 6) np.ndarray
        Section force per timestep per IP in element-local frame; columns
        ordered ``(N, Vy, Vz, T, My, Mz)``.
    gp_xi : (n_ip,) np.ndarray
        Natural coordinates of the IPs, ascending in ``[-1, +1]``.
    xi : float
        Natural coordinate at which to evaluate.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps, 6)`` — interpolated section force in local frame.
        Constant extrapolation when ``xi`` falls outside the IP envelope
        (useful at element ends when the integration rule omits ξ = ±1,
        e.g. Gauss-Legendre 2-point).
    """
    n_ip = gp_xi.size
    if n_ip == 0:
        raise ValueError("gp_xi is empty — no integration points to interpolate.")
    if section_force.shape[1] != n_ip:
        raise ValueError(
            f"section_force shape {section_force.shape} disagrees with "
            f"gp_xi size {n_ip}."
        )

    if xi <= gp_xi[0]:
        return section_force[:, 0, :].copy()
    if xi >= gp_xi[-1]:
        return section_force[:, -1, :].copy()

    a_idx = int(np.searchsorted(gp_xi, xi, side="right") - 1)
    b_idx = a_idx + 1
    denom = float(gp_xi[b_idx] - gp_xi[a_idx])
    alpha = float((xi - gp_xi[a_idx]) / denom)
    return (1.0 - alpha) * section_force[:, a_idx, :] + alpha * section_force[:, b_idx, :]


def _rotate_section_force_and_apply_side(
    f_local: np.ndarray,
    R: np.ndarray,
    kept_side_is_node1: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``(n_steps, 6)`` local section force to global ``(F, M)``.

    The OpenSees ``section.force`` convention reports the section
    internal force as ``F_sec_+→-`` — the force the +x face applies to
    the -x face — in element-local frame. After rotating to global, the
    cut resultant (force the discarded side exerts on the kept side) is:

    - ``+F_sec_+→-`` when node 1 is on the kept side (cut force equals
      what the +x = discarded side applies to the -x = kept side),
    - ``-F_sec_+→-`` when node 2 is on the kept side (Newton 3rd law).

    Parameters
    ----------
    f_local : (n_steps, 6) np.ndarray
        Section force in local frame ``(N, Vy, Vz, T, My, Mz)``.
    R : (3, 3) np.ndarray
        Local-to-global rotation matrix: ``v_global = R @ v_local``.
    kept_side_is_node1 : bool
        Whether node 1 of the beam lies on the kept side of the cut.

    Returns
    -------
    (F, M) tuple of (n_steps, 3) np.ndarray
        Cut force and moment at the intersection point, in global frame.
    """
    F_local = f_local[:, 0:3]
    M_local = f_local[:, 3:6]
    # Batched rotation: v_global_row = R @ v_local_row, expressed as
    # right-multiplication by R.T over the row axis.
    F_global = F_local @ R.T
    M_global = M_local @ R.T
    if kept_side_is_node1:
        return F_global, M_global
    return -F_global, -M_global


def _section_force_columns(ip_idx: int) -> list[str]:
    """Column names for a single integration point's section force.

    Convention (matches ``ElementResults.at_ip``): ``P_ip<k>``, ``Vy_ip<k>``,
    ``Vz_ip<k>``, ``T_ip<k>``, ``My_ip<k>``, ``Mz_ip<k>``.
    """
    return [
        f"P_ip{ip_idx}",
        f"Vy_ip{ip_idx}",
        f"Vz_ip{ip_idx}",
        f"T_ip{ip_idx}",
        f"My_ip{ip_idx}",
        f"Mz_ip{ip_idx}",
    ]
