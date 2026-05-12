"""Beam local-axis fallback — OpenSees ``vecxz`` convention.

Adapted from apeGmsh ``viewers/diagrams/_beam_geometry.py``.

This module is the **fallback** path for computing a beam's local frame
from just its endpoint coordinates and an optional ``vecxz`` reference
vector. The **primary** path in STKO_to_python — used by
``plotting/beam_solid.py``, ``cuts/kernels/beam_resultant.py``, and
``cuts/kernels/shell.py`` — is the quaternion stored in
``.cdata`` ``*LOCAL_AXES``, read by
:meth:`STKO_to_python.model.cdata_reader.CDataReader.rotation_matrix`.
The quaternion path is more accurate (it's the exact frame OpenSees
used during analysis, including Corotational and user-supplied
``vecxz``).

Use this module when:

* The dataset has no `.cdata` sidecar (external inputs, partial fixtures).
* You only have endpoint coordinates (custom datasets, unit tests).
* You need a *recomputed* frame for a hypothetical beam (e.g. a
  Phase 3 `DiagramLayer` rendering against a user-supplied vector).

Convention (OpenSees ``Linear`` / ``PDelta`` / ``Corotational``
``geomTransf``):

* ``x_local = (j - i) / |j - i|``
* ``z_local = (vecxz - (vecxz · x_local) * x_local) / |…|``
  (Gram-Schmidt: the part of ``vecxz`` perpendicular to ``x_local``).
* ``y_local = z_local × x_local`` (right-hand rule).

The user-supplied ``vecxz`` is a vector lying in the local x-z plane.
When omitted, :func:`default_vecxz` mimics the typical structural
default: global Z for non-vertical beams, global X for vertical.

This module is pure numpy. It does **not** depend on
``MPCODataSet`` — a higher-level adapter that prefers the quaternion
path when available will land in ``viewer/core/datasource.py`` in
Phase 2.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# Above this |dot| with global +Z the beam axis is treated as vertical
# and ``default_vecxz`` switches to global +X so Gram-Schmidt has a
# non-parallel reference.
_VERTICAL_TOL: float = 0.99

# Below this magnitude a vector is treated as numerically zero. The
# value matches apeGmsh; ``1e-12`` is safely below any structural-scale
# coordinate magnitude we expect.
_DEGENERATE_EPS: float = 1e-12


__all__ = [
    "default_vecxz",
    "compute_local_axes",
    "station_position",
]


def default_vecxz(x_local: ndarray) -> ndarray:
    """Sensible fallback ``vecxz`` when the user provides none.

    Args:
        x_local: Unit vector along the beam's local x-axis.

    Returns:
        ``[0, 0, 1]`` (global Z) for a beam that is not (nearly)
        parallel to global Z, otherwise ``[1, 0, 0]`` (global X). The
        switch at the vertical case ensures the subsequent Gram-Schmidt
        in :func:`compute_local_axes` produces a non-degenerate
        ``z_local``.
    """
    x = np.asarray(x_local, dtype=np.float64)
    if abs(np.dot(x, np.array([0.0, 0.0, 1.0]))) > _VERTICAL_TOL:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def compute_local_axes(
    coord_i: ndarray,
    coord_j: ndarray,
    vecxz: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray, float]:
    """Return ``(x_local, y_local, z_local, length)`` for a beam.

    All returned axes are unit vectors. ``(x, y, z)`` is right-handed.

    Args:
        coord_i: Endpoint coordinates of node ``i`` — ``(3,)``.
        coord_j: Endpoint coordinates of node ``j`` — ``(3,)``.
        vecxz: Reference vector lying in the local x-z plane. ``None``
            selects :func:`default_vecxz`. When the user-supplied
            vector is (almost) parallel to ``x_local``, the routine
            silently falls back to :func:`default_vecxz` and only
            raises if that *also* fails (unreachable in practice — see
            note below).

    Returns:
        ``(x_local, y_local, z_local, length)``:

        * ``x_local`` — unit vector along ``(j - i)``.
        * ``y_local`` — unit vector perpendicular to ``x_local``,
          obtained via right-hand rule ``z × x``.
        * ``z_local`` — unit vector perpendicular to ``x_local``,
          aligned with the projection of ``vecxz`` onto the plane
          perpendicular to ``x_local``.
        * ``length`` — Euclidean distance from ``i`` to ``j``.

    Raises:
        ValueError: When endpoints coincide (zero beam length), or
            when both the user's ``vecxz`` and :func:`default_vecxz`
            are parallel to ``x_local``. The second case is defensive
            only — :func:`default_vecxz` returns a vector perpendicular
            to whichever cardinal axis ``x_local`` aligns with, so the
            inner failure cannot be reached in practice.
    """
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    raw_x = cj - ci
    L = float(np.linalg.norm(raw_x))
    if L <= _DEGENERATE_EPS:
        raise ValueError(
            "Beam endpoints coincide — cannot build a local frame."
        )
    x_local = raw_x / L

    if vecxz is None:
        vecxz = default_vecxz(x_local)
    v = np.asarray(vecxz, dtype=np.float64)

    # z_local: Gram-Schmidt — part of vecxz perpendicular to x_local.
    proj = float(np.dot(v, x_local)) * x_local
    z_raw = v - proj
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < _DEGENERATE_EPS:
        # vecxz parallel to x_local — try the default reference.
        fallback = default_vecxz(x_local)
        proj = float(np.dot(fallback, x_local)) * x_local
        z_raw = fallback - proj
        z_norm = float(np.linalg.norm(z_raw))
        if z_norm < _DEGENERATE_EPS:
            raise ValueError(
                "Could not derive a non-degenerate z_local; "
                f"x_local={x_local}, vecxz={v}."
            )
    z_local = z_raw / z_norm

    # y_local: right-hand rule.
    y_local = np.cross(z_local, x_local)
    y_local /= float(np.linalg.norm(y_local))

    return x_local, y_local, z_local, L


def station_position(
    coord_i: ndarray,
    coord_j: ndarray,
    natural_coord: float,
) -> ndarray:
    """Position along the beam at a natural coordinate ``ξ ∈ [-1, +1]``.

    Args:
        coord_i: Endpoint coordinates of node ``i`` — ``(3,)``.
        coord_j: Endpoint coordinates of node ``j`` — ``(3,)``.
        natural_coord: Natural coordinate along the beam axis. ``-1``
            maps to node ``i``; ``+1`` maps to node ``j``. Values
            outside ``[-1, +1]`` are allowed and extrapolate linearly
            — the function does not clamp.

    Returns:
        ``(3,)`` global-coordinate position. Linear interpolation
        matches OpenSees integration-point placement on a 1-D parent
        element with linear shape functions.
    """
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    t = (1.0 + float(natural_coord)) / 2.0
    return ci + t * (cj - ci)
