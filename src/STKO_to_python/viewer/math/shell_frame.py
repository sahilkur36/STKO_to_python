"""Shell local-frame fallback + rotation-matrix ↔ quaternion utility.

Adapted from apeGmsh ``results/_shell_geometry.py``.

The **primary** shell-local-frame path in STKO_to_python — used by
``cuts/kernels/shell.py`` and any other consumer that has a dataset
in hand — is the per-element quaternion from STKO's `.cdata`
``*LOCAL_AXES`` section, read by
:meth:`STKO_to_python.model.cdata_reader.CDataReader.rotation_matrix`.
That quaternion is the exact frame OpenSees used during analysis.

This module is the **fallback** path: it derives the same frame from
just the element's corner-node coordinates. Use it when:

* The dataset has no `.cdata` sidecar (external datasets, partial
  fixtures).
* You only have node coordinates (custom inputs, unit tests).
* You want to *recompute* the frame for a hypothetical shell (e.g. a
  Phase 3 layer-stack viewer rendering against synthetic geometry).

Convention (matches OpenSees ASDShellQ4 / ShellMITC4 / etc.):

* Quads: ``x_local = (n2 − n1) / |…|``, ``z_local = (e1 × e2) / |…|``
  where ``e2 = n4 − n1``; ``y_local = z_local × x_local``.
* Triangles: same recipe with ``e2 = n3 − n1``.
* 9-node shells: use corner nodes 1–4 (internal nodes don't affect the
  element-local frame).

The returned ``R`` is shape ``(3, 3)`` with **columns** equal to the
local basis vectors expressed in the global frame — matching STKO's
existing :func:`STKO_to_python.quaternion_to_rotation_matrix` so
``v_global = R @ v_local``.

This module is pure numpy. It does **not** depend on
``MPCODataSet``. A higher-level adapter that prefers the
quaternion-from-`.cdata` path when available will land in
``viewer/core/datasource.py`` in Phase 2.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# Shell class lists kept in sync with ``cuts/kernels/shell.py``.
_QUAD_SHELL_CLASSES: frozenset[str] = frozenset(
    {"ShellMITC4", "ShellDKGQ", "ShellNLDKGQ", "ASDShellQ4"}
)
_TRI_SHELL_CLASSES: frozenset[str] = frozenset(
    {"ShellDKGT", "ShellNLDKGT", "ASDShellT3"}
)
_QUAD9_SHELL_CLASSES: frozenset[str] = frozenset({"ShellMITC9"})


# Below this norm, vectors are treated as numerically degenerate.
_DEGENERATE_EPS: float = 1e-14


__all__ = [
    "shell_local_axes",
    "rotation_matrix_to_quaternion",
    "shell_quaternion",
]


def _frame_edges(
    node_coords: ndarray, class_name: str,
) -> tuple[ndarray, ndarray]:
    """Pick the two edge vectors that define the local frame.

    Quads (incl. 9-node): ``(n2 − n1, n4 − n1)``.
    Triangles: ``(n2 − n1, n3 − n1)``.
    """
    if class_name in _QUAD_SHELL_CLASSES or class_name in _QUAD9_SHELL_CLASSES:
        if node_coords.shape[0] < 4:
            raise ValueError(
                f"{class_name} expects ≥4 corner nodes; got "
                f"node_coords.shape={node_coords.shape}."
            )
        n1 = node_coords[0]
        n2 = node_coords[1]
        n4 = node_coords[3]
        return (n2 - n1, n4 - n1)
    if class_name in _TRI_SHELL_CLASSES:
        if node_coords.shape[0] < 3:
            raise ValueError(
                f"{class_name} expects ≥3 corner nodes; got "
                f"node_coords.shape={node_coords.shape}."
            )
        n1 = node_coords[0]
        n2 = node_coords[1]
        n3 = node_coords[2]
        return (n2 - n1, n3 - n1)
    raise ValueError(
        f"Unsupported shell class for local axes: {class_name!r}. "
        f"Known classes: {sorted(_QUAD_SHELL_CLASSES | _TRI_SHELL_CLASSES | _QUAD9_SHELL_CLASSES)}."
    )


def shell_local_axes(
    node_coords: ndarray, class_name: str,
) -> ndarray:
    """Return the ``(3, 3)`` rotation matrix for a shell's local frame.

    Args:
        node_coords: ``(n_nodes, 3)`` global coordinates of the shell's
            corner nodes, in OpenSees connectivity order.
        class_name: OpenSees shell class string (e.g.
            ``"ASDShellQ4"``). Class names with the MPCO tag prefix
            (e.g. ``"203-ASDShellQ4"``) should be stripped by the
            caller before calling.

    Returns:
        Rotation matrix ``R`` of shape ``(3, 3)`` with **columns**
        equal to the local basis vectors in global coordinates::

            R[:, 0] = x_local in global frame
            R[:, 1] = y_local in global frame
            R[:, 2] = z_local in global frame

        Matches the convention of
        :func:`STKO_to_python.quaternion_to_rotation_matrix` so
        ``v_global = R @ v_local``.

    Raises:
        ValueError: When ``class_name`` is not a recognised shell, the
            node-count is insufficient for the class, the first edge
            has zero length, or the two edges are collinear (zero
            cross-product magnitude).
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    e1, e2 = _frame_edges(coords, class_name)

    e1_norm = float(np.linalg.norm(e1))
    if e1_norm < _DEGENERATE_EPS:
        raise ValueError(
            f"Degenerate shell ({class_name}): first edge has zero length."
        )
    x_local = e1 / e1_norm

    z_raw = np.cross(e1, e2)
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < _DEGENERATE_EPS:
        raise ValueError(
            f"Degenerate shell ({class_name}): edges are collinear "
            f"(zero normal)."
        )
    z_local = z_raw / z_norm

    # y_local is unit by construction (z and x are orthonormal).
    y_local = np.cross(z_local, x_local)

    # Columns = local basis in global frame (STKO convention).
    return np.column_stack([x_local, y_local, z_local])


def rotation_matrix_to_quaternion(R: ndarray) -> ndarray:
    """Convert a ``(3, 3)`` rotation matrix to a scalar-first quaternion.

    Inverse of :func:`STKO_to_python.quaternion_to_rotation_matrix`.
    Round-tripping ``R → q → R`` reproduces the input within numerical
    precision; the intermediate quaternion is determined up to a global
    sign (``q`` and ``−q`` represent the same rotation, so the
    intermediate value may flip sign without affecting the round trip).

    Uses the Shepperd trace-pivot algorithm: pick the branch whose
    pivot is largest in magnitude to maximise numerical stability
    across all four quadrant cases (near-identity, near-180° about
    each principal axis).

    Args:
        R: ``(3, 3)`` rotation matrix in STKO convention — columns are
            the body-frame basis vectors expressed in the world frame,
            so ``v_world = R @ v_body``.

    Returns:
        ``(4,)`` ``float64`` quaternion ``(qw, qx, qy, qz)`` — same
        component order as STKO's `.cdata` ``*LOCAL_AXES`` storage.

    Raises:
        ValueError: If ``R`` is not shape ``(3, 3)``.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.shape != (3, 3):
        raise ValueError(f"R must be (3, 3); got {R_arr.shape}.")

    trace = R_arr[0, 0] + R_arr[1, 1] + R_arr[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_arr[2, 1] - R_arr[1, 2]) * s
        y = (R_arr[0, 2] - R_arr[2, 0]) * s
        z = (R_arr[1, 0] - R_arr[0, 1]) * s
    elif R_arr[0, 0] > R_arr[1, 1] and R_arr[0, 0] > R_arr[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_arr[0, 0] - R_arr[1, 1] - R_arr[2, 2])
        w = (R_arr[2, 1] - R_arr[1, 2]) / s
        x = 0.25 * s
        y = (R_arr[0, 1] + R_arr[1, 0]) / s
        z = (R_arr[0, 2] + R_arr[2, 0]) / s
    elif R_arr[1, 1] > R_arr[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R_arr[1, 1] - R_arr[0, 0] - R_arr[2, 2])
        w = (R_arr[0, 2] - R_arr[2, 0]) / s
        x = (R_arr[0, 1] + R_arr[1, 0]) / s
        y = 0.25 * s
        z = (R_arr[1, 2] + R_arr[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R_arr[2, 2] - R_arr[0, 0] - R_arr[1, 1])
        w = (R_arr[1, 0] - R_arr[0, 1]) / s
        x = (R_arr[0, 2] + R_arr[2, 0]) / s
        y = (R_arr[1, 2] + R_arr[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def shell_quaternion(
    node_coords: ndarray, class_name: str,
) -> ndarray:
    """Shell local-frame as a quaternion in one call.

    Composition of :func:`shell_local_axes` and
    :func:`rotation_matrix_to_quaternion`.

    Args:
        node_coords: ``(n_nodes, 3)`` global corner coordinates.
        class_name: OpenSees shell class string.

    Returns:
        ``(4,)`` ``float64`` quaternion ``(qw, qx, qy, qz)`` matching
        STKO's `.cdata` ``*LOCAL_AXES`` storage convention.
    """
    R = shell_local_axes(node_coords, class_name)
    return rotation_matrix_to_quaternion(R)
