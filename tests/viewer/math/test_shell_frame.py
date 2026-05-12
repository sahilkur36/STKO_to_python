"""Tests for :mod:`STKO_to_python.viewer.math.shell_frame`.

The module is the apeGmsh-derived shell-local-frame **fallback** for
shells that do not have a ``.cdata`` ``*LOCAL_AXES`` quaternion
available, plus the inverse of
:func:`STKO_to_python.quaternion_to_rotation_matrix`.

Coverage:

* :func:`shell_local_axes` produces a right-handed orthonormal frame
  for quads and triangles in arbitrary orientations.
* The convention matches STKO — columns are the local basis in global
  coordinates, so ``v_global = R @ v_local``.
* Degenerate geometries (collinear edges, zero-length first edge)
  raise ``ValueError``.
* :func:`rotation_matrix_to_quaternion` round-trips through STKO's
  :func:`quaternion_to_rotation_matrix` for the identity rotation, a
  90° rotation about each principal axis (trace > 0 branch), and a
  180° rotation about each principal axis (one of the three diagonal
  branches each).
* :func:`shell_quaternion` composes the two correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import quaternion_to_rotation_matrix
from STKO_to_python.viewer.math.shell_frame import (
    rotation_matrix_to_quaternion,
    shell_local_axes,
    shell_quaternion,
)


# --------------------------------------------------------------------- #
# shell_local_axes — basic geometry                                     #
# --------------------------------------------------------------------- #


def _assert_orthonormal_right_handed(
    R: np.ndarray, *, atol: float = 1e-12,
) -> None:
    """Check ``R`` is a proper rotation matrix (orthonormal, det = +1)."""
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    np.testing.assert_allclose(np.linalg.norm(x), 1.0, atol=atol)
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=atol)
    np.testing.assert_allclose(np.linalg.norm(z), 1.0, atol=atol)
    np.testing.assert_allclose(np.dot(x, y), 0.0, atol=atol)
    np.testing.assert_allclose(np.dot(y, z), 0.0, atol=atol)
    np.testing.assert_allclose(np.dot(x, z), 0.0, atol=atol)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=atol)


def test_shell_local_axes_quad_in_xy_plane_is_identity() -> None:
    """A unit-square ASDShellQ4 in the xy-plane gives the identity matrix."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    R = shell_local_axes(coords, "ASDShellQ4")
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_shell_local_axes_tri_in_xy_plane_is_identity() -> None:
    """A unit-right-triangle ASDShellT3 in the xy-plane gives identity."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    R = shell_local_axes(coords, "ASDShellT3")
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_shell_local_axes_quad_in_xz_plane() -> None:
    """A quad in the xz-plane: x_local = +X, z_local = −Y, y_local = +Z."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    R = shell_local_axes(coords, "ASDShellQ4")
    # e1 = +X, e2 = +Z. z_raw = e1 × e2 = +X × +Z = −Y. z_local = −Y.
    # y_local = z × x = −Y × +X = +Z.
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R[:, 1], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(R[:, 2], [0.0, -1.0, 0.0], atol=1e-12)
    _assert_orthonormal_right_handed(R)


@pytest.mark.parametrize(
    "class_name, nodes",
    [
        # Quad variants
        ("ASDShellQ4", [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
        ("ShellMITC4", [[0,0,0], [2,0,0], [2,3,1], [0,3,1]]),
        ("ShellDKGQ",  [[1,2,3], [4,5,3], [4,5,6], [1,2,6]]),
        # Tri variants
        ("ASDShellT3", [[0,0,0], [1,0,0], [0,1,0]]),
        ("ShellDKGT",  [[1,1,1], [2,1,1], [1,2,1]]),
        # 9-node uses corners 1..4 only
        ("ShellMITC9", [[0,0,0], [1,0,0], [1,1,0], [0,1,0],
                        [0.5,0,0], [1,0.5,0], [0.5,1,0], [0,0.5,0],
                        [0.5,0.5,0]]),
    ],
)
def test_shell_local_axes_orthonormal_for_all_classes(
    class_name: str, nodes: list[list[float]],
) -> None:
    R = shell_local_axes(np.array(nodes, dtype=np.float64), class_name)
    _assert_orthonormal_right_handed(R, atol=1e-10)


def test_shell_local_axes_v_global_equals_R_times_v_local() -> None:
    """STKO convention: applying ``R`` to a local vector gives global."""
    # Quad rotated so x_local = +Y, y_local = -X, z_local = +Z
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    R = shell_local_axes(coords, "ASDShellQ4")
    # x_local in local frame is (1, 0, 0). Its global expression must
    # equal column 0 of R.
    v_local_x = np.array([1.0, 0.0, 0.0])
    v_global_x = R @ v_local_x
    np.testing.assert_allclose(v_global_x, R[:, 0], atol=1e-12)
    # And R[:, 0] should be +Y for this configuration.
    np.testing.assert_allclose(R[:, 0], [0.0, 1.0, 0.0], atol=1e-12)


# --------------------------------------------------------------------- #
# shell_local_axes — error paths                                        #
# --------------------------------------------------------------------- #


def test_shell_local_axes_collinear_quad_raises() -> None:
    """Nodes on a single line have a zero cross-product → ValueError."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="collinear"):
        shell_local_axes(coords, "ASDShellQ4")


def test_shell_local_axes_zero_first_edge_raises() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="zero length"):
        shell_local_axes(coords, "ASDShellQ4")


def test_shell_local_axes_unknown_class_raises() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="Unsupported shell class"):
        shell_local_axes(coords, "999-NotAShellClass")


def test_shell_local_axes_too_few_nodes_for_quad_raises() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],   # only 3 nodes
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="≥4 corner nodes"):
        shell_local_axes(coords, "ASDShellQ4")


def test_shell_local_axes_too_few_nodes_for_tri_raises() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # only 2 nodes
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="≥3 corner nodes"):
        shell_local_axes(coords, "ASDShellT3")


# --------------------------------------------------------------------- #
# rotation_matrix_to_quaternion — branch coverage + round-trip           #
# --------------------------------------------------------------------- #


def _round_trip(R: np.ndarray) -> np.ndarray:
    """``R → q → R``; result should equal the input."""
    q = rotation_matrix_to_quaternion(R)
    return quaternion_to_rotation_matrix(q)


def test_rotation_matrix_to_quaternion_identity() -> None:
    q = rotation_matrix_to_quaternion(np.eye(3))
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_rotation_matrix_to_quaternion_round_trips_identity() -> None:
    np.testing.assert_allclose(_round_trip(np.eye(3)), np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "axis_name, R",
    [
        # 90° about +x: trace > 0 branch
        ("90 about +x",
         np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])),
        # 90° about +y
        ("90 about +y",
         np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])),
        # 90° about +z
        ("90 about +z",
         np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])),
        # 180° about +x: triggers R[0,0]-max branch (R[0,0]=1, others=-1, trace=-1)
        ("180 about +x",
         np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])),
        # 180° about +y: triggers R[1,1]-max branch
        ("180 about +y",
         np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])),
        # 180° about +z: triggers R[2,2]-max branch
        ("180 about +z",
         np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])),
    ],
)
def test_rotation_matrix_to_quaternion_round_trips_all_branches(
    axis_name: str, R: np.ndarray,
) -> None:
    """All four Shepperd branches reproduce the input under round-trip."""
    np.testing.assert_allclose(_round_trip(R), R, atol=1e-12)


def test_rotation_matrix_to_quaternion_round_trips_random_rotations() -> None:
    """Compose three axis rotations into a non-trivial R; round-trip closes."""
    # Build R = Rz @ Ry @ Rx for some non-special angles.
    a, b, c = 0.3, 0.7, 1.1
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    np.testing.assert_allclose(_round_trip(R), R, atol=1e-12)


def test_rotation_matrix_to_quaternion_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"R must be \(3, 3\)"):
        rotation_matrix_to_quaternion(np.eye(4))


def test_rotation_matrix_to_quaternion_returns_unit_quaternion() -> None:
    """Output quaternion has unit norm for any rotation matrix."""
    R = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],   # 120° about (1,1,1)
        dtype=np.float64,
    )
    q = rotation_matrix_to_quaternion(R)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-12)


# --------------------------------------------------------------------- #
# shell_quaternion — composition                                        #
# --------------------------------------------------------------------- #


def test_shell_quaternion_identity_for_quad_in_xy_plane() -> None:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    q = shell_quaternion(coords, "ASDShellQ4")
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_shell_quaternion_round_trips_to_local_axes() -> None:
    """``shell_quaternion → quaternion_to_rotation_matrix`` recovers R."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.5],
            [1.5, 2.5, 1.0],
            [-0.5, 1.5, 0.5],
        ],
        dtype=np.float64,
    )
    R = shell_local_axes(coords, "ASDShellQ4")
    q = shell_quaternion(coords, "ASDShellQ4")
    R_check = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(R_check, R, atol=1e-12)
