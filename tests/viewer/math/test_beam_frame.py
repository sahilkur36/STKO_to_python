"""Tests for :mod:`STKO_to_python.viewer.math.beam_frame`.

The module is the apeGmsh-derived ``vecxz`` Gram-Schmidt fallback for
beams that do not have a ``.cdata`` ``*LOCAL_AXES`` quaternion
available. Tests cover:

* :func:`default_vecxz` switches between global Z and global X at the
  vertical threshold.
* :func:`compute_local_axes` produces a right-handed orthonormal frame
  for any beam orientation, with the expected geometric properties.
* :func:`compute_local_axes` raises on coincident endpoints.
* :func:`compute_local_axes` falls back silently when the user's
  ``vecxz`` is parallel to ``x_local``.
* :func:`station_position` linearly interpolates between endpoints.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.viewer.math.beam_frame import (
    compute_local_axes,
    default_vecxz,
    station_position,
)


# --------------------------------------------------------------------- #
# default_vecxz                                                         #
# --------------------------------------------------------------------- #


def test_default_vecxz_horizontal_beam_returns_global_Z() -> None:
    x_local = np.array([1.0, 0.0, 0.0])
    np.testing.assert_array_equal(default_vecxz(x_local), [0.0, 0.0, 1.0])


def test_default_vecxz_y_aligned_beam_returns_global_Z() -> None:
    x_local = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(default_vecxz(x_local), [0.0, 0.0, 1.0])


def test_default_vecxz_vertical_beam_returns_global_X() -> None:
    x_local = np.array([0.0, 0.0, 1.0])
    np.testing.assert_array_equal(default_vecxz(x_local), [1.0, 0.0, 0.0])


def test_default_vecxz_near_vertical_beam_returns_global_X() -> None:
    # cos(0.1 rad) ~ 0.995 > 0.99 threshold
    x_local = np.array([np.sin(0.1), 0.0, np.cos(0.1)])
    np.testing.assert_array_equal(default_vecxz(x_local), [1.0, 0.0, 0.0])


def test_default_vecxz_tilted_beam_returns_global_Z() -> None:
    # cos(20 deg) ~ 0.94 < 0.99 threshold — not vertical enough.
    angle = np.deg2rad(20.0)
    x_local = np.array([np.sin(angle), 0.0, np.cos(angle)])
    np.testing.assert_array_equal(default_vecxz(x_local), [0.0, 0.0, 1.0])


# --------------------------------------------------------------------- #
# compute_local_axes — basic geometry                                   #
# --------------------------------------------------------------------- #


def _assert_orthonormal_right_handed(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, *, atol: float = 1e-12,
) -> None:
    """All axes unit-norm; pairwise orthogonal; (x, y, z) right-handed."""
    np.testing.assert_allclose(np.linalg.norm(x), 1.0, atol=atol)
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=atol)
    np.testing.assert_allclose(np.linalg.norm(z), 1.0, atol=atol)
    np.testing.assert_allclose(np.dot(x, y), 0.0, atol=atol)
    np.testing.assert_allclose(np.dot(y, z), 0.0, atol=atol)
    np.testing.assert_allclose(np.dot(x, z), 0.0, atol=atol)
    # Right-handed: det([x|y|z]) == +1.
    det = float(np.linalg.det(np.column_stack([x, y, z])))
    np.testing.assert_allclose(det, 1.0, atol=atol)


def test_compute_local_axes_horizontal_beam_along_X() -> None:
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    # y = z × x = +Z × +X = +Y
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(L, 5.0)
    _assert_orthonormal_right_handed(x, y, z)


def test_compute_local_axes_horizontal_beam_along_Y() -> None:
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 3.0, 0.0]),
    )
    np.testing.assert_allclose(x, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    # y = z × x = +Z × +Y = -X
    np.testing.assert_allclose(y, [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(L, 3.0)
    _assert_orthonormal_right_handed(x, y, z)


def test_compute_local_axes_vertical_beam() -> None:
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 4.0]),
    )
    np.testing.assert_allclose(x, [0.0, 0.0, 1.0])
    # Default vecxz for vertical beam is +X, projected onto plane
    # perpendicular to +Z stays +X.
    np.testing.assert_allclose(z, [1.0, 0.0, 0.0])
    # y = z × x = +X × +Z = -Y
    np.testing.assert_allclose(y, [0.0, -1.0, 0.0])
    np.testing.assert_allclose(L, 4.0)
    _assert_orthonormal_right_handed(x, y, z)


def test_compute_local_axes_diagonal_beam_remains_orthonormal() -> None:
    x, y, z, L = compute_local_axes(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 6.0, 9.0]),
    )
    expected_length = np.linalg.norm([3.0, 4.0, 6.0])
    np.testing.assert_allclose(L, expected_length)
    _assert_orthonormal_right_handed(x, y, z)


@pytest.mark.parametrize(
    "i, j",
    [
        ([0, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [0, 1, 0]),
        ([0, 0, 0], [1, 1, 0]),
        ([0, 0, 0], [1, 1, 1]),
        ([10, -5, 3], [-2, 7, 11]),
        ([0, 0, 0], [0, 0, -1]),   # downward
    ],
)
def test_compute_local_axes_invariants_for_arbitrary_endpoints(
    i: list[float], j: list[float],
) -> None:
    """For any non-degenerate endpoint pair, the frame is orthonormal RH."""
    x, y, z, L = compute_local_axes(np.array(i), np.array(j))
    _assert_orthonormal_right_handed(x, y, z, atol=1e-10)
    np.testing.assert_allclose(L, np.linalg.norm(np.array(j) - np.array(i)))
    # x_local must align with (j - i).
    raw = np.array(j) - np.array(i)
    np.testing.assert_allclose(x, raw / np.linalg.norm(raw), atol=1e-12)


# --------------------------------------------------------------------- #
# compute_local_axes — custom vecxz                                     #
# --------------------------------------------------------------------- #


def test_compute_local_axes_respects_user_vecxz() -> None:
    """A user vecxz in the +Y direction tilts z_local for a +X beam."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        vecxz=np.array([0.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(x, [1.0, 0.0, 0.0])
    # vecxz = +Y is already perpendicular to +X, so z_local = +Y.
    np.testing.assert_allclose(z, [0.0, 1.0, 0.0])
    # y_local = z × x = +Y × +X = -Z
    np.testing.assert_allclose(y, [0.0, 0.0, -1.0])
    _assert_orthonormal_right_handed(x, y, z)


def test_compute_local_axes_parallel_vecxz_falls_back_silently() -> None:
    """User vecxz parallel to x_local triggers the default fallback."""
    x, y, z, L = compute_local_axes(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        vecxz=np.array([3.0, 0.0, 0.0]),  # parallel to +X (== x_local)
    )
    # default_vecxz for horizontal beam is +Z; result should be the
    # same as if we had passed no vecxz at all.
    np.testing.assert_allclose(z, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(y, [0.0, 1.0, 0.0])
    _assert_orthonormal_right_handed(x, y, z)


# --------------------------------------------------------------------- #
# compute_local_axes — error paths                                      #
# --------------------------------------------------------------------- #


def test_compute_local_axes_coincident_endpoints_raises() -> None:
    with pytest.raises(ValueError, match="endpoints coincide"):
        compute_local_axes(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
        )


def test_compute_local_axes_near_coincident_endpoints_raises() -> None:
    """Within _DEGENERATE_EPS the beam is treated as zero-length."""
    with pytest.raises(ValueError, match="endpoints coincide"):
        compute_local_axes(
            np.array([0.0, 0.0, 0.0]),
            np.array([1e-15, 0.0, 0.0]),
        )


# --------------------------------------------------------------------- #
# station_position                                                      #
# --------------------------------------------------------------------- #


def test_station_position_at_minus_one_returns_node_i() -> None:
    ci = np.array([1.0, 2.0, 3.0])
    cj = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(station_position(ci, cj, -1.0), ci)


def test_station_position_at_plus_one_returns_node_j() -> None:
    ci = np.array([1.0, 2.0, 3.0])
    cj = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(station_position(ci, cj, +1.0), cj)


def test_station_position_at_zero_returns_midpoint() -> None:
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([4.0, 6.0, 8.0])
    np.testing.assert_allclose(station_position(ci, cj, 0.0), [2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "xi, expected_t",
    [
        (-0.5, 0.25),
        (+0.5, 0.75),
        (+1.0, 1.0),
        (-1.0, 0.0),
    ],
)
def test_station_position_t_mapping(xi: float, expected_t: float) -> None:
    """``ξ -> t = (1 + ξ) / 2`` mapping reproduces OpenSees IP placement."""
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([10.0, 0.0, 0.0])
    pos = station_position(ci, cj, xi)
    np.testing.assert_allclose(pos[0], 10.0 * expected_t)


def test_station_position_does_not_clamp_outside_minus_one_plus_one() -> None:
    """Values outside ``[-1, +1]`` extrapolate linearly without clamping."""
    ci = np.array([0.0, 0.0, 0.0])
    cj = np.array([2.0, 0.0, 0.0])
    # xi = 3 -> t = 2 -> position = ci + 2 * (cj - ci) = (4, 0, 0).
    np.testing.assert_allclose(station_position(ci, cj, 3.0), [4.0, 0.0, 0.0])
