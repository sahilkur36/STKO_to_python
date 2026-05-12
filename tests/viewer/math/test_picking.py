"""Tests for :mod:`STKO_to_python.viewer.math.picking`.

The module is the pure-numpy projection kernel that drives 3D
box-pick selection. The tests use **constructed** projection matrices
(no VTK) so they can run without any rendering deps installed.

Coverage:

* ``world_to_ndc`` correctness on identity and translation matrices,
  perspective division behaviour, and the divide-by-zero guard.
* ``world_to_display`` maps NDC ``[-1, +1]`` to the viewport correctly
  for both full-window and partial-viewport cases.
* ``points_in_box`` normalizes corner ordering and is inclusive at
  the boundary.
* ``world_points_in_box`` composes correctly end-to-end.
* Empty inputs return empty outputs throughout.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.viewer.math.picking import (
    points_in_box,
    world_points_in_box,
    world_to_display,
    world_to_ndc,
)


# --------------------------------------------------------------------- #
# world_to_ndc                                                          #
# --------------------------------------------------------------------- #


def test_world_to_ndc_identity_matrix_preserves_xy() -> None:
    """Identity 4×4 → NDC equals world (after trivial w=1 division)."""
    M = np.eye(4)
    pts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [+1.0, +1.0, 0.0],
            [+0.5, -0.25, 0.0],
        ],
        dtype=np.float64,
    )
    ndc = world_to_ndc(pts, M)
    np.testing.assert_allclose(ndc, pts, atol=1e-12)


def test_world_to_ndc_empty_input_returns_empty_output() -> None:
    M = np.eye(4)
    out = world_to_ndc(np.zeros((0, 3)), M)
    assert out.shape == (0, 3)


def test_world_to_ndc_translation_shifts_xy() -> None:
    """A translation matrix shifts the projected coordinates by the same amount."""
    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, -0.25],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    ndc = world_to_ndc(pts, M)
    np.testing.assert_allclose(ndc[0, :2], [0.5, -0.25], atol=1e-12)


def test_world_to_ndc_perspective_division_applied() -> None:
    """Setting ``w = 2`` halves the projected xy coordinates."""
    # Matrix that sets clip-space w := 2 (no other transform).
    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],   # row → w = 2 for any point
        ],
        dtype=np.float64,
    )
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    ndc = world_to_ndc(pts, M)
    np.testing.assert_allclose(ndc[0], [0.5, 1.0, 1.5], atol=1e-12)


def test_world_to_ndc_zero_w_uses_safe_divisor() -> None:
    """A point that produces w == 0 doesn't raise; gets w := 1 fallback."""
    # Matrix that sets clip-space w := 0 for any point.
    M = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],   # w = 0
        ],
        dtype=np.float64,
    )
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    ndc = world_to_ndc(pts, M)
    # With safe_w == 1, NDC equals the un-divided clip coords.
    np.testing.assert_allclose(ndc[0], [1.0, 2.0, 3.0], atol=1e-12)
    assert np.all(np.isfinite(ndc))


# --------------------------------------------------------------------- #
# world_to_display                                                      #
# --------------------------------------------------------------------- #


def test_world_to_display_identity_full_viewport() -> None:
    """NDC [-1, +1] maps linearly to the full window."""
    M = np.eye(4)
    pts = np.array(
        [
            [-1.0, -1.0, 0.0],   # bottom-left of viewport
            [+1.0, +1.0, 0.0],   # top-right of viewport
            [0.0, 0.0, 0.0],     # centre
            [+0.5, -0.5, 0.0],   # offset
        ],
        dtype=np.float64,
    )
    display = world_to_display(pts, M, window_size=(100, 200))

    np.testing.assert_allclose(display[0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(display[1], [100.0, 200.0], atol=1e-12)
    np.testing.assert_allclose(display[2], [50.0, 100.0], atol=1e-12)
    np.testing.assert_allclose(display[3], [75.0, 50.0], atol=1e-12)


def test_world_to_display_partial_viewport_respects_offset() -> None:
    """A right-half viewport places NDC -1 at the window midpoint."""
    M = np.eye(4)
    pts = np.array([[-1.0, 0.0, 0.0], [+1.0, 0.0, 0.0]], dtype=np.float64)
    display = world_to_display(
        pts, M, window_size=(100, 100), viewport=(0.5, 0.0, 1.0, 1.0),
    )
    np.testing.assert_allclose(display[0], [50.0, 50.0], atol=1e-12)
    np.testing.assert_allclose(display[1], [100.0, 50.0], atol=1e-12)


def test_world_to_display_empty_input_returns_empty_output() -> None:
    M = np.eye(4)
    out = world_to_display(np.zeros((0, 3)), M, window_size=(640, 480))
    assert out.shape == (0, 2)


# --------------------------------------------------------------------- #
# points_in_box                                                         #
# --------------------------------------------------------------------- #


def test_points_in_box_basic() -> None:
    xy = np.array(
        [
            [10.0, 10.0],     # inside
            [50.0, 50.0],     # inside (centre)
            [99.0, 99.0],     # inside (near corner)
            [101.0, 50.0],    # outside (right)
            [50.0, -1.0],     # outside (below)
        ],
        dtype=np.float64,
    )
    mask = points_in_box(xy, (0.0, 0.0, 100.0, 100.0))
    np.testing.assert_array_equal(mask, [True, True, True, False, False])


def test_points_in_box_reversed_corners() -> None:
    """Box passed as (x1, y1, x0, y0) → same mask as the forward order."""
    xy = np.array([[5.0, 5.0], [50.0, 50.0]], dtype=np.float64)
    forward = points_in_box(xy, (0.0, 0.0, 100.0, 100.0))
    reversed_ = points_in_box(xy, (100.0, 100.0, 0.0, 0.0))
    np.testing.assert_array_equal(forward, reversed_)


def test_points_in_box_boundary_inclusive() -> None:
    """Points lying exactly on the rectangle edge are considered inside."""
    xy = np.array(
        [
            [0.0, 0.0],       # bottom-left corner
            [100.0, 50.0],    # right edge
            [50.0, 0.0],      # bottom edge
            [100.0, 100.0],   # top-right corner
        ],
        dtype=np.float64,
    )
    mask = points_in_box(xy, (0.0, 0.0, 100.0, 100.0))
    np.testing.assert_array_equal(mask, [True, True, True, True])


def test_points_in_box_empty_input() -> None:
    mask = points_in_box(np.zeros((0, 2)), (0.0, 0.0, 100.0, 100.0))
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_points_in_box_degenerate_rectangle_includes_only_the_line() -> None:
    """A zero-area box (x0 == x1) only matches points on that line."""
    xy = np.array(
        [
            [50.0, 50.0],   # on the line
            [50.1, 50.0],   # just off
        ],
        dtype=np.float64,
    )
    mask = points_in_box(xy, (50.0, 0.0, 50.0, 100.0))
    np.testing.assert_array_equal(mask, [True, False])


# --------------------------------------------------------------------- #
# world_points_in_box                                                   #
# --------------------------------------------------------------------- #


def test_world_points_in_box_end_to_end_identity() -> None:
    """Identity projection on a 100×100 window with a centred 40px box."""
    M = np.eye(4)
    # Mix of inside and outside-the-NDC-frustum points.
    pts = np.array(
        [
            [-0.5, -0.5, 0.0],   # NDC (-0.5, -0.5) → display (25, 25)
            [+0.5, +0.5, 0.0],   # NDC (+0.5, +0.5) → display (75, 75)
            [+0.9, +0.9, 0.0],   # display (95, 95) — outside the box
            [-0.9, -0.9, 0.0],   # display (5, 5) — outside the box
        ],
        dtype=np.float64,
    )
    # Box from (30, 30) to (80, 80) in display pixels.
    mask = world_points_in_box(
        pts, M, window_size=(100, 100), box=(30.0, 30.0, 80.0, 80.0),
    )
    np.testing.assert_array_equal(mask, [False, True, False, False])


def test_world_points_in_box_empty_input() -> None:
    M = np.eye(4)
    mask = world_points_in_box(
        np.zeros((0, 3)), M, window_size=(100, 100), box=(0, 0, 50, 50),
    )
    assert mask.shape == (0,)


def test_world_points_in_box_perspective_division_affects_result() -> None:
    """A point further from the camera shrinks toward the screen centre."""
    # Standard symmetric perspective projection with a 90 deg FOV. The
    # canonical matrix (OpenGL convention) places the near plane at
    # z = -1 and the far plane at z = -100.
    near, far = 1.0, 100.0
    aspect = 1.0
    f = 1.0 / np.tan(np.deg2rad(45.0))
    M = np.array(
        [
            [f / aspect, 0.0, 0.0,                              0.0],
            [0.0,        f,   0.0,                              0.0],
            [0.0,        0.0, (far + near) / (near - far),      (2.0 * far * near) / (near - far)],
            [0.0,        0.0, -1.0,                             0.0],
        ],
        dtype=np.float64,
    )
    # Two points at the same world (x, y) but different depths. The
    # further one (more negative z) should project closer to (0, 0).
    pts = np.array(
        [
            [0.5, 0.5, -2.0],
            [0.5, 0.5, -50.0],
        ],
        dtype=np.float64,
    )
    display = world_to_display(pts, M, window_size=(200, 200))
    # Near point projects further from screen centre than far point.
    near_offset = np.linalg.norm(display[0] - [100.0, 100.0])
    far_offset = np.linalg.norm(display[1] - [100.0, 100.0])
    assert near_offset > far_offset
