"""Vectorized world → screen projection for box-pick selection.

Adapted from apeGmsh ``viewers/core/results_pick.py`` (the pure-math
projection kernel, without the VTK interactor wiring).

Pure-numpy primitives that drive 3D selection workflows: project a
batch of world-space points to screen-space coordinates in one matmul,
then test inclusion against a rectangle. ~40× faster than the
per-point ``vtkRenderer.WorldToDisplay`` loop at 100k+ points — the
gap matters when a user box-selects across a million Gauss markers on
a solid model.

This module does **not** depend on ``pyvista``, ``vtk``, or any
renderer. Callers (Phase 4 Qt picking controller) are responsible for:

1. Extracting the composite view-projection matrix from the renderer
   (in VTK: ``camera.GetCompositeProjectionTransformMatrix(aspect, 0, 1)``).
2. Extracting the window size and viewport bounds from the renderer.
3. Passing all three plus the candidate point batch into the functions
   here, and acting on the returned mask.

Convention
----------

* World coordinates are right-handed 3-D.
* ``projection_matrix`` is the row-major 4×4 composite matrix mapping
  world coordinates to clip space — i.e. the matrix VTK returns from
  ``GetCompositeProjectionTransformMatrix``. Same convention as OpenGL.
* NDC (normalized device coordinates) span ``[-1, +1]`` on the x and
  y axes for points inside the view frustum. The z axis carries depth
  (passed through for callers that care; not used here).
* Display pixels have origin at the bottom-left of the viewport, x
  increasing right, y increasing up — VTK's convention. (Qt's window
  uses top-left; the caller flips y if needed.)
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


__all__ = [
    "world_to_ndc",
    "world_to_display",
    "points_in_box",
    "world_points_in_box",
]


def world_to_ndc(
    world_points: ndarray,
    projection_matrix: ndarray,
) -> ndarray:
    """Project a batch of world-space points to normalized device coords.

    Args:
        world_points: ``(N, 3)`` array of world-space positions.
        projection_matrix: ``(4, 4)`` row-major composite view-projection
            matrix. Same orientation VTK and OpenGL use: applying
            ``clip = M @ [x, y, z, 1]`` produces clip-space coords whose
            xy ∈ ``[-w, +w]`` are visible.

    Returns:
        ``(N, 3)`` array of NDC coordinates. Visible points have ``x``,
        ``y`` in ``[-1, +1]``; the ``z`` column carries depth.

        Points whose homogeneous-coordinate ``w`` is exactly zero
        (e.g. exactly on the camera plane) receive a divisor of ``1``
        instead — they will not project sensibly but the function does
        not raise. Callers should mask such points by checking
        ``|ndc| <= 1`` or by inspecting depth.

    Empty input ``(0, 3)`` returns an empty ``(0, 3)`` array.
    """
    pts = np.asarray(world_points, dtype=np.float64)
    M = np.asarray(projection_matrix, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Build homogeneous coords ``(N, 4)`` with w == 1.
    homog = np.empty((n, 4), dtype=np.float64)
    homog[:, :3] = pts
    homog[:, 3] = 1.0
    # ``clip = homog @ M.T`` is the row-major form of ``clip_col = M @ homog_col``.
    clip = homog @ M.T
    w = clip[:, 3:4]
    # Avoid divide-by-zero for w == 0 (point on the camera plane).
    safe_w = np.where(w == 0.0, 1.0, w)
    return clip[:, :3] / safe_w


def world_to_display(
    world_points: ndarray,
    projection_matrix: ndarray,
    window_size: tuple[int, int],
    viewport: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> ndarray:
    """Project world points to display-pixel coordinates.

    Args:
        world_points: ``(N, 3)`` world-space positions.
        projection_matrix: ``(4, 4)`` composite view-projection matrix.
            See :func:`world_to_ndc`.
        window_size: ``(width, height)`` of the render window, in
            pixels.
        viewport: ``(x_min, y_min, x_max, y_max)`` viewport bounds as
            fractions of the window, in ``[0, 1]``. Defaults to the
            full window.

    Returns:
        ``(N, 2)`` array of display-pixel ``(x, y)`` coordinates with
        VTK's bottom-left origin. Empty input returns ``(0, 2)``.
    """
    n = np.asarray(world_points, dtype=np.float64).shape[0]
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    ndc = world_to_ndc(world_points, projection_matrix)
    win_w, win_h = float(window_size[0]), float(window_size[1])
    vp_x0 = float(viewport[0]) * win_w
    vp_y0 = float(viewport[1]) * win_h
    vp_w = (float(viewport[2]) - float(viewport[0])) * win_w
    vp_h = (float(viewport[3]) - float(viewport[1])) * win_h

    out = np.empty((n, 2), dtype=np.float64)
    out[:, 0] = vp_x0 + (ndc[:, 0] * 0.5 + 0.5) * vp_w
    out[:, 1] = vp_y0 + (ndc[:, 1] * 0.5 + 0.5) * vp_h
    return out


def points_in_box(
    display_xy: ndarray,
    box: tuple[float, float, float, float],
) -> ndarray:
    """Boolean mask: which display-space points fall inside the rectangle.

    Args:
        display_xy: ``(N, 2)`` array of display-space ``(x, y)``
            coordinates.
        box: ``(x0, y0, x1, y1)`` corners of the rectangle. Either
            diagonal works — the function normalizes so a "drag from
            top-right to bottom-left" produces the same mask as a drag
            in the opposite direction. Boundary points are treated as
            inside.

    Returns:
        ``(N,)`` bool mask. Empty input returns an empty mask.
    """
    xy = np.asarray(display_xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return (
        (xy[:, 0] >= bx0) & (xy[:, 0] <= bx1)
        & (xy[:, 1] >= by0) & (xy[:, 1] <= by1)
    )


def world_points_in_box(
    world_points: ndarray,
    projection_matrix: ndarray,
    window_size: tuple[int, int],
    box: tuple[float, float, float, float],
    viewport: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> ndarray:
    """End-to-end: project then test box inclusion in one call.

    Equivalent to::

        display = world_to_display(world_points, M, window_size, viewport)
        mask = points_in_box(display, box)

    Provided as a convenience for the hot path of rubber-band box-pick
    selection, where the caller does not need the intermediate display
    coordinates.

    Args:
        world_points: ``(N, 3)`` world-space positions.
        projection_matrix: ``(4, 4)`` composite view-projection matrix.
        window_size: ``(width, height)`` of the render window in pixels.
        box: ``(x0, y0, x1, y1)`` rectangle in display pixels.
        viewport: ``(x_min, y_min, x_max, y_max)`` fraction-of-window
            viewport bounds.

    Returns:
        ``(N,)`` bool mask. Empty input returns an empty mask.
    """
    display = world_to_display(
        world_points, projection_matrix, window_size, viewport,
    )
    return points_in_box(display, box)
