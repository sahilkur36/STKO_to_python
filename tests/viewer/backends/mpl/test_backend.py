"""Tests for :class:`MplBackend`.

The backend is tested in isolation — no layer, no real STKO data. Each
test exercises one method on the protocol via a fresh scene handle,
verifies the matplotlib artifacts it produced, then closes the figure
to keep memory bounded.

Matplotlib runs under the ``Agg`` backend in CI (set in the workflow
``env: MPLBACKEND: Agg``) so no display is required.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from STKO_to_python.viewer.backends.mpl import MplBackend, MplSceneHandle
from STKO_to_python.viewer.core import (
    BBox,
    Backend,
    BackendCapabilityError,
    CameraSpec,
    SceneStyle,
)


# Run under Agg in the test suite so figures don't try to open windows.
matplotlib.use("Agg", force=True)


@pytest.fixture
def backend() -> MplBackend:
    return MplBackend()


@pytest.fixture
def scene_2d(backend: MplBackend) -> MplSceneHandle:
    handle = backend.make_scene(is_3d=False, off_screen=True)
    yield handle
    plt.close(handle.fig)


@pytest.fixture
def scene_3d(backend: MplBackend) -> MplSceneHandle:
    handle = backend.make_scene(is_3d=True, off_screen=True)
    yield handle
    plt.close(handle.fig)


# --------------------------------------------------------------------- #
# Protocol conformance                                                  #
# --------------------------------------------------------------------- #


def test_mpl_backend_satisfies_backend_protocol() -> None:
    assert isinstance(MplBackend(), Backend)


def test_mpl_backend_static_attributes() -> None:
    assert MplBackend.name == "mpl"
    assert MplBackend.is_3d_capable is True
    assert MplBackend.is_interactive is True


# --------------------------------------------------------------------- #
# Scene lifecycle                                                       #
# --------------------------------------------------------------------- #


def test_make_scene_2d_returns_handle(backend: MplBackend) -> None:
    handle = backend.make_scene(is_3d=False, off_screen=True)
    try:
        assert isinstance(handle, MplSceneHandle)
        assert isinstance(handle.fig, Figure)
        assert handle.is_3d is False
        assert handle.off_screen is True
    finally:
        plt.close(handle.fig)


def test_make_scene_3d_uses_3d_projection(backend: MplBackend) -> None:
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        assert handle.is_3d is True
        assert handle.ax.name == "3d"
    finally:
        plt.close(handle.fig)


def test_make_scene_default_args_are_2d_onscreen(backend: MplBackend) -> None:
    handle = backend.make_scene()
    try:
        assert handle.is_3d is False
        assert handle.off_screen is False
    finally:
        plt.close(handle.fig)


def test_set_bounds_applies_xlim_ylim_in_2d(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    backend.set_bounds(scene_2d, BBox(-1, -2, -3, 4, 5, 6))
    assert scene_2d.ax.get_xlim() == (-1.0, 4.0)
    assert scene_2d.ax.get_ylim() == (-2.0, 5.0)


def test_set_bounds_applies_zlim_in_3d(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    backend.set_bounds(scene_3d, BBox(-1, -2, -3, 4, 5, 6))
    assert scene_3d.ax.get_zlim() == (-3.0, 6.0)


def test_set_camera_is_noop_in_2d(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    """2-D axes don't have a 3-D camera; the call must succeed silently."""
    backend.set_camera(
        scene_2d,
        CameraSpec(position=(0, 0, 5), focal_point=(0, 0, 0), view_up=(0, 1, 0)),
    )


def test_set_camera_in_3d_updates_view_angles(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    """A camera with position above and to the side rotates the view."""
    backend.set_camera(
        scene_3d,
        CameraSpec(
            position=(1.0, 1.0, 1.0), focal_point=(0, 0, 0), view_up=(0, 0, 1),
        ),
    )
    # The exact elev/azim depend on the conversion math; verify they
    # are within sensible bounds (not the matplotlib defaults).
    elev = scene_3d.ax.elev
    azim = scene_3d.ax.azim
    assert 0.0 < elev < 90.0
    assert 0.0 < azim < 90.0


def test_set_style_applies_background_and_grid(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    style = SceneStyle(background="#cccccc", grid=True, font_size=12)
    backend.set_style(scene_2d, style)
    # Both figure and axes get the background.
    assert scene_2d.fig.get_facecolor()[:3] == pytest.approx((0.8, 0.8, 0.8), abs=0.01)
    assert scene_2d.ax.xaxis._major_tick_kw.get("gridOn", False) is not False


# --------------------------------------------------------------------- #
# add_segments                                                          #
# --------------------------------------------------------------------- #


def test_add_segments_2d_returns_line_collection(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    segs = np.array(
        [[[0, 0, 0], [1, 1, 0]], [[1, 1, 0], [2, 0, 0]]], dtype=np.float64,
    )
    actor = backend.add_segments(scene_2d, segs, color="red", width=2.0, alpha=0.5, label=None)
    assert isinstance(actor, LineCollection)
    assert actor in scene_2d.ax.collections


def test_add_segments_3d_returns_line3d_collection(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    segs = np.array(
        [[[0, 0, 0], [1, 1, 1]]], dtype=np.float64,
    )
    actor = backend.add_segments(scene_3d, segs, color="blue", width=1.0, alpha=1.0, label=None)
    assert isinstance(actor, Line3DCollection)


def test_add_segments_empty_2d_is_a_valid_collection(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    actor = backend.add_segments(scene_2d, np.zeros((0, 2, 3)), color=None, width=None, alpha=None, label=None)
    assert isinstance(actor, LineCollection)


# --------------------------------------------------------------------- #
# add_points                                                            #
# --------------------------------------------------------------------- #


def test_add_points_2d_returns_scatter_collection(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    pts = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.float64)
    actor = backend.add_points(scene_2d, pts, color="green", size=10.0)
    assert isinstance(actor, PathCollection)


def test_add_points_3d_returns_3d_scatter_collection(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    actor = backend.add_points(scene_3d, pts, color="green", size=10.0)
    # In 3-D matplotlib, ax.scatter returns Path3DCollection (subclass
    # of PathCollection).
    assert isinstance(actor, PathCollection)


def test_add_points_with_scalars_drives_color(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    pts = np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0]], dtype=np.float64)
    scalars = np.array([0.1, 0.5, 0.9])
    actor = backend.add_points(scene_2d, pts, scalars=scalars, cmap="viridis")
    np.testing.assert_array_equal(actor.get_array(), scalars)


# --------------------------------------------------------------------- #
# add_polygons                                                          #
# --------------------------------------------------------------------- #


def test_add_polygons_2d_returns_poly_collection(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    polys = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)]
    actor = backend.add_polygons(scene_2d, polys, edge_color="black")
    assert isinstance(actor, PolyCollection)


def test_add_polygons_3d_returns_poly3d_collection(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    polys = [np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)]
    actor = backend.add_polygons(scene_3d, polys, edge_color="black")
    assert isinstance(actor, Poly3DCollection)


def test_add_polygons_with_values_sets_array(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    polys = [
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=np.float64),
    ]
    values = np.array([0.2, 0.8])
    actor = backend.add_polygons(scene_2d, polys, values=values, cmap="plasma")
    np.testing.assert_array_equal(actor.get_array(), values)


# --------------------------------------------------------------------- #
# add_arrows                                                            #
# --------------------------------------------------------------------- #


def test_add_arrows_2d_uses_quiver(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    origins = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.float64)
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    actor = backend.add_arrows(scene_2d, origins, vectors, scale=1.0, color="red")
    assert actor is not None


def test_add_arrows_3d_uses_quiver(
    backend: MplBackend, scene_3d: MplSceneHandle,
) -> None:
    origins = np.array([[0, 0, 0]], dtype=np.float64)
    vectors = np.array([[1, 1, 1]], dtype=np.float64)
    actor = backend.add_arrows(scene_3d, origins, vectors, scale=1.0, color="blue")
    assert actor is not None


# --------------------------------------------------------------------- #
# In-place actor updates                                                #
# --------------------------------------------------------------------- #


def test_update_scalars_on_polygon_collection(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    polys = [
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
    ]
    actor = backend.add_polygons(scene_2d, polys, values=np.array([0.0]), cmap="viridis")
    backend.update_scalars(actor, np.array([0.7]))
    np.testing.assert_array_equal(actor.get_array(), [0.7])


def test_update_scalars_on_unsupported_actor_raises(backend: MplBackend) -> None:
    class NoSetArray:
        pass
    with pytest.raises(BackendCapabilityError):
        backend.update_scalars(NoSetArray(), np.array([1.0]))


def test_update_points_on_line_collection_2d(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    actor = backend.add_segments(
        scene_2d,
        np.array([[[0, 0, 0], [1, 1, 0]]], dtype=np.float64),
        color=None, width=None, alpha=None, label=None,
    )
    new_segs = np.array([[[0, 0], [2, 2]]], dtype=np.float64)
    backend.update_points(actor, new_segs)
    np.testing.assert_array_equal(
        np.asarray(actor.get_segments()[0]), new_segs[0],
    )


def test_update_points_on_unsupported_actor_raises(backend: MplBackend) -> None:
    class Unrelated:
        pass
    with pytest.raises(BackendCapabilityError):
        backend.update_points(Unrelated(), np.zeros((1, 2)))


def test_set_visible_toggles_artist_visibility(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    actor = backend.add_segments(
        scene_2d,
        np.array([[[0, 0, 0], [1, 1, 0]]], dtype=np.float64),
        color=None, width=None, alpha=None, label=None,
    )
    backend.set_visible(actor, False)
    assert actor.get_visible() is False
    backend.set_visible(actor, True)
    assert actor.get_visible() is True


def test_remove_drops_actor_from_axes(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    actor = backend.add_segments(
        scene_2d,
        np.array([[[0, 0, 0], [1, 1, 0]]], dtype=np.float64),
        color=None, width=None, alpha=None, label=None,
    )
    assert actor in scene_2d.ax.collections
    backend.remove(scene_2d, actor)
    assert actor not in scene_2d.ax.collections


# --------------------------------------------------------------------- #
# Output                                                                #
# --------------------------------------------------------------------- #


def test_show_is_noop_when_off_screen(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    """``off_screen`` scenes must not call ``plt.show``."""
    # If show() were called and tried to open a display, the Agg backend
    # would handle it silently — but the contract is "no-op." We verify
    # the call completes without error.
    backend.show(scene_2d)  # off_screen=True per fixture


def test_save_writes_file(
    backend: MplBackend, scene_2d: MplSceneHandle, tmp_path,
) -> None:
    out = tmp_path / "scene.png"
    backend.add_segments(
        scene_2d,
        np.array([[[0, 0, 0], [1, 1, 0]]], dtype=np.float64),
        color=None, width=None, alpha=None, label=None,
    )
    backend.save(scene_2d, out, dpi=72)
    assert out.exists()
    assert out.stat().st_size > 0


def test_snapshot_returns_rgb_uint8_array(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    arr = backend.snapshot(scene_2d)
    assert arr.dtype == np.uint8
    assert arr.ndim == 3
    assert arr.shape[-1] == 3  # RGB, no alpha


def test_snapshot_dimensions_match_figure_pixels(
    backend: MplBackend, scene_2d: MplSceneHandle,
) -> None:
    """Snapshot height/width align with the figure's pixel size."""
    arr = backend.snapshot(scene_2d)
    fig_w, fig_h = scene_2d.fig.canvas.get_width_height()
    # Sometimes mpl returns the buffer at a slightly different size on
    # high-DPI canvases — allow ±1 px tolerance.
    assert abs(arr.shape[1] - fig_w) <= 1
    assert abs(arr.shape[0] - fig_h) <= 1
