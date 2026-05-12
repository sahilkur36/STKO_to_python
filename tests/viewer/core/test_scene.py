"""Tests for the :class:`Scene` orchestrator."""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.viewer.core import BBox, Scene, SceneStyle

from .conftest import RecordingLayer, StubBackend, StubDataSource


# --------------------------------------------------------------------- #
# Construction + handle creation                                        #
# --------------------------------------------------------------------- #


def test_scene_constructs_without_calling_backend(
    stub_backend, stub_source,
) -> None:
    """The backend isn't touched until something asks for the handle."""
    Scene(stub_backend, stub_source)
    assert stub_backend.make_scene_calls == 0


def test_scene_handle_is_lazy(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    assert stub_backend.make_scene_calls == 0
    _ = scene.handle
    assert stub_backend.make_scene_calls == 1
    _ = scene.handle  # cached
    assert stub_backend.make_scene_calls == 1


def test_scene_handle_set_style_runs_once(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source, style=SceneStyle(background="black"))
    _ = scene.handle
    assert len(stub_backend.set_style_calls) == 1
    style_passed = stub_backend.set_style_calls[0][1]
    assert style_passed.background == "black"


def test_scene_default_style_is_empty_scene_style(
    stub_backend, stub_source,
) -> None:
    scene = Scene(stub_backend, stub_source)
    assert scene.style == SceneStyle()


# --------------------------------------------------------------------- #
# Layer management                                                      #
# --------------------------------------------------------------------- #


def test_scene_add_attaches_layer_and_appends(
    stub_backend, stub_source,
) -> None:
    scene = Scene(stub_backend, stub_source)
    layer = RecordingLayer(name="A")
    returned = scene.add(layer)
    assert returned is layer
    assert layer in scene
    assert len(scene) == 1
    assert layer.is_attached
    assert len(layer.attach_calls) == 1


def test_scene_add_forces_handle_before_layer_attach(
    stub_backend, stub_source,
) -> None:
    """Layers expect ``scene.handle`` to exist when ``attach`` runs."""
    scene = Scene(stub_backend, stub_source)
    layer = RecordingLayer()
    scene.add(layer)
    # The scene handle was created (so layer.attach could use backend).
    assert stub_backend.make_scene_calls == 1


def test_scene_add_after_set_step_advances_new_layer(
    stub_backend, stub_source,
) -> None:
    """A layer added after the cursor has moved is fast-forwarded."""
    scene = Scene(stub_backend, stub_source)
    scene.set_step(7)
    layer = RecordingLayer()
    scene.add(layer)
    assert layer.update_calls == [7]


def test_scene_remove_detaches_layer(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    layer = RecordingLayer()
    scene.add(layer)
    scene.remove(layer)
    assert layer not in scene
    assert layer.detach_calls == 1
    assert layer.is_attached is False


def test_scene_remove_unknown_layer_raises(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    orphan = RecordingLayer(name="orphan")
    with pytest.raises(ValueError, match="not in this scene"):
        scene.remove(orphan)


# --------------------------------------------------------------------- #
# Step / stage cursor                                                   #
# --------------------------------------------------------------------- #


def test_scene_set_step_propagates_to_every_layer(
    stub_backend, stub_source,
) -> None:
    scene = Scene(stub_backend, stub_source)
    a = RecordingLayer(name="A")
    b = RecordingLayer(name="B")
    scene.add(a)
    scene.add(b)
    scene.set_step(3)
    assert a.update_calls == [3]
    assert b.update_calls == [3]
    assert scene.current_step == 3


def test_scene_set_step_fires_invisible_layers_too(
    stub_backend, stub_source,
) -> None:
    """Per the perf contract, invisible layers still get fresh data."""
    scene = Scene(stub_backend, stub_source)
    layer = RecordingLayer(visible=False)
    scene.add(layer)
    scene.set_step(5)
    assert layer.update_calls == [5]


def test_scene_set_stage_stores_stage(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    scene.set_stage("STAGE_1")
    assert scene.current_stage == "STAGE_1"


# --------------------------------------------------------------------- #
# Bounds + output                                                       #
# --------------------------------------------------------------------- #


def test_scene_fit_bounds_pulls_bbox_from_source(
    stub_backend, stub_source,
) -> None:
    scene = Scene(stub_backend, stub_source)
    scene.fit_bounds()
    assert len(stub_backend.set_bounds_calls) == 1
    _, bbox_passed = stub_backend.set_bounds_calls[0]
    assert bbox_passed == BBox(-1, -2, -3, 4, 5, 6)


def test_scene_show_delegates_to_backend(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    scene.show()
    assert stub_backend.show_calls == 1


def test_scene_save_passes_dpi(stub_backend, stub_source, tmp_path) -> None:
    scene = Scene(stub_backend, stub_source)
    out = tmp_path / "frame.png"
    scene.save(out, dpi=150)
    assert len(stub_backend.save_calls) == 1
    scene_arg, path_arg, dpi_arg = stub_backend.save_calls[0]
    assert path_arg == out
    assert dpi_arg == 150


def test_scene_snapshot_returns_backend_array(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    arr = scene.snapshot()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape[-1] == 3


# --------------------------------------------------------------------- #
# Container protocol                                                    #
# --------------------------------------------------------------------- #


def test_scene_iteration_preserves_insertion_order(
    stub_backend, stub_source,
) -> None:
    scene = Scene(stub_backend, stub_source)
    a = RecordingLayer(name="A")
    b = RecordingLayer(name="B")
    c = RecordingLayer(name="C")
    scene.add(a); scene.add(b); scene.add(c)
    assert list(scene) == [a, b, c]


def test_scene_len_matches_layer_count(stub_backend, stub_source) -> None:
    scene = Scene(stub_backend, stub_source)
    assert len(scene) == 0
    scene.add(RecordingLayer())
    assert len(scene) == 1
