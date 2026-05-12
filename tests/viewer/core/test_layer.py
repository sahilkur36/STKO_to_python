"""Tests for the :class:`Layer` abstract base."""
from __future__ import annotations

import pytest

from STKO_to_python.viewer.core import Layer, LayerStyle, SelectionSpec

from .conftest import RecordingLayer


def test_layer_cannot_be_instantiated_directly() -> None:
    """``Layer`` is abstract; instantiating it must fail."""
    with pytest.raises(TypeError):
        Layer()  # type: ignore[abstract]


def test_recording_layer_is_concrete_and_instantiable() -> None:
    layer = RecordingLayer()
    assert isinstance(layer, Layer)


def test_layer_default_name_is_class_name() -> None:
    layer = RecordingLayer()
    assert layer.name == "RecordingLayer"


def test_layer_explicit_name_overrides_default() -> None:
    layer = RecordingLayer(name="my-mesh")
    assert layer.name == "my-mesh"


def test_layer_default_selection_is_empty_spec() -> None:
    layer = RecordingLayer()
    assert layer.selection == SelectionSpec.empty()
    assert layer.selection.is_empty()


def test_layer_explicit_selection_is_preserved() -> None:
    spec = SelectionSpec(node_ids=(1, 2, 3))
    layer = RecordingLayer(selection=spec)
    assert layer.selection == spec


def test_layer_default_style_is_empty_layer_style() -> None:
    layer = RecordingLayer()
    assert layer.style == LayerStyle()


def test_layer_visibility_defaults_true() -> None:
    layer = RecordingLayer()
    assert layer.visible is True


def test_layer_z_order_defaults_zero() -> None:
    layer = RecordingLayer()
    assert layer.z_order == 0


def test_layer_is_attached_property_reflects_lifecycle() -> None:
    layer = RecordingLayer()
    assert layer.is_attached is False
    layer.attach(scene="fake-scene", source="fake-source")  # type: ignore[arg-type]
    assert layer.is_attached is True
    layer.detach()
    assert layer.is_attached is False


def test_layer_subclass_must_provide_three_abstract_methods() -> None:
    """A subclass missing any of attach/update_to_step/detach can't instantiate."""

    class MissingDetach(Layer):
        kind = "missing"

        def attach(self, scene, source):  # type: ignore[no-untyped-def]
            pass

        def update_to_step(self, step: int) -> None:
            pass

    with pytest.raises(TypeError):
        MissingDetach()  # type: ignore[abstract]


def test_layer_kind_is_a_class_attribute_default() -> None:
    """``Layer.kind`` defaults to ``"layer"``; subclasses override."""
    assert Layer.kind == "layer"
    assert RecordingLayer.kind == "recording"
