"""Tests for :class:`LayerStyle` and :class:`SceneStyle`."""
from __future__ import annotations

import pytest

from STKO_to_python.viewer.core import LayerStyle, SceneStyle


def test_layer_style_defaults_are_none() -> None:
    style = LayerStyle()
    assert style.color is None
    assert style.linewidth is None
    assert style.alpha is None


def test_layer_style_is_frozen() -> None:
    style = LayerStyle(color="red")
    with pytest.raises(Exception):
        style.color = "blue"  # type: ignore[misc]


def test_layer_style_merge_layer_wins_when_set() -> None:
    base = LayerStyle(color="blue", linewidth=1.0)
    override = LayerStyle(color="red")
    merged = override.merge(base)
    assert merged.color == "red"       # override wins
    assert merged.linewidth == 1.0     # inherited from base


def test_layer_style_merge_inherits_all_when_override_is_empty() -> None:
    base = LayerStyle(color="blue", linewidth=2.5, alpha=0.5)
    merged = LayerStyle().merge(base)
    assert merged.color == "blue"
    assert merged.linewidth == 2.5
    assert merged.alpha == 0.5


def test_layer_style_merge_does_not_mutate_inputs() -> None:
    base = LayerStyle(color="blue")
    override = LayerStyle(color="red")
    merged = override.merge(base)
    assert base.color == "blue"
    assert override.color == "red"
    assert merged.color == "red"


def test_scene_style_get_defaults_for_missing_kind_returns_empty() -> None:
    scene = SceneStyle()
    empty = scene.get_defaults_for("nonexistent_layer_kind")
    assert isinstance(empty, LayerStyle)
    assert empty == LayerStyle()


def test_scene_style_get_defaults_for_known_kind() -> None:
    scene = SceneStyle(
        layer_defaults={"mesh": LayerStyle(color="lightgray", linewidth=0.5)},
    )
    default = scene.get_defaults_for("mesh")
    assert default.color == "lightgray"
    assert default.linewidth == 0.5


def test_scene_style_with_layer_default_returns_new_instance() -> None:
    original = SceneStyle()
    updated = original.with_layer_default("mesh", LayerStyle(color="red"))
    # Original unchanged.
    assert original.layer_defaults == {}
    # New scene has the new default.
    assert updated.get_defaults_for("mesh").color == "red"


def test_scene_style_with_layer_default_overrides_existing() -> None:
    scene = SceneStyle(
        layer_defaults={"mesh": LayerStyle(color="blue")},
    )
    updated = scene.with_layer_default("mesh", LayerStyle(color="red"))
    assert updated.get_defaults_for("mesh").color == "red"
