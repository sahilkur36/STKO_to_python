"""Tests for :class:`SelectionSpec`."""
from __future__ import annotations

import pytest

from STKO_to_python.viewer.core import SelectionSpec


def test_selection_spec_empty_classmethod() -> None:
    spec = SelectionSpec.empty()
    assert spec.is_empty()
    assert spec == SelectionSpec()


def test_selection_spec_default_construction_is_empty() -> None:
    spec = SelectionSpec()
    assert spec.is_empty()


def test_selection_spec_is_not_empty_when_any_field_set() -> None:
    assert not SelectionSpec(selection_set_name="slab").is_empty()
    assert not SelectionSpec(node_ids=(1, 2, 3)).is_empty()
    assert not SelectionSpec(element_type="203-ASDShellQ4").is_empty()


def test_selection_spec_is_frozen() -> None:
    spec = SelectionSpec(node_ids=(1, 2))
    with pytest.raises(Exception):
        spec.node_ids = (3,)  # type: ignore[misc]


def test_selection_spec_is_hashable() -> None:
    a = SelectionSpec(node_ids=(1, 2, 3))
    b = SelectionSpec(node_ids=(1, 2, 3))
    c = SelectionSpec(node_ids=(1, 2, 4))
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)
    # Suitable as a dict / set key — used downstream as cache key.
    assert {a, b, c} == {a, c}


def test_selection_spec_combines_multiple_fields_via_and() -> None:
    """Multiple non-None fields are AND-combined at resolution time.

    This test pins the documented semantics — actual resolution happens
    in the adapter layer; here we just confirm both fields persist.
    """
    spec = SelectionSpec(
        selection_set_name="slab",
        element_type="203-ASDShellQ4",
    )
    assert spec.selection_set_name == "slab"
    assert spec.element_type == "203-ASDShellQ4"
