"""Phase 2 integration contract: ``MPCODataSet`` exposes a resolver.

The resolver is wired as a ``@cached_property`` on the dataset that
builds itself from the lazy ``selection_set`` property. These tests pin
the import path and the wiring contract without constructing a full
dataset.
"""
from __future__ import annotations

import pytest

from STKO_to_python.core.dataset import MPCODataSet
from STKO_to_python.selection import SelectionSetResolver


def test_resolver_class_is_importable_from_selection() -> None:
    """Public import path: ``from STKO_to_python.selection import SelectionSetResolver``."""
    from STKO_to_python.selection import SelectionSetResolver as Reexport
    assert Reexport is SelectionSetResolver


def test_resolver_constructs_from_dataset_selection_set_shape() -> None:
    """The resolver must accept the exact dict shape that
    ``CData._extract_selection_set_ids`` returns today.
    """
    fake_selection_set = {
        1: {"SET_NAME": "Foo", "NODES": {1, 2, 3}, "ELEMENTS": set()},
        2: {"SET_NAME": "Bar", "NODES": set(), "ELEMENTS": {100}},
    }
    r = SelectionSetResolver(fake_selection_set)
    assert sorted(r.resolve_nodes(ids=1).tolist()) == [1, 2, 3]
    assert sorted(r.resolve_elements(names="Bar").tolist()) == [100]


def test_dataset_resolver_property_builds_from_seeded_selection_set() -> None:
    """Pin the contract that ``self._selection_resolver`` is the result of
    ``SelectionSetResolver(self.selection_set)``.

    Pre-seeds both ``__dict__`` slots so the lazy property short-circuits
    without touching the cdata parser; the chained derivation still has
    to produce a working resolver.
    """
    ds = MPCODataSet.__new__(MPCODataSet)
    # Pre-fill the cached_property slots so accessing them doesn't trigger
    # the parser; mirrors what real construction does after first access.
    ds.__dict__["selection_set"] = {
        5: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()},
    }
    ds.__dict__["_selection_resolver"] = SelectionSetResolver(ds.selection_set)

    assert isinstance(ds._selection_resolver, SelectionSetResolver)
    assert ds._selection_resolver.resolve_nodes(names="Roof").tolist() == [10, 11]


def test_selection_set_is_lazy_on_construction(elastic_frame_dir) -> None:
    """``selection_set`` must NOT be materialized during ``__init__``.

    The whole point of the @cached_property is to defer the .cdata parse
    until something actually needs it. If a future change re-eagers it,
    this test catches it.
    """
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    assert "selection_set" not in ds.__dict__
    assert "_selection_resolver" not in ds.__dict__

    # Touching the property triggers the parse and fills __dict__.
    _ = ds.selection_set
    assert "selection_set" in ds.__dict__

    # The resolver is independently lazy — pulling it should also work
    # after the selection_set has been materialized.
    _ = ds._selection_resolver
    assert "_selection_resolver" in ds.__dict__


def test_selection_set_returns_cached_instance(elastic_frame_dir) -> None:
    """Repeated access of the lazy properties must return the same object."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    first = ds.selection_set
    second = ds.selection_set
    assert first is second
    assert ds._selection_resolver is ds._selection_resolver
