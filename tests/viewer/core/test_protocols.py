"""Tests for the :class:`Backend` and :class:`DataSource` protocols.

Both are ``runtime_checkable`` Protocols, so an unrelated class that
implements the right methods satisfies ``isinstance(stub, Protocol)``.
The conftest stubs (``StubBackend``, ``StubDataSource``) are the proof
of concept that the protocol surfaces are non-trivially satisfiable.
"""
from __future__ import annotations

from STKO_to_python.viewer.core import Backend, DataSource


def test_stub_backend_satisfies_backend_protocol(stub_backend) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(stub_backend, Backend)


def test_stub_source_satisfies_datasource_protocol(stub_source) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(stub_source, DataSource)


def test_unrelated_class_does_not_satisfy_backend_protocol() -> None:
    class NotABackend:
        name = "fake"
    assert not isinstance(NotABackend(), Backend)


def test_unrelated_class_does_not_satisfy_datasource_protocol() -> None:
    class NotASource:
        @property
        def dataset(self):
            return None
    # Missing required methods (node_coords, etc.) — Protocol rejects.
    assert not isinstance(NotASource(), DataSource)


def test_backend_protocol_methods_are_declared() -> None:
    """Pin the protocol surface so accidental removals fail loudly."""
    expected_methods = {
        "make_scene", "set_bounds", "set_camera", "set_style",
        "add_segments", "add_points", "add_polygons", "add_arrows",
        "update_scalars", "update_points", "set_visible", "remove",
        "show", "save", "snapshot",
    }
    declared = set(dir(Backend))
    missing = expected_methods - declared
    assert not missing, f"Backend protocol missing methods: {sorted(missing)}"


def test_datasource_protocol_methods_are_declared() -> None:
    expected_methods = {
        "node_coords", "element_centroids", "model_bbox",
        "n_steps", "time",
        "resolve_node_ids", "resolve_element_ids",
    }
    declared = set(dir(DataSource))
    missing = expected_methods - declared
    assert not missing, f"DataSource protocol missing methods: {sorted(missing)}"
