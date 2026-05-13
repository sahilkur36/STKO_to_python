"""Tests for :class:`MPCODataSourceAdapter`.

The adapter is the only viewer-side module that talks to STKO's
DataFrame world, so it gets two flavors of coverage:

* fake-dataset unit tests — fast, exercise every branch of the
  adapter without depending on any HDF5 fixture;
* a real-fixture smoke test against ``elastic_frame_dir`` to verify
  the adapter binds to the attribute names that the real dataset
  actually exposes (the live contract).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.viewer.core import (
    BBox,
    DataSource,
    MPCODataSourceAdapter,
    SelectionSpec,
)


# --------------------------------------------------------------------- #
# Fake dataset helpers — avoid any HDF5 dependency.
# --------------------------------------------------------------------- #


class _FakeResolver:
    """Mimic the small surface of :class:`SelectionSetResolver` used here."""

    def __init__(
        self,
        node_sets: Dict[str, List[int]] | None = None,
        elem_sets: Dict[str, List[int]] | None = None,
    ) -> None:
        self._node_sets = node_sets or {}
        self._elem_sets = elem_sets or {}

    def resolve_nodes(self, *, names=None, ids=None, explicit_ids=None):
        return self._resolve(self._node_sets, names, ids, "NODES")

    def resolve_elements(self, *, names=None, ids=None, explicit_ids=None):
        return self._resolve(self._elem_sets, names, ids, "ELEMENTS")

    def _resolve(self, store, names, ids, payload_key):
        gathered: list[np.ndarray] = []
        for n in names or ():
            key = str(n).strip().lower()
            if key not in store:
                raise ValueError(f"Selection set name not found: {n!r}")
            gathered.append(np.asarray(store[key], dtype=np.int64))
        for sid in ids or ():
            key = f"#{int(sid)}"
            if key not in store:
                raise ValueError(f"Selection set {sid} empty or missing {payload_key}")
            gathered.append(np.asarray(store[key], dtype=np.int64))
        if not gathered:
            raise ValueError("Provide names, ids, and/or explicit_ids — got none.")
        return np.unique(np.concatenate(gathered))


def _make_fake_dataset(
    *,
    node_rows: list[tuple[int, float, float, float]] | None = None,
    elem_rows: list[tuple[int, str, float, float, float]] | None = None,
    model_stages: list[str] | None = None,
    number_of_steps: dict[str, int] | None = None,
    time_df: pd.DataFrame | None = None,
    resolver: _FakeResolver | None = None,
) -> Any:
    """Build a duck-typed stand-in for :class:`MPCODataSet`.

    Carries only the attributes :class:`MPCODataSourceAdapter` reads —
    everything else is left out so any drift in the contract surfaces
    as a clear ``AttributeError``.
    """
    if node_rows is None:
        node_rows = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 2.0, 0.0),
            (4, 0.0, 2.0, 0.5),
        ]
    if elem_rows is None:
        # (element_id, element_type, centroid_x, centroid_y, centroid_z)
        elem_rows = [
            (10, "5-ElasticBeam3d", 0.5, 0.0, 0.0),
            (11, "203-ASDShellQ4", 0.5, 1.0, 0.25),
            (12, "203-ASDShellQ4", 0.5, 1.5, 0.25),
            (13, "8-Brick", 0.5, 1.5, 0.5),
        ]
    if model_stages is None:
        model_stages = ["STAGE_0"]
    if number_of_steps is None:
        number_of_steps = {s: 3 for s in model_stages}
    if time_df is None:
        rows = []
        for stage in model_stages:
            for step in range(number_of_steps[stage]):
                rows.append({"MODEL_STAGE": stage, "STEP": step, "TIME": float(step) * 0.1})
        time_df = pd.DataFrame(rows).set_index(["MODEL_STAGE", "STEP"]).sort_index()

    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "x", "y", "z"])
    elements_df = pd.DataFrame(
        elem_rows,
        columns=["element_id", "element_type", "centroid_x", "centroid_y", "centroid_z"],
    )

    fake = SimpleNamespace(
        nodes_info={"dataframe": nodes_df},
        elements_info={"dataframe": elements_df},
        model_stages=list(model_stages),
        number_of_steps=dict(number_of_steps),
        time=time_df,
    )
    fake._selection_resolver = resolver if resolver is not None else _FakeResolver()
    return fake


# --------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------- #


def test_adapter_satisfies_datasource_protocol() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    assert isinstance(adapter, DataSource)


def test_adapter_exposes_underlying_dataset() -> None:
    ds = _make_fake_dataset()
    adapter = MPCODataSourceAdapter(ds)
    assert adapter.dataset is ds


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #


def test_node_coords_full_dataset() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    coords = adapter.node_coords()
    assert coords.shape == (4, 3)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(coords[3], [0.0, 2.0, 0.5])


def test_node_coords_preserves_caller_order() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    coords = adapter.node_coords(np.array([3, 1, 2]))
    np.testing.assert_allclose(coords[0], [1.0, 2.0, 0.0])  # id=3
    np.testing.assert_allclose(coords[1], [0.0, 0.0, 0.0])  # id=1
    np.testing.assert_allclose(coords[2], [1.0, 0.0, 0.0])  # id=2


def test_node_coords_empty_id_array_returns_empty() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    coords = adapter.node_coords(np.array([], dtype=np.int64))
    assert coords.shape == (0, 3)


def test_node_coords_raises_on_unknown_id() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    with pytest.raises(KeyError, match="Node id 999"):
        adapter.node_coords(np.array([1, 999]))


def test_node_coords_on_empty_dataset() -> None:
    fake = _make_fake_dataset(node_rows=[])
    adapter = MPCODataSourceAdapter(fake)
    coords = adapter.node_coords()
    assert coords.shape == (0, 3)


def test_element_centroids_full_dataset() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    cents = adapter.element_centroids()
    assert cents.shape == (4, 3)


def test_element_centroids_preserves_caller_order() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    cents = adapter.element_centroids(np.array([12, 10]))
    np.testing.assert_allclose(cents[0], [0.5, 1.5, 0.25])
    np.testing.assert_allclose(cents[1], [0.5, 0.0, 0.0])


def test_element_centroids_raises_on_unknown_id() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    with pytest.raises(KeyError, match="Element id 999"):
        adapter.element_centroids(np.array([10, 999]))


def test_model_bbox_spans_every_node() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    bbox = adapter.model_bbox()
    assert bbox == BBox(0.0, 0.0, 0.0, 1.0, 2.0, 0.5)


def test_model_bbox_is_cached() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    bbox1 = adapter.model_bbox()
    bbox2 = adapter.model_bbox()
    assert bbox1 is bbox2  # cached BBox dataclass, returned by reference


def test_model_bbox_on_empty_dataset_is_origin() -> None:
    fake = _make_fake_dataset(node_rows=[])
    adapter = MPCODataSourceAdapter(fake)
    assert adapter.model_bbox() == BBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------- #
# Time axis
# --------------------------------------------------------------------- #


def test_n_steps_default_stage() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset(number_of_steps={"STAGE_0": 5}))
    assert adapter.n_steps() == 5


def test_n_steps_named_stage() -> None:
    adapter = MPCODataSourceAdapter(
        _make_fake_dataset(
            model_stages=["A", "B"],
            number_of_steps={"A": 2, "B": 7},
        )
    )
    assert adapter.n_steps("A") == 2
    assert adapter.n_steps("B") == 7


def test_n_steps_unknown_stage_raises() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    with pytest.raises(KeyError, match="MISSING"):
        adapter.n_steps("MISSING")


def test_time_returns_step_ascending_values() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    t = adapter.time()
    np.testing.assert_allclose(t, [0.0, 0.1, 0.2])


def test_time_unknown_stage_raises() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    with pytest.raises(KeyError, match="MISSING"):
        adapter.time("MISSING")


# --------------------------------------------------------------------- #
# Selection resolution — empty / id-only specs
# --------------------------------------------------------------------- #


def test_resolve_node_ids_empty_spec_returns_all_nodes() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    out = adapter.resolve_node_ids(SelectionSpec.empty())
    np.testing.assert_array_equal(out, [1, 2, 3, 4])


def test_resolve_element_ids_empty_spec_returns_all_elements() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    out = adapter.resolve_element_ids(SelectionSpec.empty())
    np.testing.assert_array_equal(out, [10, 11, 12, 13])


def test_resolve_node_ids_by_explicit_ids() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(node_ids=(2, 4))
    np.testing.assert_array_equal(adapter.resolve_node_ids(spec), [2, 4])


def test_resolve_node_ids_drops_ids_not_in_dataset() -> None:
    """Resolver is best-effort; explicit ids missing from the model drop out."""
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(node_ids=(2, 999))
    np.testing.assert_array_equal(adapter.resolve_node_ids(spec), [2])


def test_resolve_element_ids_by_explicit_ids() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(element_ids=(10, 12))
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [10, 12])


# --------------------------------------------------------------------- #
# Selection resolution — element_type filter
# --------------------------------------------------------------------- #


def test_resolve_element_ids_by_element_type_base() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(element_type="203-ASDShellQ4")
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [11, 12])


def test_resolve_element_ids_by_element_type_decorated() -> None:
    """The decorated form ``...[4n]`` matches the base type the same way."""
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(element_type="203-ASDShellQ4[4n]")
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [11, 12])


def test_resolve_element_ids_by_element_type_multiple() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(element_type=("203-ASDShellQ4", "5-ElasticBeam3d"))
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [10, 11, 12])


def test_resolve_element_ids_combines_type_and_ids_via_and() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(
        element_type="203-ASDShellQ4",
        element_ids=(11, 13),  # 13 is a Brick — should drop out
    )
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [11])


# --------------------------------------------------------------------- #
# Selection resolution — named sets
# --------------------------------------------------------------------- #


def test_resolve_node_ids_via_selection_set_name() -> None:
    resolver = _FakeResolver(node_sets={"slab": [1, 2, 3]})
    adapter = MPCODataSourceAdapter(_make_fake_dataset(resolver=resolver))
    spec = SelectionSpec(selection_set_name="slab")
    np.testing.assert_array_equal(adapter.resolve_node_ids(spec), [1, 2, 3])


def test_resolve_element_ids_via_selection_set_id() -> None:
    resolver = _FakeResolver(elem_sets={"#7": [11, 12]})
    adapter = MPCODataSourceAdapter(_make_fake_dataset(resolver=resolver))
    spec = SelectionSpec(selection_set_id=7)
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [11, 12])


def test_resolve_combines_selection_set_with_explicit_ids_via_and() -> None:
    resolver = _FakeResolver(node_sets={"slab": [1, 2, 3]})
    adapter = MPCODataSourceAdapter(_make_fake_dataset(resolver=resolver))
    spec = SelectionSpec(selection_set_name="slab", node_ids=(2, 3, 4))
    # 4 is in the dataset but not in the slab set → dropped.
    # 1 is in the slab set but not in the explicit list → dropped.
    np.testing.assert_array_equal(adapter.resolve_node_ids(spec), [2, 3])


def test_node_resolution_ignores_element_only_fields() -> None:
    """``element_type`` is element-only — passing it to a node query is a no-op."""
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(node_ids=(1, 2), element_type="203-ASDShellQ4")
    np.testing.assert_array_equal(adapter.resolve_node_ids(spec), [1, 2])


def test_element_resolution_ignores_node_only_fields() -> None:
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec(element_ids=(10, 11), node_ids=(1,))
    np.testing.assert_array_equal(adapter.resolve_element_ids(spec), [10, 11])


# --------------------------------------------------------------------- #
# Caching behavior
# --------------------------------------------------------------------- #


def test_selection_results_are_cached_per_spec() -> None:
    """A repeated selection skips the resolver — pin the cache contract.

    Mutating the underlying resolver between calls would normally
    change the answer; the cache keeps the first answer.
    """
    resolver = _FakeResolver(node_sets={"slab": [1, 2, 3]})
    adapter = MPCODataSourceAdapter(_make_fake_dataset(resolver=resolver))
    spec = SelectionSpec(selection_set_name="slab")
    first = adapter.resolve_node_ids(spec)
    resolver._node_sets["slab"] = [4]  # would change the answer if not cached
    second = adapter.resolve_node_ids(spec)
    np.testing.assert_array_equal(first, second)


def test_selection_result_is_a_defensive_copy() -> None:
    """Callers may mutate; the cached array must not change."""
    adapter = MPCODataSourceAdapter(_make_fake_dataset())
    spec = SelectionSpec.empty()
    out1 = adapter.resolve_node_ids(spec)
    out1[0] = -1
    out2 = adapter.resolve_node_ids(spec)
    np.testing.assert_array_equal(out2, [1, 2, 3, 4])


# --------------------------------------------------------------------- #
# Real-fixture smoke test — guards the actual MPCODataSet contract.
# --------------------------------------------------------------------- #


def test_adapter_against_elastic_frame_fixture(elastic_frame_dir: Path) -> None:
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    adapter = MPCODataSourceAdapter(ds)

    # Protocol conformance against the live dataset.
    assert isinstance(adapter, DataSource)
    assert adapter.dataset is ds

    # Geometry consistency with the dataset's own info dicts.
    nodes_df = ds.nodes_info["dataframe"]
    elements_df = ds.elements_info["dataframe"]
    coords = adapter.node_coords()
    assert coords.shape == (len(nodes_df), 3)
    np.testing.assert_allclose(
        coords, nodes_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    )
    centroids = adapter.element_centroids()
    assert centroids.shape == (len(elements_df), 3)

    # Subset lookup preserves caller order.
    ids = nodes_df["node_id"].to_numpy()[:3][::-1]
    sub = adapter.node_coords(ids)
    np.testing.assert_allclose(
        sub,
        nodes_df.set_index("node_id").loc[ids, ["x", "y", "z"]].to_numpy(
            dtype=np.float64
        ),
    )

    # Model bbox tracks the node table.
    bbox = adapter.model_bbox()
    assert bbox.x_min == pytest.approx(float(nodes_df["x"].min()))
    assert bbox.z_max == pytest.approx(float(nodes_df["z"].max()))

    # Time axis matches the dataset.
    stage0 = ds.model_stages[0]
    assert adapter.n_steps() == int(ds.number_of_steps[stage0])
    np.testing.assert_allclose(
        adapter.time(), ds.time.loc[stage0]["TIME"].to_numpy(dtype=np.float64)
    )

    # Empty spec yields every entity, sorted ascending.
    all_nodes = adapter.resolve_node_ids(SelectionSpec.empty())
    np.testing.assert_array_equal(
        all_nodes, np.sort(nodes_df["node_id"].to_numpy(dtype=np.int64))
    )
    all_elems = adapter.resolve_element_ids(SelectionSpec.empty())
    np.testing.assert_array_equal(
        all_elems, np.sort(elements_df["element_id"].to_numpy(dtype=np.int64))
    )

    # element_type filter matches the existing ``ds.plot.mesh`` semantics.
    beam_ids_via_adapter = adapter.resolve_element_ids(
        SelectionSpec(element_type="5-ElasticBeam3d")
    )
    beam_ids_expected = elements_df.loc[
        elements_df["element_type"].str.startswith("5-ElasticBeam3d"), "element_id"
    ].to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(beam_ids_via_adapter, np.sort(beam_ids_expected))
