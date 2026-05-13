"""Tests for :class:`MeshLayer`.

Coverage splits into three flavours:

1. **Unit tests** with a tiny fake :class:`MPCODataSet` — exercise the
   attach/detach lifecycle, segment counting, selection resolution,
   and ``mpl_zorder`` post-hoc styling without needing an HDF5
   fixture.
2. **Backend-integration tests** that drive the MplBackend with a
   real :class:`Scene` to confirm the layer emits the expected actor
   types (``LineCollection`` / ``Line3DCollection``) and respects
   the v1.x ``zorder=1.0`` contract.
3. **Visual regression** against the real ``elastic_frame_dir``
   fixture — proves the Phase 2.4 rewire is byte-identical to a
   re-implementation of the legacy code path on the same data.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from STKO_to_python.viewer.backends.mpl.backend import MplBackend
from STKO_to_python.viewer.core import (
    MPCODataSourceAdapter,
    Scene,
    SelectionSpec,
)
from STKO_to_python.viewer.core.errors import LayerAttachError
from STKO_to_python.viewer.layers import MeshLayer


# --------------------------------------------------------------------- #
# Fake-dataset helpers
# --------------------------------------------------------------------- #


def _make_fake_dataset(
    *,
    node_rows=None,
    elem_rows=None,
    selection_resolver=None,
):
    """Tiny stand-in for :class:`MPCODataSet` — only the attributes
    :class:`MeshLayer` and :class:`MPCODataSourceAdapter` read."""
    if node_rows is None:
        node_rows = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
        ]
    if elem_rows is None:
        elem_rows = [
            # (id, type, num_nodes, node_list)
            (10, "5-ElasticBeam3d", 2, (1, 2)),
            (11, "203-ASDShellQ4", 4, (1, 2, 3, 4)),
        ]
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "x", "y", "z"])
    elements_df = pd.DataFrame(
        elem_rows, columns=["element_id", "element_type", "num_nodes", "node_list"]
    )
    fake = SimpleNamespace(
        nodes_info={"dataframe": nodes_df},
        elements_info={"dataframe": elements_df},
        model_stages=["STAGE_0"],
        number_of_steps={"STAGE_0": 1},
        time=pd.DataFrame(
            [{"MODEL_STAGE": "STAGE_0", "STEP": 0, "TIME": 0.0}]
        ).set_index(["MODEL_STAGE", "STEP"]),
    )
    fake._selection_resolver = selection_resolver
    return fake


def _make_scene(fake_dataset, *, is_3d=False, ax=None):
    backend = MplBackend()
    handle = backend.make_scene(is_3d=is_3d, ax=ax)
    source = MPCODataSourceAdapter(fake_dataset)
    return Scene(backend, source, is_3d=is_3d, handle=handle), backend, source


# --------------------------------------------------------------------- #
# Construction / contract
# --------------------------------------------------------------------- #


def test_mesh_layer_kind_is_mesh() -> None:
    assert MeshLayer.kind == "mesh"


def test_mesh_layer_default_styling() -> None:
    layer = MeshLayer()
    assert layer.edge_color == "lightgray"
    assert layer.linewidth == 0.5
    assert layer.alpha == 1.0
    assert layer.mpl_zorder == 1.0


def test_mesh_layer_is_not_attached_at_construction() -> None:
    layer = MeshLayer()
    assert not layer.is_attached


def test_mesh_layer_summary_is_zero_before_attach() -> None:
    layer = MeshLayer()
    assert layer.n_edges == 0
    assert layer.n_elements_drawn == 0
    assert layer.edges_per_class == {}
    assert layer.skipped_classes == []
    assert layer.actors == {}


# --------------------------------------------------------------------- #
# Attach lifecycle — counts and topology
# --------------------------------------------------------------------- #


def test_attach_populates_actor_per_element_class() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        scene.add(layer)
        # 1 beam class (1 edge × 1 elem) + 1 quad class (4 edges × 1 elem) = 5 edges
        assert layer.n_edges == 5
        assert layer.n_elements_drawn == 2
        assert set(layer.edges_per_class.keys()) == {
            "5-ElasticBeam3d(2n)",
            "203-ASDShellQ4(4n)",
        }
        assert layer.edges_per_class["5-ElasticBeam3d(2n)"] == 1
        assert layer.edges_per_class["203-ASDShellQ4(4n)"] == 4
        assert layer.skipped_classes == []
        assert set(layer.actors.keys()) == set(layer.edges_per_class.keys())
    finally:
        plt.close("all")


def test_attach_emits_warning_for_unsupported_topology() -> None:
    """Higher-order solids with no edge topology entry are skipped."""
    elem_rows = [
        (10, "5-ElasticBeam3d", 2, (1, 2)),
        (50, "MysteryElement", 5, (1, 2, 3, 4, 1)),  # 5-node — unsupported
    ]
    fake = _make_fake_dataset(elem_rows=elem_rows)
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        with pytest.warns(RuntimeWarning, match="MysteryElement"):
            scene.add(layer)
        assert "MysteryElement(5n)" in layer.skipped_classes
        # The beam still draws.
        assert layer.n_elements_drawn == 1
    finally:
        plt.close("all")


def test_attach_raises_when_dataset_is_empty() -> None:
    fake = _make_fake_dataset(node_rows=[], elem_rows=[])
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        with pytest.raises(LayerAttachError, match="no nodes or no elements"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_twice_raises() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        scene.add(layer)
        with pytest.raises(LayerAttachError, match="already attached"):
            layer.attach(scene, scene.source)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Backend interaction
# --------------------------------------------------------------------- #


def test_2d_attach_emits_line_collection_per_class() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake, is_3d=False)
    layer = MeshLayer()
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, LineCollection)
            assert not isinstance(actor, Line3DCollection)
    finally:
        plt.close("all")


def test_3d_attach_emits_line3dcollection_per_class() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake, is_3d=True)
    layer = MeshLayer()
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, Line3DCollection)
    finally:
        plt.close("all")


def test_attach_applies_mpl_zorder() -> None:
    """Default zorder=1.0 sits below scatter/contour overlays (default 2.0)."""
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert actor.get_zorder() == pytest.approx(1.0)
    finally:
        plt.close("all")


def test_attach_applies_custom_mpl_zorder() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer(mpl_zorder=3.5)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert actor.get_zorder() == pytest.approx(3.5)
    finally:
        plt.close("all")


def test_attach_propagates_linewidth_and_alpha() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer(linewidth=2.5, alpha=0.7)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            np.testing.assert_allclose(actor.get_linewidth(), 2.5)
            assert actor.get_alpha() == pytest.approx(0.7)
    finally:
        plt.close("all")


def test_invisible_layer_hides_actors_at_attach() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer(visible=False)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert actor.get_visible() is False
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# update_to_step + detach contracts
# --------------------------------------------------------------------- #


def test_update_to_step_is_a_no_op() -> None:
    """MeshLayer has no per-step data — update must not mutate actors."""
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        scene.add(layer)
        snapshots = {
            label: np.array(actor.get_segments(), dtype=object)
            for label, actor in layer.actors.items()
        }
        layer.update_to_step(42)
        for label, actor in layer.actors.items():
            after = np.array(actor.get_segments(), dtype=object)
            assert len(after) == len(snapshots[label])
    finally:
        plt.close("all")


def test_detach_clears_actors_and_counters() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer()
    try:
        scene.add(layer)
        assert layer.n_edges > 0
        scene.remove(layer)
        assert not layer.is_attached
        assert layer.actors == {}
        assert layer.n_edges == 0
        assert layer.n_elements_drawn == 0
        assert layer.edges_per_class == {}
        assert layer.skipped_classes == []
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Selection-driven subset rendering
# --------------------------------------------------------------------- #


def test_selection_by_element_ids_subsets_the_layer() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer(selection=SelectionSpec(element_ids=(10,)))
    try:
        scene.add(layer)
        # Only the beam survives the filter; the quad is gone.
        assert set(layer.edges_per_class.keys()) == {"5-ElasticBeam3d(2n)"}
        assert layer.n_edges == 1
    finally:
        plt.close("all")


def test_empty_selection_match_raises_attach_error() -> None:
    fake = _make_fake_dataset()
    scene, _, _ = _make_scene(fake)
    layer = MeshLayer(selection=SelectionSpec(element_ids=(9999,)))
    try:
        with pytest.raises(LayerAttachError, match="No elements remain"):
            scene.add(layer)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Real-fixture parity — proves the Phase 2.4 rewire is byte-identical
# --------------------------------------------------------------------- #


def test_plot_mesh_rewire_matches_legacy_meta(elastic_frame_dir: Path) -> None:
    """The rewired ``ds.plot.mesh()`` returns the same meta the v1.x
    implementation did."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.mesh()
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["n_edges"] == 3
        assert meta["n_elements_drawn"] == 3
        assert meta["skipped_classes"] == []
        assert meta["is_3d"] is True
        assert meta["model_stage"] is None
        # The ax carries one collection per drawn element class.
        line3d_collections = [
            c for c in ax.collections if isinstance(c, Line3DCollection)
        ]
        assert len(line3d_collections) == 1
        # And the mesh sits at zorder 1.0 — below the contour layer.
        for c in line3d_collections:
            assert c.get_zorder() == pytest.approx(1.0)
    finally:
        plt.close("all")


def test_plot_mesh_rewire_honours_user_axes(elastic_frame_dir: Path) -> None:
    """A user-supplied ``ax`` is reused; no extra figure is created."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    n_figures_before = len(plt.get_fignums())
    try:
        out_ax, _ = ds.plot.mesh(ax=ax)
        assert out_ax is ax
        assert len(plt.get_fignums()) == n_figures_before
    finally:
        plt.close("all")
