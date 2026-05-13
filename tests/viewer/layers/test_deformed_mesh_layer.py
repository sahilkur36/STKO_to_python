"""Tests for :class:`DeformedMeshLayer`.

Three flavours of coverage:

1. **Unit tests** against a fake :class:`MPCODataSet` with a
   monkey-patched ``_displacement_at_step`` so the layer can be
   exercised without an HDF5 fixture.
2. **Backend-integration tests** that drive the real :class:`Scene`
   and :class:`MplBackend` to confirm actor types, zorder, in-place
   ``update_to_step`` semantics, and detach cleanup.
3. **Real-fixture parity** against ``elastic_frame_dir`` — proves the
   Phase 2.5 rewire of ``ds.plot.deformed_shape`` keeps the legacy
   meta contract intact and that animation via
   :meth:`DeformedMeshLayer.update_to_step` produces the same
   geometry the legacy ``ds.plot.deformed_shape(step=N)`` produces.
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
from STKO_to_python.viewer.layers import DeformedMeshLayer
from STKO_to_python.viewer.layers import deformed_mesh as deformed_mesh_module


# --------------------------------------------------------------------- #
# Fake-dataset helpers
# --------------------------------------------------------------------- #


def _make_fake_dataset(
    *,
    node_rows=None,
    elem_rows=None,
):
    """Tiny stand-in for :class:`MPCODataSet`."""
    if node_rows is None:
        node_rows = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
        ]
    if elem_rows is None:
        elem_rows = [
            (10, "5-ElasticBeam3d", 2, (1, 2)),
            (11, "203-ASDShellQ4", 4, (1, 2, 3, 4)),
        ]
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "x", "y", "z"])
    elements_df = pd.DataFrame(
        elem_rows,
        columns=["element_id", "element_type", "num_nodes", "node_list"],
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
    fake._selection_resolver = None
    return fake


def _make_scene(fake_dataset, *, is_3d=False, ax=None):
    backend = MplBackend()
    handle = backend.make_scene(is_3d=is_3d, ax=ax)
    source = MPCODataSourceAdapter(fake_dataset)
    return Scene(backend, source, is_3d=is_3d, handle=handle)


def _disp_factory(disp_per_step):
    """Build a ``_displacement_at_step`` stand-in keyed on ``step``.

    ``disp_per_step``: ``{step: {node_id: ndarray(3)}}``.
    """

    def _stub(dataset, *, model_stage, step):
        if step not in disp_per_step:
            raise ValueError(
                f"step={step} not present in DISPLACEMENT for stage "
                f"{model_stage!r}."
            )
        return {nid: np.asarray(v, dtype=np.float64)
                for nid, v in disp_per_step[step].items()}

    return _stub


# --------------------------------------------------------------------- #
# Construction / contract
# --------------------------------------------------------------------- #


def test_kind_is_deformed_mesh() -> None:
    assert DeformedMeshLayer.kind == "deformed_mesh"


def test_default_styling_matches_legacy_plot_deformed_shape() -> None:
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0)
    assert layer.edge_color == "C0"
    assert layer.linewidth == 1.2
    assert layer.alpha == 1.0
    assert layer.mpl_zorder == 2.0


def test_summary_is_empty_before_attach() -> None:
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=5, scale=10.0)
    assert layer.current_step is None
    assert layer.segment_count == 0
    assert layer.edges_per_class == {}
    assert layer.skipped_classes == []
    assert layer.actors == {}
    assert layer.original_coords == {}
    assert layer.deformed_coords == {}


def test_init_preserves_model_stage_scale_step() -> None:
    layer = DeformedMeshLayer(
        model_stage="MODEL_STAGE[1]", step=7, scale=50.0,
    )
    assert layer.model_stage == "MODEL_STAGE[1]"
    assert layer.scale == 50.0
    # _initial_step is internal; current_step is exposed.
    assert layer.current_step is None  # not yet attached


# --------------------------------------------------------------------- #
# Attach lifecycle — coords + segments
# --------------------------------------------------------------------- #


def test_attach_with_scale_zero_collapses_to_original(monkeypatch) -> None:
    """``scale=0`` must skip the DISPLACEMENT fetch entirely."""

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("displacement fetch must not run when scale=0")

    monkeypatch.setattr(
        deformed_mesh_module, "_displacement_at_step", _should_not_be_called,
    )

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=5, scale=0.0)
    try:
        scene.add(layer)
        for nid, orig in layer.original_coords.items():
            np.testing.assert_array_equal(layer.deformed_coords[nid], orig)
        assert layer.current_step == 5
    finally:
        plt.close("all")


def test_attach_with_none_step_collapses_to_original(monkeypatch) -> None:
    """``step=None`` is the same shortcut — useful for the
    ``show_undeformed`` overlay case in plot_deformed_shape."""

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("displacement fetch must not run when step=None")

    monkeypatch.setattr(
        deformed_mesh_module, "_displacement_at_step", _should_not_be_called,
    )

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=None, scale=1.0)
    try:
        scene.add(layer)
        for nid, orig in layer.original_coords.items():
            np.testing.assert_array_equal(layer.deformed_coords[nid], orig)
    finally:
        plt.close("all")


def test_attach_applies_displacement_at_initial_step(monkeypatch) -> None:
    """deformed_coords[nid] == original_coords[nid] + scale * disp."""
    disp_per_step = {
        5: {
            1: np.array([0.1, 0.0, 0.0]),
            2: np.array([0.2, 0.0, 0.0]),
            3: np.array([0.3, 0.0, 0.0]),
            4: np.array([0.4, 0.0, 0.0]),
        }
    }
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory(disp_per_step),
    )

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=5, scale=10.0)
    try:
        scene.add(layer)
        np.testing.assert_allclose(
            layer.deformed_coords[2],
            np.array([1.0 + 10.0 * 0.2, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            layer.deformed_coords[3],
            np.array([1.0 + 10.0 * 0.3, 1.0, 0.0]),
        )
        assert layer.current_step == 5
    finally:
        plt.close("all")


def test_attach_skips_nodes_without_displacement(monkeypatch) -> None:
    """Missing nodes use their original coordinates — same shortcut
    as the legacy renderer."""
    disp_per_step = {
        5: {1: np.array([0.5, 0.0, 0.0])}  # only node 1 has disp
    }
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory(disp_per_step),
    )

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=5, scale=1.0)
    try:
        scene.add(layer)
        np.testing.assert_allclose(layer.deformed_coords[1], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(layer.deformed_coords[2], [1.0, 0.0, 0.0])  # unchanged
    finally:
        plt.close("all")


def test_attach_populates_actor_per_element_class(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({5: {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0], 4: [0, 0, 0]}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=5, scale=1.0)
    try:
        scene.add(layer)
        assert set(layer.actors.keys()) == {
            "5-ElasticBeam3d(2n)",
            "203-ASDShellQ4(4n)",
        }
        # 1 beam edge + 4 quad edges = 5 segments.
        assert layer.segment_count == 5
    finally:
        plt.close("all")


def test_attach_raises_when_dataset_is_empty() -> None:
    fake = _make_fake_dataset(node_rows=[], elem_rows=[])
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=0.0)
    try:
        with pytest.raises(LayerAttachError, match="no nodes or no elements"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_twice_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: [0, 0, 0] for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        with pytest.raises(LayerAttachError, match="already attached"):
            layer.attach(scene, scene.source)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Backend interaction
# --------------------------------------------------------------------- #


def test_2d_attach_emits_line_collection_per_class(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: [0, 0, 0] for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake, is_3d=False)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, LineCollection)
            assert not isinstance(actor, Line3DCollection)
    finally:
        plt.close("all")


def test_3d_attach_emits_line3dcollection_per_class(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: [0, 0, 0] for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake, is_3d=True)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, Line3DCollection)
    finally:
        plt.close("all")


def test_default_mpl_zorder_sits_above_undeformed_overlay(monkeypatch) -> None:
    """zorder=2.0 keeps the deformed wireframe in front of the
    undeformed overlay (which sits at 1.0)."""
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: [0, 0, 0] for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert actor.get_zorder() == pytest.approx(2.0)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# update_to_step — the apeGmsh perf contract
# --------------------------------------------------------------------- #


def test_update_to_step_mutates_segments_without_recreating_actors(
    monkeypatch,
) -> None:
    """The perf contract: a step change must update existing actors,
    not recreate them. We verify by holding the actor instance across
    a step change and re-checking ``id()``."""
    disp = {
        0: {nid: np.zeros(3) for nid in (1, 2, 3, 4)},
        5: {
            1: np.array([0.1, 0.0, 0.0]),
            2: np.array([0.2, 0.0, 0.0]),
            3: np.array([0.3, 0.0, 0.0]),
            4: np.array([0.4, 0.0, 0.0]),
        },
    }
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory(disp),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=10.0)
    try:
        scene.add(layer)
        beam_actor_before = layer.actors["5-ElasticBeam3d(2n)"]
        # Snapshot the beam's segment ([node 1, node 2]).
        before = np.asarray(beam_actor_before.get_segments())
        layer.update_to_step(5)
        beam_actor_after = layer.actors["5-ElasticBeam3d(2n)"]
        # Same Python object — actor was not recreated.
        assert beam_actor_after is beam_actor_before
        # Segments moved with the displacement.
        after = np.asarray(beam_actor_after.get_segments())
        assert not np.allclose(before, after)
        assert layer.current_step == 5
        # And the deformed_coords dict tracks the new step.
        np.testing.assert_allclose(
            layer.deformed_coords[2], np.array([1.0 + 10.0 * 0.2, 0.0, 0.0]),
        )
    finally:
        plt.close("all")


def test_update_to_step_is_no_op_when_step_unchanged(monkeypatch) -> None:
    call_count = {"n": 0}
    real_factory = _disp_factory({0: {nid: np.zeros(3) for nid in (1, 2, 3, 4)}})

    def _counting_fetch(*args, **kwargs):
        call_count["n"] += 1
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(
        deformed_mesh_module, "_displacement_at_step", _counting_fetch,
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        n_attach = call_count["n"]
        layer.update_to_step(0)  # same step
        assert call_count["n"] == n_attach  # no extra fetch
    finally:
        plt.close("all")


def test_update_to_step_no_op_before_attach() -> None:
    """A pre-attach update_to_step must be silent, not crash."""
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    layer.update_to_step(5)  # nothing happens — not attached
    assert layer.current_step is None


# --------------------------------------------------------------------- #
# detach
# --------------------------------------------------------------------- #


def test_detach_clears_actors_and_state(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: np.zeros(3) for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(model_stage="STAGE_0", step=0, scale=1.0)
    try:
        scene.add(layer)
        assert layer.segment_count > 0
        scene.remove(layer)
        assert not layer.is_attached
        assert layer.actors == {}
        assert layer.segment_count == 0
        assert layer.edges_per_class == {}
        assert layer.original_coords == {}
        assert layer.deformed_coords == {}
        assert layer.current_step is None
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Selection-driven subset rendering
# --------------------------------------------------------------------- #


def test_selection_by_element_ids_subsets_the_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        deformed_mesh_module,
        "_displacement_at_step",
        _disp_factory({0: {nid: np.zeros(3) for nid in (1, 2, 3, 4)}}),
    )
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = DeformedMeshLayer(
        model_stage="STAGE_0", step=0, scale=1.0,
        selection=SelectionSpec(element_ids=(10,)),
    )
    try:
        scene.add(layer)
        assert set(layer.edges_per_class.keys()) == {"5-ElasticBeam3d(2n)"}
        assert layer.segment_count == 1
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Real-fixture parity — proves the Phase 2.5 rewire is byte-identical
# --------------------------------------------------------------------- #


def test_plot_deformed_shape_rewire_returns_legacy_meta(
    elastic_frame_dir: Path,
) -> None:
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.deformed_shape(
            model_stage="MODEL_STAGE[1]", step=5, scale=10.0,
        )
        # Legacy meta surface.
        assert set(meta.keys()) >= {
            "deformed_coords",
            "original_coords",
            "edges_per_class",
            "segment_count",
            "skipped_classes",
            "is_3d",
            "scale",
            "step",
            "model_stage",
        }
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["segment_count"] == 3
        assert meta["skipped_classes"] == []
        assert meta["is_3d"] is True
        assert meta["scale"] == 10.0
        assert meta["step"] == 5
        assert meta["model_stage"] == "MODEL_STAGE[1]"
        # The undeformed overlay sits below the deformed mesh.
        line3d_collections = [
            c for c in ax.collections if isinstance(c, Line3DCollection)
        ]
        zorders = sorted(c.get_zorder() for c in line3d_collections)
        assert zorders == [pytest.approx(1.0), pytest.approx(2.0)]
    finally:
        plt.close("all")


def test_animation_step_change_matches_one_shot_renders(
    elastic_frame_dir: Path,
) -> None:
    """update_to_step on a live layer must produce the same geometry
    as a fresh ``ds.plot.deformed_shape(step=N)`` would. This is the
    animation contract."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    backend = MplBackend()
    fig = plt.figure()
    user_ax = fig.add_subplot(111, projection="3d")
    handle = backend.make_scene(is_3d=True, ax=user_ax)
    source = MPCODataSourceAdapter(ds)
    scene = Scene(backend, source, is_3d=True, handle=handle)

    try:
        layer = DeformedMeshLayer(
            model_stage="MODEL_STAGE[1]", step=2, scale=10.0,
        )
        scene.add(layer)
        coords_step2 = layer.deformed_coords
        # Advance to step 7 — same actor mutated in place.
        layer.update_to_step(7)
        coords_step7 = layer.deformed_coords
        assert layer.current_step == 7

        # Independent one-shot render at step 7 — must match the
        # animated layer's coords.
        _, meta_step7_one_shot = ds.plot.deformed_shape(
            model_stage="MODEL_STAGE[1]", step=7, scale=10.0,
            show_undeformed=False,
        )
        for nid, xyz in meta_step7_one_shot["deformed_coords"].items():
            np.testing.assert_allclose(
                coords_step7[int(nid)], xyz, atol=1e-12,
            )
        # The two animation snapshots were independent dicts — the
        # ``deformed_coords`` property returns a defensive copy, so
        # update_to_step must not have aliased into the earlier
        # snapshot.
        assert coords_step2 is not coords_step7
    finally:
        plt.close("all")
