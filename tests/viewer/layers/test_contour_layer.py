"""Tests for :class:`ContourLayer`.

Three flavours of coverage:

1. **Unit tests** with a tiny fake :class:`MPCODataSet` — exercise
   attach, class-bucketing, skipped classes, clim auto-derivation,
   static vs callable scalar binding, and lookup-failure errors.
2. **Backend-integration tests** that drive the real :class:`Scene`
   + :class:`MplBackend` to confirm actor types
   (:class:`~matplotlib.collections.PolyCollection`), zorder, clim,
   and in-place ``update_scalars`` on the same actor instance.
3. **Real-fixture parity** against the multi-partition QuadFrame —
   proves the layer renders shells from a live ``MPCODataSet``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from STKO_to_python.viewer.backends.mpl.backend import MplBackend
from STKO_to_python.viewer.core import (
    MPCODataSourceAdapter,
    Scene,
    SelectionSpec,
)
from STKO_to_python.viewer.core.errors import LayerAttachError
from STKO_to_python.viewer.layers import ContourLayer


# --------------------------------------------------------------------- #
# Fake-dataset helpers
# --------------------------------------------------------------------- #


def _make_fake_dataset(
    *,
    node_rows=None,
    elem_rows=None,
):
    if node_rows is None:
        node_rows = [
            (1, 0.0, 0.0, 0.0),
            (2, 1.0, 0.0, 0.0),
            (3, 1.0, 1.0, 0.0),
            (4, 0.0, 1.0, 0.0),
            (5, 2.0, 0.0, 0.0),
            (6, 2.0, 1.0, 0.0),
        ]
    if elem_rows is None:
        elem_rows = [
            # (id, type, num_nodes, node_list)
            (10, "5-ElasticBeam3d", 2, (1, 2)),  # skipped (line)
            (11, "203-ASDShellQ4", 4, (1, 2, 3, 4)),
            (12, "203-ASDShellQ4", 4, (2, 5, 6, 3)),
            (13, "204-ASDShellT3", 3, (1, 2, 4)),
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
        number_of_steps={"STAGE_0": 5},
        time=pd.DataFrame(
            [{"MODEL_STAGE": "STAGE_0", "STEP": i, "TIME": float(i) * 0.1}
             for i in range(5)]
        ).set_index(["MODEL_STAGE", "STEP"]),
    )
    fake._selection_resolver = None
    return fake


def _make_scene(fake_dataset, *, is_3d=False):
    backend = MplBackend()
    handle = backend.make_scene(is_3d=is_3d)
    source = MPCODataSourceAdapter(fake_dataset)
    return Scene(backend, source, is_3d=is_3d, handle=handle)


# --------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------- #


def test_kind_is_contour() -> None:
    assert ContourLayer.kind == "contour"


def test_is_time_varying_reflects_input() -> None:
    layer_static = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    assert layer_static.is_time_varying is False
    layer_dynamic = ContourLayer(scalars=lambda step: {11: 1.0})
    assert layer_dynamic.is_time_varying is True


def test_summary_is_empty_before_attach() -> None:
    layer = ContourLayer(scalars={11: 1.0})
    assert layer.current_step is None
    assert layer.skipped_classes == []
    assert layer.n_faces == 0
    assert layer.actors == {}


# --------------------------------------------------------------------- #
# Attach lifecycle
# --------------------------------------------------------------------- #


def test_attach_buckets_renderable_elements_per_class() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    scalars = {11: 1.0, 12: 2.0, 13: 3.0}
    layer = ContourLayer(scalars=scalars)
    try:
        scene.add(layer)
        # Two quads + one tri = 3 faces drawn.
        assert layer.n_faces == 3
        actors = layer.actors
        assert "203-ASDShellQ4(4n)" in actors
        assert "204-ASDShellT3(3n)" in actors
        # Beams (2-node) are skipped.
        assert layer.skipped_classes == ["5-ElasticBeam3d(2n)"]
    finally:
        plt.close("all")


def test_attach_emits_warning_for_skipped_lines() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    try:
        with pytest.warns(RuntimeWarning, match="ElasticBeam3d"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_raises_when_no_renderable_classes() -> None:
    """A line-only dataset has nothing to fill — must raise."""
    elem_rows = [(10, "5-ElasticBeam3d", 2, (1, 2))]
    fake = _make_fake_dataset(elem_rows=elem_rows)
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={10: 0.0})
    try:
        with pytest.raises(LayerAttachError, match="no renderable element classes"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_raises_when_dataset_is_empty() -> None:
    fake = _make_fake_dataset(node_rows=[], elem_rows=[])
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={})
    try:
        with pytest.raises(LayerAttachError, match="no nodes or no elements"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_twice_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 0.0, 12: 0.0, 13: 0.0})
    try:
        scene.add(layer)
        with pytest.raises(LayerAttachError, match="already attached"):
            layer.attach(scene, scene.source)
    finally:
        plt.close("all")


def test_attach_missing_element_in_scalar_dict_raises() -> None:
    """A rendered element with no scalar entry must fail loud, not silent."""
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 1.0})  # 12 and 13 missing
    try:
        with pytest.raises(KeyError, match="element id 1[23]"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_attach_clim_auto_freezes_to_initial_data() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 1.0, 12: 5.0, 13: 3.0})
    try:
        scene.add(layer)
        assert layer.clim == (1.0, 5.0)
    finally:
        plt.close("all")


def test_attach_clim_explicit_overrides_auto() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(
        scalars={11: 1.0, 12: 5.0, 13: 3.0}, clim=(0.0, 10.0),
    )
    try:
        scene.add(layer)
        assert layer.clim == (0.0, 10.0)
    finally:
        plt.close("all")


def test_attach_clim_handles_constant_field() -> None:
    """A degenerate single-value field gets a non-zero range so colormaps work."""
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 7.0, 12: 7.0, 13: 7.0})
    try:
        scene.add(layer)
        vmin, vmax = layer.clim
        assert vmin == 7.0
        assert vmax > vmin
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Backend interaction (matplotlib)
# --------------------------------------------------------------------- #


def test_2d_attach_emits_polycollection_per_class() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake, is_3d=False)
    layer = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, PolyCollection)
            assert not isinstance(actor, Poly3DCollection)
    finally:
        plt.close("all")


def test_3d_attach_emits_poly3dcollection_per_class() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake, is_3d=True)
    layer = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert isinstance(actor, Poly3DCollection)
    finally:
        plt.close("all")


def test_attach_applies_mpl_zorder() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(
        scalars={11: 1.0, 12: 2.0, 13: 3.0}, mpl_zorder=3.0,
    )
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            assert actor.get_zorder() == pytest.approx(3.0)
    finally:
        plt.close("all")


def test_attach_applies_clim_to_actor() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(
        scalars={11: 1.0, 12: 5.0, 13: 3.0}, clim=(0.0, 10.0),
    )
    try:
        scene.add(layer)
        for actor in layer.actors.values():
            vmin, vmax = actor.get_clim()
            assert vmin == pytest.approx(0.0)
            assert vmax == pytest.approx(10.0)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# update_to_step
# --------------------------------------------------------------------- #


def test_static_scalars_update_is_noop() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    try:
        scene.add(layer)
        assert layer.current_step is None  # static layer never tracks step
        layer.update_to_step(3)
        assert layer.current_step is None
    finally:
        plt.close("all")


def test_callable_scalars_update_mutates_in_place() -> None:
    """Same actor across step changes — apeGmsh perf contract."""
    fake = _make_fake_dataset()
    scene = _make_scene(fake)

    def scalars_at(step):
        # Linear ramp so we can verify values change.
        return {11: float(step) + 1.0,
                12: float(step) + 2.0,
                13: float(step) + 3.0}

    layer = ContourLayer(scalars=scalars_at, step=0)
    try:
        scene.add(layer)
        actor_q4_before = layer.actors["203-ASDShellQ4(4n)"]
        layer.update_to_step(5)
        actor_q4_after = layer.actors["203-ASDShellQ4(4n)"]
        # Same actor object — no recreation.
        assert actor_q4_after is actor_q4_before
        assert layer.current_step == 5
        # Scalar array reflects step 5 values.
        np.testing.assert_allclose(
            actor_q4_after.get_array(), [6.0, 7.0],  # 11→6.0, 12→7.0
        )
    finally:
        plt.close("all")


def test_callable_update_same_step_is_noop() -> None:
    call_log = {"n": 0}

    def scalars_at(step):
        call_log["n"] += 1
        return {11: 1.0, 12: 2.0, 13: 3.0}

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars=scalars_at, step=0)
    try:
        scene.add(layer)
        n_attach = call_log["n"]
        layer.update_to_step(0)
        assert call_log["n"] == n_attach
    finally:
        plt.close("all")


def test_pre_attach_update_is_silent() -> None:
    layer = ContourLayer(scalars=lambda s: {11: 1.0})
    layer.update_to_step(5)
    assert layer.current_step is None


# --------------------------------------------------------------------- #
# Selection + detach
# --------------------------------------------------------------------- #


def test_selection_subsets_rendered_elements() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    # Only request element 11.
    layer = ContourLayer(
        scalars={11: 1.0},
        selection=SelectionSpec(element_ids=(11,)),
    )
    try:
        scene.add(layer)
        assert layer.n_faces == 1
        assert layer.skipped_classes == []  # beam wasn't part of the selection
    finally:
        plt.close("all")


def test_empty_selection_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(
        scalars={}, selection=SelectionSpec(element_ids=(9999,)),
    )
    try:
        with pytest.raises(LayerAttachError, match="No elements remain"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_detach_clears_state_and_removes_actors() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = ContourLayer(scalars={11: 1.0, 12: 2.0, 13: 3.0})
    try:
        scene.add(layer)
        assert layer.n_faces > 0
        scene.remove(layer)
        assert not layer.is_attached
        assert layer.n_faces == 0
        assert layer.skipped_classes == []
        assert layer.actors == {}
        assert layer.current_step is None
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Solid element coverage — bricks render 6 faces each
# --------------------------------------------------------------------- #


def test_brick_emits_six_faces_per_element() -> None:
    """An 8-node brick contributes 6 quad faces, all sharing the cell value."""
    node_rows = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    elem_rows = [(100, "Brick", 8, (1, 2, 3, 4, 5, 6, 7, 8))]
    fake = _make_fake_dataset(node_rows=node_rows, elem_rows=elem_rows)
    scene = _make_scene(fake, is_3d=True)
    layer = ContourLayer(scalars={100: 0.5})
    try:
        scene.add(layer)
        assert layer.n_faces == 6
        # rendered_element_ids reports the unique element ids, not faces.
        rendered = layer.rendered_element_ids
        assert rendered["Brick(8n)"].tolist() == [100]
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Real-fixture parity
# --------------------------------------------------------------------- #


def test_contour_layer_renders_quad_frame(quad_frame_dir: Path) -> None:
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    try:
        # QuadFrame has 625 ASDShellQ4 + 75 beams. Provide a dummy scalar
        # per shell — the test only verifies the renderer wires up.
        df_elements = ds.elements_info["dataframe"]
        shell_ids = df_elements.loc[
            df_elements["element_type"] == "203-ASDShellQ4", "element_id"
        ].to_numpy(dtype=np.int64)
        scalars = {int(eid): float(i) for i, eid in enumerate(shell_ids)}

        backend = MplBackend()
        handle = backend.make_scene(is_3d=True)
        source = MPCODataSourceAdapter(ds)
        scene = Scene(backend, source, is_3d=True, handle=handle)
        layer = ContourLayer(
            scalars=scalars,
            selection=SelectionSpec(element_type="203-ASDShellQ4"),
        )
        scene.add(layer)
        # One face per shell.
        assert layer.n_faces == 625
        # Beams aren't in the selection so they're not flagged as skipped.
        assert layer.skipped_classes == []
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# PyVista backend integration (gated on the [viewer-3d] extra)
# --------------------------------------------------------------------- #


def test_contour_layer_renders_through_pyvista_backend() -> None:
    """The layer is backend-agnostic — same Scene + layer code, just a
    different backend instance."""
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)
        layer = ContourLayer(
            scalars={11: 1.0, 12: 2.0, 13: 3.0},
            cmap="viridis",
        )
        scene.add(layer)
        # Three faces total — two quads + one tri.
        assert layer.n_faces == 3
        # PyVista's add_polygons produces a _PvActorRef per class. Verify
        # the cell_data scalars are bound to the dataset for the quad class.
        from STKO_to_python.viewer.backends.pyvista.backend import _PvActorRef

        for ref in layer.actors.values():
            assert isinstance(ref, _PvActorRef)
            assert ref.scalar_field == "values"
    finally:
        handle.plotter.close()


def test_pyvista_contour_in_place_update_keeps_same_actor() -> None:
    """apeGmsh perf contract on PyVista: update_scalars mutates the same
    cell_data array — actor instance is preserved across steps."""
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)

        def scalars_at(step):
            return {11: float(step) + 1.0,
                    12: float(step) + 2.0,
                    13: float(step) + 3.0}

        layer = ContourLayer(scalars=scalars_at, step=0)
        scene.add(layer)
        ref_before = layer.actors["203-ASDShellQ4(4n)"]
        layer.update_to_step(5)
        ref_after = layer.actors["203-ASDShellQ4(4n)"]
        assert ref_after is ref_before  # actor not recreated
        # Step 5 values: 11 -> 6.0, 12 -> 7.0
        np.testing.assert_allclose(
            ref_after.dataset.cell_data["values"], [6.0, 7.0],
        )
    finally:
        handle.plotter.close()


# --------------------------------------------------------------------- #
# Nodal topology (Phase 3.0c)
# --------------------------------------------------------------------- #


def test_topology_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="topology must be"):
        ContourLayer(scalars={1: 0.0}, topology="bogus")


def test_topology_default_is_cell() -> None:
    layer = ContourLayer(scalars={11: 0.0})
    assert layer.topology == "cell"


def test_nodal_mode_on_mpl_raises_backend_capability_error() -> None:
    """MplBackend doesn't support per-vertex polygon coloring —
    nodal contour on mpl must fail loud with a clear message."""
    from STKO_to_python.viewer.core.errors import BackendCapabilityError

    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    # One value per node in the fake dataset (6 nodes).
    nodal_scalars = {1: 0.0, 2: 0.5, 3: 1.0, 4: 0.5, 5: 0.7, 6: 0.9}
    layer = ContourLayer(scalars=nodal_scalars, topology="nodal")
    try:
        with pytest.raises(BackendCapabilityError, match="per-vertex"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_nodal_mode_missing_node_in_scalar_dict_raises() -> None:
    """Same loud-failure contract as cell mode, but for node ids."""
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    # Leaves out one of the nodes a rendered face references.
    nodal_scalars = {1: 0.0, 2: 0.5}  # missing 3, 4, 5, 6
    layer = ContourLayer(scalars=nodal_scalars, topology="nodal")
    try:
        # On mpl the BackendCapabilityError from add_polygons fires first
        # at the call site; the KeyError contract is exercised by the
        # PyVista nodal tests below where the path can actually proceed
        # far enough to evaluate scalars.
        with pytest.raises(Exception):
            scene.add(layer)
    finally:
        plt.close("all")


def test_pyvista_nodal_contour_binds_point_data() -> None:
    """PyVista path stores per-vertex scalars in point_data, not cell_data."""
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend
    from STKO_to_python.viewer.backends.pyvista.backend import _PvActorRef

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)

        nodal = {1: 0.0, 2: 1.0, 3: 2.0, 4: 3.0, 5: 4.0, 6: 5.0}
        layer = ContourLayer(scalars=nodal, topology="nodal")
        scene.add(layer)

        # The Q4 class: two quads share an edge (nodes 2,3). Vertex stream
        # is the flattened concat of each face's 4 corners.
        # Quad 11 (1,2,3,4) -> [0.0, 1.0, 2.0, 3.0]
        # Quad 12 (2,5,6,3) -> [1.0, 4.0, 5.0, 2.0]
        q4_ref = layer.actors["203-ASDShellQ4(4n)"]
        assert isinstance(q4_ref, _PvActorRef)
        assert q4_ref.scalar_field == "point_values"
        np.testing.assert_allclose(
            q4_ref.dataset.point_data["point_values"],
            [0.0, 1.0, 2.0, 3.0,    # quad 11
             1.0, 4.0, 5.0, 2.0],   # quad 12
        )
        # Shared corner node 2 contributed identical values (1.0) to both
        # quads — that's what makes the rendering visually continuous.
    finally:
        handle.plotter.close()


def test_pyvista_nodal_contour_in_place_update() -> None:
    """update_scalars must dispatch to point_data, not cell_data."""
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)

        def scalars_at(step):
            base = float(step)
            return {1: base, 2: base + 1, 3: base + 2,
                    4: base + 3, 5: base + 4, 6: base + 5}

        layer = ContourLayer(scalars=scalars_at, topology="nodal", step=0)
        scene.add(layer)
        ref_before = layer.actors["203-ASDShellQ4(4n)"]
        layer.update_to_step(10)
        ref_after = layer.actors["203-ASDShellQ4(4n)"]
        assert ref_after is ref_before  # actor preserved
        assert layer.current_step == 10
        # cell_data MUST be untouched; point_data has the new step-10 values.
        assert "values" not in ref_after.dataset.cell_data
        # Step 10 quad 11 (1,2,3,4): 10, 11, 12, 13
        np.testing.assert_allclose(
            ref_after.dataset.point_data["point_values"][:4],
            [10.0, 11.0, 12.0, 13.0],
        )
    finally:
        handle.plotter.close()


def test_pyvista_nodal_contour_clim_autofreezes_to_data_range() -> None:
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)
        nodal = {1: -2.0, 2: 0.0, 3: 5.0, 4: 1.0, 5: 3.0, 6: 7.0}
        layer = ContourLayer(scalars=nodal, topology="nodal")
        scene.add(layer)
        vmin, vmax = layer.clim
        assert vmin == pytest.approx(-2.0)
        assert vmax == pytest.approx(7.0)
    finally:
        handle.plotter.close()


def test_pyvista_nodal_contour_missing_node_raises() -> None:
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    fake = _make_fake_dataset()
    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        source = MPCODataSourceAdapter(fake)
        scene = Scene(backend, source, is_3d=True, handle=handle)
        layer = ContourLayer(
            scalars={1: 0.0, 2: 1.0},  # 3-6 missing
            topology="nodal",
        )
        with pytest.raises(KeyError, match="node id"):
            scene.add(layer)
    finally:
        handle.plotter.close()


def test_pyvista_backend_polygons_both_values_and_point_values_raises() -> None:
    """The protocol forbids supplying both at once."""
    pv = pytest.importorskip("pyvista")
    from STKO_to_python.viewer.backends.pyvista import PyVistaBackend

    backend = PyVistaBackend()
    handle = backend.make_scene(is_3d=True, off_screen=True)
    try:
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="not both"):
            backend.add_polygons(
                handle, [tri],
                values=np.array([1.0]),
                point_values=np.array([0.0, 1.0, 2.0]),
            )
    finally:
        handle.plotter.close()
