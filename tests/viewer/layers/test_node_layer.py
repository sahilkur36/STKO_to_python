"""Tests for :class:`NodeLayer`.

Coverage:

1. Construction validation — scalar binding is all-or-nothing.
2. Position-only mode — static layer, ``update_to_step`` is a no-op.
3. Scalar-bound mode — initial scalars at attach, in-place updates
   via ``backend.update_scalars`` on step changes; magnitude and
   single-component selectors both work.
4. Selection subsetting (matches the same SelectionSpec contract the
   other layers use).
5. Detach cleanup.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection

from STKO_to_python.viewer.backends.mpl.backend import MplBackend
from STKO_to_python.viewer.core import (
    MPCODataSourceAdapter,
    Scene,
    SelectionSpec,
)
from STKO_to_python.viewer.core.errors import LayerAttachError
from STKO_to_python.viewer.layers import NodeLayer


# --------------------------------------------------------------------- #
# Fake-dataset helpers
# --------------------------------------------------------------------- #


class _FakeNodalResults:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def _make_disp_df(
    node_ids,
    *,
    n_steps: int = 5,
    vectors: dict[int, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build a ``(node_id, step) -> (DISPLACEMENT, 1/2/3)`` DataFrame.

    ``vectors``: optional ``{node_id: ndarray(n_steps, 3)}`` — when a
    node is missing it gets zeros.
    """
    rows = []
    for nid in node_ids:
        per_node = (
            vectors[int(nid)]
            if vectors and int(nid) in vectors
            else np.zeros((n_steps, 3), dtype=np.float64)
        )
        for step in range(n_steps):
            rows.append(
                {
                    "node_id": int(nid),
                    "step": int(step),
                    "c1": float(per_node[step, 0]),
                    "c2": float(per_node[step, 1]),
                    "c3": float(per_node[step, 2]),
                }
            )
    df = pd.DataFrame(rows).set_index(["node_id", "step"])
    df.columns = pd.MultiIndex.from_tuples(
        [("DISPLACEMENT", "1"), ("DISPLACEMENT", "2"), ("DISPLACEMENT", "3")],
        names=["result", "component"],
    )
    return df


def _make_fake_dataset(*, displacement_df: pd.DataFrame | None = None):
    node_rows = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
    ]
    elem_rows = [
        (10, "5-ElasticBeam3d", 2, (1, 2)),
        (11, "203-ASDShellQ4", 4, (1, 2, 3, 4)),
    ]
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "x", "y", "z"])
    elements_df = pd.DataFrame(
        elem_rows,
        columns=["element_id", "element_type", "num_nodes", "node_list"],
    )

    if displacement_df is None:
        displacement_df = _make_disp_df([1, 2, 3, 4])

    def _fake_get_nodal_results(*, results_name, model_stage, node_ids):
        return _FakeNodalResults(displacement_df)

    nodes_obj = SimpleNamespace(get_nodal_results=_fake_get_nodal_results)

    fake = SimpleNamespace(
        nodes=nodes_obj,
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


def test_kind_is_node() -> None:
    assert NodeLayer.kind == "node"


def test_default_construction_is_position_only() -> None:
    layer = NodeLayer()
    assert layer.is_scalar_bound is False
    assert layer.result_name is None
    assert layer.size == 20.0
    assert layer.mpl_zorder == 2.5


def test_partial_scalar_binding_raises() -> None:
    """Scalar binding is all-or-nothing — partial binding is rejected."""
    with pytest.raises(ValueError, match="all-or-nothing|requires result_name"):
        NodeLayer(result_name="DISPLACEMENT")
    with pytest.raises(ValueError, match="requires result_name"):
        NodeLayer(result_name="DISPLACEMENT", component="1")
    with pytest.raises(ValueError, match="requires result_name"):
        NodeLayer(component="magnitude", model_stage="STAGE_0", step=0)


def test_full_scalar_binding_is_accepted() -> None:
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="1",
        model_stage="STAGE_0",
        step=0,
    )
    assert layer.is_scalar_bound is True


# --------------------------------------------------------------------- #
# Position-only mode
# --------------------------------------------------------------------- #


def test_position_only_attach_places_one_point_per_node() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer()
    try:
        scene.add(layer)
        assert isinstance(layer.actor, PathCollection)
        assert layer.node_ids.tolist() == [1, 2, 3, 4]
        assert layer.coords.shape == (4, 3)
        assert layer.scalars is None
    finally:
        plt.close("all")


def test_position_only_update_to_step_is_no_op() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer()
    try:
        scene.add(layer)
        actor_before = layer.actor
        layer.update_to_step(3)
        assert layer.actor is actor_before
        assert layer.current_step is None  # never set in position-only mode
    finally:
        plt.close("all")


def test_position_only_attach_applies_mpl_zorder() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer(mpl_zorder=4.0)
    try:
        scene.add(layer)
        assert layer.actor.get_zorder() == pytest.approx(4.0)
    finally:
        plt.close("all")


def test_position_only_attach_respects_visible_false() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer(visible=False)
    try:
        scene.add(layer)
        assert layer.actor.get_visible() is False
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Scalar-bound mode
# --------------------------------------------------------------------- #


def test_scalar_bound_attach_fetches_initial_scalars() -> None:
    # Make node 2 at step 3 have displacement (1, 0, 0).
    vectors = {2: np.zeros((5, 3))}
    vectors[2][3] = np.array([1.0, 0.0, 0.0])
    df = _make_disp_df([1, 2, 3, 4], vectors=vectors)
    fake = _make_fake_dataset(displacement_df=df)

    scene = _make_scene(fake)
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="1",
        model_stage="STAGE_0",
        step=3,
    )
    try:
        scene.add(layer)
        scalars = layer.scalars
        assert scalars is not None
        # node 2 (row 1 in [1,2,3,4]) has component 1 = 1.0
        assert scalars[1] == pytest.approx(1.0)
        # others are zero
        assert scalars[0] == pytest.approx(0.0)
        assert scalars[2] == pytest.approx(0.0)
        assert scalars[3] == pytest.approx(0.0)
        assert layer.current_step == 3
    finally:
        plt.close("all")


def test_scalar_bound_magnitude_computes_norm() -> None:
    vectors = {3: np.zeros((5, 3))}
    vectors[3][0] = np.array([3.0, 4.0, 0.0])  # magnitude = 5
    df = _make_disp_df([1, 2, 3, 4], vectors=vectors)
    fake = _make_fake_dataset(displacement_df=df)

    scene = _make_scene(fake)
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="magnitude",
        model_stage="STAGE_0",
        step=0,
    )
    try:
        scene.add(layer)
        scalars = layer.scalars
        # node 3 is index 2 in [1,2,3,4]
        assert scalars[2] == pytest.approx(5.0)
        assert scalars[0] == pytest.approx(0.0)
    finally:
        plt.close("all")


def test_scalar_bound_unknown_component_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="bogus",
        model_stage="STAGE_0",
        step=0,
    )
    try:
        with pytest.raises(ValueError, match="bogus"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_scalar_bound_update_to_step_replaces_scalars_in_place() -> None:
    """update_to_step must call backend.update_scalars on the SAME
    actor — no recreation."""
    vectors = {
        1: np.zeros((5, 3)),
        2: np.zeros((5, 3)),
    }
    vectors[1][2] = np.array([10.0, 0.0, 0.0])
    vectors[2][4] = np.array([20.0, 0.0, 0.0])
    df = _make_disp_df([1, 2, 3, 4], vectors=vectors)
    fake = _make_fake_dataset(displacement_df=df)

    scene = _make_scene(fake)
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="1",
        model_stage="STAGE_0",
        step=2,
    )
    try:
        scene.add(layer)
        actor_at_step2 = layer.actor
        assert layer.scalars[0] == pytest.approx(10.0)

        layer.update_to_step(4)
        # Same actor object — no recreation.
        assert layer.actor is actor_at_step2
        assert layer.current_step == 4
        # Scalars reflect step 4.
        assert layer.scalars[1] == pytest.approx(20.0)
        assert layer.scalars[0] == pytest.approx(0.0)
    finally:
        plt.close("all")


def test_scalar_bound_update_to_step_unchanged_is_no_op() -> None:
    call_log = {"n": 0}

    def _counting(*, results_name, model_stage, node_ids):
        call_log["n"] += 1
        return _FakeNodalResults(_make_disp_df([1, 2, 3, 4]))

    fake = _make_fake_dataset()
    fake.nodes = SimpleNamespace(get_nodal_results=_counting)
    scene = _make_scene(fake)
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="1",
        model_stage="STAGE_0",
        step=2,
    )
    try:
        scene.add(layer)
        n_attach = call_log["n"]
        layer.update_to_step(2)
        assert call_log["n"] == n_attach
    finally:
        plt.close("all")


def test_pre_attach_update_to_step_is_silent() -> None:
    layer = NodeLayer(
        result_name="DISPLACEMENT",
        component="1",
        model_stage="STAGE_0",
        step=0,
    )
    layer.update_to_step(5)
    assert layer.current_step is None


# --------------------------------------------------------------------- #
# Selection subsetting
# --------------------------------------------------------------------- #


def test_selection_by_node_ids_subsets_layer() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer(selection=SelectionSpec(node_ids=(2, 4)))
    try:
        scene.add(layer)
        assert layer.node_ids.tolist() == [2, 4]
        assert layer.coords.shape == (2, 3)
    finally:
        plt.close("all")


def test_empty_selection_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer(selection=SelectionSpec(node_ids=(9999,)))
    try:
        with pytest.raises(LayerAttachError, match="No nodes remain"):
            scene.add(layer)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# Detach
# --------------------------------------------------------------------- #


def test_detach_clears_state() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer()
    try:
        scene.add(layer)
        assert layer.is_attached
        assert layer.node_ids.size == 4
        scene.remove(layer)
        assert not layer.is_attached
        assert layer.node_ids.size == 0
        assert layer.coords.size == 0
        assert layer.scalars is None
        assert layer.actor is None
    finally:
        plt.close("all")


def test_attach_twice_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = NodeLayer()
    try:
        scene.add(layer)
        with pytest.raises(LayerAttachError, match="already attached"):
            layer.attach(scene, scene.source)
    finally:
        plt.close("all")
