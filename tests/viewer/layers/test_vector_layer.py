"""Tests for :class:`VectorLayer`.

Coverage:

1. Construction validation — ``result_name`` and ``model_stage``
   are required; ``n_components`` must be ≥ 1.
2. Attach — origins from node coords, vectors fetched from the
   result, padded/truncated to ``n_components``.
3. ``update_to_step`` — remove + re-add (Phase 2.6 perf gap;
   documented in the layer's class docstring). Pin the contract:
   actor instance changes across step updates so callers can detect
   the remove-re-add lifecycle.
4. Selection subsetting.
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

from STKO_to_python.viewer.backends.mpl.backend import MplBackend
from STKO_to_python.viewer.core import (
    MPCODataSourceAdapter,
    Scene,
    SelectionSpec,
)
from STKO_to_python.viewer.core.errors import LayerAttachError
from STKO_to_python.viewer.layers import VectorLayer


# --------------------------------------------------------------------- #
# Fake-dataset helpers (same shape as test_node_layer.py)
# --------------------------------------------------------------------- #


class _FakeNodalResults:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


def _make_disp_df(
    node_ids,
    *,
    n_steps: int = 5,
    vectors: dict[int, np.ndarray] | None = None,
    n_components: int = 3,
) -> pd.DataFrame:
    rows = []
    for nid in node_ids:
        per_node = (
            vectors[int(nid)]
            if vectors and int(nid) in vectors
            else np.zeros((n_steps, n_components), dtype=np.float64)
        )
        for step in range(n_steps):
            row = {"node_id": int(nid), "step": int(step)}
            for c in range(n_components):
                row[f"c{c + 1}"] = float(per_node[step, c])
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["node_id", "step"])
    df.columns = pd.MultiIndex.from_tuples(
        [("DISPLACEMENT", str(c + 1)) for c in range(n_components)],
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
    ]
    nodes_df = pd.DataFrame(node_rows, columns=["node_id", "x", "y", "z"])
    elements_df = pd.DataFrame(
        elem_rows,
        columns=["element_id", "element_type", "num_nodes", "node_list"],
    )

    if displacement_df is None:
        displacement_df = _make_disp_df([1, 2, 3, 4])

    def _fake(*, results_name, model_stage, node_ids):
        return _FakeNodalResults(displacement_df)

    fake = SimpleNamespace(
        nodes=SimpleNamespace(get_nodal_results=_fake),
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


def test_kind_is_vector() -> None:
    assert VectorLayer.kind == "vector"


def test_construction_requires_result_name() -> None:
    with pytest.raises(ValueError, match="requires result_name"):
        VectorLayer(result_name="", model_stage="STAGE_0", step=0)


def test_construction_requires_model_stage() -> None:
    with pytest.raises(ValueError, match="requires result_name"):
        VectorLayer(result_name="DISPLACEMENT", model_stage="", step=0)


def test_construction_rejects_zero_n_components() -> None:
    with pytest.raises(ValueError, match="n_components"):
        VectorLayer(
            result_name="DISPLACEMENT", model_stage="STAGE_0",
            step=0, n_components=0,
        )


def test_construction_preserves_config() -> None:
    layer = VectorLayer(
        result_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        step=7,
        scale=50.0,
        n_components=2,
        color="red",
    )
    assert layer.result_name == "DISPLACEMENT"
    assert layer.model_stage == "MODEL_STAGE[1]"
    assert layer.scale == 50.0
    assert layer.n_components == 2
    assert layer.color == "red"


# --------------------------------------------------------------------- #
# Attach
# --------------------------------------------------------------------- #


def test_attach_places_arrow_origins_at_node_coords() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
    )
    try:
        scene.add(layer)
        assert layer.node_ids.tolist() == [1, 2, 3, 4]
        np.testing.assert_allclose(layer.origins[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(layer.origins[1], [1.0, 0.0, 0.0])
        assert layer.actor is not None
        assert layer.current_step == 0
    finally:
        plt.close("all")


def test_attach_extracts_vectors_at_initial_step() -> None:
    vectors = {2: np.zeros((5, 3))}
    vectors[2][3] = np.array([0.5, 1.5, -0.25])
    df = _make_disp_df([1, 2, 3, 4], vectors=vectors)
    fake = _make_fake_dataset(displacement_df=df)
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=3,
    )
    try:
        scene.add(layer)
        # node 2 is row 1.
        np.testing.assert_allclose(layer.vectors[1], [0.5, 1.5, -0.25])
        # others are zero.
        np.testing.assert_allclose(layer.vectors[0], [0.0, 0.0, 0.0])
    finally:
        plt.close("all")


def test_attach_pads_short_results_to_n_components() -> None:
    """A 2-component result must be zero-padded to length 3 by default."""
    df = _make_disp_df([1, 2, 3, 4], n_components=2)
    fake = _make_fake_dataset(displacement_df=df)
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
    )
    try:
        scene.add(layer)
        assert layer.vectors.shape == (4, 3)
        np.testing.assert_allclose(layer.vectors[:, 2], 0.0)
    finally:
        plt.close("all")


def test_attach_truncates_long_results_to_n_components() -> None:
    df = _make_disp_df([1, 2, 3, 4], n_components=5)
    fake = _make_fake_dataset(displacement_df=df)
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
        n_components=3,
    )
    try:
        scene.add(layer)
        assert layer.vectors.shape == (4, 3)
    finally:
        plt.close("all")


# --------------------------------------------------------------------- #
# update_to_step — perf gap (remove + re-add)
# --------------------------------------------------------------------- #


def test_update_to_step_swaps_actor_and_advances_step() -> None:
    """Phase 2.6 perf gap: update_to_step removes + re-adds the
    arrow actor. Test pins this contract — the actor *instance*
    changes, but vectors track the new step."""
    vectors = {2: np.zeros((5, 3))}
    vectors[2][1] = np.array([1.0, 0.0, 0.0])
    vectors[2][4] = np.array([0.0, 2.0, 0.0])
    df = _make_disp_df([1, 2, 3, 4], vectors=vectors)
    fake = _make_fake_dataset(displacement_df=df)
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=1,
    )
    try:
        scene.add(layer)
        actor_at_step1 = layer.actor
        np.testing.assert_allclose(layer.vectors[1], [1.0, 0.0, 0.0])

        layer.update_to_step(4)
        # Different actor instance — see class docstring.
        assert layer.actor is not actor_at_step1
        assert layer.current_step == 4
        np.testing.assert_allclose(layer.vectors[1], [0.0, 2.0, 0.0])
    finally:
        plt.close("all")


def test_update_to_step_no_op_on_unchanged_step() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=2,
    )
    try:
        scene.add(layer)
        actor_before = layer.actor
        layer.update_to_step(2)
        assert layer.actor is actor_before
    finally:
        plt.close("all")


def test_pre_attach_update_to_step_is_silent() -> None:
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
    )
    layer.update_to_step(3)
    assert layer.current_step is None


# --------------------------------------------------------------------- #
# Selection + detach
# --------------------------------------------------------------------- #


def test_selection_subsets_arrows() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
        selection=SelectionSpec(node_ids=(1, 3)),
    )
    try:
        scene.add(layer)
        assert layer.node_ids.tolist() == [1, 3]
        assert layer.origins.shape == (2, 3)
        assert layer.vectors.shape == (2, 3)
    finally:
        plt.close("all")


def test_empty_selection_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
        selection=SelectionSpec(node_ids=(9999,)),
    )
    try:
        with pytest.raises(LayerAttachError, match="No nodes remain"):
            scene.add(layer)
    finally:
        plt.close("all")


def test_detach_clears_state_and_removes_actor() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
    )
    try:
        scene.add(layer)
        assert layer.is_attached
        scene.remove(layer)
        assert not layer.is_attached
        assert layer.actor is None
        assert layer.node_ids.size == 0
        assert layer.origins.size == 0
        assert layer.vectors.size == 0
        assert layer.current_step is None
    finally:
        plt.close("all")


def test_attach_twice_raises() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
    )
    try:
        scene.add(layer)
        with pytest.raises(LayerAttachError, match="already attached"):
            layer.attach(scene, scene.source)
    finally:
        plt.close("all")


def test_invisible_on_attach_hides_actor() -> None:
    fake = _make_fake_dataset()
    scene = _make_scene(fake)
    layer = VectorLayer(
        result_name="DISPLACEMENT", model_stage="STAGE_0", step=0,
        visible=False,
    )
    try:
        scene.add(layer)
        # Quiver inherits the matplotlib Artist API (set_visible /
        # get_visible).
        assert layer.actor.get_visible() is False
    finally:
        plt.close("all")
