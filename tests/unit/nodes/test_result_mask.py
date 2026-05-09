"""Unit tests for ``NodeResultMask`` and the ``nr.where(...)`` chain.

Builds a small synthetic :class:`NodalResults` (3 nodes x 5 steps,
3-component DISPLACEMENT) so each reduction's expected output is
hand-checkable. No HDF5 fixture required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.results.nodal_results_dataclass import NodalResults
from STKO_to_python.nodes.result_mask import NodeResultMask


# ---------------------------------------------------------------------- #
# Fixture                                                                #
# ---------------------------------------------------------------------- #

@pytest.fixture
def nr() -> NodalResults:
    """3 nodes × 5 steps × 3 DISPLACEMENT components."""
    nids = (1, 2, 3)
    steps = list(range(5))
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Per node, per step values for components 1, 2, 3.
    # Designed so the |U|-magnitude peak is well-separated:
    #   node 1 → mostly small, peak |U| = 5 at step 4 (3,4,0)
    #   node 2 → biggest, peak |U| = 13 at step 2 (5,12,0)
    #   node 3 → tiny, peak |U| = 1 at step 0 (1,0,0)
    u1 = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
    ]
    u2 = [
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0],
        [5.0, 12.0, 0.0],
        [3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    u3 = [
        [1.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]
    per_node = {1: u1, 2: u2, 3: u3}

    rows: list[dict] = []
    for n in nids:
        for s in steps:
            ux, uy, uz = per_node[n][s]
            rows.append(
                {
                    "node_id": n,
                    "step": s,
                    ("DISPLACEMENT", 1): ux,
                    ("DISPLACEMENT", 2): uy,
                    ("DISPLACEMENT", 3): uz,
                }
            )
    df = pd.DataFrame(rows).set_index(["node_id", "step"]).sort_index()
    df.columns = pd.MultiIndex.from_tuples(
        df.columns.tolist(), names=("result", "component")
    )

    coords = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "z": [0.0, 0.0, 0.0]},
        index=pd.Index(nids, name="node_id"),
    )

    return NodalResults(
        df=df,
        time=time,
        name="synthetic",
        nodes_ids=nids,
        nodes_info=coords,
        results_components=("DISPLACEMENT|1", "DISPLACEMENT|2", "DISPLACEMENT|3"),
        model_stages=("MODEL_STAGE[1]",),
        stage_step_ranges={"MODEL_STAGE[1]": (0, 5)},
    )


# ---------------------------------------------------------------------- #
# Component path                                                          #
# ---------------------------------------------------------------------- #

def test_component_peak_threshold(nr: NodalResults):
    mask = nr.where().component("DISPLACEMENT", 2).peak().gt(5.0)
    ids = set(mask.ids().tolist())
    # Component 2 peaks: node1=4, node2=12, node3=0. Only node 2 > 5.
    assert ids == {2}


def test_component_abs_peak(nr: NodalResults):
    mask = nr.where().component("DISPLACEMENT", 1).abs_peak().ge(3.0)
    ids = set(mask.ids().tolist())
    # |comp1| peaks: node1=3, node2=5, node3=1. Threshold ≥3.
    assert ids == {1, 2}


def test_component_at_step(nr: NodalResults):
    mask = nr.where().component("DISPLACEMENT", 1).at_step(2).gt(1.5)
    ids = set(mask.ids().tolist())
    # At step 2, comp 1: node1=2, node2=5, node3=0.5.
    assert ids == {1, 2}


def test_component_unknown_raises(nr: NodalResults):
    with pytest.raises(ValueError, match="not found"):
        nr.where().component("DISPLACEMENT", 99)


# ---------------------------------------------------------------------- #
# Magnitude path                                                          #
# ---------------------------------------------------------------------- #

def test_magnitude_default_uses_all_components(nr: NodalResults):
    mask = nr.where().magnitude("DISPLACEMENT").peak().gt(10.0)
    ids = set(mask.ids().tolist())
    # |U| peaks: node1=5, node2=13, node3=1. Threshold > 10 → only node 2.
    assert ids == {2}


def test_magnitude_planar_components(nr: NodalResults):
    mask = nr.where().magnitude("DISPLACEMENT", components=(1, 2)).peak().ge(5.0)
    ids = set(mask.ids().tolist())
    # Same since comp 3 is zero throughout — node 1 (5) and node 2 (13).
    assert ids == {1, 2}


def test_magnitude_unknown_result_raises(nr: NodalResults):
    with pytest.raises(ValueError, match="no components"):
        nr.where().magnitude("VELOCITY")


# ---------------------------------------------------------------------- #
# Time windowing                                                          #
# ---------------------------------------------------------------------- #

def test_default_time_window_propagates(nr: NodalResults):
    # Window covers steps 0..2 (time half-open [0, 3) → steps 0,1,2).
    mask = (
        nr.where(time=(0.0, 3.0))
        .component("DISPLACEMENT", 2)
        .peak()
        .gt(5.0)
    )
    ids = set(mask.ids().tolist())
    # Comp 2 peaks in window: node1=0, node2=12, node3=0. Only node 2 > 5.
    assert ids == {2}


def test_explicit_time_overrides_default(nr: NodalResults):
    # Default window of [0, 3); override with the full series.
    mask = (
        nr.where(time=(0.0, 3.0))
        .component("DISPLACEMENT", 1)
        .peak(time=None)
        .gt(2.5)
    )
    ids = set(mask.ids().tolist())
    # Full-window comp 1 peaks: node1=3, node2=5, node3=1.
    assert ids == {1, 2}


# ---------------------------------------------------------------------- #
# Boolean composition + nr[mask]                                          #
# ---------------------------------------------------------------------- #

def test_mask_and(nr: NodalResults):
    a = nr.where().magnitude("DISPLACEMENT").peak().gt(2.0)
    b = nr.where().component("DISPLACEMENT", 1).peak().gt(2.5)
    ids = set((a & b).ids().tolist())
    # |U| > 2: nodes 1, 2; comp 1 > 2.5: nodes 1, 2 → AND = {1,2}.
    assert ids == {1, 2}


def test_mask_or(nr: NodalResults):
    a = nr.where().component("DISPLACEMENT", 1).peak().gt(4.0)  # node 2
    b = nr.where().component("DISPLACEMENT", 2).peak().ge(4.0)  # nodes 1 and 2
    ids = set((a | b).ids().tolist())
    assert ids == {1, 2}


def test_mask_invert(nr: NodalResults):
    a = nr.where().magnitude("DISPLACEMENT").peak().gt(10.0)
    not_a = ~a
    ids = set(not_a.ids().tolist())
    # ~{2} → {1, 3}
    assert ids == {1, 3}


def test_apply_returns_trimmed_nodal_results(nr: NodalResults):
    mask = nr.where().magnitude("DISPLACEMENT").peak().gt(10.0)
    trimmed = nr[mask]
    assert isinstance(trimmed, NodalResults)
    assert tuple(trimmed.info.nodes_ids) == (2,)
    # df should only contain rows for node 2.
    nids = set(trimmed.df.index.get_level_values("node_id").unique().tolist())
    assert nids == {2}


def test_apply_empty_mask(nr: NodalResults):
    mask = nr.where().magnitude("DISPLACEMENT").peak().gt(1e9)
    trimmed = nr[mask]
    assert isinstance(trimmed, NodalResults)
    assert tuple(trimmed.info.nodes_ids) == ()
    assert trimmed.df.empty


# ---------------------------------------------------------------------- #
# predicate escape hatch                                                  #
# ---------------------------------------------------------------------- #

def test_predicate_per_node_array(nr: NodalResults):
    # Per-node bool array — same length as info.nodes_ids.
    mask = nr.where().predicate(lambda df: np.array([True, False, True]))
    ids = set(mask.ids().tolist())
    assert ids == {1, 3}


def test_predicate_per_row_array(nr: NodalResults):
    # Per-row bool array — collapsed via any over (node_id, step).
    arr = np.zeros(len(nr.df), dtype=bool)
    arr[0] = True  # one row of node 1 — any() → node 1 included.
    mask = nr.where().predicate(lambda df: arr)
    ids = set(mask.ids().tolist())
    assert ids == {1}


def test_predicate_shape_mismatch_raises(nr: NodalResults):
    with pytest.raises(ValueError, match="returned shape"):
        nr.where().predicate(lambda df: np.ones(99, dtype=bool))


# ---------------------------------------------------------------------- #
# __getitem__ rejects non-mask                                            #
# ---------------------------------------------------------------------- #

def test_getitem_non_mask_raises(nr: NodalResults):
    with pytest.raises(TypeError, match="NodeResultMask"):
        nr[42]
