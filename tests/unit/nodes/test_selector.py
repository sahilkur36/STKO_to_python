"""Unit tests for ``NodeSelector``.

Synthetic, self-contained ``NodeManager`` stand-in — no on-disk
fixtures. Geometry is a uniform 4 x 4 x 4 lattice of nodes plus a
12-element beam grid wiring adjacent x-neighbours along z=0, y=0 so
that ``attached_to`` has a non-trivial connectivity to chase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.nodes.selector import NodeSelector


# ---------------------------------------------------------------------- #
# Fixtures                                                                #
# ---------------------------------------------------------------------- #

def _build_index() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes_df, elements_df).

    Nodes: 4 x 4 x 4 integer-coord lattice (64 nodes), node_id from 1.
    Elements: 12 line elements wiring (ix, 0, 0) -> (ix+1, 0, 0) for
    ix in 0..2, repeated for y in {0} and z in {0}, but here just one
    row of 3 along z=0 for simplicity. Plus 3 elements at z=2 and 3 at
    z=3 so attached_to spans more than one z-level.
    """
    nodes: list[dict] = []
    nid = 1
    coord_to_id: dict[tuple[int, int, int], int] = {}
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                coord_to_id[(ix, iy, iz)] = nid
                nodes.append(
                    {
                        "node_id": nid,
                        "file_id": 0,
                        "index": nid - 1,
                        "x": float(ix),
                        "y": float(iy),
                        "z": float(iz),
                    }
                )
                nid += 1
    df_nodes = pd.DataFrame(nodes)

    # Elements: a few line elements running along x at z=0, z=2, z=3.
    elem_rows: list[dict] = []
    eid = 1
    for iz in (0, 2, 3):
        for ix in range(3):
            n1 = coord_to_id[(ix, 0, iz)]
            n2 = coord_to_id[(ix + 1, 0, iz)]
            elem_rows.append(
                {
                    "element_id": eid,
                    "element_idx": eid - 1,
                    "file_id": 0,
                    "element_type": "64-DispBeamColumn3d",
                    "decorated_type": "64-DispBeamColumn3d[1000:0]",
                    "node_list": (n1, n2),
                    "num_nodes": 2,
                    "centroid_x": float(ix) + 0.5,
                    "centroid_y": 0.0,
                    "centroid_z": float(iz),
                }
            )
            eid += 1
    df_elems = pd.DataFrame(elem_rows)
    return df_nodes, df_elems


@dataclass
class _MockSelResolver:
    node_sets: dict  # case-folded name -> np.ndarray
    id_sets: dict  # int -> np.ndarray

    def resolve_nodes(
        self,
        *,
        names: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[int]] = None,
        explicit_ids: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        gathered: list[np.ndarray] = []
        if names:
            for n in names:
                key = str(n).strip().lower()
                if key not in self.node_sets:
                    raise ValueError(f"unknown set: {n}")
                gathered.append(self.node_sets[key])
        if ids:
            for sid in ids:
                if int(sid) not in self.id_sets:
                    raise ValueError(f"unknown set id: {sid}")
                gathered.append(self.id_sets[int(sid)])
        if explicit_ids is not None:
            gathered.append(np.asarray(list(explicit_ids), dtype=np.int64))
        if not gathered:
            raise ValueError("no inputs")
        return np.unique(np.concatenate(gathered))


class _MockDataset:
    def __init__(self, df_nodes, df_elems, resolver):
        self.nodes_info = {"dataframe": df_nodes}
        self.elements_info = {"dataframe": df_elems}
        self._selection_resolver = resolver


class _MockManager:
    def __init__(self, df_nodes, df_elems, resolver):
        self._df = df_nodes
        self.dataset = _MockDataset(df_nodes, df_elems, resolver)

    def _ensure_node_index_df(self) -> pd.DataFrame:
        return self._df


@pytest.fixture
def mgr() -> _MockManager:
    df_nodes, df_elems = _build_index()
    # "Roof" → all nodes at z=3; "Base" → all nodes at z=0.
    roof = df_nodes[df_nodes["z"] == 3.0]["node_id"].to_numpy(np.int64)
    base = df_nodes[df_nodes["z"] == 0.0]["node_id"].to_numpy(np.int64)
    resolver = _MockSelResolver(
        node_sets={"roof": roof, "base": base},
        id_sets={1: roof, 2: base},
    )
    return _MockManager(df_nodes, df_elems, resolver)


# ---------------------------------------------------------------------- #
# Anchors                                                                 #
# ---------------------------------------------------------------------- #

def test_select_with_no_anchor_universe_is_all_nodes(mgr: _MockManager):
    sel = NodeSelector(mgr)
    assert sel.count() == 64
    # Mask still works without an anchor (universe = all nodes).
    m = sel.mask()
    assert m.size == 64
    assert m.all()


def test_from_selection_by_name(mgr: _MockManager):
    sel = NodeSelector(mgr).from_selection("Roof")
    assert sel.count() == 16
    df = sel.df()
    assert (df["z"].to_numpy() == 3.0).all()


def test_from_selection_by_id(mgr: _MockManager):
    sel = NodeSelector(mgr).from_selection(2)  # base
    assert sel.count() == 16
    assert (sel.df()["z"].to_numpy() == 0.0).all()


def test_with_ids_anchor(mgr: _MockManager):
    ids = [1, 2, 3, 4]
    sel = NodeSelector(mgr).with_ids(ids)
    out = sel.ids().tolist()
    assert sorted(out) == sorted(ids)


# ---------------------------------------------------------------------- #
# Spatial primitives                                                      #
# ---------------------------------------------------------------------- #

def test_within_box(mgr: _MockManager):
    sel = NodeSelector(mgr).within_box(min=(0, 0, 0), max=(1, 1, 1))
    # 2*2*2 = 8 nodes
    assert sel.count() == 8


def test_within_distance(mgr: _MockManager):
    sel = NodeSelector(mgr).within_distance((0, 0, 0), radius=1.0)
    ids = set(sel.ids().tolist())
    # Nodes at distance 0 (origin) and 1 (axis-aligned neighbours).
    # Coords with sum of squares <= 1: (0,0,0), (1,0,0), (0,1,0), (0,0,1).
    assert len(ids) == 4


def test_nearest_to_k(mgr: _MockManager):
    sel = NodeSelector(mgr).nearest_to((0.1, 0.1, 0.1), k=3)
    ids = sel.ids().tolist()
    # First should be (0,0,0); next two are axis-aligned neighbours.
    assert len(ids) == 3
    df = sel.df().reset_index(drop=True)
    # Distance ordering preserved.
    assert df.iloc[0]["x"] == 0.0
    assert df.iloc[0]["y"] == 0.0
    assert df.iloc[0]["z"] == 0.0


def test_on_plane_axis_aligned(mgr: _MockManager):
    sel = NodeSelector(mgr).on_plane(z=2.0)
    df = sel.df()
    assert (df["z"].to_numpy() == 2.0).all()
    assert sel.count() == 16


def test_at_level_z(mgr: _MockManager):
    sel = NodeSelector(mgr).at_level("z", 2.0)
    df = sel.df()
    assert (df["z"].to_numpy() == 2.0).all()
    assert sel.count() == 16


def test_coord_in_axis(mgr: _MockManager):
    sel = NodeSelector(mgr).coord_in("x", lo=1.0, hi=2.0)
    df = sel.df()
    xs = df["x"].to_numpy()
    assert ((xs >= 1.0) & (xs <= 2.0)).all()


def test_near_line(mgr: _MockManager):
    # Line along the x-axis at y=0, z=0
    sel = NodeSelector(mgr).near_line((0, 0, 0), (3, 0, 0), radius=0.0)
    df = sel.df()
    # Only nodes with y=0 and z=0 (the line itself) qualify.
    assert (df["y"].to_numpy() == 0.0).all()
    assert (df["z"].to_numpy() == 0.0).all()


def test_where_predicate(mgr: _MockManager):
    sel = NodeSelector(mgr).where(
        lambda df: (df["x"].to_numpy() ** 2 + df["y"].to_numpy() ** 2) < 1.5
    )
    df = sel.df()
    # x²+y² < 1.5 → (x,y) ∈ {(0,0), (1,0), (0,1)} = 3 combos × 4 z-levels.
    assert sel.count() == 12
    assert df["x"].max() <= 1.0


def test_where_predicate_shape_mismatch_raises(mgr: _MockManager):
    sel = NodeSelector(mgr).where(lambda df: np.ones(3, dtype=bool))
    with pytest.raises(ValueError, match="predicate returned shape"):
        sel.df()


# ---------------------------------------------------------------------- #
# attached_to                                                             #
# ---------------------------------------------------------------------- #

def test_attached_to_explicit_ids(mgr: _MockManager):
    # Element 1 connects nodes (0,0,0) and (1,0,0).
    sel = NodeSelector(mgr).attached_to(element_ids=[1])
    ids = set(sel.ids().tolist())
    df_nodes = mgr._df
    n1 = int(df_nodes[
        (df_nodes["x"] == 0) & (df_nodes["y"] == 0) & (df_nodes["z"] == 0)
    ]["node_id"].iloc[0])
    n2 = int(df_nodes[
        (df_nodes["x"] == 1) & (df_nodes["y"] == 0) & (df_nodes["z"] == 0)
    ]["node_id"].iloc[0])
    assert ids == {n1, n2}


def test_attached_to_requires_one_source(mgr: _MockManager):
    with pytest.raises(ValueError, match="exactly one"):
        NodeSelector(mgr).attached_to()


# ---------------------------------------------------------------------- #
# Boolean composition                                                     #
# ---------------------------------------------------------------------- #

def test_intersection(mgr: _MockManager):
    a = NodeSelector(mgr).from_selection("Roof")
    b = NodeSelector(mgr).at_level("x", 0.0)
    ids = set((a & b).ids().tolist())
    # roof (z=3) ∩ x=0 → 4 nodes (one per y).
    assert len(ids) == 4


def test_union(mgr: _MockManager):
    a = NodeSelector(mgr).from_selection("Roof")
    b = NodeSelector(mgr).from_selection("Base")
    ids = set((a | b).ids().tolist())
    # Roof + Base = 32 nodes total.
    assert len(ids) == 32


def test_negation_universe_is_anchor(mgr: _MockManager):
    a = NodeSelector(mgr).from_selection("Roof").at_level("x", 0.0)
    # The universe is the Roof set; negation gives the rest of the roof.
    not_a = ~a
    ids = set(not_a.ids().tolist())
    # Roof has 16 nodes; 4 at x=0; complement has 12.
    assert len(ids) == 12


def test_negation_without_anchor_raises(mgr: _MockManager):
    sel = NodeSelector(mgr).at_level("z", 0.0)
    with pytest.raises(ValueError, match="anchor"):
        ~sel


def test_combinator_cannot_take_anchor_methods(mgr: _MockManager):
    a = NodeSelector(mgr).from_selection("Roof")
    b = NodeSelector(mgr).from_selection("Base")
    with pytest.raises(TypeError, match="combined selector"):
        (a & b).from_selection("Roof")


def test_mask_indexed_by_universe(mgr: _MockManager):
    sel = NodeSelector(mgr).from_selection("Roof").at_level("x", 0.0)
    m = sel.mask()
    # Mask indexed by the universe (Roof = 16 nodes).
    assert m.size == 16
    assert m.sum() == 4


def test_repr_includes_anchor_and_ops(mgr: _MockManager):
    sel = (
        NodeSelector(mgr)
        .from_selection("Roof")
        .within_box(min=(0, 0, 0), max=(3, 3, 3))
    )
    r = repr(sel)
    assert "NodeSelector" in r
    assert "from_selection" in r
    assert "WithinBoxOp" in r


def test_selectors_are_immutable(mgr: _MockManager):
    a = NodeSelector(mgr).from_selection("Roof")
    b = a.at_level("x", 0.0)
    # Adding the op did not mutate ``a``.
    assert a.count() == 16
    assert b.count() == 4
