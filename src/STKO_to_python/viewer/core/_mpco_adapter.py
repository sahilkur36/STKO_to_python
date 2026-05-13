"""Concrete :class:`DataSource` over an :class:`MPCODataSet`.

This is the **one** place where viewer code touches STKO's pandas
DataFrame world. Every other viewer module operates against the
:class:`DataSource` protocol declared in :mod:`.datasource`, which
keeps layers backend- and data-source-agnostic. A future non-MPCO
source (e.g. a streaming OpenSeesPy adapter) just implements the
protocol — the layers stay unchanged.

See ``docs/viewer/01-architecture.md`` §5 and
``docs/viewer/02-porting-from-apegmsh.md`` §8 for the rationale.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence, Tuple, Union

import numpy as np

from .selection import SelectionSpec
from .types import BBox

if TYPE_CHECKING:
    from ...core.dataset import MPCODataSet


_StrOrTuple = Union[str, Sequence[str]]
_IntOrTuple = Union[int, np.integer, Sequence[int]]


class MPCODataSourceAdapter:
    """:class:`DataSource` implementation wrapping a :class:`MPCODataSet`.

    Translates the viewer's array-oriented protocol into DataFrame
    queries against the dataset's existing caches. The adapter keeps a
    reference to the dataset but never mutates it; one adapter can be
    reused across many plotting calls.

    Caches
    ------
    The adapter memoizes derived structures that are pure functions of
    the dataset's geometry:

    * ``node_id -> row`` and ``element_id -> row`` lookup maps, lazily
      built on the first ``node_coords(ids=...)`` /
      ``element_centroids(ids=...)`` call;
    * the model bounding box;
    * resolved id arrays keyed by ``SelectionSpec`` (which is frozen and
      hashable) — so a layer that re-asks for the same selection across
      steps pays the resolver cost once.

    Parameters
    ----------
    dataset:
        The :class:`MPCODataSet` to adapt.
    """

    __slots__ = (
        "_dataset",
        "_node_id_to_row",
        "_elem_id_to_row",
        "_bbox",
        "_selection_cache",
    )

    def __init__(self, dataset: "MPCODataSet") -> None:
        self._dataset = dataset
        self._node_id_to_row: dict[int, int] | None = None
        self._elem_id_to_row: dict[int, int] | None = None
        self._bbox: BBox | None = None
        self._selection_cache: dict[tuple[str, SelectionSpec], np.ndarray] = {}

    @property
    def dataset(self) -> "MPCODataSet":
        return self._dataset

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def node_coords(self, ids: np.ndarray | None = None) -> np.ndarray:
        """``(N, 3)`` node coordinates.

        Parameters
        ----------
        ids:
            Optional node IDs. ``None`` returns coordinates for every
            node in dataset order. A non-empty array returns rows in
            the **caller's** order — so ``node_coords([3, 1, 2])`` and
            ``node_coords([1, 2, 3])`` produce different orderings even
            though both name the same three nodes.

        Raises
        ------
        KeyError
            If any element of ``ids`` is not present in the dataset.
        """
        df = self._dataset.nodes_info["dataframe"]
        if df.empty:
            return np.zeros((0, 3), dtype=np.float64)
        coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if ids is None:
            return coords
        ids_arr = np.asarray(ids, dtype=np.int64).ravel()
        if ids_arr.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        mapping = self._node_index_map()
        rows = np.empty(ids_arr.size, dtype=np.int64)
        for i, nid in enumerate(ids_arr):
            try:
                rows[i] = mapping[int(nid)]
            except KeyError as exc:
                raise KeyError(f"Node id {int(nid)} not in dataset") from exc
        return coords[rows]

    def element_centroids(self, ids: np.ndarray | None = None) -> np.ndarray:
        """``(E, 3)`` element centroids.

        Same ``ids`` semantics as :meth:`node_coords` — caller order is
        preserved, missing IDs raise.
        """
        df = self._dataset.elements_info["dataframe"]
        if df.empty:
            return np.zeros((0, 3), dtype=np.float64)
        coords = df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(
            dtype=np.float64
        )
        if ids is None:
            return coords
        ids_arr = np.asarray(ids, dtype=np.int64).ravel()
        if ids_arr.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        mapping = self._element_index_map()
        rows = np.empty(ids_arr.size, dtype=np.int64)
        for i, eid in enumerate(ids_arr):
            try:
                rows[i] = mapping[int(eid)]
            except KeyError as exc:
                raise KeyError(f"Element id {int(eid)} not in dataset") from exc
        return coords[rows]

    def model_bbox(self) -> BBox:
        """Axis-aligned bounding box over every node in the model.

        Cached after the first call. Empty datasets return an all-zero
        :class:`BBox` (so layers can still call ``set_bounds`` without
        special-casing the empty path).
        """
        if self._bbox is None:
            df = self._dataset.nodes_info["dataframe"]
            if df.empty:
                self._bbox = BBox(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                xs = df["x"].to_numpy(dtype=np.float64)
                ys = df["y"].to_numpy(dtype=np.float64)
                zs = df["z"].to_numpy(dtype=np.float64)
                self._bbox = BBox(
                    float(xs.min()),
                    float(ys.min()),
                    float(zs.min()),
                    float(xs.max()),
                    float(ys.max()),
                    float(zs.max()),
                )
        return self._bbox

    # ------------------------------------------------------------------ #
    # Time axis
    # ------------------------------------------------------------------ #
    def n_steps(self, stage: str | None = None) -> int:
        """Number of analysis steps in ``stage`` (defaults to stage 0)."""
        stage = self._resolve_stage(stage)
        try:
            return int(self._dataset.number_of_steps[stage])
        except KeyError as exc:
            raise KeyError(
                f"Stage {stage!r} not present in dataset.number_of_steps"
            ) from exc

    def time(self, stage: str | None = None) -> np.ndarray:
        """``(T,)`` time values for ``stage`` (defaults to stage 0).

        Returns the ``TIME`` column from ``dataset.time`` restricted to
        ``stage``, in ``STEP``-ascending order — same convention the
        rest of the library uses for time-history queries.
        """
        stage = self._resolve_stage(stage)
        df = self._dataset.time
        try:
            stage_df = df.loc[stage]
        except KeyError as exc:
            raise KeyError(f"Stage {stage!r} not present in dataset.time") from exc
        return stage_df["TIME"].to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Selection resolution
    # ------------------------------------------------------------------ #
    def resolve_node_ids(self, spec: SelectionSpec) -> np.ndarray:
        """Return the node IDs matching ``spec`` as a sorted ``int64`` array.

        Fields are AND-combined: a node is returned only when every
        non-``None`` criterion of ``spec`` accepts it. Within a single
        criterion (e.g. a tuple of selection-set names), members are
        union-combined — same semantics as
        :class:`~STKO_to_python.selection.SelectionSetResolver`.

        ``spec.element_type`` and ``spec.element_ids`` are element-only
        filters and are ignored when resolving nodes.
        """
        key = ("node", spec)
        cached = self._selection_cache.get(key)
        if cached is not None:
            return cached.copy()

        all_ids = self._all_node_ids()
        if spec.is_empty():
            self._selection_cache[key] = all_ids
            return all_ids.copy()

        masks: list[np.ndarray] = []
        names, sids = _selection_set_inputs(spec)
        if names or sids:
            resolver = self._dataset._selection_resolver
            from_sel = resolver.resolve_nodes(
                names=names if names else None,
                ids=sids if sids else None,
            )
            masks.append(from_sel)
        if spec.node_ids is not None:
            masks.append(np.asarray(spec.node_ids, dtype=np.int64).ravel())

        out = _intersect_with_universe(masks, all_ids)
        self._selection_cache[key] = out
        return out.copy()

    def resolve_element_ids(self, spec: SelectionSpec) -> np.ndarray:
        """Return the element IDs matching ``spec`` as a sorted ``int64`` array.

        ``spec.node_ids`` is a node-only filter and is ignored here.
        ``spec.element_type`` matches on the *base* type — the prefix
        before ``[`` — so both ``"203-ASDShellQ4"`` and
        ``"203-ASDShellQ4[4n]"`` select the same elements (parallel to
        ``ds.plot.mesh``'s ``element_type=`` argument).
        """
        key = ("element", spec)
        cached = self._selection_cache.get(key)
        if cached is not None:
            return cached.copy()

        all_ids = self._all_element_ids()
        if spec.is_empty():
            self._selection_cache[key] = all_ids
            return all_ids.copy()

        masks: list[np.ndarray] = []
        names, sids = _selection_set_inputs(spec)
        if names or sids:
            resolver = self._dataset._selection_resolver
            from_sel = resolver.resolve_elements(
                names=names if names else None,
                ids=sids if sids else None,
            )
            masks.append(from_sel)
        if spec.element_ids is not None:
            masks.append(np.asarray(spec.element_ids, dtype=np.int64).ravel())
        if spec.element_type is not None:
            masks.append(self._filter_by_element_type(spec.element_type))

        out = _intersect_with_universe(masks, all_ids)
        self._selection_cache[key] = out
        return out.copy()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_stage(self, stage: str | None) -> str:
        if stage is not None:
            return stage
        stages = self._dataset.model_stages
        if not stages:
            raise ValueError("Dataset has no model stages")
        return stages[0]

    def _all_node_ids(self) -> np.ndarray:
        df = self._dataset.nodes_info["dataframe"]
        if df.empty:
            return np.zeros(0, dtype=np.int64)
        return df["node_id"].to_numpy(dtype=np.int64)

    def _all_element_ids(self) -> np.ndarray:
        df = self._dataset.elements_info["dataframe"]
        if df.empty:
            return np.zeros(0, dtype=np.int64)
        return df["element_id"].to_numpy(dtype=np.int64)

    def _node_index_map(self) -> dict[int, int]:
        if self._node_id_to_row is None:
            df = self._dataset.nodes_info["dataframe"]
            self._node_id_to_row = {
                int(nid): i for i, nid in enumerate(df["node_id"].to_numpy())
            }
        return self._node_id_to_row

    def _element_index_map(self) -> dict[int, int]:
        if self._elem_id_to_row is None:
            df = self._dataset.elements_info["dataframe"]
            self._elem_id_to_row = {
                int(eid): i for i, eid in enumerate(df["element_id"].to_numpy())
            }
        return self._elem_id_to_row

    def _filter_by_element_type(self, element_type: _StrOrTuple) -> np.ndarray:
        df = self._dataset.elements_info["dataframe"]
        types: Tuple[str, ...]
        if isinstance(element_type, str):
            types = (element_type,)
        else:
            types = tuple(str(t) for t in element_type)
        bases = [t.split("[")[0] for t in types]
        col = df["element_type"]
        mask = np.zeros(len(df), dtype=bool)
        for base in bases:
            mask |= col.str.startswith(base).to_numpy()
        return df.loc[mask, "element_id"].to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------- #
# Module-private helpers
# ---------------------------------------------------------------------- #
def _selection_set_inputs(
    spec: SelectionSpec,
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Split a spec's selection-set fields into ``(names, ids)`` tuples."""
    raw_name = spec.selection_set_name
    if raw_name is None:
        names: Tuple[str, ...] = ()
    elif isinstance(raw_name, str):
        names = (raw_name,)
    else:
        names = tuple(str(n) for n in raw_name)

    raw_id = spec.selection_set_id
    if raw_id is None:
        sids: Tuple[int, ...] = ()
    elif isinstance(raw_id, (int, np.integer)):
        sids = (int(raw_id),)
    else:
        sids = tuple(int(s) for s in raw_id)

    return names, sids


def _intersect_with_universe(
    masks: Iterable[np.ndarray], universe: np.ndarray
) -> np.ndarray:
    """AND-combine ``masks`` then clip to entities the dataset knows about.

    ``np.intersect1d`` deduplicates and sorts, which is the canonical
    shape for a resolved selection — downstream layers don't rely on
    insertion order, and a stable contract makes the cache key
    well-defined.
    """
    mask_list = list(masks)
    if not mask_list:
        return universe.copy()
    out = mask_list[0]
    for m in mask_list[1:]:
        out = np.intersect1d(out, m, assume_unique=False)
    return np.intersect1d(out, universe, assume_unique=False)


__all__ = ["MPCODataSourceAdapter"]
