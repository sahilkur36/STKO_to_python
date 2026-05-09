"""NodeResultMask — threshold / time-window filters on :class:`NodalResults`.

Mirrors :class:`STKO_to_python.elements.result_mask.ResultMask` for the
nodal side. The selector layer (:class:`NodeSelector`) decides *which
nodes* before any HDF5 read; this layer decides *which of the fetched
nodes satisfy a value condition*. Output is a per-node boolean mask
that can be combined with ``&`` / ``|`` / ``~`` and applied to the
parent ``NodalResults`` via ``nr[mask]``.

Differences from the element side
---------------------------------
* :class:`NodalResults` uses a ``(result, component)`` :class:`MultiIndex`
  on its columns. Pick a column with a single call:
  ``nr.where(...).component("DISPLACEMENT", 1)``.
* :meth:`_ResultQuery.magnitude` reduces a vector
  (``sqrt(sum(comp_i**2))``) per ``(node_id, step)`` before time-axis
  reductions — the most common pick for "filter nodes by ``|U|`` peak".

The chain shape is::

    nr.where(time=...)                      # default time window (optional)
      .component("DISPLACEMENT", 1)         # or .magnitude("DISPLACEMENT")
      .abs_peak(time=...)                   # reduction over time → scalar/node
      .gt(0.05)                             # comparator → NodeResultMask
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

# Time-grammar resolver is shared with the element side. Importing
# preserves a single source of truth for the spec semantics.
from ..elements.result_mask import resolve_step_indices

if TYPE_CHECKING:  # pragma: no cover
    from ..results.nodal_results_dataclass import NodalResults


TimeSpec = Union[
    None,
    int,
    float,
    slice,
    Tuple[float, float],
    Sequence[int],
    Sequence[float],
    np.ndarray,
]


# Sentinel for "no override; use the chain's default time window".
_DEFAULT: Any = object()


# ---------------------------------------------------------------------- #
# NodeResultMask                                                          #
# ---------------------------------------------------------------------- #

class NodeResultMask:
    """Per-node boolean mask, composable via ``& / | / ~``.

    Created by the comparator step of a ``nr.where().<...>`` chain or
    by the ``predicate(fn)`` escape hatch. Apply via ``nr[mask]`` (a
    fresh :class:`NodalResults`) or extract the matching ids via
    :meth:`ids`.
    """

    __slots__ = ("_nr", "_series")

    def __init__(self, nr: "NodalResults", series: pd.Series) -> None:
        if not isinstance(series, pd.Series):
            raise TypeError("NodeResultMask: series must be a pandas Series.")
        s = series.astype("boolean").fillna(False).astype(bool)
        canonical_idx = pd.Index(
            np.asarray(_node_ids_from_results(nr), dtype=np.int64)
        )
        s = s.reindex(canonical_idx, fill_value=False)
        s.name = "mask"
        s.index.name = "node_id"
        self._nr = nr
        self._series = s

    # ------------------------------------------------------------------ #
    # Outputs                                                            #
    # ------------------------------------------------------------------ #
    def mask(self) -> pd.Series:
        """Boolean Series indexed by ``node_id``."""
        return self._series.copy()

    def ids(self) -> np.ndarray:
        """``int64`` array of node IDs where the mask is ``True``."""
        return self._series.index[self._series].to_numpy(dtype=np.int64)

    def count(self) -> int:
        """Number of nodes where the mask is ``True``."""
        return int(self._series.sum())

    def apply(self) -> "NodalResults":
        """Return a fresh :class:`NodalResults` trimmed to matched ids."""
        return _subset_nr(self._nr, self.ids())

    # ------------------------------------------------------------------ #
    # Boolean composition                                                 #
    # ------------------------------------------------------------------ #
    def __and__(self, other: "NodeResultMask") -> "NodeResultMask":
        if not isinstance(other, NodeResultMask):
            return NotImplemented  # type: ignore[return-value]
        if self._nr is not other._nr:
            raise ValueError(
                "Cannot AND masks from different NodalResults instances."
            )
        return NodeResultMask(self._nr, self._series & other._series)

    def __or__(self, other: "NodeResultMask") -> "NodeResultMask":
        if not isinstance(other, NodeResultMask):
            return NotImplemented  # type: ignore[return-value]
        if self._nr is not other._nr:
            raise ValueError(
                "Cannot OR masks from different NodalResults instances."
            )
        return NodeResultMask(self._nr, self._series | other._series)

    def __invert__(self) -> "NodeResultMask":
        return NodeResultMask(self._nr, ~self._series)

    def __repr__(self) -> str:
        return (
            f"NodeResultMask(n_true={self.count()}, "
            f"n_total={len(self._series)})"
        )

    def __len__(self) -> int:
        return int(self.count())


# ---------------------------------------------------------------------- #
# _ScalarPerNode — reduced values, exposes comparators                   #
# ---------------------------------------------------------------------- #

class _ScalarPerNode:
    """One float per node, ready for a comparator step."""

    __slots__ = ("_nr", "_values")

    def __init__(self, nr: "NodalResults", values: pd.Series) -> None:
        self._nr = nr
        canonical_idx = pd.Index(
            np.asarray(_node_ids_from_results(nr), dtype=np.int64)
        )
        self._values = values.reindex(canonical_idx)
        self._values.index.name = "node_id"

    def values(self) -> pd.Series:
        """The underlying scalar Series (indexed by ``node_id``)."""
        return self._values.copy()

    def gt(self, value: float) -> NodeResultMask:
        return NodeResultMask(self._nr, self._values > value)

    def lt(self, value: float) -> NodeResultMask:
        return NodeResultMask(self._nr, self._values < value)

    def ge(self, value: float) -> NodeResultMask:
        return NodeResultMask(self._nr, self._values >= value)

    def le(self, value: float) -> NodeResultMask:
        return NodeResultMask(self._nr, self._values <= value)

    def between(self, lo: float, hi: float, *, inclusive: bool = True) -> NodeResultMask:
        if inclusive:
            s = (self._values >= lo) & (self._values <= hi)
        else:
            s = (self._values > lo) & (self._values < hi)
        return NodeResultMask(self._nr, s)

    def outside(self, lo: float, hi: float, *, inclusive: bool = False) -> NodeResultMask:
        if inclusive:
            s = (self._values <= lo) | (self._values >= hi)
        else:
            s = (self._values < lo) | (self._values > hi)
        return NodeResultMask(self._nr, s)

    def eq(self, value: float, *, atol: float = 0.0) -> NodeResultMask:
        if atol == 0.0:
            return NodeResultMask(self._nr, self._values == value)
        return self.near(value, atol=atol)

    def near(self, value: float, *, atol: float) -> NodeResultMask:
        return NodeResultMask(self._nr, (self._values - value).abs() <= atol)

    def __repr__(self) -> str:
        return f"_ScalarPerNode(n={len(self._values)})"


# ---------------------------------------------------------------------- #
# _PerStepSeries — Series indexed by (node_id, step), exposes reductions #
# ---------------------------------------------------------------------- #

class _PerStepSeries:
    """Backbone for both :class:`_ComponentQuery` and :class:`_MagnitudeQuery`.

    Holds a ``(node_id, step)``-indexed Series of values plus the chain's
    default time window, and implements the time-axis reductions and
    ``over_threshold`` / ``predicate`` paths.
    """

    __slots__ = ("_nr", "_series", "_default_time", "_label")

    def __init__(
        self,
        nr: "NodalResults",
        series: pd.Series,
        default_time: TimeSpec,
        label: str,
    ) -> None:
        self._nr = nr
        self._series = series
        self._default_time = default_time
        self._label = label

    # ------------------------------------------------------------------ #
    # Reductions                                                          #
    # ------------------------------------------------------------------ #
    def at_step(self, step: int) -> _ScalarPerNode:
        s = self._slice_steps(np.asarray([int(step)], dtype=np.int64))
        return _ScalarPerNode(self._nr, s.droplevel("step"))

    def at_time(self, t: float) -> _ScalarPerNode:
        idx = resolve_step_indices(float(t), self._time_arr())
        if idx.size == 0:
            raise ValueError(f"at_time({t}): no step found.")
        s = self._slice_steps(np.asarray([int(idx[0])], dtype=np.int64))
        return _ScalarPerNode(self._nr, s.droplevel("step"))

    def peak(self, *, time: Any = _DEFAULT) -> _ScalarPerNode:
        return self._reduce_over_time(time, "max")

    def trough(self, *, time: Any = _DEFAULT) -> _ScalarPerNode:
        return self._reduce_over_time(time, "min")

    def abs_peak(self, *, time: Any = _DEFAULT) -> _ScalarPerNode:
        return self._reduce_over_time(time, "abs_max")

    def mean(self, *, time: Any = _DEFAULT) -> _ScalarPerNode:
        return self._reduce_over_time(time, "mean")

    def residual(self, *, time: Any = _DEFAULT) -> _ScalarPerNode:
        return self._reduce_over_time(time, "last")

    def over_threshold(
        self,
        value: float,
        *,
        time: Any = _DEFAULT,
    ) -> _ScalarPerNode:
        """Fraction of steps in the window where value > ``threshold``.

        Chain a comparator (e.g. ``.gt(0.1)``) to mask nodes that
        spend at least 10% of the window above the threshold.
        """
        eff = self._effective_time(time)
        steps = resolve_step_indices(eff, self._time_arr())
        if steps.size == 0:
            raise ValueError("over_threshold: empty step window.")
        sub = self._slice_steps(steps)
        ser = (sub > value).groupby(level="node_id").mean()
        return _ScalarPerNode(self._nr, ser)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _time_arr(self) -> np.ndarray:
        t = self._nr.time
        if not isinstance(t, np.ndarray):
            t = np.asarray(t, dtype=np.float64)
        return t

    def _effective_time(self, time: Any) -> TimeSpec:
        return self._default_time if time is _DEFAULT else time

    def _slice_steps(self, steps: np.ndarray) -> pd.Series:
        lvl = self._series.index.get_level_values("step")
        return self._series.loc[lvl.isin(steps)]

    def _reduce_over_time(self, time: TimeSpec, op: str) -> _ScalarPerNode:
        eff = self._effective_time(time)
        steps = resolve_step_indices(eff, self._time_arr())
        if steps.size == 0:
            raise ValueError(f"{op}: empty step window.")
        sub = self._slice_steps(steps)
        if op == "max":
            ser = sub.groupby(level="node_id").max()
        elif op == "min":
            ser = sub.groupby(level="node_id").min()
        elif op == "abs_max":
            ser = sub.abs().groupby(level="node_id").max()
        elif op == "mean":
            ser = sub.groupby(level="node_id").mean()
        elif op == "last":
            sorted_ser = (
                sub.reset_index()
                .sort_values("step")
                .set_index(["node_id", "step"])
                .iloc[:, 0]
            )
            ser = sorted_ser.groupby(level="node_id").last()
        else:
            raise ValueError(f"unknown reduction {op!r}")
        return _ScalarPerNode(self._nr, ser)

    def __repr__(self) -> str:
        return f"_PerStepSeries({self._label!r})"


class _ComponentQuery(_PerStepSeries):
    """One ``(result, component)`` column from a :class:`NodalResults`."""

    @classmethod
    def from_column(
        cls,
        nr: "NodalResults",
        result_name: str,
        component: Any,
        default_time: TimeSpec,
    ) -> "_ComponentQuery":
        col = _resolve_component_column(nr, result_name, component)
        ser = nr.df[col]
        if isinstance(ser, pd.DataFrame):
            # Defensive: should not happen since column tuple is exact.
            ser = ser.iloc[:, 0]
        ser.name = f"{result_name}|{component}"
        label = f"{result_name}|{component}"
        return cls(nr, ser, default_time, label)


class _MagnitudeQuery(_PerStepSeries):
    """Vector-magnitude reduction over a result's components.

    Produces one scalar per ``(node_id, step)`` via
    ``sqrt(sum(comp_i**2))``, which then flows through the same
    time-axis reductions as :class:`_ComponentQuery`.
    """

    @classmethod
    def from_components(
        cls,
        nr: "NodalResults",
        result_name: str,
        components: Sequence[Any],
        default_time: TimeSpec,
    ) -> "_MagnitudeQuery":
        # Empty `components` means "use every component for this result"
        # — the common 3-DOF |U| / |V| / |A| case.
        if not components:
            cols = _list_components_for_result(nr, result_name)
            if not cols:
                raise ValueError(
                    f"magnitude: result {result_name!r} has no components."
                )
        else:
            cols = [
                _resolve_component_column(nr, result_name, c) for c in components
            ]
        sub = nr.df[cols]
        # Square + sum across columns, then sqrt — preserves the
        # (node_id, step) MultiIndex on the resulting Series.
        sq = sub.pow(2).sum(axis=1)
        mag = np.sqrt(sq)
        mag.name = f"|{result_name}|"
        label = f"|{result_name}|"
        return cls(nr, mag, default_time, label)


# ---------------------------------------------------------------------- #
# _ResultQuery — nr.where() entry point                                   #
# ---------------------------------------------------------------------- #

class _ResultQuery:
    """Entry point produced by :meth:`NodalResults.where`."""

    __slots__ = ("_nr", "_default_time")

    def __init__(self, nr: "NodalResults", default_time: TimeSpec) -> None:
        self._nr = nr
        self._default_time = default_time

    def component(self, result_name: str, component: Any) -> _ComponentQuery:
        """Pick one ``(result, component)`` column (e.g. ``("DISPLACEMENT", 1)``)."""
        return _ComponentQuery.from_column(
            self._nr, str(result_name), component, self._default_time
        )

    def magnitude(
        self,
        result_name: str,
        *,
        components: Sequence[Any] = (),
    ) -> _MagnitudeQuery:
        """Pick the vector magnitude of a result over given components.

        With ``components=()`` (default) every component for the result
        is used — the natural choice for 3-DOF nodal vectors like
        ``DISPLACEMENT`` / ``VELOCITY`` / ``ACCELERATION``. Pass an
        explicit list to restrict to a subset (e.g. ``components=(1, 2)``
        for a planar magnitude).
        """
        return _MagnitudeQuery.from_components(
            self._nr, str(result_name), tuple(components), self._default_time
        )

    def predicate(
        self,
        fn: Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]],
    ) -> NodeResultMask:
        """Escape hatch — ``fn`` receives the parent ``df`` and must
        return a bool mask aligned with ``node_id`` (length =
        ``n_nodes``) or aligned with the full ``(node_id, step)``
        index (in which case it is reduced via ``any``).
        """
        df = self._nr.df
        node_ids = _node_ids_from_results(self._nr)
        n_nodes = len(node_ids)
        out = fn(df)
        arr = np.asarray(out)
        if arr.shape == (n_nodes,):
            ser = pd.Series(
                arr.astype(bool),
                index=pd.Index(np.asarray(node_ids, dtype=np.int64), name="node_id"),
            )
        elif arr.shape == (len(df),):
            tmp = pd.Series(arr.astype(bool), index=df.index)
            ser = tmp.groupby(level="node_id").any()
        else:
            raise ValueError(
                f"predicate(fn): returned shape {arr.shape}; expected "
                f"({n_nodes},) or ({len(df)},)."
            )
        return NodeResultMask(self._nr, ser)

    def __repr__(self) -> str:
        return f"_ResultQuery(default_time={self._default_time!r})"


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #

def _node_ids_from_results(nr: "NodalResults") -> Tuple[int, ...]:
    """Pull the canonical sorted node-id tuple from a ``NodalResults``."""
    info = getattr(nr, "info", None)
    nids = getattr(info, "nodes_ids", None) if info is not None else None
    if nids:
        return tuple(int(x) for x in nids)
    # Fallback: derive from the dataframe index.
    idx = nr.df.index
    if isinstance(idx, pd.MultiIndex):
        names = list(idx.names) if idx.names else []
        if "node_id" in names:
            level = names.index("node_id")
        elif idx.nlevels == 2:
            level = 0
        elif idx.nlevels == 3:
            level = 1
        else:
            raise ValueError(
                "NodalResults.df index has unexpected nlevels "
                f"{idx.nlevels}; cannot infer node_id level."
            )
        vals = idx.get_level_values(level).unique().to_numpy(dtype=np.int64)
        vals.sort()
        return tuple(int(x) for x in vals)
    return ()


def _resolve_component_column(
    nr: "NodalResults",
    result_name: str,
    component: Any,
) -> Tuple[str, Any]:
    """Resolve ``(result_name, component)`` to an exact column tuple.

    Accepts a ``component`` that is either an ``int`` (1-based index as
    stored in the column MultiIndex) or any value comparing equal to
    one of the stored component labels via ``str``-coercion.
    """
    cols = nr.df.columns
    if not isinstance(cols, pd.MultiIndex):
        raise TypeError(
            "component(...): NodalResults has single-level columns; "
            "the (result, component) two-arg form requires a MultiIndex. "
            "Use predicate(fn) or build the mask from .ids() directly."
        )
    rname = str(result_name)
    direct = (rname, component)
    if direct in cols:
        return direct
    # Try str-coerced match
    comp_str = str(component)
    for c0, c1 in cols:
        if str(c0) == rname and str(c1) == comp_str:
            return (c0, c1)
    available = sorted(
        {str(c1) for (c0, c1) in cols if str(c0) == rname}
    )
    raise ValueError(
        f"component {component!r} not found for result {result_name!r}. "
        f"Available components: {available}"
    )


def _list_components_for_result(
    nr: "NodalResults",
    result_name: str,
) -> list[Tuple[str, Any]]:
    """Return every column tuple under a given result name."""
    cols = nr.df.columns
    if not isinstance(cols, pd.MultiIndex):
        return []
    rname = str(result_name)
    return [(c0, c1) for (c0, c1) in cols if str(c0) == rname]


# ---------------------------------------------------------------------- #
# Subsetting helper — fresh NodalResults from a mask                      #
# ---------------------------------------------------------------------- #

def _subset_nr(nr: "NodalResults", ids: np.ndarray) -> "NodalResults":
    """Build a fresh :class:`NodalResults` keeping only ``ids``.

    Trims ``df``, ``info.nodes_ids``, and ``info.nodes_info`` while
    preserving every other field (time, stage metadata, plot settings,
    selection sets, analysis time, size).
    """
    from ..results.nodal_results_dataclass import NodalResults

    keep = np.asarray(sorted(int(x) for x in ids), dtype=np.int64)

    idx = nr.df.index
    if not isinstance(idx, pd.MultiIndex):
        raise TypeError(
            "NodalResults.df must have a MultiIndex with a 'node_id' level."
        )
    names = list(idx.names) if idx.names else []
    if "node_id" in names:
        node_level = names.index("node_id")
    elif idx.nlevels == 2:
        node_level = 0
    elif idx.nlevels == 3:
        node_level = 1
    else:
        raise ValueError(
            f"NodalResults.df index has unexpected nlevels {idx.nlevels}."
        )

    if keep.size == 0:
        new_df = nr.df.iloc[0:0]
    else:
        lvl = idx.get_level_values(node_level)
        new_df = nr.df.loc[lvl.isin(keep)]

    info = nr.info
    nodes_info_df = getattr(info, "nodes_info", None)
    if isinstance(nodes_info_df, pd.DataFrame) and not nodes_info_df.empty:
        # ``nodes_info`` is indexed by node_id (set up in
        # ``_fetch_nodal_results_uncached``), so a label-based loc is
        # safe; missing ids would raise but we filter to the kept set.
        common = nodes_info_df.index.intersection(keep)
        new_nodes_info = nodes_info_df.loc[common]
    else:
        new_nodes_info = nodes_info_df

    return NodalResults(
        df=new_df,
        time=nr.time,
        name=nr.name,
        nodes_ids=tuple(int(x) for x in keep),
        nodes_info=new_nodes_info,
        results_components=tuple(info.results_components or ()),
        model_stages=tuple(info.model_stages or ()),
        stage_step_ranges=dict(info.stage_step_ranges or {}),
        plot_settings=nr.plot_settings,
        selection_set=info.selection_set,
        analysis_time=info.analysis_time,
        size=info.size,
    )


__all__ = ["NodeResultMask"]
