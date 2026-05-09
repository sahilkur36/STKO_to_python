"""NodeSelector — lazy, chainable, composable node-id queries.

Mirrors :class:`STKO_to_python.elements.selector.ElementSelector` for
nodes. Resolution to an id array is deferred until :meth:`ids` /
:meth:`mask` / :meth:`df` / :meth:`count` is called.

The composition primitives are:

* **Chain** — ``sel.from_selection(...).within_box(...).nearest_to(...)`` —
  each call AND-narrows the candidate set in source order.
* **Boolean algebra** — ``a & b``, ``a | b``, ``~a`` — operate on the
  resolved id sets via ``np.intersect1d`` / ``np.union1d`` /
  ``np.setdiff1d`` against an anchor-bound universe.

Universe rule
-------------
Without an anchor, the universe is *all* nodes in the model — spatial
primitives still work fine. ``~sel`` requires an explicit anchor
(``from_selection`` / ``with_ids``) so that "complement" has a
well-defined meaning; otherwise it would silently negate against the
whole model and produce a surprisingly large set.

Combinators inherit a derived universe:

* ``(a & b)._universe`` = ``a._universe ∩ b._universe``
* ``(a | b)._universe`` = ``a._universe ∪ b._universe``
* ``(~a)._universe``  = ``a._universe``  (idempotent)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .node_manager import NodeManager
    from ..elements.selector import ElementSelector


ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------- #
# Filter ops                                                             #
# ---------------------------------------------------------------------- #

class _FilterOp:
    """Marker base. Each op narrows a DataFrame; ops compose by stacking."""

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class _WithinBoxOp(_FilterOp):
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        bmin = np.asarray(self.bbox_min, dtype=np.float64)
        bmax = np.asarray(self.bbox_max, dtype=np.float64)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()
        mask = (
            (x >= bmin[0]) & (x <= bmax[0]) &
            (y >= bmin[1]) & (y <= bmax[1]) &
            (z >= bmin[2]) & (z <= bmax[2])
        )
        return df.loc[mask]


@dataclass(frozen=True)
class _WithinDistanceOp(_FilterOp):
    point: Tuple[float, float, float]
    radius: float

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        p = np.asarray(self.point, dtype=np.float64)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()
        d = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2)
        return df.loc[d <= self.radius]


@dataclass(frozen=True)
class _NearestToOp(_FilterOp):
    point: Tuple[float, float, float]
    k: int

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty or self.k <= 0:
            return df.iloc[0:0]
        p = np.asarray(self.point, dtype=np.float64)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()
        d = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2)
        if self.k >= len(df):
            return df.iloc[np.argsort(d, kind="stable")]
        idx = np.argpartition(d, self.k - 1)[: self.k]
        idx = idx[np.argsort(d[idx], kind="stable")]
        return df.iloc[idx]


@dataclass(frozen=True)
class _OnPlaneOp(_FilterOp):
    point: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    tol: float

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        p = np.asarray(self.point, dtype=np.float64)
        n = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            raise ValueError("on_plane: normal vector has zero length.")
        n = n / norm
        coords = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        signed = (coords - p) @ n
        return df.loc[np.abs(signed) <= self.tol]


@dataclass(frozen=True)
class _NearLineOp(_FilterOp):
    p0: Tuple[float, float, float]
    p1: Tuple[float, float, float]
    radius: float

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        a = np.asarray(self.p0, dtype=np.float64)
        b = np.asarray(self.p1, dtype=np.float64)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-30:
            raise ValueError("near_line: p0 and p1 coincide.")
        c = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        t = np.clip(((c - a) @ ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(c - proj, axis=1)
        return df.loc[d <= self.radius]


@dataclass(frozen=True)
class _CoordInOp(_FilterOp):
    axis: str
    lo: Optional[float]
    hi: Optional[float]

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        col = self.axis.lower()
        if col not in {"x", "y", "z"}:
            raise ValueError(f"coord_in: unknown axis {self.axis!r}.")
        v = df[col].to_numpy()
        mask = np.ones(len(v), dtype=bool)
        if self.lo is not None:
            mask &= v >= self.lo
        if self.hi is not None:
            mask &= v <= self.hi
        return df.loc[mask]


@dataclass(frozen=True)
class _AtLevelOp(_FilterOp):
    axis: str
    value: float
    tol: float

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        col = self.axis.lower()
        if col not in {"x", "y", "z"}:
            raise ValueError(f"at_level: unknown axis {self.axis!r}.")
        v = df[col].to_numpy()
        return df.loc[np.abs(v - self.value) <= self.tol]


@dataclass(frozen=True)
class _AttachedToOp(_FilterOp):
    """Keep nodes that appear in the ``node_list`` of any of a given set
    of element ids.

    The element id list is captured at op-construction time. Resolution
    against the elements_info table happens at :meth:`apply` so
    selectors stay lazy.
    """
    element_ids: Tuple[int, ...]

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty or not self.element_ids:
            return df.iloc[0:0]
        elements_info = getattr(manager.dataset, "elements_info", None)
        if not isinstance(elements_info, dict):
            raise ValueError(
                "attached_to: dataset has no elements_info table; "
                "cannot resolve element connectivity."
            )
        df_elem = elements_info.get("dataframe")
        if df_elem is None or df_elem.empty:
            raise ValueError(
                "attached_to: elements_info dataframe is empty."
            )
        eid_set = set(int(x) for x in self.element_ids)
        sub = df_elem[df_elem["element_id"].isin(eid_set)]
        if sub.empty:
            return df.iloc[0:0]
        attached: set[int] = set()
        for nl in sub["node_list"]:
            for n in nl:
                attached.add(int(n))
        if not attached:
            return df.iloc[0:0]
        return df.loc[df["node_id"].isin(attached)]


@dataclass(frozen=True)
class _PredicateOp(_FilterOp):
    fn: Callable[[pd.DataFrame], np.ndarray]

    def apply(self, df: pd.DataFrame, manager: "NodeManager") -> pd.DataFrame:
        if df.empty:
            return df
        mask = self.fn(df)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (len(df),):
            raise ValueError(
                f"where(fn): predicate returned shape {mask_arr.shape}, "
                f"expected ({len(df)},)."
            )
        return df.loc[mask_arr]


# ---------------------------------------------------------------------- #
# Anchor                                                                 #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Anchor:
    selection: Optional[Tuple[Any, ...]] = None  # names or set ids
    explicit_ids: Optional[Tuple[int, ...]] = None

    def is_set(self) -> bool:
        return self.selection is not None or self.explicit_ids is not None

    def with_(
        self,
        *,
        selection: Optional[Tuple[Any, ...]] = None,
        explicit_ids: Optional[Tuple[int, ...]] = None,
    ) -> "_Anchor":
        return _Anchor(
            selection=selection if selection is not None else self.selection,
            explicit_ids=(
                explicit_ids
                if explicit_ids is not None
                else self.explicit_ids
            ),
        )


# ---------------------------------------------------------------------- #
# NodeSelector                                                           #
# ---------------------------------------------------------------------- #

class NodeSelector:
    """Lazy, immutable node-id query.

    Construct via :meth:`NodeManager.select` and chain primitives.
    Resolution to ids is deferred until :meth:`ids`, :meth:`mask`,
    :meth:`df`, or :meth:`count` is called.

    Examples
    --------
    >>> sel = (dataset.nodes.select()
    ...        .from_selection("Roof")
    ...        .within_box(min=(0, 0, 0), max=(10, 10, 30))
    ...        .nearest_to((5, 5, 30), k=4))
    >>> ids = sel.ids()
    >>> df  = sel.df()
    >>> mask = sel.mask()
    >>> n   = sel.count()

    Composition
    -----------
    >>> a = dataset.nodes.select().from_selection("Core").at_level("z", 30.0)
    >>> b = dataset.nodes.select().from_selection("Perimeter").at_level("z", 30.0)
    >>> (a & b).ids()        # intersection
    >>> (a | b).ids()        # union
    >>> (~a).ids()           # complement within a's anchor universe
    """

    def __init__(
        self,
        manager: "NodeManager",
        *,
        anchor: Optional[_Anchor] = None,
        ops: Tuple[_FilterOp, ...] = (),
    ) -> None:
        self._manager = manager
        self._anchor = anchor or _Anchor()
        self._ops = ops

    # ------------------------------------------------------------------ #
    # Anchor primitives                                                   #
    # ------------------------------------------------------------------ #

    def from_selection(
        self,
        name_or_id: Union[str, int, Sequence[Union[str, int]]],
    ) -> "NodeSelector":
        """Anchor universe to one or more selection sets (by name or id)."""
        if isinstance(name_or_id, (str, int, np.integer)):
            sel_tup: Tuple[Any, ...] = (name_or_id,)
        else:
            sel_tup = tuple(name_or_id)
        existing = self._anchor.selection or ()
        return NodeSelector(
            self._manager,
            anchor=self._anchor.with_(selection=existing + sel_tup),
            ops=self._ops,
        )

    def with_ids(
        self, ids: Union[int, Sequence[int], np.ndarray]
    ) -> "NodeSelector":
        """Anchor universe to an explicit set of node ids."""
        if isinstance(ids, (int, np.integer)):
            arr = (int(ids),)
        else:
            arr = tuple(int(x) for x in np.asarray(ids).ravel())
        existing = self._anchor.explicit_ids or ()
        return NodeSelector(
            self._manager,
            anchor=self._anchor.with_(explicit_ids=existing + arr),
            ops=self._ops,
        )

    # ------------------------------------------------------------------ #
    # Spatial primitives                                                  #
    # ------------------------------------------------------------------ #

    def within_box(
        self,
        *,
        min: ArrayLike,
        max: ArrayLike,
    ) -> "NodeSelector":
        """Keep nodes whose coordinates lie inside an axis-aligned bounding box."""
        op = _WithinBoxOp(
            bbox_min=tuple(np.asarray(min, dtype=np.float64).tolist()),
            bbox_max=tuple(np.asarray(max, dtype=np.float64).tolist()),
        )
        return self._with_op(op)

    def within_distance(
        self, point: ArrayLike, radius: float
    ) -> "NodeSelector":
        """Keep nodes within ``radius`` of ``point``."""
        if radius < 0:
            raise ValueError("within_distance: radius must be non-negative.")
        op = _WithinDistanceOp(
            point=tuple(np.asarray(point, dtype=np.float64).tolist()),
            radius=float(radius),
        )
        return self._with_op(op)

    def nearest_to(self, point: ArrayLike, k: int = 1) -> "NodeSelector":
        """Keep the ``k`` nodes nearest to ``point``.

        Result rows are sorted by ascending distance (deterministic, with
        a stable tie-break on the original row order).
        """
        if k < 0:
            raise ValueError("nearest_to: k must be non-negative.")
        op = _NearestToOp(
            point=tuple(np.asarray(point, dtype=np.float64).tolist()),
            k=int(k),
        )
        return self._with_op(op)

    def on_plane(
        self,
        *,
        point: Optional[ArrayLike] = None,
        normal: Optional[ArrayLike] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        tol: float = 1e-6,
    ) -> "NodeSelector":
        """Keep nodes lying on a plane within ``tol`` (signed distance).

        Either pass ``point`` + ``normal`` for a general plane, or one
        of ``x=``, ``y=``, ``z=`` for an axis-aligned plane (equivalent
        to :meth:`at_level` with the same axis).
        """
        axis_args = sum(arg is not None for arg in (x, y, z))
        if axis_args > 1:
            raise ValueError("on_plane: pass at most one of x=, y=, z=.")
        if axis_args == 1 and (point is not None or normal is not None):
            raise ValueError(
                "on_plane: cannot mix x/y/z= with point/normal."
            )
        if axis_args == 1:
            if x is not None:
                p, n = (x, 0.0, 0.0), (1.0, 0.0, 0.0)
            elif y is not None:
                p, n = (0.0, y, 0.0), (0.0, 1.0, 0.0)
            else:
                p, n = (0.0, 0.0, z), (0.0, 0.0, 1.0)  # type: ignore[arg-type]
        else:
            if point is None or normal is None:
                raise ValueError(
                    "on_plane: pass point=+normal= or one of x=/y=/z=."
                )
            p = tuple(np.asarray(point, dtype=np.float64).tolist())  # type: ignore[assignment]
            n = tuple(np.asarray(normal, dtype=np.float64).tolist())  # type: ignore[assignment]
        op = _OnPlaneOp(point=p, normal=n, tol=float(tol))  # type: ignore[arg-type]
        return self._with_op(op)

    def near_line(
        self,
        p0: ArrayLike,
        p1: ArrayLike,
        radius: float,
    ) -> "NodeSelector":
        """Keep nodes within ``radius`` of the line segment from ``p0`` to ``p1``."""
        if radius < 0:
            raise ValueError("near_line: radius must be non-negative.")
        op = _NearLineOp(
            p0=tuple(np.asarray(p0, dtype=np.float64).tolist()),
            p1=tuple(np.asarray(p1, dtype=np.float64).tolist()),
            radius=float(radius),
        )
        return self._with_op(op)

    def coord_in(
        self,
        axis: str,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> "NodeSelector":
        """Keep nodes with ``<axis>`` coordinate in ``[lo, hi]``.

        Either bound may be ``None`` for a one-sided constraint. ``axis``
        is one of ``"x"``, ``"y"``, ``"z"``.
        """
        if lo is None and hi is None:
            raise ValueError("coord_in: pass at least one of lo=, hi=.")
        return self._with_op(_CoordInOp(axis=str(axis), lo=lo, hi=hi))

    def at_level(
        self,
        axis: str = "z",
        value: Optional[float] = None,
        *,
        tol: float = 1e-6,
    ) -> "NodeSelector":
        """Keep nodes with ``<axis>`` coordinate equal to ``value`` (within ``tol``).

        Convenience wrapper for "story floor" / "support row" picks. Use
        :meth:`coord_in` for ranges, :meth:`on_plane` for off-axis planes.
        """
        if value is None:
            raise ValueError("at_level: ``value`` is required.")
        return self._with_op(
            _AtLevelOp(axis=str(axis), value=float(value), tol=float(tol))
        )

    def attached_to(
        self,
        *,
        element_ids: Union[int, Sequence[int], np.ndarray, None] = None,
        element_selector: Optional["ElementSelector"] = None,
    ) -> "NodeSelector":
        """Keep nodes that participate in the connectivity of the given elements.

        Pass either ``element_ids=`` (explicit ids) or
        ``element_selector=`` (resolved at chain time). Closes the loop
        between node and element selectors — e.g. "all nodes attached to
        the columns inside this story".
        """
        if (element_ids is None) == (element_selector is None):
            raise ValueError(
                "attached_to: pass exactly one of element_ids= or "
                "element_selector=."
            )
        if element_selector is not None:
            ids_arr = element_selector.ids()
        else:
            if isinstance(element_ids, (int, np.integer)):
                ids_arr = np.asarray([int(element_ids)], dtype=np.int64)
            else:
                ids_arr = np.asarray(element_ids, dtype=np.int64).ravel()
        eid_tup = tuple(int(x) for x in ids_arr)
        return self._with_op(_AttachedToOp(element_ids=eid_tup))

    def where(
        self, fn: Callable[[pd.DataFrame], np.ndarray]
    ) -> "NodeSelector":
        """Predicate escape hatch.

        ``fn`` receives the candidate node-index DataFrame (columns
        ``node_id, file_id, index, x, y, z``) and must return a boolean
        array of equal length.

        Example
        -------
        >>> sel.where(lambda df: (df["x"]**2 + df["y"]**2) < 25.0)
        """
        return self._with_op(_PredicateOp(fn=fn))

    # ------------------------------------------------------------------ #
    # Resolution                                                          #
    # ------------------------------------------------------------------ #

    def df(self) -> pd.DataFrame:
        """Return the node-index DataFrame matching this selector."""
        df = self._resolve_universe_df()
        for op in self._ops:
            df = op.apply(df, self._manager)
            if df.empty:
                break
        return df

    def ids(self) -> np.ndarray:
        """Return matched node ids as ``int64`` array.

        Order matches the underlying op chain — for ``nearest_to`` this
        is by ascending distance; otherwise by ascending node_id.
        """
        df = self.df()
        if df.empty:
            return np.empty(0, dtype=np.int64)
        return df["node_id"].to_numpy(dtype=np.int64)

    def mask(self) -> pd.Series:
        """Boolean Series indexed by node_id over this selector's universe.

        With no anchor the universe is *all* nodes in the model.
        """
        uni = self._universe_ids(allow_all=True)
        matched = set(int(x) for x in self.ids().tolist())
        values = np.fromiter(
            (int(nid) in matched for nid in uni), dtype=bool, count=uni.size
        )
        return pd.Series(values, index=uni, name="mask")

    def count(self) -> int:
        """Number of matched nodes."""
        return int(self.ids().size)

    # ------------------------------------------------------------------ #
    # Boolean composition                                                 #
    # ------------------------------------------------------------------ #

    def __and__(self, other: "NodeSelector") -> "NodeSelector":
        if not isinstance(other, NodeSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="and", parts=(self, other))

    def __or__(self, other: "NodeSelector") -> "NodeSelector":
        if not isinstance(other, NodeSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="or", parts=(self, other))

    def __invert__(self) -> "NodeSelector":
        if not self._anchor.is_set():
            raise ValueError(
                "Cannot negate a selector without a from_selection/"
                "with_ids anchor — call .from_selection(...) or "
                ".with_ids(...) first to define the universe."
            )
        return _CombinedSelector(self._manager, kind="not", parts=(self,))

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _with_op(self, op: _FilterOp) -> "NodeSelector":
        return NodeSelector(
            self._manager, anchor=self._anchor, ops=self._ops + (op,)
        )

    def _resolve_universe_df(self) -> pd.DataFrame:
        df = self._manager._ensure_node_index_df()
        a = self._anchor
        if a.selection is not None:
            names: list[str] = []
            ids: list[int] = []
            for s in a.selection:
                if isinstance(s, (int, np.integer)):
                    ids.append(int(s))
                else:
                    names.append(str(s))
            sel_ids = self._manager.dataset._selection_resolver.resolve_nodes(
                names=names or None, ids=ids or None
            )
            df = df[df["node_id"].isin(sel_ids)]
        if a.explicit_ids is not None:
            df = df[df["node_id"].isin(a.explicit_ids)]
        return df

    def _universe_ids(self, *, allow_all: bool = False) -> np.ndarray:
        """Return the node-id universe for this selector.

        ``allow_all=True`` falls back to "all nodes in the model" when
        no anchor is set (used by :meth:`mask`). Negation paths require
        an explicit anchor and pass ``allow_all=False`` (the default).
        """
        if not self._anchor.is_set():
            if not allow_all:
                raise ValueError(
                    "Selector has no anchor; cannot compute a universe."
                )
            return (
                self._manager._ensure_node_index_df()["node_id"]
                .to_numpy(dtype=np.int64)
            )
        return (
            self._resolve_universe_df()["node_id"]
            .to_numpy(dtype=np.int64)
        )

    # ------------------------------------------------------------------ #
    # Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        bits: list[str] = []
        if self._anchor.selection:
            bits.append(f"from_selection={list(self._anchor.selection)!r}")
        if self._anchor.explicit_ids:
            bits.append(f"with_ids({len(self._anchor.explicit_ids)})")
        for op in self._ops:
            bits.append(type(op).__name__.strip("_"))
        return f"NodeSelector({', '.join(bits)})"


# ---------------------------------------------------------------------- #
# Combined (AND/OR/NOT) selector                                         #
# ---------------------------------------------------------------------- #

class _CombinedSelector(NodeSelector):
    """Boolean combinator over child selectors."""

    def __init__(
        self,
        manager: "NodeManager",
        *,
        kind: str,
        parts: Tuple[NodeSelector, ...],
    ) -> None:
        self._manager = manager
        self._anchor = _Anchor()
        self._ops = ()
        if kind not in {"and", "or", "not"}:
            raise ValueError(f"_CombinedSelector: unknown kind {kind!r}.")
        if kind == "not" and len(parts) != 1:
            raise ValueError("_CombinedSelector('not'): expects one part.")
        if kind in {"and", "or"} and len(parts) < 2:
            raise ValueError(f"_CombinedSelector({kind!r}): expects ≥2 parts.")
        self._kind = kind
        self._parts = parts

    def ids(self) -> np.ndarray:
        if self._kind == "and":
            arrs = [p.ids() for p in self._parts]
            return reduce(np.intersect1d, arrs) if arrs else np.empty(0, np.int64)
        if self._kind == "or":
            arrs = [p.ids() for p in self._parts]
            return (
                reduce(np.union1d, arrs) if arrs else np.empty(0, np.int64)
            )
        # not
        inner = self._parts[0]
        uni = inner._universe_ids(allow_all=False)
        return np.setdiff1d(uni, inner.ids(), assume_unique=False)

    def df(self) -> pd.DataFrame:
        ids = self.ids()
        if ids.size == 0:
            return self._manager._ensure_node_index_df().iloc[0:0]
        df_all = self._manager._ensure_node_index_df()
        return df_all[df_all["node_id"].isin(ids)]

    def mask(self) -> pd.Series:
        uni = self._universe_ids(allow_all=True)
        matched = set(int(x) for x in self.ids().tolist())
        values = np.fromiter(
            (int(nid) in matched for nid in uni), dtype=bool, count=uni.size
        )
        return pd.Series(values, index=uni, name="mask")

    def count(self) -> int:
        return int(self.ids().size)

    def _universe_ids(self, *, allow_all: bool = False) -> np.ndarray:
        unis = [p._universe_ids(allow_all=allow_all) for p in self._parts]
        if self._kind == "and":
            return reduce(np.intersect1d, unis) if unis else np.empty(0, np.int64)
        if self._kind == "or":
            return reduce(np.union1d, unis) if unis else np.empty(0, np.int64)
        return unis[0]  # not

    def __and__(self, other: "NodeSelector") -> "NodeSelector":
        if not isinstance(other, NodeSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="and", parts=(self, other))

    def __or__(self, other: "NodeSelector") -> "NodeSelector":
        if not isinstance(other, NodeSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="or", parts=(self, other))

    def __invert__(self) -> "NodeSelector":
        for p in self._parts:
            try:
                p._universe_ids(allow_all=False)
            except ValueError as e:
                raise ValueError(
                    "Cannot negate a combinator whose parts lack anchors. "
                    f"Inner error: {e}"
                ) from e
        return _CombinedSelector(self._manager, kind="not", parts=(self,))

    def from_selection(self, name_or_id):  # type: ignore[override]
        raise TypeError(
            "Cannot call .from_selection() on a combined selector; "
            "anchor each leaf selector before combining."
        )

    def with_ids(self, ids):  # type: ignore[override]
        raise TypeError(
            "Cannot call .with_ids() on a combined selector; anchor "
            "each leaf selector before combining."
        )

    def _with_op(self, op: _FilterOp) -> "NodeSelector":
        raise TypeError(
            "Cannot append filter ops directly to a combined selector. "
            "Apply primitives to leaf selectors before combining, or "
            "pass the combined result through .ids() and rebuild."
        )

    def __repr__(self) -> str:
        op = {"and": " & ", "or": " | "}.get(self._kind)
        if self._kind == "not":
            return f"~({self._parts[0]!r})"
        return "(" + op.join(repr(p) for p in self._parts) + ")"  # type: ignore[arg-type]


__all__ = ["NodeSelector"]
