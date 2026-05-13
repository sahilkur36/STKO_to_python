"""Deformed-mesh visualization for an :class:`MPCODataSet`.

Renders the model at any step with the nodal displacement field
applied to the original node coordinates. Element edges are drawn
class-by-class:

* 2-node line elements (beams)         → 1 segment per element.
* 3-node shell triangles               → 3 edges (closed loop).
* 4-node shell quads                   → 4 edges (closed loop).
* 8-node bricks                        → 12 edges (cube outline).

Anything else with a known node count falls back to a "convex polygon"
edge loop (n edges connecting consecutive nodes); higher-order solids
that don't fit one of the cases above are skipped with a warning.

**Phase 2.5 internals.** As of v1.11, both :func:`plot_deformed_shape`
and :func:`plot_undeformed_shape` flow their rendering through the
viewer's ``Scene + Layer + Backend`` machinery
(:class:`~STKO_to_python.viewer.layers.DeformedMeshLayer` /
:class:`~STKO_to_python.viewer.layers.MeshLayer` over
:class:`~STKO_to_python.viewer.backends.mpl.backend.MplBackend`).
Public signatures, returns, and visual output are unchanged. The
private helpers (:func:`_build_segments`, :func:`_decide_3d`,
:func:`_autoscale_axes`, :func:`_displacement_at_step`) stay here for
now because both the layer code and the v1.x companions
(:func:`plot_beam_solids_deformed`) import them; they relocate to
:mod:`viewer.math` in a later phase. See ``docs/viewer/00-roadmap.md``
Phase 2.5.

The public entry points sit on :class:`STKO_to_python.plotting.plot.Plot`
(``ds.plot.deformed_shape`` / ``ds.plot.undeformed_shape``); this module
is the implementation, not the user-facing API.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


# Brick edge topology — 12 edges of the hex.
_BRICK_EDGES: Tuple[Tuple[int, int], ...] = (
    # bottom face
    (0, 1), (1, 2), (2, 3), (3, 0),
    # top face
    (4, 5), (5, 6), (6, 7), (7, 4),
    # vertical pillars
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _edge_topology(num_nodes: int) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Edge connectivity for an element with ``num_nodes`` nodes.

    Returns ``None`` for unsupported topologies — callers warn and
    skip rather than draw something misleading.
    """
    if num_nodes == 2:
        return ((0, 1),)
    if num_nodes == 3:
        return ((0, 1), (1, 2), (2, 0))
    if num_nodes == 4:
        # Treated as a 4-node shell quad. 4-node tetrahedra exist but
        # aren't covered by this initial scope.
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    if num_nodes == 8:
        return _BRICK_EDGES
    return None


def _class_label(element_type: str, num_nodes: int) -> str:
    """Human-readable bucket label for ``meta["edges_per_class"]``."""
    base = str(element_type).split("[", 1)[0]
    return f"{base}({num_nodes}n)"


def _is_solid_topology(num_nodes: int) -> bool:
    """True for element classes that force 3D rendering.

    Only 8-node bricks qualify under the current scope; 2/3/4-node
    elements may sit in a 2D plane or be embedded in 3D depending on
    their coordinates.
    """
    return num_nodes == 8


def _displacement_at_step(
    dataset: "MPCODataSet",
    *,
    model_stage: str,
    step: int,
) -> Dict[int, np.ndarray]:
    """Fetch DISPLACEMENT for every node at ``(model_stage, step)``.

    Returns a ``{node_id: np.ndarray of length 3}`` map; missing
    components (2-D models) are zero-padded.
    """
    df_nodes = dataset.nodes_info["dataframe"]
    all_node_ids = df_nodes["node_id"].astype(np.int64).tolist()

    nr = dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage=model_stage,
        node_ids=all_node_ids,
    )

    df = nr.df
    # Index is (node_id, step) for a single stage.
    try:
        snap = df.xs(int(step), level="step")
    except KeyError as err:
        raise ValueError(
            f"step={step} not present in DISPLACEMENT for stage "
            f"{model_stage!r}."
        ) from err

    cols = snap.columns
    if isinstance(cols, pd.MultiIndex):
        # (result, component); pull the DISPLACEMENT block.
        comp_cols = [c for c in cols if str(c[0]) == "DISPLACEMENT"]
        if not comp_cols:
            raise ValueError("DISPLACEMENT not present in fetched results.")
        snap = snap.loc[:, comp_cols]
        # Drop the leading level so the columns are just component IDs.
        snap.columns = [c[1] for c in snap.columns]

    # Ensure numeric component ordering 1, 2, 3 (or whatever is present).
    sorted_cols = sorted(snap.columns, key=lambda c: int(c) if str(c).isdigit() else c)
    snap = snap.loc[:, sorted_cols]

    arr = snap.to_numpy(dtype=np.float64)
    n_comp = arr.shape[1]
    if n_comp < 3:
        pad = np.zeros((arr.shape[0], 3 - n_comp), dtype=np.float64)
        arr = np.hstack([arr, pad])
    elif n_comp > 3:
        arr = arr[:, :3]

    return {int(nid): arr[i] for i, nid in enumerate(snap.index.to_numpy())}


def _build_segments(
    *,
    df_elements: pd.DataFrame,
    coord_lookup: Dict[int, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], List[str]]:
    """Build segment endpoint arrays grouped by element class.

    Returns:
        segments_per_class : {class_label: ndarray of shape (n_seg, 2, 3)}
        edges_per_class    : {class_label: edges-per-element count}
        skipped_classes    : list of class labels that were skipped
    """
    segments_per_class: Dict[str, List[np.ndarray]] = {}
    edges_per_class: Dict[str, int] = {}
    skipped: List[str] = []

    for (etype, n_nodes), group in df_elements.groupby(
        ["element_type", "num_nodes"], sort=False
    ):
        topo = _edge_topology(int(n_nodes))
        label = _class_label(str(etype), int(n_nodes))
        if topo is None:
            skipped.append(label)
            continue

        edges_per_class[label] = len(topo)
        seg_chunks: List[np.ndarray] = []
        for node_list in group["node_list"]:
            try:
                pts = np.array(
                    [coord_lookup[int(nid)] for nid in node_list],
                    dtype=np.float64,
                )
            except KeyError:
                # Element references an unknown node — skip individually.
                continue
            if pts.shape[0] != int(n_nodes):
                continue
            for i, j in topo:
                seg_chunks.append(np.stack([pts[i], pts[j]]))
        if seg_chunks:
            segments_per_class[label] = np.asarray(seg_chunks, dtype=np.float64)

    return segments_per_class, edges_per_class, skipped


def _decide_3d(
    df_elements: pd.DataFrame,
    df_nodes: pd.DataFrame,
    *,
    z_tol: float = 1e-9,
) -> bool:
    """Return True if the rendering should be 3D.

    True when either any solid element is present, or the original
    node coordinates span Z (i.e. not strictly planar in XY).
    """
    if (df_elements["num_nodes"] == 8).any():
        return True
    z = df_nodes["z"].to_numpy(dtype=np.float64)
    if z.size == 0:
        return False
    return float(z.max() - z.min()) > z_tol


def _draw_segments(
    ax: Any,
    segments: np.ndarray,
    *,
    is_3d: bool,
    color: Any,
    linewidth: float,
    alpha: float,
    label: Optional[str] = None,
    zorder: float = 2.0,
) -> None:
    """Add a single segment batch to ``ax`` using the right collection."""
    if segments.size == 0:
        return
    if is_3d:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        coll = Line3DCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    else:
        from matplotlib.collections import LineCollection

        # Drop Z for 2D.
        coll = LineCollection(
            segments[:, :, :2],
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    if label is not None:
        coll.set_label(label)
    ax.add_collection(coll)


def _autoscale_axes(
    ax: Any,
    coords: np.ndarray,
    *,
    is_3d: bool,
    pad: float = 0.05,
) -> None:
    """Set axis limits from the rendered coordinate cloud."""
    if coords.size == 0:
        return
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    if is_3d:
        dz = (zmax - zmin) or 1.0
        ax.set_zlim(zmin - pad * dz, zmax + pad * dz)


def plot_deformed_shape(
    dataset: "MPCODataSet",
    *,
    model_stage: str,
    step: int,
    scale: float = 1.0,
    ax: Any = None,
    show_undeformed: bool = True,
    color: Any = "C0",
    undeformed_color: Any = "0.7",
    linewidth: float = 1.2,
    undeformed_linewidth: float = 0.8,
    alpha: float = 1.0,
    undeformed_alpha: float = 0.6,
    title: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Plot the deformed mesh at one step.

    Parameters
    ----------
    dataset : MPCODataSet
        Source dataset.
    model_stage : str
        Stage name (e.g. ``"MODEL_STAGE[1]"``).
    step : int
        Step index inside the stage.
    scale : float, default 1.0
        Displacement amplification. ``1.0`` is true-to-life; for
        elastic models 50–1000× is typical.
    ax : matplotlib Axes or None
        Existing axes. When ``None``, a new figure is created. The
        renderer auto-selects 2D/3D from the model geometry; if you
        pass ``ax``, you are responsible for matching its projection.
    show_undeformed : bool, default True
        Overlay the original mesh in light gray.
    color, undeformed_color : matplotlib color
        Edge colors for the deformed and undeformed meshes.
    linewidth, undeformed_linewidth : float
        Edge line widths.
    alpha, undeformed_alpha : float
        Edge alpha values.
    title : str or None
        Optional axes title.

    Returns
    -------
    (ax, meta) : tuple
        ``meta`` carries:

        * ``deformed_coords`` — ``{node_id: ndarray(3)}``
        * ``original_coords`` — ``{node_id: ndarray(3)}``
        * ``edges_per_class`` — ``{class_label: edges_per_element}``
        * ``segment_count`` — total segments drawn (deformed)
        * ``skipped_classes`` — class labels with unsupported topology
        * ``is_3d`` — whether the axes are 3D
        * ``scale``, ``step``, ``model_stage``
    """
    from ..viewer.backends.mpl.backend import MplBackend
    from ..viewer.core import MPCODataSourceAdapter, Scene
    from ..viewer.layers import DeformedMeshLayer, MeshLayer

    df_nodes = dataset.nodes_info["dataframe"]
    df_elements = dataset.elements_info["dataframe"]
    if df_nodes.empty or df_elements.empty:
        raise ValueError("Dataset has no nodes or no elements to draw.")

    is_3d = _decide_3d(df_elements, df_nodes)

    # Phase 2.5 rewire — undeformed overlay rides on MeshLayer, the
    # deformed mesh on DeformedMeshLayer. The handle is pre-allocated
    # so the caller-supplied ``ax`` is honoured and the default
    # SceneStyle is **not** applied (would mutate plt.rcParams as a
    # side effect; the v1.x renderer never did that).
    backend = MplBackend()
    handle = backend.make_scene(is_3d=is_3d, ax=ax)
    source = MPCODataSourceAdapter(dataset)
    scene = Scene(backend, source, is_3d=is_3d, handle=handle)

    if show_undeformed:
        undeformed_layer = MeshLayer(
            edge_color=undeformed_color,
            linewidth=undeformed_linewidth,
            alpha=undeformed_alpha,
            mpl_zorder=1.0,
        )
        scene.add(undeformed_layer)

    deformed_layer = DeformedMeshLayer(
        model_stage=model_stage,
        step=int(step) if step is not None else None,
        scale=float(scale),
        edge_color=color,
        linewidth=linewidth,
        alpha=alpha,
        mpl_zorder=2.0,
    )
    scene.add(deformed_layer)

    # Framing — keep the v1.x behaviour of including BOTH the deformed
    # and original coords when ``show_undeformed`` is on, so the bbox
    # accommodates the entire motion path.
    deformed = deformed_layer.deformed_coords
    original = deformed_layer.original_coords
    if show_undeformed:
        coord_stack = np.vstack(
            [
                np.stack(list(deformed.values())),
                np.stack(list(original.values())),
            ]
        )
    else:
        coord_stack = np.stack(list(deformed.values()))

    ax = handle.ax
    _autoscale_axes(ax, coord_stack, is_3d=is_3d)

    if not is_3d:
        ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if is_3d:
        ax.set_zlabel("z")

    if title is not None:
        ax.set_title(title)

    meta: Dict[str, Any] = {
        "deformed_coords": deformed,
        "original_coords": original,
        "edges_per_class": deformed_layer.edges_per_class,
        "segment_count": deformed_layer.segment_count,
        "skipped_classes": deformed_layer.skipped_classes,
        "is_3d": is_3d,
        "scale": float(scale),
        "step": int(step) if step is not None else None,
        "model_stage": model_stage,
    }
    return ax, meta


def plot_undeformed_shape(
    dataset: "MPCODataSet",
    *,
    ax: Any = None,
    color: Any = "0.3",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    title: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Plot the original (undeformed) mesh.

    Equivalent to :func:`plot_deformed_shape` with ``scale=0`` and
    ``show_undeformed=False``, but without fetching DISPLACEMENT.
    """
    from ..viewer.backends.mpl.backend import MplBackend
    from ..viewer.core import MPCODataSourceAdapter, Scene
    from ..viewer.layers import MeshLayer

    df_nodes = dataset.nodes_info["dataframe"]
    df_elements = dataset.elements_info["dataframe"]
    if df_nodes.empty or df_elements.empty:
        raise ValueError("Dataset has no nodes or no elements to draw.")

    is_3d = _decide_3d(df_elements, df_nodes)

    # Phase 2.5 rewire — undeformed mesh through MeshLayer. The legacy
    # _draw_segments default zorder is 2.0, so we override MeshLayer's
    # default (1.0) to preserve the v1.x z-stack.
    backend = MplBackend()
    handle = backend.make_scene(is_3d=is_3d, ax=ax)
    source = MPCODataSourceAdapter(dataset)
    scene = Scene(backend, source, is_3d=is_3d, handle=handle)

    layer = MeshLayer(
        edge_color=color,
        linewidth=linewidth,
        alpha=alpha,
        mpl_zorder=2.0,
    )
    scene.add(layer)

    # Build the ``original_coords`` dict the meta surface promises.
    original = {
        int(row.node_id): np.array([row.x, row.y, row.z], dtype=np.float64)
        for row in df_nodes.itertuples(index=False)
    }
    coord_stack = np.stack(list(original.values()))

    ax = handle.ax
    _autoscale_axes(ax, coord_stack, is_3d=is_3d)

    if not is_3d:
        ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if is_3d:
        ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)

    meta: Dict[str, Any] = {
        "original_coords": original,
        "edges_per_class": layer.edges_per_class,
        "segment_count": layer.n_edges,
        "skipped_classes": layer.skipped_classes,
        "is_3d": is_3d,
    }
    return ax, meta


__all__ = ["plot_deformed_shape", "plot_undeformed_shape"]
