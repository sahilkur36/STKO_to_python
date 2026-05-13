"""``ContourLayer`` — scalar field over the mesh, cell- or nodal-topology.

Renders filled polygons over the model surface, colored by a caller-
supplied scalar map. Implements two of the five paths from the
directive (per ``docs/viewer/02-porting-from-apegmsh.md`` §4 /
apeGmsh's ``viewers/diagrams/_contour.py``):

* **cell** (Phase 3.0b) — one scalar per element; rendered through
  ``backend.add_polygons(values=...)``.
* **nodal** (Phase 3.0c) — one scalar per node; rendered through
  ``backend.add_polygons(point_values=...)`` for smooth Gouraud-style
  interpolation. PyVista native; MplBackend raises
  :class:`BackendCapabilityError` because matplotlib's
  ``PolyCollection`` is per-cell only.

The remaining three paths from the directive are upstream
aggregations of these two:

* GP-cell-averaged — one scalar per element from averaged per-GP
  values; same renderer path as ``cell``.
* GP-node-extrap-smooth — extrapolate GPs to nodes via
  :func:`STKO_to_python.viewer.math.gauss_extrapolation.extrapolate_to_nodes_averaged`,
  then render as ``nodal``. Phase 3.0d.
* GP-node-discrete — extrapolate per-element, render each element's
  corner values as a separate discontinuous polygon. Phase 3.0e.

Scalars contract
----------------

For ``topology="cell"``, the caller supplies scalars as
``dict[element_id, float]``. For ``topology="nodal"``, the caller
supplies ``dict[node_id, float]``. In both cases the input is either:

* a **static** dict — the layer is non-time-varying and
  :meth:`update_to_step` is a no-op;
* a **callable** ``(step: int) -> dict`` invoked at attach (with the
  initial step) and at every :meth:`update_to_step` — same dict
  shape, fresh per call.

Both shapes are dict-based on purpose: the caller doesn't need to
know the renderer's element / vertex ordering, and a missing key is
a hard error (no silent data dropouts). Nodal-mode lookups happen
per face vertex against the **node** the vertex came from — shared
nodes across neighbouring faces resolve to the same scalar value,
which is what makes the rendering visually smooth despite the
polydata storing duplicated vertices per polygon.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import numpy as np
import pandas as pd

from ..core.errors import LayerAttachError
from ..core.layer import Layer

if TYPE_CHECKING:
    from ..core.datasource import DataSource
    from ..core.scene import Scene
    from ..core.selection import SelectionSpec


# ---------------------------------------------------------------------- #
# Face topology — which corner indices form a filled face per element class
# ---------------------------------------------------------------------- #


# Tri / quad / hex face encoding. Each element class maps to a tuple of
# faces; each face is a tuple of corner indices in CCW (or VTK-compatible)
# winding.
_FACE_TOPOLOGY: dict[int, Tuple[Tuple[int, ...], ...]] = {
    3: ((0, 1, 2),),
    4: ((0, 1, 2, 3),),
    # 8-node hex — six quad faces of the cube. Winding here is
    # outward-facing in the standard OpenSees node ordering; for
    # filled-color contour rendering only the closed-loop is required
    # so the winding direction isn't load-bearing.
    8: (
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 3, 7, 4),
        (1, 5, 6, 2),
        (0, 4, 5, 1),
        (2, 6, 7, 3),
    ),
}


def _face_topology(num_nodes: int) -> Tuple[Tuple[int, ...], ...] | None:
    """Face topology for an element of ``num_nodes`` corners. ``None`` if
    unsupported (e.g. line elements)."""
    return _FACE_TOPOLOGY.get(int(num_nodes))


def _class_label(element_type: str, num_nodes: int) -> str:
    """Bucket label matching MeshLayer's convention."""
    base = str(element_type).split("[", 1)[0]
    return f"{base}({num_nodes}n)"


# ---------------------------------------------------------------------- #
# ContourLayer
# ---------------------------------------------------------------------- #


ScalarsInput = Mapping[int, float] | Callable[[int], Mapping[int, float]]


class ContourLayer(Layer):
    """Scalar contour over the model surface, cell- or nodal-topology.

    Parameters
    ----------
    scalars:
        Either a static ``dict[int, float]`` or a callable
        ``(step) -> dict[int, float]``. The dict keys are
        **element ids** when ``topology="cell"`` and **node ids**
        when ``topology="nodal"``. Each rendered element / node must
        have an entry; missing keys raise :class:`KeyError`.
    topology:
        ``"cell"`` (default) for one scalar per element, rendered
        through ``backend.add_polygons(values=...)``. ``"nodal"`` for
        one scalar per node, rendered through
        ``backend.add_polygons(point_values=...)`` for smooth
        Gouraud-style interpolation. The nodal path is supported by
        :class:`~STKO_to_python.viewer.backends.pyvista.PyVistaBackend`;
        :class:`~STKO_to_python.viewer.backends.mpl.MplBackend` raises
        :class:`BackendCapabilityError` because
        ``PolyCollection`` is per-cell only.
    step:
        Initial step. Only consulted when ``scalars`` is a callable —
        the static-dict path ignores it.
    cmap:
        Colormap name forwarded to the backend.
    clim:
        ``(vmin, vmax)`` color limits. When ``None`` the layer auto-
        determines limits from the attach-step data and **freezes**
        them so the colorbar means the same thing across animation
        steps. Set explicitly for predictable color comparisons.
    edge_color:
        Optional edge color drawn on top of the filled faces.
    name, selection, visible, z_order:
        Forwarded to :class:`Layer`.
    mpl_zorder:
        matplotlib z-order applied post-hoc via ``actor.set_zorder``.
        Default ``1.5`` sits between :class:`MeshLayer` (1.0) and
        scatter / diagram overlays (≥ 2.0) so the contour fills are
        visible without obscuring later passes.

    Skipped element classes
    -----------------------

    Line elements (2-node) and any class without a face topology in
    :data:`_FACE_TOPOLOGY` are skipped with a ``RuntimeWarning``.
    Their labels surface as :attr:`skipped_classes`. Solids (8-node
    bricks) are rendered as six quad faces per cell — the same
    scalar repeated across each face, which is geometrically
    correct (the brick interior is a single field value) but wastes
    fragment work on hidden interior faces. The Phase 3.X
    boundary-extraction layer reduces this.

    Per-step contract
    -----------------

    When ``scalars`` is callable, :meth:`update_to_step` re-invokes
    it, re-indexes into the rendered-face order, and calls
    ``backend.update_scalars`` on the existing actors — no actor
    recreation, satisfies the apeGmsh perf contract. When
    ``scalars`` is a static dict, :meth:`update_to_step` is a no-op.
    """

    kind: str = "contour"

    def __init__(
        self,
        *,
        scalars: ScalarsInput,
        topology: str = "cell",
        step: int = 0,
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        edge_color: Any = None,
        name: str | None = None,
        selection: "SelectionSpec | None" = None,
        visible: bool = True,
        z_order: int = 0,
        mpl_zorder: float = 1.5,
    ) -> None:
        super().__init__(
            name=name, selection=selection, visible=visible, z_order=z_order,
        )
        if topology not in ("cell", "nodal"):
            raise ValueError(
                f"topology must be 'cell' or 'nodal', got {topology!r}."
            )
        self.topology = topology
        self._scalars_input: ScalarsInput = scalars
        self._initial_step = int(step) if step is not None else 0
        self._current_step: int | None = None
        self.cmap = cmap
        self.clim = clim
        self.edge_color = edge_color
        self.mpl_zorder = mpl_zorder

        # Populated at attach.
        # Each entry: (label, polygons, source_element_ids,
        # face_node_ids, actor).
        # - source_element_ids parallels ``polygons`` per class — one
        #   entry per drawn face. A brick contributes 6 polygon entries
        #   all pointing at its element id.
        # - face_node_ids is a length-n_faces list of (M,) int64 arrays:
        #   the node ids associated with each face vertex, used to look
        #   scalars up by node id in the nodal-topology path.
        self._classes: list[dict[str, Any]] = []
        self._skipped_classes: list[str] = []

    # ------------------------------------------------------------------ #
    # Read-only summary surface
    # ------------------------------------------------------------------ #
    @property
    def is_time_varying(self) -> bool:
        """``True`` if ``scalars`` was given as a callable."""
        return callable(self._scalars_input)

    @property
    def current_step(self) -> int | None:
        """Step the layer is currently rendering, or ``None`` before attach."""
        return self._current_step

    @property
    def skipped_classes(self) -> list[str]:
        """Class labels skipped because their topology isn't supported."""
        return list(self._skipped_classes)

    @property
    def rendered_element_ids(self) -> dict[str, np.ndarray]:
        """``{class_label: element_ids that contributed at least one face}``."""
        out: dict[str, np.ndarray] = {}
        for entry in self._classes:
            ids = entry["source_element_ids"]
            # Each element can contribute multiple faces (bricks=6); pull the
            # unique set in insertion order.
            unique, idx = np.unique(ids, return_index=True)
            out[entry["label"]] = unique[np.argsort(idx)]
        return out

    @property
    def n_faces(self) -> int:
        """Total polygon faces drawn across every class."""
        return sum(len(entry["polygons"]) for entry in self._classes)

    @property
    def actors(self) -> dict[str, Any]:
        """``{class_label: backend actor}`` mapping. For inspection only."""
        return {entry["label"]: entry["actor"] for entry in self._classes}

    # ------------------------------------------------------------------ #
    # Layer lifecycle
    # ------------------------------------------------------------------ #
    def attach(self, scene: "Scene", source: "DataSource") -> None:
        if self.is_attached:
            raise LayerAttachError(f"Layer {self.name!r} is already attached.")
        self._scene = scene
        self._source = source

        ds = source.dataset
        df_nodes = ds.nodes_info["dataframe"]
        df_elements = ds.elements_info["dataframe"]
        if df_nodes.empty or df_elements.empty:
            raise LayerAttachError(
                "Dataset has no nodes or no elements to draw."
            )

        # Resolve selection → element subset (parallel to MeshLayer).
        if self.selection.is_empty():
            df_filtered = df_elements
        else:
            kept = source.resolve_element_ids(self.selection)
            if kept.size == 0:
                raise LayerAttachError(
                    f"No elements remain after filtering for {self.selection!r}."
                )
            kept_set = {int(i) for i in kept}
            df_filtered = df_elements[df_elements["element_id"].isin(kept_set)]
            if df_filtered.empty:
                raise LayerAttachError(
                    f"No elements remain after filtering for {self.selection!r}."
                )

        coord_lookup = {
            int(row.node_id): np.array([row.x, row.y, row.z], dtype=np.float64)
            for row in df_nodes.itertuples(index=False)
        }

        # Build per-class polygon buckets.
        skipped: list[str] = []
        classes: list[dict[str, Any]] = []
        for (etype, n_nodes), group in df_filtered.groupby(
            ["element_type", "num_nodes"], sort=False,
        ):
            topo = _face_topology(int(n_nodes))
            label = _class_label(str(etype), int(n_nodes))
            if topo is None:
                skipped.append(label)
                continue

            polygons: list[np.ndarray] = []
            src_ids: list[int] = []
            face_node_ids: list[np.ndarray] = []
            for row in group.itertuples(index=False):
                node_list = row.node_list
                try:
                    pts = np.array(
                        [coord_lookup[int(nid)] for nid in node_list],
                        dtype=np.float64,
                    )
                except KeyError:
                    # Element references an unknown node — skip.
                    continue
                if pts.shape[0] != int(n_nodes):
                    continue
                node_list_arr = np.asarray(
                    [int(nid) for nid in node_list], dtype=np.int64,
                )
                for face_idx in topo:
                    idx = list(face_idx)
                    polygons.append(pts[idx])
                    src_ids.append(int(row.element_id))
                    face_node_ids.append(node_list_arr[idx])

            if not polygons:
                continue
            classes.append(
                {
                    "label": label,
                    "polygons": polygons,
                    "source_element_ids": np.asarray(src_ids, dtype=np.int64),
                    "face_node_ids": face_node_ids,
                    "actor": None,  # filled below
                }
            )
        self._classes = classes
        self._skipped_classes = skipped

        if not self._classes:
            raise LayerAttachError(
                "ContourLayer has no renderable element classes "
                "(line elements are skipped; need shells or solids)."
            )

        # Initial scalars for every drawn face / per-vertex stream, in
        # the same class / polygon order as the geometry. Auto-frozen
        # clim when not given.
        initial_values = self._values_at_step(self._initial_step)
        if self.clim is None:
            global_values = np.concatenate(
                [arr for arr in initial_values.values() if arr.size]
            ) if initial_values else np.asarray([], dtype=np.float64)
            if global_values.size:
                vmin = float(np.nanmin(global_values))
                vmax = float(np.nanmax(global_values))
                # Avoid a degenerate single-value clim — colormaps need a
                # non-zero range.
                if vmin == vmax:
                    vmax = vmin + 1.0
                self.clim = (vmin, vmax)

        # Create actors. The two topologies route through different
        # add_polygons kwargs; the backend protocol guarantees that
        # passing both at once is a ValueError, so each entry is
        # one-or-the-other.
        backend = scene.backend
        handle = scene.handle
        for entry in self._classes:
            scalars = initial_values[entry["label"]]
            if self.topology == "cell":
                actor = backend.add_polygons(
                    handle,
                    entry["polygons"],
                    values=scalars,
                    cmap=self.cmap,
                    edge_color=self.edge_color,
                )
            else:  # "nodal"
                actor = backend.add_polygons(
                    handle,
                    entry["polygons"],
                    point_values=scalars,
                    cmap=self.cmap,
                    edge_color=self.edge_color,
                )
            entry["actor"] = actor
            self._apply_post_hoc(actor)

        if not self.visible:
            for entry in self._classes:
                backend.set_visible(entry["actor"], False)

        if skipped:
            warnings.warn(
                "[contour] Skipped element classes with unsupported "
                f"topology: {skipped}",
                RuntimeWarning,
            )

        # Track step only for the callable-binding mode — static layers
        # have a single fixed value per element and ``current_step`` is
        # meaningless for them. ``update_to_step`` is a no-op in static
        # mode either way; gating on None here keeps the public state
        # honest.
        if self.is_time_varying:
            self._current_step = self._initial_step

    def update_to_step(self, step: int) -> None:
        """Re-fetch scalars at ``step`` (no-op for the static-dict mode)."""
        if not self.is_attached:
            return
        if not self.is_time_varying:
            return
        if step == self._current_step:
            return
        values_per_class = self._values_at_step(step)
        backend = self._scene.backend  # type: ignore[union-attr]
        for entry in self._classes:
            actor = entry["actor"]
            if actor is None:
                continue
            backend.update_scalars(actor, values_per_class[entry["label"]])
        self._current_step = step

    def detach(self) -> None:
        scene = self._scene
        if scene is not None:
            for entry in self._classes:
                actor = entry["actor"]
                if actor is not None:
                    scene.backend.remove(scene.handle, actor)
        self._classes = []
        self._skipped_classes = []
        self._current_step = None
        self._scene = None
        self._source = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _scalars_dict_at(self, step: int) -> Mapping[int, float]:
        if callable(self._scalars_input):
            result = self._scalars_input(int(step))
        else:
            result = self._scalars_input
        if not isinstance(result, Mapping):
            raise TypeError(
                f"ContourLayer.scalars must be a Mapping or callable returning "
                f"a Mapping; got {type(result).__name__}."
            )
        return result

    def _values_at_step(self, step: int) -> dict[str, np.ndarray]:
        """Resolve the per-class scalar arrays for ``step``.

        Cell topology: ``{class_label: (n_faces,) float64 array}`` — one
        value per face, looked up by ``source_element_ids``. Bricks have
        six faces but a single per-element value broadcast across all six.

        Nodal topology: ``{class_label: (sum_face_M,) float64 array}`` —
        the flattened per-vertex stream in the same order as the
        concatenated polygon vertex stream that ``backend.add_polygons``
        consumes. Shared nodes resolve to identical values, which is
        what makes the rendering visually smooth.
        """
        scalars = self._scalars_dict_at(step)
        out: dict[str, np.ndarray] = {}
        for entry in self._classes:
            if self.topology == "cell":
                src_ids = entry["source_element_ids"]
                try:
                    arr = np.asarray(
                        [scalars[int(eid)] for eid in src_ids],
                        dtype=np.float64,
                    )
                except KeyError as exc:
                    missing = int(exc.args[0]) if exc.args else None
                    raise KeyError(
                        f"ContourLayer.scalars is missing element id "
                        f"{missing} (class {entry['label']!r}, step {step})."
                    ) from exc
                out[entry["label"]] = arr
            else:  # "nodal"
                # Flatten the per-face node_id stream and look up each
                # vertex's scalar by node id.
                vertices: list[float] = []
                for face_nids in entry["face_node_ids"]:
                    for nid in face_nids:
                        try:
                            vertices.append(float(scalars[int(nid)]))
                        except KeyError as exc:
                            raise KeyError(
                                f"ContourLayer.scalars is missing node id "
                                f"{int(nid)} (class {entry['label']!r}, "
                                f"step {step})."
                            ) from exc
                out[entry["label"]] = np.asarray(vertices, dtype=np.float64)
        return out

    def _apply_post_hoc(self, actor: Any) -> None:
        """Apply renderer-specific styling that doesn't fit the protocol."""
        if hasattr(actor, "set_zorder"):
            actor.set_zorder(self.mpl_zorder)
        if self.clim is not None and hasattr(actor, "set_clim"):
            actor.set_clim(*self.clim)


__all__ = ["ContourLayer"]
