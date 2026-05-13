"""``MeshLayer`` — element-edge wireframe over a :class:`Scene`.

This is the first concrete :class:`~STKO_to_python.viewer.core.Layer`.
It wraps the v1.x ``ds.plot.mesh`` rendering so the same edge-drawing
pipeline now flows through Scene + DataSource + Backend.

Edge topology, the segment builder, and the 2-D/3-D decision are
borrowed from the existing
:mod:`STKO_to_python.plotting.deformed_shape` helpers — that module
has been the home of these utilities since v1.4; lifting them into
:mod:`viewer.math` will land alongside the deformed-mesh layer in a
later phase. Until then, the temporary import direction
``viewer.layers → plotting`` is intentional and reviewer-acknowledged.

The layer is **static** — it has no per-step data, so
:meth:`update_to_step` is a no-op. Time-varying counterparts
(``DeformedMeshLayer``) land in Phase 2.5.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

# Temporary import from plotting/ — see module docstring.
from ...plotting.deformed_shape import _build_segments
from ..core.errors import LayerAttachError
from ..core.layer import Layer

if TYPE_CHECKING:
    from ..core.datasource import DataSource
    from ..core.scene import Scene
    from ..core.selection import SelectionSpec


class MeshLayer(Layer):
    """Element-edge wireframe layer.

    Draws the model's element edges class-by-class through the scene's
    backend, matching the v1.x ``ds.plot.mesh`` output:

    * one ``LineCollection`` / ``Line3DCollection`` per element class;
    * caller-controlled colour, line width, and alpha;
    * matplotlib ``zorder`` defaulting to 1.0 so contour and scatter
      overlays (matplotlib ``zorder`` 2.0) draw on top.

    The matplotlib-specific ``mpl_zorder`` knob is intentionally
    backend-specific: PyVista and Trame use depth ordering rather
    than a scalar ``zorder``, so the parameter only takes effect on
    actors that expose ``set_zorder``.

    Parameters
    ----------
    name, selection, visible, z_order:
        Forwarded to :class:`Layer`.
    edge_color, linewidth, alpha:
        Edge styling. Defaults match the v1.x ``ds.plot.mesh`` API
        (``"lightgray"``, ``0.5``, ``1.0``).
    mpl_zorder:
        Matplotlib z-order for the segment collections. Applied
        post-hoc via ``actor.set_zorder`` so non-matplotlib backends
        are unaffected. Default ``1.0`` matches the legacy mesh
        renderer, which sits below scatter / contour overlays.
    """

    kind: str = "mesh"

    def __init__(
        self,
        *,
        name: str | None = None,
        selection: "SelectionSpec | None" = None,
        visible: bool = True,
        z_order: int = 0,
        edge_color: Any = "lightgray",
        linewidth: float = 0.5,
        alpha: float = 1.0,
        mpl_zorder: float = 1.0,
    ) -> None:
        super().__init__(
            name=name, selection=selection, visible=visible, z_order=z_order,
        )
        self.edge_color = edge_color
        self.linewidth = linewidth
        self.alpha = alpha
        self.mpl_zorder = mpl_zorder

        self._actors_per_class: dict[str, Any] = {}
        self._edges_per_class: dict[str, int] = {}
        self._skipped_classes: list[str] = []
        self._n_edges: int = 0
        self._n_elements_drawn: int = 0

    # ------------------------------------------------------------------ #
    # Read-only summary (drives plot_mesh's meta dict)
    # ------------------------------------------------------------------ #
    @property
    def n_edges(self) -> int:
        """Total number of segments drawn across every element class."""
        return self._n_edges

    @property
    def n_elements_drawn(self) -> int:
        """Element rows that contributed at least one segment."""
        return self._n_elements_drawn

    @property
    def edges_per_class(self) -> dict[str, int]:
        """``{class_label: edges_per_element}`` for every drawn class."""
        return dict(self._edges_per_class)

    @property
    def skipped_classes(self) -> list[str]:
        """Class labels skipped because their topology isn't supported."""
        return list(self._skipped_classes)

    @property
    def actors(self) -> dict[str, Any]:
        """``{class_label: backend actor}`` mapping. For inspection only."""
        return dict(self._actors_per_class)

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

        # Resolve selection → element subset.
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

        segs_per_class, edges_per_class, skipped = _build_segments(
            df_elements=df_filtered, coord_lookup=coord_lookup,
        )
        self._edges_per_class = dict(edges_per_class)
        self._skipped_classes = list(skipped)

        backend = scene.backend
        handle = scene.handle
        n_edges = 0
        n_elems_drawn = 0
        for label, segs in segs_per_class.items():
            actor = backend.add_segments(
                handle, segs,
                color=self.edge_color,
                width=self.linewidth,
                alpha=self.alpha,
            )
            # zorder is matplotlib-specific; PyVista uses depth ordering.
            # Apply post-hoc so non-matplotlib backends are unaffected.
            if hasattr(actor, "set_zorder"):
                actor.set_zorder(self.mpl_zorder)
            self._actors_per_class[label] = actor
            n_edges += int(segs.shape[0])
            epe = edges_per_class.get(label, 0)
            if epe:
                n_elems_drawn += int(segs.shape[0] // epe)
        self._n_edges = n_edges
        self._n_elements_drawn = n_elems_drawn

        if skipped:
            warnings.warn(
                "[mesh] Skipped element classes with unsupported topology: "
                f"{skipped}",
                RuntimeWarning,
            )

        # Initial visibility — mirror the public attribute.
        if not self.visible:
            for actor in self._actors_per_class.values():
                backend.set_visible(actor, False)

    def update_to_step(self, step: int) -> None:
        """No-op — the mesh has no per-step data."""
        return None

    def detach(self) -> None:
        scene = self._scene
        if scene is not None:
            for actor in self._actors_per_class.values():
                scene.backend.remove(scene.handle, actor)
        self._actors_per_class.clear()
        self._edges_per_class.clear()
        self._skipped_classes.clear()
        self._n_edges = 0
        self._n_elements_drawn = 0
        self._scene = None
        self._source = None


__all__ = ["MeshLayer"]
