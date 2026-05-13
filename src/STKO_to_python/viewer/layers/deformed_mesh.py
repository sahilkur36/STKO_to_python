"""``DeformedMeshLayer`` — element edges at a deformed step.

Time-varying counterpart to :class:`MeshLayer`. Fetches the nodal
displacement field at a given ``(model_stage, step)``, builds the
deformed coordinates as ``original + scale * displacement``, then
draws element edges through the scene's backend.

Per the perf contract in ``docs/viewer/01-architecture.md`` §4 and
``docs/viewer/02-porting-from-apegmsh.md`` §6, :meth:`update_to_step`
mutates pre-allocated actor data via ``backend.update_points`` rather
than recreating actors. That keeps the per-step cost dominated by the
displacement fetch (which the query engine caches) plus an
O(n_segments) coord scatter — not by VTK/matplotlib actor allocation.

Like :class:`MeshLayer`, this layer borrows
:func:`_build_segments` from
:mod:`STKO_to_python.plotting.deformed_shape` and the displacement
fetch from the same module. Those helpers move into
:mod:`viewer.math` in a later phase.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

# Temporary imports from plotting/ — see module docstring.
from ...plotting.deformed_shape import _build_segments, _displacement_at_step
from ..core.errors import LayerAttachError
from ..core.layer import Layer

if TYPE_CHECKING:
    import pandas as pd

    from ..core.datasource import DataSource
    from ..core.scene import Scene
    from ..core.selection import SelectionSpec


class DeformedMeshLayer(Layer):
    """Element-edge wireframe at ``original + scale * displacement(step)``.

    The layer fixes ``model_stage`` and ``scale`` at construction;
    ``step`` is the initial step and may be advanced via
    :meth:`update_to_step` (drives time scrubbers and animation
    export). Each step change re-fetches displacements through the
    dataset's cached :class:`~STKO_to_python.nodes.node_manager.NodeManager`
    and updates segment endpoints in place — no actor recreation.

    Parameters
    ----------
    model_stage:
        Stage name (e.g. ``"MODEL_STAGE[1]"``).
    step:
        Initial step inside the stage. May be advanced later via
        :meth:`update_to_step`.
    scale:
        Displacement amplification. ``1.0`` is true-to-life; ``0.0``
        collapses to the undeformed configuration (no DISPLACEMENT
        fetch happens in that case — same shortcut as the v1.x
        renderer).
    name, selection, visible, z_order:
        Forwarded to :class:`Layer`.
    edge_color, linewidth, alpha:
        Edge styling. Defaults match the v1.x
        ``ds.plot.deformed_shape`` API (``"C0"``, ``1.2``, ``1.0``).
    mpl_zorder:
        Matplotlib z-order for the segment collections. Default
        ``2.0`` sits **above** the undeformed overlay (which the
        v1.x renderer draws at ``zorder=1.0``).
    """

    kind: str = "deformed_mesh"

    def __init__(
        self,
        *,
        model_stage: str,
        step: int,
        scale: float = 1.0,
        name: str | None = None,
        selection: "SelectionSpec | None" = None,
        visible: bool = True,
        z_order: int = 0,
        edge_color: Any = "C0",
        linewidth: float = 1.2,
        alpha: float = 1.0,
        mpl_zorder: float = 2.0,
    ) -> None:
        super().__init__(
            name=name, selection=selection, visible=visible, z_order=z_order,
        )
        self.model_stage = str(model_stage)
        self.scale = float(scale)
        self.edge_color = edge_color
        self.linewidth = linewidth
        self.alpha = alpha
        self.mpl_zorder = mpl_zorder

        self._initial_step: int | None = (
            int(step) if step is not None else None
        )
        self._current_step: int | None = None

        # Populated at attach.
        self._df_filtered: "pd.DataFrame | None" = None
        self._original_coords: dict[int, np.ndarray] = {}
        self._deformed_coords: dict[int, np.ndarray] = {}
        self._actors_per_class: dict[str, Any] = {}
        self._edges_per_class: dict[str, int] = {}
        self._skipped_classes: list[str] = []
        self._segment_count: int = 0

    # ------------------------------------------------------------------ #
    # Read-only summary surface
    # ------------------------------------------------------------------ #
    @property
    def current_step(self) -> int | None:
        """The step the layer is currently rendering, or ``None`` before attach."""
        return self._current_step

    @property
    def segment_count(self) -> int:
        """Total number of deformed segments across all element classes."""
        return self._segment_count

    @property
    def edges_per_class(self) -> dict[str, int]:
        """``{class_label: edges_per_element}`` for every drawn class."""
        return dict(self._edges_per_class)

    @property
    def skipped_classes(self) -> list[str]:
        """Class labels skipped because their topology isn't supported."""
        return list(self._skipped_classes)

    @property
    def original_coords(self) -> dict[int, np.ndarray]:
        """Original node coordinates, ``{node_id: ndarray(3)}``."""
        return {nid: xyz.copy() for nid, xyz in self._original_coords.items()}

    @property
    def deformed_coords(self) -> dict[int, np.ndarray]:
        """Current deformed coordinates, ``{node_id: ndarray(3)}``."""
        return {nid: xyz.copy() for nid, xyz in self._deformed_coords.items()}

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

        self._df_filtered = df_filtered
        self._original_coords = {
            int(r.node_id): np.array([r.x, r.y, r.z], dtype=np.float64)
            for r in df_nodes.itertuples(index=False)
        }

        # Compute the initial step's deformed coordinates and build
        # segments + actors. Subsequent set_step calls only call
        # update_points on these actors — no actor recreation.
        self._compute_deformed_for_step(self._initial_step)
        segs_per_class, edges_per_class, skipped = _build_segments(
            df_elements=self._df_filtered, coord_lookup=self._deformed_coords,
        )
        self._edges_per_class = dict(edges_per_class)
        self._skipped_classes = list(skipped)

        backend = scene.backend
        handle = scene.handle
        seg_count = 0
        for label, segs in segs_per_class.items():
            actor = backend.add_segments(
                handle, segs,
                color=self.edge_color,
                width=self.linewidth,
                alpha=self.alpha,
            )
            if hasattr(actor, "set_zorder"):
                actor.set_zorder(self.mpl_zorder)
            self._actors_per_class[label] = actor
            seg_count += int(segs.shape[0])
        self._segment_count = seg_count

        if skipped:
            warnings.warn(
                "[deformed_mesh] Skipped element classes with unsupported "
                f"topology: {skipped}",
                RuntimeWarning,
            )

        if not self.visible:
            for actor in self._actors_per_class.values():
                backend.set_visible(actor, False)

        self._current_step = self._initial_step

    def update_to_step(self, step: int) -> None:
        """Advance to ``step`` via in-place segment mutation.

        No-op when the layer is unattached or already at ``step``.
        """
        if not self.is_attached:
            return
        if step == self._current_step:
            return

        self._compute_deformed_for_step(step)
        segs_per_class, _, _ = _build_segments(
            df_elements=self._df_filtered, coord_lookup=self._deformed_coords,
        )
        backend = self._scene.backend  # type: ignore[union-attr]
        seg_count = 0
        for label, segs in segs_per_class.items():
            actor = self._actors_per_class.get(label)
            if actor is None:
                # New class showed up at this step (e.g. element-by-element
                # activation). Allocating a new actor would violate the
                # no-recreation contract, so we skip — animations should
                # be defined on a fixed topology.
                continue
            backend.update_points(actor, segs)
            seg_count += int(segs.shape[0])
        self._segment_count = seg_count
        self._current_step = step

    def detach(self) -> None:
        scene = self._scene
        if scene is not None:
            for actor in self._actors_per_class.values():
                scene.backend.remove(scene.handle, actor)
        self._actors_per_class.clear()
        self._original_coords = {}
        self._deformed_coords = {}
        self._edges_per_class = {}
        self._skipped_classes = []
        self._segment_count = 0
        self._df_filtered = None
        self._current_step = None
        self._scene = None
        self._source = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _compute_deformed_for_step(self, step: int | None) -> None:
        if self.scale == 0.0 or step is None:
            self._deformed_coords = {
                nid: xyz.copy() for nid, xyz in self._original_coords.items()
            }
            return
        if self._source is None:
            raise LayerAttachError(
                "Cannot compute deformed coordinates: layer is not attached."
            )
        disp = _displacement_at_step(
            self._source.dataset, model_stage=self.model_stage, step=int(step),
        )
        deformed: dict[int, np.ndarray] = {}
        scale = float(self.scale)
        for nid, xyz in self._original_coords.items():
            d = disp.get(nid)
            if d is None:
                deformed[nid] = xyz.copy()
            else:
                deformed[nid] = xyz + scale * d
        self._deformed_coords = deformed


__all__ = ["DeformedMeshLayer"]
