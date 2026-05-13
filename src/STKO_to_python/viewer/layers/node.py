"""``NodeLayer`` + ``VectorLayer`` — point cloud and arrow glyphs at node positions.

These are the **first greenfield layers** — neither has a v1.x
``ds.plot.*`` counterpart to rewire, so they introduce new capability
rather than re-routing existing code. The first user-facing entry
points are expected to land in Phase 3 (PyVista backend) alongside
the 3-D scene catalog; Phase 2.6 ships the layers themselves with
matplotlib coverage so the contract can be exercised in unit tests
ahead of the 3-D work.

Both layers fetch nodal results through ``source.dataset.nodes``
(the same path :func:`~STKO_to_python.plotting.deformed_shape._displacement_at_step`
uses). That coupling is the same temporary
``viewer → plotting``-style coupling
:class:`~STKO_to_python.viewer.layers.DeformedMeshLayer` carries —
it relocates to the DataSource protocol (``nodal_values`` /
``nodal_vectors``) in a later phase when more layers need it.

Class split (per
``docs/viewer/01-architecture.md`` §8 and the directive's §5.4):

* :class:`NodeLayer` — point cloud at node positions. Optional scalar
  coloring (``result_name`` + ``component``). Time-varying only when
  a scalar is bound; otherwise :meth:`Layer.update_to_step` is a
  no-op.
* :class:`VectorLayer` — arrow glyphs at node positions. The vector
  field is REQUIRED — the layer carries a ``(result_name,
  model_stage, step)`` binding and a scale multiplier.

Phase 2.6 perf note: :class:`VectorLayer`'s ``update_to_step``
removes and re-adds the arrow actor — matplotlib's ``Quiver`` has no
general in-place vector setter (especially in 3-D), and the
``Backend.update_arrows`` primitive has not landed yet. The animation
cost is O(N_arrows) per step until Phase 3 lands the primitive. For
the 2-D notebook case this is acceptable; the perf contract matters
for the Phase 4 GUI scrubber where every animation frame is on a
budget.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..core.errors import LayerAttachError
from ..core.layer import Layer
from ..core.selection import SelectionSpec

if TYPE_CHECKING:
    from ..core.datasource import DataSource
    from ..core.scene import Scene


# ---------------------------------------------------------------------- #
# Shared internals
# ---------------------------------------------------------------------- #
def _fetch_nodal_snapshot(
    dataset: Any,
    *,
    result_name: str,
    model_stage: str,
    step: int,
    node_ids: np.ndarray,
) -> pd.DataFrame:
    """Pull a single ``(stage, step)`` snapshot of a nodal result.

    Returns a DataFrame whose index is ``node_id`` (matching the
    caller's order) and whose columns are component keys (typically
    ``"1"``, ``"2"``, ``"3"`` for a vector result). Wraps the same
    path :func:`STKO_to_python.plotting.deformed_shape._displacement_at_step`
    uses so the query-engine cache is shared.
    """
    nr = dataset.nodes.get_nodal_results(
        results_name=result_name,
        model_stage=model_stage,
        node_ids=[int(nid) for nid in node_ids],
    )
    df = nr.df
    try:
        snap = df.xs(int(step), level="step")
    except KeyError as exc:
        raise ValueError(
            f"step={step} not present in {result_name} for stage "
            f"{model_stage!r}."
        ) from exc

    cols = snap.columns
    if isinstance(cols, pd.MultiIndex):
        comp_cols = [c for c in cols if str(c[0]) == result_name]
        if not comp_cols:
            raise ValueError(
                f"Result {result_name!r} not present in fetched results."
            )
        snap = snap.loc[:, comp_cols]
        snap.columns = [c[1] for c in snap.columns]

    # Reindex to caller's node order so the returned array aligns with
    # the layer's pre-allocated geometry.
    snap = snap.reindex([int(nid) for nid in node_ids])
    return snap


def _component_to_scalar(snap: pd.DataFrame, component: Any) -> np.ndarray:
    """Reduce a nodal-result snapshot to a ``(N,)`` scalar array.

    ``component`` may be:

    * The literal string ``"magnitude"`` — compute
      ``sqrt(sum_i c_i^2)`` across every available component.
    * An integer or numeric string (``1``, ``"1"``, ``2``, …) —
      select the matching column.
    """
    if component == "magnitude":
        arr = snap.to_numpy(dtype=np.float64)
        return np.linalg.norm(arr, axis=1)
    col_key = str(component)
    for c in snap.columns:
        if str(c) == col_key:
            return snap[c].to_numpy(dtype=np.float64)
    raise ValueError(
        f"Component {component!r} not in available columns "
        f"{[str(c) for c in snap.columns]}."
    )


def _snap_to_vector(snap: pd.DataFrame, n_components: int) -> np.ndarray:
    """Reduce a nodal-result snapshot to a ``(N, n_components)`` vector array.

    Components are sorted numerically when possible (``1``, ``2``,
    ``3``) so ``v[:, 0]`` is always the first component. Missing
    components are zero-padded; extra components are truncated.
    """
    sorted_cols = sorted(
        snap.columns,
        key=lambda c: (int(c) if str(c).isdigit() else 1_000_000, str(c)),
    )
    snap = snap.loc[:, sorted_cols]
    arr = snap.to_numpy(dtype=np.float64)
    n_have = arr.shape[1] if arr.ndim == 2 else 0
    if n_have < n_components:
        pad = np.zeros(
            (arr.shape[0], n_components - n_have), dtype=np.float64,
        )
        arr = np.hstack([arr, pad]) if n_have else pad
    elif n_have > n_components:
        arr = arr[:, :n_components]
    return arr


# ---------------------------------------------------------------------- #
# NodeLayer
# ---------------------------------------------------------------------- #
class NodeLayer(Layer):
    """Point cloud at node positions, optional scalar coloring.

    Modes (selected at construction):

    * **Position-only** — ``result_name=None``. Static point cloud at
      the original node coordinates with uniform ``color``.
      :meth:`update_to_step` is a no-op.
    * **Scalar-bound** — ``result_name`` and ``component`` provided.
      Points are coloured by a per-node scalar derived from a nodal
      result at ``(model_stage, step)``. :meth:`update_to_step`
      re-fetches scalars at the new step and calls
      ``backend.update_scalars`` in place.

    Parameters
    ----------
    result_name, component, model_stage, step:
        Scalar binding. All four are required together if any are
        provided. ``component`` may be an integer / numeric string
        (selecting one column from a vector result) or the literal
        ``"magnitude"`` (compute ``sqrt(sum c_i^2)`` across every
        available component).
    color:
        Uniform colour for the position-only mode. Ignored when
        ``scalars`` are bound.
    size:
        Marker size (matplotlib ``s=`` units).
    cmap, vmin, vmax:
        Colormap controls. Used only in scalar-bound mode.
    name, selection, visible, z_order:
        Forwarded to :class:`Layer`.
    mpl_zorder:
        Matplotlib z-order applied post-hoc via ``actor.set_zorder``.
        Default ``2.5`` sits above MeshLayer (``1.0``) and at the
        same level as a typical mesh overlay.
    """

    kind: str = "node"

    def __init__(
        self,
        *,
        result_name: str | None = None,
        component: Any = None,
        model_stage: str | None = None,
        step: int | None = None,
        color: Any = "C0",
        size: float = 20.0,
        cmap: str | None = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        name: str | None = None,
        selection: "SelectionSpec | None" = None,
        visible: bool = True,
        z_order: int = 0,
        mpl_zorder: float = 2.5,
    ) -> None:
        super().__init__(
            name=name, selection=selection, visible=visible, z_order=z_order,
        )
        # Scalar binding is all-or-nothing.
        bound_any = any(
            v is not None for v in (result_name, component, model_stage, step)
        )
        bound_all = all(
            v is not None for v in (result_name, component, model_stage)
        )
        if bound_any and not bound_all:
            raise ValueError(
                "Scalar-bound NodeLayer requires result_name, component, "
                "AND model_stage (and step at construction); got "
                f"result_name={result_name!r}, component={component!r}, "
                f"model_stage={model_stage!r}, step={step!r}."
            )

        self.result_name = result_name
        self.component = component
        self.model_stage = model_stage
        self._initial_step = step
        self._current_step: int | None = None

        self.color = color
        self.size = float(size)
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax
        self.mpl_zorder = mpl_zorder

        # Populated at attach.
        self._actor: Any = None
        self._node_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._coords: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._scalars: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Read-only summary surface
    # ------------------------------------------------------------------ #
    @property
    def is_scalar_bound(self) -> bool:
        return self.result_name is not None

    @property
    def current_step(self) -> int | None:
        return self._current_step

    @property
    def node_ids(self) -> np.ndarray:
        return self._node_ids.copy()

    @property
    def coords(self) -> np.ndarray:
        return self._coords.copy()

    @property
    def scalars(self) -> np.ndarray | None:
        return None if self._scalars is None else self._scalars.copy()

    @property
    def actor(self) -> Any:
        return self._actor

    # ------------------------------------------------------------------ #
    # Layer lifecycle
    # ------------------------------------------------------------------ #
    def attach(self, scene: "Scene", source: "DataSource") -> None:
        if self.is_attached:
            raise LayerAttachError(f"Layer {self.name!r} is already attached.")
        self._scene = scene
        self._source = source

        ids = source.resolve_node_ids(self.selection)
        if ids.size == 0:
            raise LayerAttachError(
                f"No nodes remain after filtering for {self.selection!r}."
            )
        self._node_ids = ids
        self._coords = source.node_coords(ids)

        scalars: np.ndarray | None = None
        if self.is_scalar_bound:
            scalars = self._fetch_scalars(self._initial_step)
            self._scalars = scalars

        actor = scene.backend.add_points(
            scene.handle,
            self._coords,
            color=self.color if scalars is None else None,
            size=self.size,
            scalars=scalars,
            cmap=self.cmap if scalars is not None else None,
        )
        self._actor = actor

        if hasattr(actor, "set_zorder"):
            actor.set_zorder(self.mpl_zorder)
        if scalars is not None and (
            self.vmin is not None or self.vmax is not None
        ):
            if hasattr(actor, "set_clim"):
                actor.set_clim(self.vmin, self.vmax)

        if not self.visible:
            scene.backend.set_visible(actor, False)

        self._current_step = self._initial_step

    def update_to_step(self, step: int) -> None:
        """Re-fetch scalars at ``step`` (no-op in position-only mode)."""
        if not self.is_attached:
            return
        if not self.is_scalar_bound:
            return
        if step == self._current_step:
            return
        scalars = self._fetch_scalars(step)
        self._scalars = scalars
        self._scene.backend.update_scalars(self._actor, scalars)  # type: ignore[union-attr]
        self._current_step = step

    def detach(self) -> None:
        scene = self._scene
        if scene is not None and self._actor is not None:
            scene.backend.remove(scene.handle, self._actor)
        self._actor = None
        self._node_ids = np.zeros(0, dtype=np.int64)
        self._coords = np.zeros((0, 3), dtype=np.float64)
        self._scalars = None
        self._current_step = None
        self._scene = None
        self._source = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fetch_scalars(self, step: int | None) -> np.ndarray:
        if step is None:
            raise LayerAttachError(
                "Scalar-bound NodeLayer requires step= at construction "
                "(or via update_to_step)."
            )
        snap = _fetch_nodal_snapshot(
            self._source.dataset,  # type: ignore[union-attr]
            result_name=self.result_name,  # type: ignore[arg-type]
            model_stage=self.model_stage,  # type: ignore[arg-type]
            step=int(step),
            node_ids=self._node_ids,
        )
        return _component_to_scalar(snap, self.component)


# ---------------------------------------------------------------------- #
# VectorLayer
# ---------------------------------------------------------------------- #
class VectorLayer(Layer):
    """Arrow glyphs at node positions, vector from a nodal result.

    Origins are the **original** node coordinates (Phase 2.6). A
    deformed-origin variant — origins follow a
    :class:`~STKO_to_python.viewer.layers.DeformedMeshLayer` — lands in
    Phase 3 alongside the PyVista backend.

    Vectors are pulled from ``(model_stage, step)`` of ``result_name``;
    missing components are zero-padded, extras truncated. Plot through
    ``backend.add_arrows`` with the configured ``scale``.

    Phase 2.6 perf gap: :meth:`update_to_step` **removes and re-adds**
    the arrow actor. matplotlib's ``Quiver`` lacks a generic in-place
    vector setter (especially in 3-D), and the
    ``Backend.update_arrows`` primitive has not landed yet. Per-step
    cost is O(N_arrows). The Phase 3 PyVista backend adds the
    primitive and this layer switches to in-place updates.

    Parameters
    ----------
    result_name, model_stage, step:
        Vector result binding. ``step`` is the initial step;
        :meth:`update_to_step` advances it.
    scale:
        Multiplier applied to the raw vector before plotting.
    n_components:
        How many components to extract per node. Default ``3``;
        missing components are zero-padded, extras truncated.
    color, cmap:
        Forwarded to ``backend.add_arrows``.
    name, selection, visible, z_order:
        Forwarded to :class:`Layer`.
    """

    kind: str = "vector"

    def __init__(
        self,
        *,
        result_name: str,
        model_stage: str,
        step: int,
        scale: float = 1.0,
        n_components: int = 3,
        color: Any = "C1",
        cmap: str | None = None,
        name: str | None = None,
        selection: "SelectionSpec | None" = None,
        visible: bool = True,
        z_order: int = 0,
    ) -> None:
        super().__init__(
            name=name, selection=selection, visible=visible, z_order=z_order,
        )
        if not result_name or not model_stage:
            raise ValueError(
                "VectorLayer requires result_name and model_stage; got "
                f"result_name={result_name!r}, model_stage={model_stage!r}."
            )
        if n_components < 1:
            raise ValueError(f"n_components must be ≥ 1; got {n_components}.")

        self.result_name = result_name
        self.model_stage = model_stage
        self.scale = float(scale)
        self.n_components = int(n_components)
        self.color = color
        self.cmap = cmap

        self._initial_step = int(step) if step is not None else None
        self._current_step: int | None = None
        self._actor: Any = None
        self._node_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self._origins: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._vectors: np.ndarray = np.zeros((0, 3), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Read-only summary
    # ------------------------------------------------------------------ #
    @property
    def current_step(self) -> int | None:
        return self._current_step

    @property
    def node_ids(self) -> np.ndarray:
        return self._node_ids.copy()

    @property
    def origins(self) -> np.ndarray:
        return self._origins.copy()

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors.copy()

    @property
    def actor(self) -> Any:
        return self._actor

    # ------------------------------------------------------------------ #
    # Layer lifecycle
    # ------------------------------------------------------------------ #
    def attach(self, scene: "Scene", source: "DataSource") -> None:
        if self.is_attached:
            raise LayerAttachError(f"Layer {self.name!r} is already attached.")
        self._scene = scene
        self._source = source

        ids = source.resolve_node_ids(self.selection)
        if ids.size == 0:
            raise LayerAttachError(
                f"No nodes remain after filtering for {self.selection!r}."
            )
        self._node_ids = ids
        self._origins = source.node_coords(ids)
        self._vectors = self._fetch_vectors(self._initial_step)

        self._actor = scene.backend.add_arrows(
            scene.handle,
            self._origins,
            self._vectors,
            scale=self.scale,
            color=self.color,
            cmap=self.cmap,
        )
        if not self.visible:
            scene.backend.set_visible(self._actor, False)
        self._current_step = self._initial_step

    def update_to_step(self, step: int) -> None:
        """Advance to ``step`` by removing + re-adding the arrow actor.

        See class docstring for the Phase 2.6 perf gap rationale.
        """
        if not self.is_attached:
            return
        if step == self._current_step:
            return
        scene = self._scene  # type: ignore[assignment]
        # Remove the existing actor before re-adding.
        scene.backend.remove(scene.handle, self._actor)
        self._vectors = self._fetch_vectors(step)
        self._actor = scene.backend.add_arrows(
            scene.handle,
            self._origins,
            self._vectors,
            scale=self.scale,
            color=self.color,
            cmap=self.cmap,
        )
        if not self.visible:
            scene.backend.set_visible(self._actor, False)
        self._current_step = step

    def detach(self) -> None:
        scene = self._scene
        if scene is not None and self._actor is not None:
            scene.backend.remove(scene.handle, self._actor)
        self._actor = None
        self._node_ids = np.zeros(0, dtype=np.int64)
        self._origins = np.zeros((0, 3), dtype=np.float64)
        self._vectors = np.zeros((0, 3), dtype=np.float64)
        self._current_step = None
        self._scene = None
        self._source = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _fetch_vectors(self, step: int | None) -> np.ndarray:
        if step is None:
            raise LayerAttachError(
                "VectorLayer requires step= at construction "
                "(or via update_to_step)."
            )
        snap = _fetch_nodal_snapshot(
            self._source.dataset,  # type: ignore[union-attr]
            result_name=self.result_name,
            model_stage=self.model_stage,
            step=int(step),
            node_ids=self._node_ids,
        )
        return _snap_to_vector(snap, self.n_components)


__all__ = ["NodeLayer", "VectorLayer"]
