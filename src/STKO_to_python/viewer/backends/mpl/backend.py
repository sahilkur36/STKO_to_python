"""Matplotlib :class:`Backend` implementation.

Maps every primitive on the
:class:`STKO_to_python.viewer.core.Backend` protocol onto matplotlib's
2-D + 3-D drawing surface. Both projections share the same primitive
signatures — the backend selects the appropriate matplotlib artist
class internally based on the scene's ``is_3d`` flag.

The :class:`MplSceneHandle` wraps the ``Figure`` + ``Axes`` pair the
backend allocates per scene; layers receive matplotlib ``Artist``
instances as ``ActorRef`` cookies. Subsequent ``update_*`` /
``set_visible`` / ``remove`` calls treat those artists as opaque
handles.

This module imports matplotlib at top level — matplotlib has always
been a base STKO_to_python dependency, so no optional-extra gating is
required.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.quiver import Quiver
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
    Poly3DCollection,
)

from ...core.errors import BackendCapabilityError
from ...core.style import SceneStyle
from ...core.types import ActorRef, BBox, CameraSpec


@dataclass
class MplSceneHandle:
    """Wraps a single matplotlib ``Figure`` + ``Axes`` pair.

    Returned by :meth:`MplBackend.make_scene` and threaded through every
    subsequent backend call. Carries the ``is_3d`` flag so per-primitive
    branching (``LineCollection`` vs ``Line3DCollection``) lives in one
    place.

    Attributes:
        fig: The matplotlib ``Figure``.
        ax: The matplotlib ``Axes`` (2-D) or ``Axes3D`` (3-D).
        is_3d: Whether the axes use the ``"3d"`` projection.
        off_screen: When ``True``, ``show()`` is a no-op so headless
            runs do not pop a window.
    """

    fig: Figure
    ax: Axes
    is_3d: bool
    off_screen: bool


class MplBackend:
    """Backend protocol satisfied via matplotlib's 2-D + 3-D primitives.

    Construction is free — the backend instance holds no scene state,
    only configuration. Scene-specific state lives in
    :class:`MplSceneHandle`, which the backend creates on demand from
    :meth:`make_scene`.
    """

    name: str = "mpl"
    is_3d_capable: bool = True
    is_interactive: bool = True

    # ----- Scene lifecycle ------------------------------------------- #

    def make_scene(
        self,
        *,
        is_3d: bool = False,
        off_screen: bool = False,
        ax: Axes | None = None,
    ) -> MplSceneHandle:
        """Allocate (or borrow) a ``Figure`` + ``Axes`` for the scene.

        The optional ``ax`` argument is a backend-specific extension
        beyond the :class:`Backend` protocol — when supplied, the
        backend wraps the caller's axes instead of creating a new
        figure. This is the path the legacy ``Plot.*`` rewires take
        to honour their ``ax=`` kwarg: callers that pass their own
        axes get a scene that draws into those axes, without an extra
        figure showing up.

        The caller is responsible for ensuring ``ax``'s projection
        matches ``is_3d`` — the backend trusts what it is given.
        """
        if ax is not None:
            return MplSceneHandle(
                fig=ax.figure, ax=ax, is_3d=is_3d, off_screen=off_screen,
            )
        if is_3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()
        return MplSceneHandle(fig=fig, ax=ax, is_3d=is_3d, off_screen=off_screen)

    def set_bounds(self, scene: MplSceneHandle, bbox: BBox) -> None:
        """Apply ``bbox`` to the axes' x/y/z limits."""
        scene.ax.set_xlim(bbox.x_min, bbox.x_max)
        scene.ax.set_ylim(bbox.y_min, bbox.y_max)
        if scene.is_3d:
            scene.ax.set_zlim(bbox.z_min, bbox.z_max)

    def set_camera(self, scene: MplSceneHandle, cam: CameraSpec) -> None:
        """Apply a saved camera state (3-D only — 2-D ignores ``cam``)."""
        if not scene.is_3d:
            return
        # Matplotlib 3D camera is parameterised by elev / azim / roll +
        # the focal point. CameraSpec is renderer-neutral, so we
        # translate position + focal_point into elev/azim. Roll is left
        # at the matplotlib default.
        pos = np.asarray(cam.position, dtype=np.float64)
        focal = np.asarray(cam.focal_point, dtype=np.float64)
        v = pos - focal
        r = float(np.linalg.norm(v))
        if r == 0.0:
            return
        elev = float(np.degrees(np.arcsin(v[2] / r)))
        azim = float(np.degrees(np.arctan2(v[1], v[0])))
        scene.ax.view_init(elev=elev, azim=azim)

    def set_style(self, scene: MplSceneHandle, style: SceneStyle) -> None:
        """Apply scene-wide styling (background colour, grid, font size)."""
        scene.fig.patch.set_facecolor(style.background)
        scene.ax.set_facecolor(style.background)
        scene.ax.grid(style.grid)
        # Font size affects all subsequent text creation; matplotlib does
        # not retroactively resize existing text, so layers that add
        # labels after set_style will pick up the new size automatically.
        plt.rcParams["font.size"] = float(style.font_size)

    # ----- Primitives ------------------------------------------------ #

    def add_segments(
        self,
        scene: MplSceneHandle,
        segments: np.ndarray,
        *,
        color: Any = None,
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
    ) -> ActorRef:
        """Add a ``LineCollection`` (2-D) or ``Line3DCollection`` (3-D)."""
        segs = np.asarray(segments, dtype=np.float64)
        if segs.size == 0:
            segs = segs.reshape(0, 2, 3 if scene.is_3d else 2)
        kwargs: dict[str, Any] = {}
        if color is not None:
            kwargs["colors"] = color
        if width is not None:
            kwargs["linewidths"] = width
        if alpha is not None:
            kwargs["alpha"] = alpha
        if label is not None:
            kwargs["label"] = label
        if scene.is_3d:
            collection = Line3DCollection(segs, **kwargs)
        else:
            # 2-D LineCollection wants (N, 2, 2). Drop the z column if
            # the caller passed 3-D segments — common case when the same
            # layer code is used against a 2-D scene.
            if segs.ndim == 3 and segs.shape[-1] == 3:
                segs = segs[:, :, :2]
            collection = LineCollection(segs, **kwargs)
        scene.ax.add_collection(collection)
        return collection

    def add_points(
        self,
        scene: MplSceneHandle,
        points: np.ndarray,
        *,
        color: Any = None,
        size: float | None = None,
        scalars: np.ndarray | None = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add a scatter ``PathCollection`` via ``ax.scatter``."""
        pts = np.asarray(points, dtype=np.float64)
        kwargs: dict[str, Any] = {}
        if size is not None:
            kwargs["s"] = size
        if cmap is not None:
            kwargs["cmap"] = cmap
        # If scalars are provided, scatter colours by scalar; otherwise
        # use the explicit colour (or matplotlib default).
        c: Any = scalars if scalars is not None else color
        if pts.size == 0:
            # Empty scatter still needs valid coords array shape.
            pts = pts.reshape(0, 3 if scene.is_3d else 2)
        if scene.is_3d:
            return scene.ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], c=c, **kwargs,
            )
        return scene.ax.scatter(pts[:, 0], pts[:, 1], c=c, **kwargs)

    def add_polygons(
        self,
        scene: MplSceneHandle,
        polygons: Any,
        *,
        values: np.ndarray | None = None,
        cmap: str | None = None,
        edge_color: Any = None,
    ) -> ActorRef:
        """Add a ``PolyCollection`` (2-D) or ``Poly3DCollection`` (3-D).

        ``polygons`` is a sequence of ``(M_i, 3)`` (or ``(M_i, 2)``)
        vertex arrays — one ``M_i``-sided polygon per entry. Matplotlib
        accepts heterogeneous M, so we do not reshape into a tensor.
        """
        kwargs: dict[str, Any] = {}
        if edge_color is not None:
            kwargs["edgecolors"] = edge_color
        if scene.is_3d:
            collection = Poly3DCollection(polygons, **kwargs)
        else:
            # 2-D collection wants (M, 2) vertex arrays; if 3-D input is
            # given, drop the z column.
            polys_2d = [np.asarray(p)[:, :2] for p in polygons]
            collection = PolyCollection(polys_2d, **kwargs)
        if values is not None:
            collection.set_array(np.asarray(values, dtype=np.float64))
            if cmap is not None:
                collection.set_cmap(cmap)
        scene.ax.add_collection(collection)
        return collection

    def add_arrows(
        self,
        scene: MplSceneHandle,
        origins: np.ndarray,
        vectors: np.ndarray,
        *,
        scale: float = 1.0,
        color: Any = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add arrow glyphs via ``ax.quiver``."""
        o = np.asarray(origins, dtype=np.float64)
        v = np.asarray(vectors, dtype=np.float64)
        kwargs: dict[str, Any] = {}
        if color is not None:
            kwargs["color"] = color
        if cmap is not None:
            kwargs["cmap"] = cmap
        if scene.is_3d:
            return scene.ax.quiver(
                o[:, 0], o[:, 1], o[:, 2],
                scale * v[:, 0], scale * v[:, 1], scale * v[:, 2],
                **kwargs,
            )
        # 2-D quiver has a different `scale` semantics; we pass arrow
        # components scaled by ``scale`` and use scale_units="xy",
        # angles="xy" so the arrows match data coordinates.
        return scene.ax.quiver(
            o[:, 0], o[:, 1],
            scale * v[:, 0], scale * v[:, 1],
            scale_units="xy", angles="xy", scale=1.0, **kwargs,
        )

    # ----- In-place actor updates ------------------------------------ #

    def update_scalars(self, actor: ActorRef, scalars: np.ndarray) -> None:
        """Mutate an actor's per-vertex scalars (drives colour mapping)."""
        if hasattr(actor, "set_array"):
            actor.set_array(np.asarray(scalars, dtype=np.float64))
            return
        raise BackendCapabilityError(
            f"Actor {type(actor).__name__} does not support scalar updates "
            f"(no set_array method)."
        )

    def update_points(self, actor: ActorRef, points: np.ndarray) -> None:
        """Mutate an actor's vertex positions.

        Handles three common cases:

        * ``LineCollection`` / ``Line3DCollection`` — replaces the
          segment array via ``set_segments``. 3-D segments (shape
          ``(N, 2, 3)``) passed to a 2-D ``LineCollection`` are
          z-dropped to ``(N, 2, 2)`` — same convention as
          :meth:`add_segments` so layers can stay shape-agnostic.
        * ``PathCollection`` (scatter) — replaces the offsets via
          ``set_offsets`` (2-D) or ``set_offsets`` + ``set_3d_properties``
          (3-D).
        * Everything else — raises ``BackendCapabilityError`` so the
          caller learns rather than silently dropping the update.
        """
        pts = np.asarray(points, dtype=np.float64)
        if isinstance(actor, Line3DCollection):
            actor.set_segments(pts)
            return
        if isinstance(actor, LineCollection):
            if pts.ndim == 3 and pts.shape[-1] == 3:
                pts = pts[:, :, :2]
            actor.set_segments(pts)
            return
        if isinstance(actor, PathCollection):
            if pts.ndim == 2 and pts.shape[1] == 2:
                actor.set_offsets(pts)
                return
            if pts.ndim == 2 and pts.shape[1] == 3:
                # 3-D PathCollection: offsets carry (x, y); z lives in
                # the depth array.
                actor.set_offsets(pts[:, :2])
                if hasattr(actor, "set_3d_properties"):
                    actor.set_3d_properties(pts[:, 2], zdir="z")
                return
        raise BackendCapabilityError(
            f"Actor {type(actor).__name__} does not support point updates."
        )

    def set_visible(self, actor: ActorRef, visible: bool) -> None:
        actor.set_visible(visible)

    def remove(self, scene: MplSceneHandle, actor: ActorRef) -> None:
        """Detach ``actor`` from the scene; idempotent if already removed."""
        try:
            actor.remove()
        except (NotImplementedError, ValueError):
            # NotImplementedError: matplotlib 3-D collections sometimes
            # cannot .remove() themselves directly; fall back to manual
            # collection list manipulation.
            try:
                scene.ax.collections.remove(actor)  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                pass

    # ----- Output ---------------------------------------------------- #

    def show(self, scene: MplSceneHandle) -> None:
        """Hand off to ``plt.show`` unless the scene is off-screen."""
        if scene.off_screen:
            return
        plt.show()

    def save(self, scene: MplSceneHandle, path: Path, *, dpi: int = 300) -> None:
        scene.fig.savefig(path, dpi=dpi)

    def snapshot(self, scene: MplSceneHandle) -> np.ndarray:
        """Return the current frame as an ``(H, W, 3)`` ``uint8`` RGB array."""
        scene.fig.canvas.draw()
        # Use the buffer_rgba route — broadly supported across mpl
        # canvases (Agg, Qt5Agg, etc.) including the headless CI backend.
        buf = np.asarray(scene.fig.canvas.buffer_rgba())
        # buffer_rgba returns (H, W, 4) uint8; drop the alpha channel.
        return buf[..., :3].copy()
