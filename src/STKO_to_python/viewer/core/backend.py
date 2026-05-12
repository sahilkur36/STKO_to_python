"""Renderer-agnostic ``Backend`` protocol.

See ``docs/viewer/01-architecture.md`` §3 for the full design
rationale. Three hard rules every concrete backend must honour:

1. Layers never import from a backend module. They reach the backend
   only through :attr:`Scene.backend`.
2. A backend that cannot do something raises
   :class:`STKO_to_python.viewer.core.errors.BackendCapabilityError`
   with a precise message. No silent fallback.
3. ``update_*`` methods mutate in place and re-render in
   O(modified data), not O(scene). This is the perf contract that
   keeps animation playback usable on big models.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .types import ActorRef, BBox, CameraSpec, SceneHandle

if TYPE_CHECKING:
    from .style import SceneStyle


@runtime_checkable
class Backend(Protocol):
    """Protocol that every renderer implements.

    Concrete implementations (``MplBackend`` in step 2,
    ``PyVistaBackend`` in step 6, ``TrameBackend`` in Phase 6) all
    satisfy this interface. The :class:`Scene` only ever calls methods
    declared here.
    """

    name: str
    is_3d_capable: bool
    is_interactive: bool

    # ----- Scene lifecycle ------------------------------------------- #

    def make_scene(
        self, *, is_3d: bool = False, off_screen: bool = False,
    ) -> SceneHandle:
        """Allocate the renderer-side state for a new scene."""

    def set_bounds(self, scene: SceneHandle, bbox: BBox) -> None:
        """Set the model bounding box (drives default camera framing)."""

    def set_camera(self, scene: SceneHandle, cam: CameraSpec) -> None:
        """Apply a saved camera state."""

    def set_style(self, scene: SceneHandle, style: "SceneStyle") -> None:
        """Apply scene-wide styling (background, grid, theme)."""

    # ----- Primitives ------------------------------------------------ #
    #
    # Every layer is composed from these primitives. Backends that
    # cannot render a primitive raise BackendCapabilityError rather
    # than degrade silently.

    def add_segments(
        self,
        scene: SceneHandle,
        segments: np.ndarray,
        *,
        color: Any = None,
        width: float | None = None,
        alpha: float | None = None,
        label: str | None = None,
    ) -> ActorRef:
        """Add a line-segment batch. ``segments`` shape ``(N, 2, 3)``."""

    def add_points(
        self,
        scene: SceneHandle,
        points: np.ndarray,
        *,
        color: Any = None,
        size: float | None = None,
        scalars: np.ndarray | None = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add a point cloud. ``points`` shape ``(N, 3)``."""

    def add_polygons(
        self,
        scene: SceneHandle,
        polygons: np.ndarray,
        *,
        values: np.ndarray | None = None,
        cmap: str | None = None,
        edge_color: Any = None,
    ) -> ActorRef:
        """Add filled polygons. ``polygons`` is a ``(N, M, 3)`` array
        of M-sided polygon vertices, or backend-specific equivalent."""

    def add_arrows(
        self,
        scene: SceneHandle,
        origins: np.ndarray,
        vectors: np.ndarray,
        *,
        scale: float = 1.0,
        color: Any = None,
        cmap: str | None = None,
    ) -> ActorRef:
        """Add an arrow glyph batch. Shapes: ``(N, 3)`` each."""

    # ----- In-place actor updates ------------------------------------ #
    #
    # The perf contract: animation calls update_* only, never add_*.

    def update_scalars(self, actor: ActorRef, scalars: np.ndarray) -> None:
        """Mutate the scalars of an existing actor in place."""

    def update_points(self, actor: ActorRef, points: np.ndarray) -> None:
        """Mutate the vertex positions of an existing actor in place."""

    def set_visible(self, actor: ActorRef, visible: bool) -> None:
        """Toggle the visibility of an actor."""

    def remove(self, scene: SceneHandle, actor: ActorRef) -> None:
        """Detach and dispose of an actor."""

    # ----- Output ---------------------------------------------------- #

    def show(self, scene: SceneHandle) -> None:
        """Hand off to the renderer's interactive show loop (if any)."""

    def save(self, scene: SceneHandle, path: Path, *, dpi: int = 300) -> None:
        """Persist the current frame to disk."""

    def snapshot(self, scene: SceneHandle) -> np.ndarray:
        """Return the current frame as an ``(H, W, 3)`` uint8 RGB array."""
