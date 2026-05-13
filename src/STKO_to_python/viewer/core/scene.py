"""``Scene`` — orchestrates a Backend, a DataSource, and a list of Layers.

One scene is **one drawable region** — either a single 2-D axes or a
single 3-D viewport. A figure with multiple panels is built from
multiple scenes (see ``docs/viewer/01-architecture.md`` §4).

This class is concrete; it contains the bookkeeping that every viewer
implementation needs. It does **not** hold any rendering code — that
lives in the concrete :class:`Backend`. Likewise it does not pull data
itself — that's the :class:`DataSource`'s job.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .style import SceneStyle

if TYPE_CHECKING:
    import numpy as np

    from .backend import Backend
    from .datasource import DataSource
    from .layer import Layer
    from .types import SceneHandle


class Scene:
    """A drawable region with an ordered layer stack.

    Args:
        backend: Renderer implementation (matplotlib, pyvista, …).
        source: Data adapter the layers will query.
        is_3d: Whether this scene wants a 3-D viewport. The backend
            may downgrade silently (or raise on backends that lack 3-D
            capability — see :attr:`Backend.is_3d_capable`).
        off_screen: When ``True`` the backend renders without showing
            a window — used by the headless CLI and CI snapshots.
        style: Scene-wide style. Defaults to a fresh
            :class:`SceneStyle()`.
        handle: Optional pre-allocated backend handle. When supplied,
            the scene **borrows** it — :meth:`Backend.make_scene` and
            :meth:`Backend.set_style` are skipped. Used by the legacy
            ``Plot.*`` rewires to thread a caller-supplied matplotlib
            ``Axes`` through the scene machinery without mutating
            global rcParams via the default style application.

    The opaque :attr:`handle` is allocated lazily on first access so
    that constructing a Scene does not yet require a renderer to be
    available; the handle is created when the first layer is added or
    when :meth:`show` / :meth:`save` is called.
    """

    def __init__(
        self,
        backend: "Backend",
        source: "DataSource",
        *,
        is_3d: bool = False,
        off_screen: bool = False,
        style: SceneStyle | None = None,
        handle: "SceneHandle | None" = None,
    ) -> None:
        self.backend = backend
        self.source = source
        self.is_3d = is_3d
        self.style = style if style is not None else SceneStyle()
        self.layers: list["Layer"] = []
        self._handle: "SceneHandle | None" = handle
        self._off_screen = off_screen
        self._current_step: int | None = None
        self._current_stage: str | None = None

    @property
    def handle(self) -> "SceneHandle":
        """Backend-allocated scene handle. Created on first access.

        If a handle was passed to the constructor, it is returned
        without invoking the backend; otherwise it is allocated lazily
        on first access and the scene style is applied.
        """
        if self._handle is None:
            self._handle = self.backend.make_scene(
                is_3d=self.is_3d, off_screen=self._off_screen,
            )
            self.backend.set_style(self._handle, self.style)
        return self._handle

    @property
    def current_step(self) -> int | None:
        return self._current_step

    @property
    def current_stage(self) -> str | None:
        return self._current_stage

    # ----- Layer management ------------------------------------------ #

    def add(self, layer: "Layer") -> "Layer":
        """Attach ``layer`` to the scene and return it (for chaining).

        If the scene already has a current step, the newly added layer
        is immediately advanced to that step so it shows up in sync
        with the rest.
        """
        # Force the backend to allocate the scene handle now, so the
        # layer's attach() can safely call backend.add_*.
        _ = self.handle
        layer.attach(self, self.source)
        self.layers.append(layer)
        if self._current_step is not None:
            layer.update_to_step(self._current_step)
        return layer

    def remove(self, layer: "Layer") -> None:
        """Detach ``layer`` and drop it from the stack."""
        if layer not in self.layers:
            raise ValueError(f"Layer {layer.name!r} is not in this scene.")
        layer.detach()
        self.layers.remove(layer)

    # ----- Time / stage cursor --------------------------------------- #

    def set_step(self, step: int) -> None:
        """Advance every layer to ``step`` via in-place actor updates.

        Fires :meth:`Layer.update_to_step` on **every** layer, including
        invisible ones — that's the apeGmsh perf contract: each layer
        keeps its per-step data ready, the backend just toggles
        visibility.
        """
        self._current_step = step
        for layer in self.layers:
            layer.update_to_step(step)

    def set_stage(self, stage: str | None) -> None:
        """Switch the active stage. Does not refresh — call ``set_step`` after."""
        self._current_stage = stage

    # ----- Camera / bounds ------------------------------------------- #

    def fit_bounds(self) -> None:
        """Pull the model bbox from the data source and apply to the backend."""
        bbox = self.source.model_bbox()
        self.backend.set_bounds(self.handle, bbox)

    # ----- Output ---------------------------------------------------- #

    def show(self) -> None:
        """Hand off to the backend's interactive show loop."""
        self.backend.show(self.handle)

    def save(self, path: Path, *, dpi: int = 300) -> None:
        """Persist the current frame to disk via the backend."""
        self.backend.save(self.handle, path, dpi=dpi)

    def snapshot(self) -> "np.ndarray":
        """Return the current frame as an ``(H, W, 3)`` uint8 RGB array."""
        return self.backend.snapshot(self.handle)

    # ----- Container protocol ---------------------------------------- #

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

    def __contains__(self, layer: object) -> bool:
        return layer in self.layers
