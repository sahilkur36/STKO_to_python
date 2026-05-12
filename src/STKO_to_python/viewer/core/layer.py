"""``Layer`` abstract base — every renderable in a :class:`Scene`.

Lifecycle (see ``docs/viewer/01-architecture.md`` §4):

1. ``__init__`` — construct with frozen config (name, selection, style,
   visibility, z-order).
2. ``attach(scene, source)`` — bind to a :class:`Scene`; build actors
   via the scene's :class:`Backend`. Called once by :meth:`Scene.add`.
3. ``update_to_step(step)`` — read from the :class:`DataSource` at the
   new step; scatter into pre-allocated actor data via in-place
   updates. **Must not** call ``backend.add_*``.
4. ``detach()`` — release actor references; the layer becomes inert.

The "no actor recreation per step" rule is the perf contract that
keeps animation playback usable on big models. Tests in step 2 onward
pin specific layers to that contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .selection import SelectionSpec
from .style import LayerStyle

if TYPE_CHECKING:
    from .datasource import DataSource
    from .scene import Scene


class Layer(ABC):
    """Base for every viewer layer.

    Attributes:
        kind: Stable string used by ``LAYER_KINDS`` catalog,
            :attr:`SceneStyle.layer_defaults`, and the
            ``SceneSpec`` round-trip. Subclasses must override this.
        name: Human-readable name (defaults to class name).
        selection: :class:`SelectionSpec` filtering which entities the
            layer renders.
        style: Per-layer :class:`LayerStyle` overrides; ``None`` fields
            inherit from the scene defaults at attach time.
        visible: Whether the layer is rendered. Toggling does **not**
            re-fetch data — flip and call ``Scene.set_step`` (or
            re-render) to refresh.
        z_order: Render order within the scene; higher = on top.
    """

    kind: str = "layer"

    def __init__(
        self,
        *,
        name: str | None = None,
        selection: SelectionSpec | None = None,
        style: LayerStyle | None = None,
        visible: bool = True,
        z_order: int = 0,
    ) -> None:
        self.name = name if name is not None else self.__class__.__name__
        self.selection = selection if selection is not None else SelectionSpec.empty()
        self.style = style if style is not None else LayerStyle()
        self.visible = visible
        self.z_order = z_order
        self._scene: "Scene | None" = None
        self._source: "DataSource | None" = None

    @property
    def is_attached(self) -> bool:
        """True once :meth:`attach` has run and before :meth:`detach`."""
        return self._scene is not None

    # ----- Subclass contract ----------------------------------------- #

    @abstractmethod
    def attach(self, scene: "Scene", source: "DataSource") -> None:
        """Bind to ``scene`` and ``source``; build initial actors.

        Concrete implementations set ``self._scene = scene`` and
        ``self._source = source`` then issue ``scene.backend.add_*``
        calls. After this method returns, the actor set is frozen.

        Raises:
            STKO_to_python.viewer.core.errors.LayerAttachError:
                When required inputs are missing on ``source``.
        """

    @abstractmethod
    def update_to_step(self, step: int) -> None:
        """In-place scalar / point mutation for animation.

        Must not call any ``backend.add_*`` method. Use
        ``backend.update_*`` to mutate pre-allocated actor data; this
        keeps the per-step cost O(modified data).
        """

    @abstractmethod
    def detach(self) -> None:
        """Release every actor reference; the layer becomes inert."""
