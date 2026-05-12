"""Hierarchical style: scene-wide defaults + per-layer overrides.

Three-level precedence ladder (matches the existing
:class:`STKO_to_python.PlotSettings` model that
``NodalResultsPlotter`` already uses):

1. **Project / dataset default** — set once on the dataset.
2. **Scene default** — :class:`SceneStyle.layer_defaults` per layer kind.
3. **Layer override** — :class:`LayerStyle` passed at construction.

Each level only sets the fields it cares about; ``None`` inherits from
the next level up. The merge semantics are the same as ``dict.update``
but field-aware.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any


@dataclass(frozen=True)
class LayerStyle:
    """Style overrides for a single layer.

    Every field is ``None`` by default — that signals "inherit from the
    scene-level default." See :meth:`merge` for how the precedence
    cascade is resolved.

    Attributes:
        color: Solid colour name, hex string, or ``(r, g, b)`` tuple.
        linewidth: Line / edge width.
        linestyle: Matplotlib-style linestyle (``"-"``, ``"--"``, …).
        marker: Matplotlib-style marker (``"."``, ``"o"``, …).
        alpha: Opacity in ``[0, 1]``.
        cmap: Colormap name for scalar-valued layers.
        clim: ``(min, max)`` colour-range overrides; ``None`` = auto.
        label: Legend label / scalar-bar title override.
    """

    color: str | tuple[float, float, float] | None = None
    linewidth: float | None = None
    linestyle: str | None = None
    marker: str | None = None
    alpha: float | None = None
    cmap: str | None = None
    clim: tuple[float, float] | None = None
    label: str | None = None

    def merge(self, base: "LayerStyle") -> "LayerStyle":
        """Return a new style with this layer's overrides applied over ``base``.

        Fields that are ``None`` on ``self`` inherit from ``base``;
        every other field overrides ``base``.
        """
        merged: dict[str, Any] = {}
        for f in fields(self):
            override = getattr(self, f.name)
            merged[f.name] = override if override is not None else getattr(base, f.name)
        return LayerStyle(**merged)


@dataclass(frozen=True)
class SceneStyle:
    """Scene-wide style settings.

    Attributes:
        background: Background colour (any value matplotlib / pyvista
            accepts).
        grid: Whether to render coordinate grid lines.
        font_size: Base font size for labels, titles, scalar bars.
        theme: ``"light"`` or ``"dark"`` — informs default palettes.
        layer_defaults: Per-layer-kind default :class:`LayerStyle`,
            keyed on the layer's ``kind`` attribute.
    """

    background: str = "white"
    grid: bool = False
    font_size: int = 10
    theme: str = "light"
    layer_defaults: dict[str, LayerStyle] = field(default_factory=dict)

    def get_defaults_for(self, layer_kind: str) -> LayerStyle:
        """Lookup helper — returns an empty :class:`LayerStyle` when absent."""
        return self.layer_defaults.get(layer_kind, LayerStyle())

    def with_layer_default(
        self, layer_kind: str, style: LayerStyle,
    ) -> "SceneStyle":
        """Return a new :class:`SceneStyle` with one layer-kind default replaced."""
        new_defaults = dict(self.layer_defaults)
        new_defaults[layer_kind] = style
        return replace(self, layer_defaults=new_defaults)
