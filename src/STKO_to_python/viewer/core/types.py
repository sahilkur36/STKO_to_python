"""Shared concrete types for the viewer core.

These are deliberately small (one dataclass per concept) so they have
no runtime dependency on any backend. ``SceneHandle`` and ``ActorRef``
are opaque aliases — the backend casts them to its native types.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


# Opaque renderer-specific handles. Concrete backends pass these through
# unchanged; layer code treats them as opaque cookies.
SceneHandle = Any
ActorRef = Any


@dataclass(frozen=True)
class BBox:
    """Axis-aligned 3-D bounding box.

    Attributes:
        x_min, y_min, z_min: Minimum corner coordinates.
        x_max, y_max, z_max: Maximum corner coordinates.
    """

    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def size(self) -> tuple[float, float, float]:
        """``(dx, dy, dz)`` side lengths."""
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )

    @property
    def center(self) -> tuple[float, float, float]:
        """Geometric centre of the box."""
        return (
            0.5 * (self.x_min + self.x_max),
            0.5 * (self.y_min + self.y_max),
            0.5 * (self.z_min + self.z_max),
        )

    @property
    def diagonal(self) -> float:
        """Euclidean diagonal length — useful as a scale reference."""
        dx, dy, dz = self.size
        return sqrt(dx * dx + dy * dy + dz * dz)


@dataclass(frozen=True)
class CameraSpec:
    """Camera state — enough to save / restore a view orientation.

    Attributes:
        position: Camera position in world space.
        focal_point: Point the camera looks at.
        view_up: Up vector for the camera.
        parallel_projection: ``True`` for orthographic, ``False`` for
            perspective.
    """

    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    view_up: tuple[float, float, float]
    parallel_projection: bool = False
