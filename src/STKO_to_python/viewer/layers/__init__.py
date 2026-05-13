"""Concrete :class:`Layer` implementations.

Layers are renderable units in a :class:`Scene`. Phase 2.4 ships the
first one — :class:`MeshLayer`, which wraps the v1.x ``ds.plot.mesh``
edge-rendering through the Scene/Backend/DataSource machinery. The
remaining layer types from the directive's catalog (deformed mesh,
node/vector, contour, gauss, diagram, fiber, …) land in Phases
2.5–3.X.

The Phase 2.4 contract is the **byte-identical refactor under the
existing API**: ``ds.plot.mesh()`` keeps its signature and visual
output; only the implementation now flows through Scene + MeshLayer +
MplBackend.
"""
from __future__ import annotations

from .mesh import MeshLayer

__all__ = ["MeshLayer"]
