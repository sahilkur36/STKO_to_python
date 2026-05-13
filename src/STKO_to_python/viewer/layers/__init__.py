"""Concrete :class:`Layer` implementations.

Layers are renderable units in a :class:`Scene`. As of v1.11 the
package ships two:

* :class:`MeshLayer` — static element-edge wireframe; powers the
  Phase 2.4 rewire of ``ds.plot.mesh`` and ``ds.plot.undeformed_shape``.
* :class:`DeformedMeshLayer` — time-varying edge wireframe at
  ``original + scale * displacement``; powers the Phase 2.5 rewire
  of ``ds.plot.deformed_shape`` and supports in-place
  :meth:`~DeformedMeshLayer.update_to_step` for animation.

The remaining catalog (node/vector, contour, gauss, diagram, fiber,
…) lands in Phases 2.6–3.X.

The Phase 2 contract is the **byte-identical refactor under the
existing API**: ``ds.plot.*`` keeps every signature and visual
output; only the implementation now flows through
Scene + Layer + MplBackend.
"""
from __future__ import annotations

from .deformed_mesh import DeformedMeshLayer
from .mesh import MeshLayer

__all__ = ["DeformedMeshLayer", "MeshLayer"]
