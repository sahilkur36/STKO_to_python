"""Concrete :class:`Layer` implementations.

Layers are renderable units in a :class:`Scene`. As of v1.12 the
package ships four:

* :class:`MeshLayer` — static element-edge wireframe; powers the
  Phase 2.4 rewire of ``ds.plot.mesh`` and ``ds.plot.undeformed_shape``.
* :class:`DeformedMeshLayer` — time-varying edge wireframe at
  ``original + scale * displacement``; powers the Phase 2.5 rewire
  of ``ds.plot.deformed_shape`` and supports in-place
  :meth:`~DeformedMeshLayer.update_to_step` for animation.
* :class:`NodeLayer` — point cloud at node positions, optionally
  scalar-coloured by a nodal result (Phase 2.6).
* :class:`VectorLayer` — arrow glyphs at node positions driven by a
  nodal vector result, with per-step updates (Phase 2.6).

The remaining catalog (contour, gauss, diagram, fiber, layer stack,
zerolength, clipping) lands in Phase 3 alongside the PyVista backend.

The Phase 2 contract is the **byte-identical refactor under the
existing API**: ``ds.plot.*`` keeps every signature and visual
output; only the implementation now flows through
Scene + Layer + MplBackend. Greenfield layers (no v1.x counterpart)
like NodeLayer and VectorLayer ship with matplotlib coverage so the
contracts are exercised in unit tests ahead of the 3-D work.
"""
from __future__ import annotations

from .deformed_mesh import DeformedMeshLayer
from .mesh import MeshLayer
from .node import NodeLayer, VectorLayer

__all__ = ["DeformedMeshLayer", "MeshLayer", "NodeLayer", "VectorLayer"]
