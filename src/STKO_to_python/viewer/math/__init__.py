"""Pure-numpy math primitives for the viewer.

Phase 1 of the viewer integration plan
(``docs/viewer/00-roadmap.md``) lifts the algorithm tier from apeGmsh:

* Gauss-point → nodal extrapolation with cross-element averaging.
* Beam local-frame fallback (vecxz Gram-Schmidt) for datasets without
  STKO `.cdata` ``*LOCAL_AXES`` quaternions — landed in a follow-up PR.
* Shell local-axes helpers — landed in a follow-up PR.
* Display-space box-pick projection math — landed in a follow-up PR.

All modules here are pure-numpy and **have no dependency on**
``pyvista``, ``vtk``, or ``PySide6``. They are safe to import under
the base install and are exercised by ``tests/viewer/math/`` without
any optional extras.
"""
from __future__ import annotations

__all__: list[str] = []
