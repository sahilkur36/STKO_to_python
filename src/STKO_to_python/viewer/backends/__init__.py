"""Concrete :class:`Backend` implementations.

Each subpackage is one renderer:

* ``mpl`` — matplotlib (Phase 2.2, this is the first one to land).
* ``pyvista`` — PyVista/VTK (Phase 3).
* ``trame`` — web viewer (Phase 6, deferred).

Importing a backend pulls its renderer; the empty top-level namespace
keeps ``import STKO_to_python.viewer.backends`` light.
"""
from __future__ import annotations

__all__: list[str] = []
