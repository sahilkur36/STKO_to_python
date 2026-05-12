"""Matplotlib backend for the viewer.

Phase 2.2 of the viewer integration plan (see
``docs/viewer/00-roadmap.md``). Implements every method on the
:class:`STKO_to_python.viewer.core.Backend` protocol against
matplotlib's 2-D and 3-D primitives.

Importing this module pulls matplotlib, which is a base dependency of
STKO_to_python — so the import is always safe (no optional-extra
gating). The viewer's smoke test only excludes ``pyvista`` / ``vtk`` /
``PySide6`` / ``trame`` / ``imageio`` from the base import; matplotlib
has always been present.

Typical usage::

    from STKO_to_python.viewer.backends.mpl import MplBackend
    from STKO_to_python.viewer.core import Scene

    scene = Scene(MplBackend(), my_source, is_3d=True)
    scene.add(SomeLayer(...))
    scene.show()
"""
from __future__ import annotations

from .backend import MplBackend, MplSceneHandle

__all__ = ["MplBackend", "MplSceneHandle"]
