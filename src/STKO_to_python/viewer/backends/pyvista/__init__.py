"""PyVista :class:`Backend` implementation.

Wraps a :class:`pyvista.Plotter` (off-screen or windowed) so the same
``Scene`` / ``Layer`` graph that drives the matplotlib backend can
render through VTK in 3-D. Gated behind the optional
``[viewer-3d]`` extra; importing this subpackage when ``pyvista``
isn't installed raises :class:`ImportError` so the lightweight
notebook path (``pip install stko_to_python``) is unaffected.
"""
from __future__ import annotations

from .backend import PvSceneHandle, PyVistaBackend

__all__ = ["PvSceneHandle", "PyVistaBackend"]
