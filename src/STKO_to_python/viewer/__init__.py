"""STKO_to_python viewer subpackage.

The viewer is intentionally empty at v1.9.0 — Phase 0 establishes the
namespace and optional extras. Subsequent phases land the algorithm
tier, the ``Scene`` / ``Layer`` / ``Backend`` machinery, and the Qt
GUI. See ``docs/viewer/00-roadmap.md`` for the phased delivery plan.

Install one of the viewer extras to enable rendering:

    pip install "stko_to_python[viewer]"            # Qt desktop GUI
    pip install "stko_to_python[viewer-3d]"         # PyVista 3D (notebook)
    pip install "stko_to_python[viewer-headless]"   # off-screen / CLI
    pip install "stko_to_python[viewer-web]"        # Trame browser (Phase 6)

This module is lazy-by-design: importing it does **not** import
``pyvista``, ``vtk``, ``PySide6`` or any other optional dependency. The
``tests/viewer/test_smoke.py`` suite enforces that contract.
"""
from __future__ import annotations

__all__: list[str] = []
