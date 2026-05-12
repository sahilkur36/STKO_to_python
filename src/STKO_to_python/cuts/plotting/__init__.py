"""Plotting subpackage for section cuts.

Two plotters live here (more land with subsequent steps of the build):

- :class:`SectionCutPlotter` — bound to a single :class:`SectionCut`,
  matplotlib by default. Step 5.
- :class:`SectionSweepPlotter` — bound to a :class:`SectionSweep`. Step 7.
- :class:`MultiCutPlotter` — bound to a :class:`MultiCutResult`. Step 8.
- Geometry view (3D model + cut plane + contributing elements,
  matplotlib + pyvista). Step 9.
"""
from __future__ import annotations

from .cut_plotter import SectionCutPlotter
from .multi_plotter import MultiCutPlotter
from .sweep_plotter import SectionSweepPlotter

__all__ = ["SectionCutPlotter", "SectionSweepPlotter", "MultiCutPlotter"]
