"""Per-element-type kernels for section-cut computation.

A kernel takes a :class:`SectionCutSpec` and an :class:`MPCODataSet`
and returns the per-element contributions plus an aggregated resultant
for one geometric family (beams, shells, solids).

This subpackage is internal — the public entry point is
:class:`SectionCut`, which composes kernel outputs.
"""
from __future__ import annotations

from .beam import BEAM_ELEMENT_CLASSES, BeamIntersection, find_beam_intersections
from .beam_resultant import BeamCutResult, compute_beam_cut
from .shell import (
    SHELL_ELEMENT_CLASSES,
    ShellCutResult,
    ShellIntersection,
    compute_shell_cut,
    find_shell_intersections,
)

__all__ = [
    "BEAM_ELEMENT_CLASSES",
    "BeamIntersection",
    "find_beam_intersections",
    "BeamCutResult",
    "compute_beam_cut",
    "SHELL_ELEMENT_CLASSES",
    "ShellIntersection",
    "find_shell_intersections",
    "ShellCutResult",
    "compute_shell_cut",
]
