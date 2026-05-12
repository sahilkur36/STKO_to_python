"""Viewer core — Backend / Scene / Layer / DataSource skeletons.

Phase 2 step 1 of the viewer integration plan (see
``docs/viewer/00-roadmap.md``). This subpackage defines the
backend-neutral contracts that every renderer implementation
satisfies. No rendering happens here; concrete backends and layer
types ride on top in subsequent steps.

Public API:

* :class:`Scene` — concrete orchestrator (Backend + DataSource + Layers).
* :class:`Layer` — abstract base every renderable subclasses.
* :class:`Backend` — runtime-checkable protocol concrete renderers implement.
* :class:`DataSource` — runtime-checkable protocol the adapter implements.
* :class:`SceneStyle`, :class:`LayerStyle` — hierarchical style dataclasses.
* :class:`SelectionSpec` — frozen, hashable selection filter.
* :class:`BBox`, :class:`CameraSpec` — geometric value types.
* :class:`BackendCapabilityError`, :class:`LayerAttachError` — exceptions.

The subpackage is pure-Python and has **no** runtime dependency on
``pyvista``, ``vtk``, ``PySide6``, or any optional extra. The viewer
smoke test (``tests/viewer/test_smoke.py``) keeps that contract honest.
"""
from __future__ import annotations

from .backend import Backend
from .datasource import DataSource
from .errors import BackendCapabilityError, LayerAttachError
from .layer import Layer
from .scene import Scene
from .selection import SelectionSpec
from .style import LayerStyle, SceneStyle
from .types import ActorRef, BBox, CameraSpec, SceneHandle

__all__ = [
    "ActorRef",
    "Backend",
    "BBox",
    "BackendCapabilityError",
    "CameraSpec",
    "DataSource",
    "Layer",
    "LayerAttachError",
    "LayerStyle",
    "Scene",
    "SceneHandle",
    "SceneStyle",
    "SelectionSpec",
]
