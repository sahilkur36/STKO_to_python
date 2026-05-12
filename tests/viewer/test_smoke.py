"""Phase 0 smoke tests for the viewer subpackage.

Phase 0 establishes the namespace and optional extras only — no
rendering, no Qt. These tests guard the contract that
``import STKO_to_python.viewer`` stays lightweight (does not pull
``pyvista`` / ``vtk`` / ``PySide6`` / ``trame`` at import time).

The two extras-gated tests (``test_viewer_3d_extra_resolves``,
``test_viewer_qt_extra_resolves``) only run when the optional deps are
installed — they verify that the resolved environment is consistent
with what the extras advertise. In the base CI matrix they skip; in
the dedicated ``viewer-extras`` CI job they run.
"""
from __future__ import annotations

import importlib
import sys

import pytest


HEAVY_DEPS = frozenset(
    {
        "pyvista",
        "vtk",
        "PySide6",
        "pyvistaqt",
        "qtpy",
        "trame",
        "trame_vtk",
        "trame_vuetify",
        "imageio",
        "imageio_ffmpeg",
    }
)


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


def test_viewer_imports_without_extras() -> None:
    """The viewer namespace must import cleanly on the base install."""
    import STKO_to_python.viewer as viewer

    assert viewer.__all__ == []


def test_viewer_import_is_light() -> None:
    """Importing the viewer must not pull heavy optional deps at import time.

    This is the core Phase 0 contract: ``pip install stko_to_python``
    (no extras) does not gain ``pyvista`` / ``vtk`` / ``PySide6`` /
    ``trame`` transitively, so importing the viewer namespace in that
    environment must not trip an ``ImportError`` — it must succeed and
    not leak any of those top-level packages into ``sys.modules``.
    """
    # Drop any heavy deps that other test modules may have imported,
    # so we get a clean read on what *the viewer import itself* pulls.
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top in HEAVY_DEPS:
            del sys.modules[name]
    sys.modules.pop("STKO_to_python.viewer", None)
    sys.modules.pop("STKO_to_python.viewer._version", None)

    import STKO_to_python.viewer  # noqa: F401

    leaked = {name.split(".")[0] for name in sys.modules} & HEAVY_DEPS
    assert not leaked, f"viewer import pulled heavy deps: {sorted(leaked)}"


def test_schema_versions_present() -> None:
    """Schema version constants exist even though no spec format does yet."""
    from STKO_to_python.viewer import _version

    assert isinstance(_version.SCENE_SPEC_SCHEMA, int)
    assert isinstance(_version.SESSION_SCHEMA, int)
    # Phase 0 values: ``0`` means "no on-disk format yet". When Phase 2
    # lands the ``SceneSpec`` format, ``SCENE_SPEC_SCHEMA`` moves to 1
    # and this assertion will need bumping.
    assert _version.SCENE_SPEC_SCHEMA == 0
    assert _version.SESSION_SCHEMA == 0


@pytest.mark.skipif(
    not (_has("pyvista") and _has("vtk")),
    reason="viewer-3d extra not installed",
)
def test_viewer_3d_extra_resolves() -> None:
    """When ``[viewer-3d]`` is installed, pyvista + vtk import successfully."""
    import pyvista  # noqa: F401
    import vtk  # noqa: F401


@pytest.mark.skipif(
    not (_has("PySide6") and _has("pyvistaqt") and _has("qtpy")),
    reason="viewer extra not installed",
)
def test_viewer_qt_extra_resolves() -> None:
    """When ``[viewer]`` is installed, the Qt stack imports successfully."""
    import PySide6  # noqa: F401
    import pyvistaqt  # noqa: F401
    import qtpy  # noqa: F401
