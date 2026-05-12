"""Viewer-specific exception types.

Defined here so callers can catch them without importing concrete
renderer modules (which may not be installed).
"""
from __future__ import annotations


class BackendCapabilityError(NotImplementedError):
    """Raised when a layer asks for a primitive its backend cannot render.

    Example: a ``VolumeLayer`` with ``mode='iso'`` against the matplotlib
    backend will surface this — the layer cannot silently downgrade, so
    the caller learns immediately and can switch backends or change the
    mode.
    """


class LayerAttachError(RuntimeError):
    """Raised when ``Layer.attach`` cannot bind the layer to its scene.

    Typical causes: the requested data source does not expose a result
    the layer needs, or a selection does not match any entities.
    """
