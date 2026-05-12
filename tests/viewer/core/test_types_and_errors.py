"""Tests for the small value types and exception classes."""
from __future__ import annotations

import math

import pytest

from STKO_to_python.viewer.core import (
    BBox,
    BackendCapabilityError,
    CameraSpec,
    LayerAttachError,
)


# --------------------------------------------------------------------- #
# BBox                                                                  #
# --------------------------------------------------------------------- #


def test_bbox_size_center_diagonal() -> None:
    bbox = BBox(-1.0, -2.0, -3.0, 4.0, 6.0, 9.0)
    assert bbox.size == (5.0, 8.0, 12.0)
    assert bbox.center == (1.5, 2.0, 3.0)
    assert math.isclose(bbox.diagonal, math.sqrt(25 + 64 + 144))


def test_bbox_zero_diagonal_is_zero() -> None:
    """A degenerate box (single point) has zero diagonal."""
    bbox = BBox(1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
    assert bbox.diagonal == 0.0


def test_bbox_is_frozen() -> None:
    bbox = BBox(0, 0, 0, 1, 1, 1)
    with pytest.raises(Exception):
        bbox.x_min = 5  # type: ignore[misc]


def test_bbox_is_hashable() -> None:
    bbox_a = BBox(0, 0, 0, 1, 1, 1)
    bbox_b = BBox(0, 0, 0, 1, 1, 1)
    bbox_c = BBox(0, 0, 0, 1, 1, 2)
    assert hash(bbox_a) == hash(bbox_b)
    assert hash(bbox_a) != hash(bbox_c)
    # Hashable means it can live in a set / dict key.
    assert {bbox_a, bbox_b, bbox_c} == {bbox_a, bbox_c}


# --------------------------------------------------------------------- #
# CameraSpec                                                            #
# --------------------------------------------------------------------- #


def test_camera_spec_holds_fields() -> None:
    cam = CameraSpec(
        position=(0.0, 0.0, 5.0),
        focal_point=(0.0, 0.0, 0.0),
        view_up=(0.0, 1.0, 0.0),
    )
    assert cam.position == (0.0, 0.0, 5.0)
    assert cam.focal_point == (0.0, 0.0, 0.0)
    assert cam.view_up == (0.0, 1.0, 0.0)
    assert cam.parallel_projection is False


def test_camera_spec_parallel_projection_overrides() -> None:
    cam = CameraSpec(
        position=(0, 0, 5),
        focal_point=(0, 0, 0),
        view_up=(0, 1, 0),
        parallel_projection=True,
    )
    assert cam.parallel_projection is True


# --------------------------------------------------------------------- #
# Exceptions                                                            #
# --------------------------------------------------------------------- #


def test_backend_capability_error_is_not_implemented_error() -> None:
    """Catching ``NotImplementedError`` also catches ``BackendCapabilityError``."""
    assert issubclass(BackendCapabilityError, NotImplementedError)
    with pytest.raises(NotImplementedError):
        raise BackendCapabilityError("test")


def test_layer_attach_error_is_runtime_error() -> None:
    assert issubclass(LayerAttachError, RuntimeError)
    with pytest.raises(RuntimeError):
        raise LayerAttachError("test")
