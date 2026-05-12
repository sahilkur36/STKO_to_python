"""Shared stubs for viewer-core unit tests.

The protocols (:class:`Backend`, :class:`DataSource`) are
``runtime_checkable``, so a hand-written stub class that implements
the right methods passes ``isinstance(stub, Protocol)`` checks. These
stubs let the Scene / Layer plumbing be tested without pulling any
optional renderer dependency.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from STKO_to_python.viewer.core import (
    BBox,
    Layer,
    SelectionSpec,
)


class StubBackend:
    """Minimal :class:`Backend` implementation that records every call."""

    name = "stub"
    is_3d_capable = False
    is_interactive = False

    def __init__(self) -> None:
        self.make_scene_calls: int = 0
        self.set_bounds_calls: list = []
        self.set_style_calls: list = []
        self.add_segments_calls: list = []
        self.add_points_calls: list = []
        self.update_scalars_calls: list = []
        self.update_points_calls: list = []
        self.show_calls: int = 0
        self.save_calls: list = []
        self._next_actor_id: int = 0

    # ----- scene lifecycle ----- #

    def make_scene(self, *, is_3d: bool = False, off_screen: bool = False) -> Any:
        self.make_scene_calls += 1
        return ("scene", self.make_scene_calls)

    def set_bounds(self, scene: Any, bbox: BBox) -> None:
        self.set_bounds_calls.append((scene, bbox))

    def set_camera(self, scene: Any, cam: Any) -> None:
        pass

    def set_style(self, scene: Any, style: Any) -> None:
        self.set_style_calls.append((scene, style))

    # ----- primitives ----- #

    def _new_actor(self, kind: str) -> Any:
        self._next_actor_id += 1
        return (kind, self._next_actor_id)

    def add_segments(self, scene, segments, **kwargs):  # type: ignore[no-untyped-def]
        self.add_segments_calls.append((scene, segments, kwargs))
        return self._new_actor("seg")

    def add_points(self, scene, points, **kwargs):  # type: ignore[no-untyped-def]
        self.add_points_calls.append((scene, points, kwargs))
        return self._new_actor("pts")

    def add_polygons(self, scene, polygons, **kwargs):  # type: ignore[no-untyped-def]
        return self._new_actor("poly")

    def add_arrows(self, scene, origins, vectors, **kwargs):  # type: ignore[no-untyped-def]
        return self._new_actor("arr")

    # ----- in-place updates ----- #

    def update_scalars(self, actor: Any, scalars: np.ndarray) -> None:
        self.update_scalars_calls.append((actor, scalars))

    def update_points(self, actor: Any, points: np.ndarray) -> None:
        self.update_points_calls.append((actor, points))

    def set_visible(self, actor: Any, visible: bool) -> None:
        pass

    def remove(self, scene: Any, actor: Any) -> None:
        pass

    # ----- output ----- #

    def show(self, scene: Any) -> None:
        self.show_calls += 1

    def save(self, scene: Any, path: Any, *, dpi: int = 300) -> None:
        self.save_calls.append((scene, path, dpi))

    def snapshot(self, scene: Any) -> np.ndarray:
        return np.zeros((1, 1, 3), dtype=np.uint8)


class StubDataSource:
    """Minimal :class:`DataSource` implementation backed by static arrays."""

    def __init__(
        self,
        *,
        bbox: BBox | None = None,
        n_steps: int = 0,
    ) -> None:
        self._bbox = bbox if bbox is not None else BBox(0, 0, 0, 1, 1, 1)
        self._n_steps = n_steps

    @property
    def dataset(self) -> Any:
        return None  # unused by the stubs; layers don't need it here

    def node_coords(self, ids: np.ndarray | None = None) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float64)

    def element_centroids(self, ids: np.ndarray | None = None) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.float64)

    def model_bbox(self) -> BBox:
        return self._bbox

    def n_steps(self, stage: str | None = None) -> int:
        return self._n_steps

    def time(self, stage: str | None = None) -> np.ndarray:
        return np.zeros(self._n_steps, dtype=np.float64)

    def resolve_node_ids(self, spec: SelectionSpec) -> np.ndarray:
        return np.zeros(0, dtype=np.int64)

    def resolve_element_ids(self, spec: SelectionSpec) -> np.ndarray:
        return np.zeros(0, dtype=np.int64)


class RecordingLayer(Layer):
    """Concrete :class:`Layer` that records every lifecycle event."""

    kind = "recording"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.attach_calls: list = []
        self.update_calls: list[int] = []
        self.detach_calls: int = 0

    def attach(self, scene: Any, source: Any) -> None:
        self._scene = scene
        self._source = source
        self.attach_calls.append((scene, source))

    def update_to_step(self, step: int) -> None:
        self.update_calls.append(step)

    def detach(self) -> None:
        self._scene = None
        self._source = None
        self.detach_calls += 1


@pytest.fixture
def stub_backend() -> StubBackend:
    return StubBackend()


@pytest.fixture
def stub_source() -> StubDataSource:
    return StubDataSource(bbox=BBox(-1, -2, -3, 4, 5, 6))
