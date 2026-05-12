"""Unit tests for STKO_to_python.cuts.plane."""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from STKO_to_python.cuts import Plane


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #
class TestConstruction:
    def test_normal_is_auto_normalized(self):
        p = Plane(point=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 5.0))
        assert p.normal == pytest.approx((0.0, 0.0, 1.0))
        assert np.linalg.norm(p.normal_arr) == pytest.approx(1.0)

    def test_point_preserved(self):
        p = Plane(point=(1.0, 2.0, 3.0), normal=(0.0, 0.0, 1.0))
        assert p.point == (1.0, 2.0, 3.0)
        np.testing.assert_allclose(p.point_arr, [1.0, 2.0, 3.0])

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError, match="nonzero"):
            Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 0.0))

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="length-3"):
            Plane(point=(0.0, 0.0), normal=(0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="length-3"):
            Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 1.0))

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError, match="finite"):
            Plane(point=(np.nan, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="finite"):
            Plane(point=(0.0, 0.0, 0.0), normal=(np.inf, 0.0, 1.0))


# ---------------------------------------------------------------------- #
# Constructors
# ---------------------------------------------------------------------- #
class TestHorizontal:
    def test_basic(self):
        p = Plane.horizontal(z=3.0)
        assert p.point == (0.0, 0.0, 3.0)
        assert p.normal == (0.0, 0.0, 1.0)

    def test_negative_z(self):
        p = Plane.horizontal(z=-1.5)
        assert p.point == (0.0, 0.0, -1.5)


class TestVertical:
    def test_x(self):
        p = Plane.vertical(axis="x", at=2.5)
        assert p.point == (2.5, 0.0, 0.0)
        assert p.normal == (1.0, 0.0, 0.0)

    def test_y(self):
        p = Plane.vertical(axis="y", at=-1.0)
        assert p.point == (0.0, -1.0, 0.0)
        assert p.normal == (0.0, 1.0, 0.0)

    def test_uppercase_accepted(self):
        p = Plane.vertical(axis="X", at=1.0)
        assert p.normal == (1.0, 0.0, 0.0)

    def test_z_rejected(self):
        # Vertical means "perpendicular to a horizontal axis"; Z is horizontal-plane normal.
        with pytest.raises(ValueError, match="'x' or 'y'"):
            Plane.vertical(axis="z", at=1.0)

    def test_garbage_axis_rejected(self):
        with pytest.raises(ValueError, match="'x' or 'y'"):
            Plane.vertical(axis="diag", at=1.0)


class TestFromThreePoints:
    def test_horizontal_recovered(self):
        # Three points in the z = 2 plane should yield a horizontal plane.
        p = Plane.from_three_points(
            (0.0, 0.0, 2.0), (1.0, 0.0, 2.0), (0.0, 1.0, 2.0),
        )
        np.testing.assert_allclose(p.normal_arr, [0.0, 0.0, 1.0], atol=1e-12)
        # signed distance of any third point in the plane should be 0.
        assert p.signed_distance(np.array([5.0, -3.0, 2.0])) == pytest.approx(0.0)

    def test_normal_hint_flips(self):
        # CCW order in the xy-plane gives +z normal.
        ccw = Plane.from_three_points(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        )
        np.testing.assert_allclose(ccw.normal_arr, [0.0, 0.0, 1.0], atol=1e-12)

        # Same points, but the hint asks for -z — normal should flip.
        flipped = Plane.from_three_points(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            normal_hint=(0.0, 0.0, -1.0),
        )
        np.testing.assert_allclose(flipped.normal_arr, [0.0, 0.0, -1.0], atol=1e-12)

    def test_normal_hint_keeps_when_aligned(self):
        p = Plane.from_three_points(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            normal_hint=(0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(p.normal_arr, [0.0, 0.0, 1.0], atol=1e-12)

    def test_collinear_raises(self):
        with pytest.raises(ValueError, match="collinear"):
            Plane.from_three_points(
                (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            )

    def test_coincident_raises(self):
        with pytest.raises(ValueError, match="collinear"):
            Plane.from_three_points(
                (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
            )


class TestHorizontalGrid:
    def test_returns_list_of_planes(self):
        planes = Plane.horizontal_grid([0.0, 3.0, 6.0, 9.0])
        assert len(planes) == 4
        for plane, z in zip(planes, [0.0, 3.0, 6.0, 9.0]):
            assert plane.point == (0.0, 0.0, z)
            assert plane.normal == (0.0, 0.0, 1.0)

    def test_empty(self):
        assert Plane.horizontal_grid([]) == []

    def test_accepts_generator(self):
        planes = Plane.horizontal_grid(float(i) for i in range(3))
        assert len(planes) == 3


# ---------------------------------------------------------------------- #
# Geometric primitives
# ---------------------------------------------------------------------- #
class TestSignedDistance:
    def test_scalar_point(self):
        p = Plane.horizontal(z=2.0)
        assert p.signed_distance(np.array([0.0, 0.0, 5.0])) == pytest.approx(3.0)
        assert p.signed_distance(np.array([1.0, 2.0, -1.0])) == pytest.approx(-3.0)

    def test_batched_points(self):
        p = Plane.horizontal(z=0.0)
        pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [3.0, 4.0, 0.5]])
        np.testing.assert_allclose(p.signed_distance(pts), [1.0, -2.0, 0.5])

    def test_returns_scalar_python_float_for_single(self):
        p = Plane.horizontal(z=0.0)
        d = p.signed_distance(np.array([0.0, 0.0, 1.0]))
        assert isinstance(d, float)

    def test_bad_shape(self):
        p = Plane.horizontal(z=0.0)
        with pytest.raises(ValueError, match="N, 3"):
            p.signed_distance(np.array([[0.0, 0.0]]))


class TestSide:
    def test_classification(self):
        p = Plane.horizontal(z=0.0)
        pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -2.0], [3.0, 4.0, 0.0]])
        np.testing.assert_array_equal(p.side(pts), [1, -1, 0])

    def test_tol_swallows_near_zero(self):
        p = Plane.horizontal(z=0.0)
        pts = np.array([[0.0, 0.0, 1e-10]])
        assert p.side(pts, tol=1e-9)[0] == 0

    def test_scalar_input_returns_scalar(self):
        p = Plane.horizontal(z=0.0)
        assert p.side(np.array([0.0, 0.0, 1.0])) == 1


class TestIntersectSegment:
    def test_crosses(self):
        p = Plane.horizontal(z=0.0)
        hit = p.intersect_segment((0.0, 0.0, -1.0), (0.0, 0.0, 3.0))
        assert hit is not None
        pt, t = hit
        np.testing.assert_allclose(pt, [0.0, 0.0, 0.0], atol=1e-12)
        assert t == pytest.approx(0.25)

    def test_oblique_crossing(self):
        p = Plane.horizontal(z=2.0)
        hit = p.intersect_segment((0.0, 0.0, 0.0), (4.0, 0.0, 4.0))
        assert hit is not None
        pt, t = hit
        np.testing.assert_allclose(pt, [2.0, 0.0, 2.0], atol=1e-12)
        assert t == pytest.approx(0.5)

    def test_endpoint_on_plane(self):
        p = Plane.horizontal(z=0.0)
        hit = p.intersect_segment((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        assert hit is not None
        _, t = hit
        assert t == pytest.approx(0.0)

    def test_segment_above_plane(self):
        p = Plane.horizontal(z=0.0)
        assert p.intersect_segment((0.0, 0.0, 1.0), (0.0, 0.0, 2.0)) is None

    def test_segment_below_plane(self):
        p = Plane.horizontal(z=0.0)
        assert p.intersect_segment((0.0, 0.0, -2.0), (0.0, 0.0, -0.5)) is None

    def test_parallel_segment_returns_none(self):
        # Segment lies parallel to the plane (constant signed distance).
        p = Plane.horizontal(z=0.0)
        assert p.intersect_segment((0.0, 0.0, 1.0), (5.0, 5.0, 1.0)) is None


class TestIntersectPolygon:
    def test_triangle_chord(self):
        # Vertical plane cutting a horizontal triangle from edge to edge.
        p = Plane.vertical(axis="x", at=0.5)
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        chord = p.intersect_polygon(tri)
        assert chord is not None
        q0, q1 = chord
        # The plane x = 0.5 cuts edges (0,0,0)->(1,0,0) at (0.5, 0, 0)
        # and (1,0,0)->(0,1,0) at (0.5, 0.5, 0).
        endpoints = sorted([tuple(np.round(q0, 6)), tuple(np.round(q1, 6))])
        expected = sorted([(0.5, 0.0, 0.0), (0.5, 0.5, 0.0)])
        assert endpoints == expected

    def test_quad_chord(self):
        # Horizontal plane cutting a vertical quad (a wall midsurface).
        p = Plane.horizontal(z=1.0)
        quad = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
        ])
        chord = p.intersect_polygon(quad)
        assert chord is not None
        q0, q1 = chord
        endpoints = sorted([tuple(np.round(q0, 6)), tuple(np.round(q1, 6))])
        expected = sorted([(0.0, 0.0, 1.0), (2.0, 0.0, 1.0)])
        assert endpoints == expected

    def test_polygon_above_plane(self):
        p = Plane.horizontal(z=0.0)
        tri = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]])
        assert p.intersect_polygon(tri) is None

    def test_polygon_below_plane(self):
        p = Plane.horizontal(z=0.0)
        tri = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, -2.0], [0.0, 1.0, -3.0]])
        assert p.intersect_polygon(tri) is None

    def test_grazes_single_vertex(self):
        # Plane touches only one vertex — no chord. Must be None.
        p = Plane.horizontal(z=0.0)
        tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        assert p.intersect_polygon(tri) is None

    def test_bad_shape_raises(self):
        p = Plane.horizontal(z=0.0)
        with pytest.raises(ValueError, match="M>=3"):
            p.intersect_polygon(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


# ---------------------------------------------------------------------- #
# Frozen / hashable / picklable
# ---------------------------------------------------------------------- #
class TestImmutability:
    def test_frozen(self):
        p = Plane.horizontal(z=3.0)
        with pytest.raises(Exception):
            p.point = (1.0, 1.0, 1.0)  # type: ignore[misc]

    def test_equality_after_normalization(self):
        a = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
        b = Plane(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 7.0))
        assert a == b

    def test_hashable(self):
        a = Plane.horizontal(z=3.0)
        b = Plane.horizontal(z=3.0)
        s = {a, b}
        assert len(s) == 1

    def test_pickle_roundtrip(self):
        original = Plane.from_three_points(
            (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),
            normal_hint=(0.0, 0.0, 1.0),
        )
        restored = pickle.loads(pickle.dumps(original))
        assert restored == original
        assert restored.normal == original.normal
