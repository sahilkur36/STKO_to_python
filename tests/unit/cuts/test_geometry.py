"""Unit tests for STKO_to_python.cuts.geometry — plane projection +
Cyrus-Beck convex-polygon clipping.

These helpers underpin the ``bounding_polygon`` feature added in v1.6.
They run without any .mpco fixture — purely synthetic geometry.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.cuts import Plane
from STKO_to_python.cuts.geometry import (
    PolygonClipper,
    _clip_point_inside_2d,
    _clip_segment_against_convex_polygon_2d,
    _plane_basis,
    _polygon_edge_normals,
    _polygon_signed_area_2d,
    _project_to_plane_basis,
    is_convex_2d,
    prepare_clipper,
)


# ---------------------------------------------------------------------- #
# Plane basis + projection
# ---------------------------------------------------------------------- #
class TestPlaneBasis:
    def test_orthonormality_horizontal(self):
        plane = Plane.horizontal(z=0.0)
        e1, e2 = _plane_basis(plane)
        # Orthogonal to normal and to each other.
        assert abs(e1 @ plane.normal_arr) < 1e-12
        assert abs(e2 @ plane.normal_arr) < 1e-12
        assert abs(e1 @ e2) < 1e-12
        # Unit length.
        assert abs(np.linalg.norm(e1) - 1.0) < 1e-12
        assert abs(np.linalg.norm(e2) - 1.0) < 1e-12

    def test_orthonormality_oblique(self):
        plane = Plane.from_three_points((0, 0, 0), (1, 1, 0), (1, 0, 1))
        e1, e2 = _plane_basis(plane)
        assert abs(e1 @ plane.normal_arr) < 1e-12
        assert abs(e2 @ plane.normal_arr) < 1e-12
        assert abs(e1 @ e2) < 1e-12

    def test_basis_deterministic(self):
        # Same plane should yield the same basis on repeat call.
        plane = Plane.horizontal(z=2.5)
        b1 = _plane_basis(plane)
        b2 = _plane_basis(plane)
        np.testing.assert_array_equal(b1[0], b2[0])
        np.testing.assert_array_equal(b1[1], b2[1])


class TestProjectToPlaneBasis:
    def test_anchor_projects_to_origin(self):
        plane = Plane.horizontal(z=5.0)
        anchor_3d = np.array(plane.point)
        pt_2d = _project_to_plane_basis(anchor_3d, plane)
        np.testing.assert_allclose(pt_2d, [0.0, 0.0], atol=1e-12)

    def test_batch_shape_preserved(self):
        plane = Plane.horizontal(z=0.0)
        pts = np.array([[1, 0, 0], [2, 3, 0], [-1, -2, 0]], dtype=float)
        proj = _project_to_plane_basis(pts, plane)
        assert proj.shape == (3, 2)

    def test_scalar_input_returns_1d(self):
        plane = Plane.horizontal(z=0.0)
        single = _project_to_plane_basis(np.array([1.0, 2.0, 0.0]), plane)
        assert single.shape == (2,)

    def test_distances_preserved(self):
        # Projection must be isometric on the plane.
        plane = Plane.from_three_points((0, 0, 0), (1, 0, 0), (0, 1, 0))
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        a2 = _project_to_plane_basis(a, plane)
        b2 = _project_to_plane_basis(b, plane)
        dist_3d = float(np.linalg.norm(a - b))
        dist_2d = float(np.linalg.norm(a2 - b2))
        assert abs(dist_3d - dist_2d) < 1e-12


# ---------------------------------------------------------------------- #
# Convexity / area / edge normals
# ---------------------------------------------------------------------- #
class TestIsConvex2D:
    def test_ccw_square_convex(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert is_convex_2d(sq)

    def test_cw_square_convex(self):
        sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert is_convex_2d(sq)

    def test_triangle_convex(self):
        tri = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        assert is_convex_2d(tri)

    def test_concave_quad_rejected(self):
        # Arrowhead / dart concave at index 2.
        dart = np.array([[0, 0], [2, 0], [1, 0.5], [1, 2]], dtype=float)
        assert not is_convex_2d(dart)

    def test_collinear_extra_vertex_allowed(self):
        # Convex pentagon with a redundant midpoint vertex on one edge.
        poly = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        assert is_convex_2d(poly)


class TestPolygonSignedArea:
    def test_unit_square_ccw_area_one(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        assert _polygon_signed_area_2d(sq) == pytest.approx(1.0)

    def test_unit_square_cw_area_neg_one(self):
        sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        assert _polygon_signed_area_2d(sq) == pytest.approx(-1.0)


class TestPolygonEdgeNormals:
    def test_unit_square_ccw_normals_point_inward(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        n = _polygon_edge_normals(sq)
        # Edge 0 is the bottom; inward = +y.
        np.testing.assert_allclose(n[0], [0, 1], atol=1e-12)
        # Edge 1 right; inward = -x.
        np.testing.assert_allclose(n[1], [-1, 0], atol=1e-12)
        # Edge 2 top; inward = -y.
        np.testing.assert_allclose(n[2], [0, -1], atol=1e-12)
        # Edge 3 left; inward = +x.
        np.testing.assert_allclose(n[3], [1, 0], atol=1e-12)

    def test_unit_square_cw_normals_also_point_inward(self):
        sq = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
        n = _polygon_edge_normals(sq)
        # Now edge 0 is the LEFT side; inward = +x.
        np.testing.assert_allclose(n[0], [1, 0], atol=1e-12)


# ---------------------------------------------------------------------- #
# Point-in-polygon
# ---------------------------------------------------------------------- #
class TestClipPointInside2D:
    @pytest.fixture
    def square(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_interior_inside(self, square):
        assert _clip_point_inside_2d(np.array([0.5, 0.5]), square)

    def test_exterior_outside(self, square):
        assert not _clip_point_inside_2d(np.array([2.0, 0.5]), square)
        assert not _clip_point_inside_2d(np.array([-1.0, 0.5]), square)
        assert not _clip_point_inside_2d(np.array([0.5, 2.0]), square)

    def test_on_edge_counts_as_inside(self, square):
        assert _clip_point_inside_2d(np.array([0.5, 0.0]), square)
        assert _clip_point_inside_2d(np.array([1.0, 0.5]), square)

    def test_vertex_counts_as_inside(self, square):
        assert _clip_point_inside_2d(np.array([0.0, 0.0]), square)
        assert _clip_point_inside_2d(np.array([1.0, 1.0]), square)


# ---------------------------------------------------------------------- #
# Cyrus-Beck segment clipping
# ---------------------------------------------------------------------- #
class TestClipSegment2D:
    @pytest.fixture
    def square(self):
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    def test_segment_through_polygon(self, square):
        # From outside-left to outside-right at y=0.5.
        q = _clip_segment_against_convex_polygon_2d(
            np.array([-1.0, 0.5]), np.array([2.0, 0.5]), square,
        )
        assert q is not None
        q0, q1, t0, t1 = q
        np.testing.assert_allclose(q0, [0.0, 0.5])
        np.testing.assert_allclose(q1, [1.0, 0.5])
        # Parameter values: enter at t=1/3, exit at t=2/3.
        assert t0 == pytest.approx(1.0 / 3.0)
        assert t1 == pytest.approx(2.0 / 3.0)

    def test_segment_fully_inside(self, square):
        q = _clip_segment_against_convex_polygon_2d(
            np.array([0.2, 0.2]), np.array([0.8, 0.8]), square,
        )
        q0, q1, t0, t1 = q
        np.testing.assert_allclose(q0, [0.2, 0.2])
        np.testing.assert_allclose(q1, [0.8, 0.8])
        assert t0 == 0.0 and t1 == 1.0

    def test_segment_fully_outside(self, square):
        q = _clip_segment_against_convex_polygon_2d(
            np.array([2.0, 2.0]), np.array([3.0, 3.0]), square,
        )
        assert q is None

    def test_segment_starting_inside(self, square):
        q = _clip_segment_against_convex_polygon_2d(
            np.array([0.5, 0.5]), np.array([2.0, 0.5]), square,
        )
        q0, q1, t0, t1 = q
        np.testing.assert_allclose(q0, [0.5, 0.5])
        np.testing.assert_allclose(q1, [1.0, 0.5])
        assert t0 == 0.0

    def test_segment_parallel_to_edge_outside(self, square):
        # Segment running along y=2 (above the square) — never enters.
        q = _clip_segment_against_convex_polygon_2d(
            np.array([-1.0, 2.0]), np.array([2.0, 2.0]), square,
        )
        assert q is None

    def test_segment_grazing_edge(self, square):
        # Segment along the top edge y=1; trims to that edge.
        q = _clip_segment_against_convex_polygon_2d(
            np.array([-1.0, 1.0]), np.array([2.0, 1.0]), square,
        )
        # Could be None (degenerate zero-area clip) or the edge itself.
        # Either is acceptable per the spec; just check it doesn't
        # produce a segment outside the polygon.
        if q is not None:
            q0, q1, _, _ = q
            assert _clip_point_inside_2d(q0, square, tol=1e-6)
            assert _clip_point_inside_2d(q1, square, tol=1e-6)


# ---------------------------------------------------------------------- #
# PolygonClipper end-to-end
# ---------------------------------------------------------------------- #
class TestPolygonClipper3D:
    @pytest.fixture
    def horizontal_clipper(self):
        plane = Plane.horizontal(z=0.0)
        square = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        return prepare_clipper(plane, square)

    def test_clipper_attributes(self, horizontal_clipper):
        # Polygon projected to (M, 2).
        assert horizontal_clipper.polygon_2d.shape == (4, 2)
        # Basis vectors are unit.
        assert abs(np.linalg.norm(horizontal_clipper.e1) - 1.0) < 1e-12
        assert abs(np.linalg.norm(horizontal_clipper.e2) - 1.0) < 1e-12

    def test_clipper_point_inside_origin(self, horizontal_clipper):
        assert horizontal_clipper.point_inside(np.array([0.0, 0.0, 0.0]))
        assert not horizontal_clipper.point_inside(np.array([2.0, 0.0, 0.0]))

    def test_clipper_segment_3d(self, horizontal_clipper):
        p0 = np.array([-2.0, 0.0, 0.0])
        p1 = np.array([2.0, 0.0, 0.0])
        clipped = horizontal_clipper.clip_segment(p0, p1)
        assert clipped is not None
        q0, q1, t0, t1 = clipped
        np.testing.assert_allclose(q0, [-1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(q1, [1.0, 0.0, 0.0], atol=1e-9)
        # Parameter values match the 3-D interpolation:
        # q = p0 + t (p1 - p0), so q at t=0.25 is x=-1, at t=0.75 is x=1.
        assert t0 == pytest.approx(0.25)
        assert t1 == pytest.approx(0.75)

    def test_clipper_segment_misses(self, horizontal_clipper):
        # Both endpoints in the polygon's exterior on the same side.
        p0 = np.array([2.0, 0.0, 0.0])
        p1 = np.array([3.0, 0.0, 0.0])
        assert horizontal_clipper.clip_segment(p0, p1) is None

    def test_oblique_plane(self):
        # 45° plane through origin.
        plane = Plane(point=(0, 0, 0), normal=(1, 0, 1))
        # Square on that plane: vertices at distances ±1 along two
        # orthogonal in-plane directions. Pick (0, 1, 0) and
        # (1, 0, -1)/√2 as orthonormal in-plane axes.
        sq3 = (
            (1 / np.sqrt(2), 1, -1 / np.sqrt(2)),
            (-1 / np.sqrt(2), 1, 1 / np.sqrt(2)),
            (-1 / np.sqrt(2), -1, 1 / np.sqrt(2)),
            (1 / np.sqrt(2), -1, -1 / np.sqrt(2)),
        )
        clipper = prepare_clipper(plane, sq3)
        # Origin must be inside the square; (10, 0, -10) lies on the
        # plane far outside.
        assert clipper.point_inside(np.array([0.0, 0.0, 0.0]))
        assert not clipper.point_inside(np.array([10 / np.sqrt(2), 0.0, -10 / np.sqrt(2)]))
