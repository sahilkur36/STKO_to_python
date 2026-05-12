"""Unit tests for STKO_to_python.cuts.kernels.solid.

Splits into:

- **Pure-math tests** (no .mpco fixture needed) covering the plane-vs-
  hex polygon math, the trilinear / linear inverse-shape-function
  helpers, and the brick stress-sampling weights.
- **Real-fixture tests** against ``solid_partition_example`` — the
  Brick + DispBeamColumn3d mixed fixture — covering the geometry phase,
  Newton's-3rd-law consistency, side-flip equivalence, and the
  mixed-kernel composition with the beam kernel.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCut, SectionCutSpec
from STKO_to_python.cuts.kernels.solid import (
    SOLID_ELEMENT_CLASSES,
    _BRICK_27IP_A,
    _brick_27ip_1d_lagrange,
    _brick_27ip_weights,
    _brick_8ip_weights,
    _CORNER_NODES_PER_CLASS,
    _NODES_PER_CLASS,
    _invert_brick_trilinear,
    _invert_tet_linear,
    _plane_polyhedron_polygon,
    _sample_solid_stress,
    _stress_voigt_to_tensor,
    _HEX_EDGES,
    _TET_EDGES,
    _strip_class_tag,
    compute_solid_cut,
    find_solid_intersections,
)


# ====================================================================== #
# Pure-math tests
# ====================================================================== #
class TestBrick8IpWeights:
    """Trilinear interpolation weights from the 8 Gauss IPs of a Brick
    to an arbitrary (ξ, η, ζ). IPs sit at ±1/√3.
    """

    def test_partition_of_unity_at_origin(self):
        w = _brick_8ip_weights(0.0, 0.0, 0.0)
        assert sum(w) == pytest.approx(1.0)

    def test_unit_response_at_each_ip(self):
        s = 1.0 / np.sqrt(3.0)
        # IP order is ξ-fastest, then η, then ζ.
        ips = [
            (-s, -s, -s), (+s, -s, -s), (-s, +s, -s), (+s, +s, -s),
            (-s, -s, +s), (+s, -s, +s), (-s, +s, +s), (+s, +s, +s),
        ]
        for k, (xi, eta, zeta) in enumerate(ips):
            w = _brick_8ip_weights(xi, eta, zeta)
            expected = np.eye(8)[k]
            np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_linear_field_recovered(self):
        # f(ξ, η, ζ) = 2 + 3ξ - 5η + 7ζ — trilinear weights reproduce
        # any tri-linear field exactly.
        s = 1.0 / np.sqrt(3.0)
        ips = np.array([
            [-s, -s, -s], [+s, -s, -s], [-s, +s, -s], [+s, +s, -s],
            [-s, -s, +s], [+s, -s, +s], [-s, +s, +s], [+s, +s, +s],
        ])
        f = lambda p: 2.0 + 3.0 * p[0] - 5.0 * p[1] + 7.0 * p[2]
        ip_values = np.array([f(p) for p in ips])
        for xi, eta, zeta in [
            (0.0, 0.0, 0.0), (0.3, -0.2, 0.5),
            (0.5, 0.5, -0.5), (-0.7, 0.7, 0.0),
        ]:
            w = _brick_8ip_weights(xi, eta, zeta)
            interp = float(w @ ip_values)
            assert interp == pytest.approx(f([xi, eta, zeta]), abs=1e-12)


class TestStressVoigtToTensor:
    def test_diagonal_components(self):
        s = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        T = _stress_voigt_to_tensor(s)
        np.testing.assert_allclose(np.diag(T), [1, 2, 3], atol=1e-12)
        # Off-diagonal entries are zero.
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            assert T[i, j] == 0.0
            assert T[j, i] == 0.0

    def test_off_diagonal_symmetry(self):
        s = np.array([0.0, 0.0, 0.0, 1.5, 2.5, 3.5])
        T = _stress_voigt_to_tensor(s)
        # sigma12 → T[0,1] = T[1,0]
        # sigma23 → T[1,2] = T[2,1]
        # sigma13 → T[0,2] = T[2,0]
        np.testing.assert_allclose(T[0, 1], 1.5, atol=1e-12)
        np.testing.assert_allclose(T[1, 0], 1.5, atol=1e-12)
        np.testing.assert_allclose(T[1, 2], 2.5, atol=1e-12)
        np.testing.assert_allclose(T[2, 1], 2.5, atol=1e-12)
        np.testing.assert_allclose(T[0, 2], 3.5, atol=1e-12)
        np.testing.assert_allclose(T[2, 0], 3.5, atol=1e-12)

    def test_traction_against_normal(self):
        # σ = diag(2, 3, 5); n = ê_z → t = σ·n = (0, 0, 5).
        s = np.array([2.0, 3.0, 5.0, 0.0, 0.0, 0.0])
        T = _stress_voigt_to_tensor(s)
        n = np.array([0.0, 0.0, 1.0])
        t = T @ n
        np.testing.assert_allclose(t, [0, 0, 5], atol=1e-12)

    def test_batched_shape(self):
        # Voigt input (n_steps, 6) → tensor (n_steps, 3, 3).
        s = np.zeros((4, 6))
        s[:, 0] = 1.0
        T = _stress_voigt_to_tensor(s)
        assert T.shape == (4, 3, 3)


class TestInvertBrickTrilinear:
    def test_invert_at_corners(self):
        # Reference cube [-1, 1]^3 in physical coords. Each corner maps
        # to its natural coords.
        node_coords = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float)
        for k, sign in enumerate([
            (-1, -1, -1), (+1, -1, -1), (+1, +1, -1), (-1, +1, -1),
            (-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1),
        ]):
            recovered = _invert_brick_trilinear(node_coords[k], node_coords)
            np.testing.assert_allclose(recovered, sign, atol=1e-9)

    def test_invert_at_centroid(self):
        node_coords = np.array([
            [0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0],
            [0, 0, 5], [4, 0, 5], [4, 3, 5], [0, 3, 5],
        ], dtype=float)
        centroid = node_coords.mean(axis=0)
        recovered = _invert_brick_trilinear(centroid, node_coords)
        np.testing.assert_allclose(recovered, [0.0, 0.0, 0.0], atol=1e-9)

    def test_invert_at_face_midpoint(self):
        # Unit cube, midpoint of the +ζ face → (0, 0, +1).
        node_coords = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float)
        top_face_mid = np.array([0.0, 0.0, 1.0])
        recovered = _invert_brick_trilinear(top_face_mid, node_coords)
        np.testing.assert_allclose(recovered, [0.0, 0.0, 1.0], atol=1e-9)


class TestInvertTetLinear:
    def test_invert_at_corners(self):
        # Reference unit tet — nodes at (0,0,0), (1,0,0), (0,1,0), (0,0,1).
        node_coords = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        natural_at_node = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        for k, expected in enumerate(natural_at_node):
            recovered = _invert_tet_linear(node_coords[k], node_coords)
            np.testing.assert_allclose(recovered, expected, atol=1e-12)

    def test_invert_at_centroid(self):
        node_coords = np.array([
            [0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4],
        ], dtype=float)
        centroid = node_coords.mean(axis=0)
        recovered = _invert_tet_linear(centroid, node_coords)
        np.testing.assert_allclose(recovered, [0.25, 0.25, 0.25], atol=1e-12)


class TestPlanePolyhedronPolygon:
    """Plane-vs-convex-polyhedron polygon math against the standard
    8-vertex unit cube.
    """

    @pytest.fixture
    def cube(self):
        return np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float)

    @pytest.fixture
    def horizontal_plane(self):
        # z = 0 plane with global Z normal; e1 = X, e2 = Y.
        return (
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    def test_horizontal_through_centroid_gives_square(self, cube, horizontal_plane):
        p, n, e1, e2 = horizontal_plane
        polygon = _plane_polyhedron_polygon(cube, _HEX_EDGES, p, n, e1, e2)
        assert polygon is not None
        assert polygon.shape == (4, 3)
        # All vertices on the plane.
        assert np.allclose(polygon[:, 2], 0.0, atol=1e-9)
        # Each vertex has |x|=|y|=1 (corners of the unit square).
        xy = polygon[:, :2]
        assert np.allclose(np.sort(np.abs(xy).ravel()), [1.0] * 8, atol=1e-9)

    def test_above_cube_returns_none(self, cube):
        polygon = _plane_polyhedron_polygon(
            cube, _HEX_EDGES,
            np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        assert polygon is None

    def test_below_cube_returns_none(self, cube):
        polygon = _plane_polyhedron_polygon(
            cube, _HEX_EDGES,
            np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        assert polygon is None

    def test_diagonal_cut_can_produce_hexagon(self, cube):
        # Plane perpendicular to (1, 1, 1) through the origin cuts the
        # cube in a regular hexagon.
        n = np.array([1.0, 1.0, 1.0])
        n /= np.linalg.norm(n)
        # Build an orthonormal in-plane basis.
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = a - n * (a @ n)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        polygon = _plane_polyhedron_polygon(
            cube, _HEX_EDGES, np.array([0.0, 0.0, 0.0]), n, e1, e2,
        )
        assert polygon is not None
        # 6 unique points expected (regular hexagon).
        assert polygon.shape == (6, 3)

    def test_face_aligned_cut_returns_four_vertices(self, cube):
        # Plane at z = +1 grazes the top face — on-plane vertices yield
        # the 4 corners of the top face.
        polygon = _plane_polyhedron_polygon(
            cube, _HEX_EDGES,
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        assert polygon is not None
        assert polygon.shape == (4, 3)
        np.testing.assert_allclose(polygon[:, 2], 1.0, atol=1e-9)

    def test_tet_cut_through_centroid(self):
        # Reference tet at (0,0,0), (1,0,0), (0,1,0), (0,0,1). Plane
        # x + y + z = 0.5 cuts through the interior; intersection is a
        # triangle.
        tet = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=float)
        n = np.array([1.0, 1.0, 1.0])
        n /= np.linalg.norm(n)
        p = 0.5 * n
        a = np.array([1.0, 0.0, 0.0])
        e1 = a - n * (a @ n)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        polygon = _plane_polyhedron_polygon(tet, _TET_EDGES, p, n, e1, e2)
        assert polygon is not None
        # Triangle: exactly 3 vertices.
        assert polygon.shape == (3, 3)


class TestSampleSolidStress:
    def test_brick_8ip_sample_recovers_ip_value(self):
        s = 1.0 / np.sqrt(3.0)
        # 1 step, 8 IPs, 6 stress components. Distinct values so a
        # wrong index pulls obvious garbage.
        stress = np.zeros((1, 8, 6))
        for k in range(8):
            for c in range(6):
                stress[0, k, c] = 10.0 * k + c
        # IP 5 is (+s, -s, +s).
        result = _sample_solid_stress(stress, +s, -s, +s, "Brick", 8)
        np.testing.assert_allclose(result[0], stress[0, 5, :], atol=1e-12)

    def test_single_ip_broadcasts(self):
        stress = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])
        # Any (ξ, η, ζ) returns the single IP value.
        result = _sample_solid_stress(stress, 0.5, -0.3, 0.7, "Brick", 1)
        np.testing.assert_allclose(result[0], stress[0, 0, :], atol=1e-12)


class TestRegistryHelpers:
    def test_strip_class_tag_decorated(self):
        assert _strip_class_tag("56-Brick") == "Brick"
        assert _strip_class_tag("203-ASDShellQ4") == "ASDShellQ4"

    def test_strip_class_tag_undecorated(self):
        assert _strip_class_tag("Brick") == "Brick"

    def test_registry_contents(self):
        # The four documented v1.7 solid classes are in the registry.
        for cls in ("Brick", "BbarBrick", "SSPbrick", "FourNodeTetrahedron"):
            assert cls in SOLID_ELEMENT_CLASSES

    def test_higher_order_hex_registered(self):
        # v1.8: Brick20 and Brick27 (plus the OpenSees aliases
        # TwentyNodeBrick / TwentySevenNodeBrick) are recognised.
        for cls in ("Brick20", "Brick27",
                    "TwentyNodeBrick", "TwentySevenNodeBrick"):
            assert cls in SOLID_ELEMENT_CLASSES

    def test_higher_order_node_counts(self):
        # 20-node hex and 27-node hex have full connectivity counts but
        # share the same 8 corner nodes for the geometry phase.
        assert _NODES_PER_CLASS["Brick20"] == 20
        assert _NODES_PER_CLASS["TwentyNodeBrick"] == 20
        assert _NODES_PER_CLASS["Brick27"] == 27
        assert _NODES_PER_CLASS["TwentySevenNodeBrick"] == 27
        for cls in ("Brick20", "TwentyNodeBrick", "Brick27", "TwentySevenNodeBrick"):
            assert _CORNER_NODES_PER_CLASS[cls] == 8


# ====================================================================== #
# Higher-order (v1.8) — 27-IP triquadratic sampling
# ====================================================================== #
class TestBrick27IpLagrange1d:
    """1-D quadratic Lagrange basis at the three Gauss-Legendre 3-pt
    nodes ``(-a, 0, +a)`` with ``a = √(3/5)``.
    """

    def test_unit_response_at_each_node(self):
        a = _BRICK_27IP_A
        # L_minus(-a) = 1; L_0(0) = 1; L_plus(+a) = 1.
        l_at_minus = _brick_27ip_1d_lagrange(-a)
        l_at_zero = _brick_27ip_1d_lagrange(0.0)
        l_at_plus = _brick_27ip_1d_lagrange(+a)
        np.testing.assert_allclose(l_at_minus, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(l_at_zero, [0.0, 1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(l_at_plus, [0.0, 0.0, 1.0], atol=1e-12)

    def test_partition_of_unity_at_arbitrary_x(self):
        for x in [-0.5, -0.1, 0.0, 0.2, 0.7]:
            s = sum(_brick_27ip_1d_lagrange(x))
            assert s == pytest.approx(1.0, abs=1e-12)


class TestBrick27IpWeights:
    """Triquadratic interpolation weights at the 27 IPs of a Brick
    (3×3×3 Gauss-Legendre).
    """

    def test_partition_of_unity_at_origin(self):
        w = _brick_27ip_weights(0.0, 0.0, 0.0)
        assert sum(w) == pytest.approx(1.0)

    def test_unit_response_at_each_ip(self):
        a = _BRICK_27IP_A
        nodes = [-a, 0.0, +a]
        # IP enumeration is xi-fastest, then eta, then zeta.
        for nzeta in range(3):
            for neta in range(3):
                for nxi in range(3):
                    idx = (nzeta * 3 + neta) * 3 + nxi
                    w = _brick_27ip_weights(
                        nodes[nxi], nodes[neta], nodes[nzeta],
                    )
                    expected = np.eye(27)[idx]
                    np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_quadratic_field_recovered(self):
        # f(ξ, η, ζ) = a quadratic polynomial in (ξ, η, ζ) — triquadratic
        # Lagrange interpolation reproduces it exactly. Build a separable
        # quadratic so it's clearly within the basis.
        a = _BRICK_27IP_A
        nodes = [-a, 0.0, +a]
        # f(x, y, z) = (1 + 2 x + 3 x²) * (4 - y + y²) * (5 + 6 z² - z)
        f = lambda p: (
            (1.0 + 2.0 * p[0] + 3.0 * p[0] ** 2)
            * (4.0 - p[1] + p[1] ** 2)
            * (5.0 + 6.0 * p[2] ** 2 - p[2])
        )
        ip_values = np.empty(27)
        for nzeta in range(3):
            for neta in range(3):
                for nxi in range(3):
                    idx = (nzeta * 3 + neta) * 3 + nxi
                    p = [nodes[nxi], nodes[neta], nodes[nzeta]]
                    ip_values[idx] = f(p)
        for xi, eta, zeta in [
            (0.0, 0.0, 0.0), (0.3, -0.2, 0.5), (-0.5, 0.5, -0.5),
        ]:
            w = _brick_27ip_weights(xi, eta, zeta)
            interp = float(w @ ip_values)
            assert interp == pytest.approx(f([xi, eta, zeta]), abs=1e-10)


class TestSample27IpSolidStress:
    def test_dispatch_to_27ip_sampler(self):
        a = _BRICK_27IP_A
        # 1 step, 27 IPs, 6 stress components. Distinct values so a
        # wrong index pulls obvious garbage.
        stress = np.zeros((1, 27, 6))
        for k in range(27):
            stress[0, k, 0] = float(k)
        # Pick IP idx = (2*3 + 1)*3 + 0 = 21 → coords (-a, 0, +a).
        result = _sample_solid_stress(
            stress, -a, 0.0, +a, "Brick27", 27,
        )
        np.testing.assert_allclose(result[0, 0], 21.0, atol=1e-12)


class TestHigherOrderHexGeometryPhase:
    """The plane-vs-polyhedron polygon math runs on the 8 corner nodes
    of a higher-order hex; midpoint / face / center nodes are ignored
    at the geometry layer.
    """

    @pytest.fixture
    def hex20_unit_cube(self):
        """Standard 20-node serendipity hex: 8 corners + 12 edge
        midpoints, sitting on a [-1, 1]^3 cube.
        """
        corners = np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ], dtype=float)
        edge_midpoints = np.array([
            [0, -1, -1], [+1, 0, -1], [0, +1, -1], [-1, 0, -1],
            [0, -1, +1], [+1, 0, +1], [0, +1, +1], [-1, 0, +1],
            [-1, -1, 0], [+1, -1, 0], [+1, +1, 0], [-1, +1, 0],
        ], dtype=float)
        return np.concatenate([corners, edge_midpoints], axis=0)

    def test_plane_polyhedron_polygon_with_corners_only(self, hex20_unit_cube):
        """Slicing the 20-node hex's coords to the 8 corners and running
        the plane-vs-polyhedron polygon math gives the same result as
        the 8-node hex case.
        """
        corner_coords = hex20_unit_cube[:8]
        polygon = _plane_polyhedron_polygon(
            corner_coords, _HEX_EDGES,
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
        )
        assert polygon is not None
        assert polygon.shape == (4, 3)
        # All vertices on z=0.
        np.testing.assert_allclose(polygon[:, 2], 0.0, atol=1e-9)


# ====================================================================== #
# Real-fixture tests (solid_partition_example)
# ====================================================================== #
@pytest.fixture
def solid_ds(solid_partition_dir) -> MPCODataSet:
    return MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)


@pytest.fixture
def all_solid_eids(solid_ds) -> tuple[int, ...]:
    df = solid_ds.elements_info["dataframe"]
    base = {c for c in SOLID_ELEMENT_CLASSES}
    is_solid = df["element_type"].map(
        lambda s: any(c == _strip_class_tag(s) for c in base)
    )
    return tuple(int(x) for x in df.loc[is_solid, "element_id"].tolist())


class TestSolidFixtureRegistry:
    """Sanity: solid_partition_example actually carries a Brick mesh."""

    def test_fixture_has_brick(self, solid_ds):
        types = list(solid_ds.unique_element_types)
        assert any("Brick" in t for t in types)


class TestFindSolidIntersections:
    def test_horizontal_cut_finds_some_solids(self, solid_ds, all_solid_eids):
        # The brick column / block spans some z-range. Cut somewhere
        # interior.
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        plane = Plane.horizontal(z=z_mid)
        spec = SectionCutSpec(plane=plane, element_ids=all_solid_eids)
        ixs = find_solid_intersections(solid_ds, spec)
        assert len(ixs) > 0
        for ix in ixs:
            assert ix.n_vertices >= 3
            assert ix.polygon_area > 1e-12

    def test_cut_far_above_returns_empty(self, solid_ds, all_solid_eids):
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_range[1] + 1e6),
            element_ids=all_solid_eids,
        )
        assert find_solid_intersections(solid_ds, spec) == []

    def test_cut_far_below_returns_empty(self, solid_ds, all_solid_eids):
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_range[0] - 1e6),
            element_ids=all_solid_eids,
        )
        assert find_solid_intersections(solid_ds, spec) == []


class TestSolidCutEndToEnd:
    def test_cut_returns_finite_force(self, solid_ds, all_solid_eids):
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_mid),
            element_ids=all_solid_eids,
        )
        cut = SectionCut.compute(spec, solid_ds, model_stage="MODEL_STAGE[1]")
        assert not cut.is_empty
        assert cut.F.shape == (cut.n_steps, 3)
        assert np.all(np.isfinite(cut.F))
        assert np.all(np.isfinite(cut.M))

    def test_consistency_check_passes(self, solid_ds, all_solid_eids):
        """Newton's 3rd law: positive + negative side cuts sum to zero.

        Independent of the load pattern — any honest kernel must
        satisfy this by construction.
        """
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_mid),
            element_ids=all_solid_eids,
        )
        cut = SectionCut.compute(spec, solid_ds, model_stage="MODEL_STAGE[1]")
        # Use a generous absolute tolerance — the stress magnitudes on
        # a real model can be order-of-magnitude ~1e6 in SI units.
        scale = max(
            1.0,
            float(np.max(np.abs(cut.F))),
            float(np.max(np.abs(cut.M))),
        )
        ok, residual = cut.consistency_check(
            solid_ds, atol=scale * 1e-3, rtol=1e-6,
        )
        assert ok, f"Residual max={np.max(np.abs(residual))} vs tol={scale * 1e-3}"

    def test_negative_side_flips_resultant(self, solid_ds, all_solid_eids):
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        plane = Plane.horizontal(z=z_mid)
        pos = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_solid_eids, side="positive"),
            solid_ds, model_stage="MODEL_STAGE[1]",
        )
        neg = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_solid_eids, side="negative"),
            solid_ds, model_stage="MODEL_STAGE[1]",
        )
        scale = max(1.0, float(np.max(np.abs(pos.F))))
        np.testing.assert_allclose(neg.F, -pos.F, atol=scale * 1e-3)

    def test_per_solid_F_has_real_values(self, solid_ds, all_solid_eids):
        """The per-element force map carries actual stress data — guard
        against a column-name typo silently zeroing the read.
        """
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_mid),
            element_ids=all_solid_eids,
        )
        cut = SectionCut.compute(spec, solid_ds, model_stage="MODEL_STAGE[1]")
        assert cut.per_solid_F  # at least one solid contributed
        # At least one solid has a non-vanishing force somewhere in the
        # time history.
        max_F = max(
            float(np.max(np.abs(Fi))) for Fi in cut.per_solid_F.values()
        )
        assert max_F > 0.0

    def test_repr_includes_solid_count(self, solid_ds, all_solid_eids):
        z_range = _brick_z_range(solid_ds, all_solid_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=z_mid),
            element_ids=all_solid_eids,
        )
        cut = SectionCut.compute(spec, solid_ds, model_stage="MODEL_STAGE[1]")
        text = repr(cut)
        n_total = len(cut.solid_intersections) + len(cut.intersections) + len(cut.shell_intersections)
        assert f"n_intersections={n_total}" in text


class TestComposedBeamSolidCut:
    """solid_partition_example mixes Brick continuum with
    DispBeamColumn3d. A cut filtered to ALL elements (not just solids)
    should produce a single composed result with contributions from
    both kernels.
    """

    def test_consistency_check_on_composed_cut(self, solid_ds):
        all_eids = tuple(
            int(x) for x in solid_ds.elements_info["dataframe"]["element_id"].tolist()
        )
        z_range = _brick_z_range(solid_ds, all_eids)
        z_mid = 0.5 * (z_range[0] + z_range[1])
        spec = SectionCutSpec(plane=Plane.horizontal(z=z_mid), element_ids=all_eids)
        cut = SectionCut.compute(spec, solid_ds, model_stage="MODEL_STAGE[1]")
        scale = max(
            1.0,
            float(np.max(np.abs(cut.F))),
            float(np.max(np.abs(cut.M))),
        )
        ok, residual = cut.consistency_check(
            solid_ds, atol=scale * 1e-3, rtol=1e-6,
        )
        assert ok, (
            f"Composed (beam + solid) consistency_check failed; max residual "
            f"{np.max(np.abs(residual))} vs tol {scale * 1e-3}."
        )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _brick_z_range(dataset: MPCODataSet, element_ids) -> tuple[float, float]:
    """Return (z_min, z_max) over the nodes of every brick in the filter."""
    df_e = dataset.elements_info["dataframe"]
    df_n = dataset.nodes_info["dataframe"]
    node_z = dict(zip(df_n["node_id"].tolist(), df_n["z"].tolist()))
    eid_set = set(int(e) for e in element_ids)
    base = {c for c in SOLID_ELEMENT_CLASSES}
    z_vals: list[float] = []
    for r in df_e.itertuples(index=False):
        if int(r.element_id) not in eid_set:
            continue
        if _strip_class_tag(str(r.element_type)) not in base:
            continue
        for nid in r.node_list:
            zv = node_z.get(int(nid))
            if zv is not None:
                z_vals.append(float(zv))
    if not z_vals:
        return (0.0, 0.0)
    return float(min(z_vals)), float(max(z_vals))
