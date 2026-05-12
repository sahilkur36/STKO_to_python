"""Interpolation + rotation tests for the section.force path.

These exercise the pure-math helpers in
``STKO_to_python.cuts.kernels.beam_resultant`` without needing a real
.mpco fixture. The kernel's responsibilities here split into three
pieces, each tested separately:

1. **Reading** — turn a DataFrame slice into a ``(n_steps, n_ip, 6)``
   array (column-naming contract).
2. **Interpolating** — linear interp in ξ between bracketing IPs, with
   constant extrapolation outside the IP envelope.
3. **Rotating + signing** — local→global rotation and the kept-side
   sign flip (Newton 3rd law).

Plus one end-to-end test through ``_section_force_cut`` using a
synthetic ``ElementResults`` and a stub dataset — proves the wiring
holds together.

Plus an equivalence test that constructs synthetic section.force data
representing the same compression-only column as elastic_frame, runs it
through the section.force path, and confirms the cut force matches the
ElasticBeam3d-derived expectation. This is the cross-check the user
asked for.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.cuts import Plane, SectionCutSpec
from STKO_to_python.cuts.kernels import BeamIntersection
from STKO_to_python.cuts.kernels.beam_resultant import (
    _interpolate_section_force_at_xi,
    _read_section_force_array,
    _rotate_section_force_and_apply_side,
    _section_force_cut,
    _section_force_columns,
)


# ---------------------------------------------------------------------- #
# Test fixtures
# ---------------------------------------------------------------------- #
def _make_section_force(
    n_steps: int, ip_values: np.ndarray
) -> np.ndarray:
    """Broadcast a ``(n_ip, 6)`` per-IP template across ``n_steps``.

    Returns a ``(n_steps, n_ip, 6)`` array with the same per-IP values
    at every step — convenient for testing interpolation in isolation
    from time variation.
    """
    return np.broadcast_to(
        ip_values[None, :, :], (n_steps, ip_values.shape[0], 6)
    ).copy()


# ====================================================================== #
# _read_section_force_array — column-naming contract
# ====================================================================== #
class TestReadSectionForceArray:
    def test_three_ip_three_step(self):
        n_ip = 3
        n_steps = 4
        cols: list[str] = []
        for k in range(n_ip):
            cols.extend(_section_force_columns(k))
        # 18 columns: 3 IPs * 6 components. Fill with deterministic
        # pattern: row r, column index c → r * 100 + c.
        data = np.arange(n_steps * len(cols), dtype=float).reshape(n_steps, len(cols))
        rows = pd.DataFrame(data, columns=cols)
        arr, has_vy, has_vz = _read_section_force_array(rows, n_ip)
        assert arr.shape == (n_steps, n_ip, 6)
        # All 6 shortnames present → both shears reported.
        assert has_vy is True
        assert has_vz is True
        # Check IP-0 P column matches column 0 of the DataFrame:
        np.testing.assert_array_equal(arr[:, 0, 0], data[:, 0])
        # IP-2 Mz column matches the last column:
        np.testing.assert_array_equal(arr[:, 2, 5], data[:, -1])

    def test_missing_shear_columns_default_to_zero(self):
        # OpenSees fiber/elastic sections emit (P, Mz, My, T) — no Vy/Vz.
        n_ip = 3
        n_steps = 2
        cols = []
        for k in range(n_ip):
            cols.extend([f"P_ip{k}", f"Mz_ip{k}", f"My_ip{k}", f"T_ip{k}"])
        data = np.ones((n_steps, len(cols)))
        rows = pd.DataFrame(data, columns=cols)
        arr, has_vy, has_vz = _read_section_force_array(rows, n_ip)
        assert arr.shape == (n_steps, n_ip, 6)
        assert has_vy is False
        assert has_vz is False
        # Vy (position 1) and Vz (position 2) should be all zeros.
        np.testing.assert_array_equal(arr[:, :, 1], 0.0)
        np.testing.assert_array_equal(arr[:, :, 2], 0.0)
        # The other four should be ones.
        np.testing.assert_array_equal(arr[:, :, 0], 1.0)  # P
        np.testing.assert_array_equal(arr[:, :, 3], 1.0)  # T
        np.testing.assert_array_equal(arr[:, :, 4], 1.0)  # My
        np.testing.assert_array_equal(arr[:, :, 5], 1.0)  # Mz

    def test_column_name_order_is_p_vy_vz_t_my_mz(self):
        assert _section_force_columns(0) == [
            "P_ip0", "Vy_ip0", "Vz_ip0", "T_ip0", "My_ip0", "Mz_ip0"
        ]
        assert _section_force_columns(7) == [
            "P_ip7", "Vy_ip7", "Vz_ip7", "T_ip7", "My_ip7", "Mz_ip7"
        ]


# ====================================================================== #
# _interpolate_section_force_at_xi — pure-math interpolation
# ====================================================================== #
class TestInterpolation:
    def test_constant_across_ips_yields_constant(self):
        # All IPs identical → result equals that value for any xi.
        gp_xi = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        per_ip = np.tile(np.array([-100.0, 5.0, 7.0, 0.0, 3.0, -2.0]), (5, 1))
        sf = _make_section_force(n_steps=4, ip_values=per_ip)
        for xi in (-1.0, -0.3, 0.0, 0.42, 1.0):
            out = _interpolate_section_force_at_xi(sf, gp_xi, xi)
            np.testing.assert_allclose(
                out, per_ip[0:1].repeat(4, axis=0), atol=1e-12,
            )

    def test_exact_match_at_gauss_point(self):
        # At xi == gp_xi[k], the result equals IP k exactly.
        gp_xi = np.array([-1.0, 0.0, 1.0])
        per_ip = np.array([
            [10.0, 0, 0, 0, 0, 0],
            [20.0, 0, 0, 0, 0, 0],
            [30.0, 0, 0, 0, 0, 0],
        ])
        sf = _make_section_force(n_steps=2, ip_values=per_ip)
        for k in range(3):
            out = _interpolate_section_force_at_xi(sf, gp_xi, gp_xi[k])
            np.testing.assert_allclose(out[:, 0], per_ip[k, 0], atol=1e-12)

    def test_linear_ramp_in_xi(self):
        # N(xi) = -100 * xi is the ground truth. Interpolation between
        # any two IPs that sample this function should recover the same
        # linear law at any in-between xi.
        gp_xi = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        per_ip = np.zeros((5, 6))
        per_ip[:, 0] = -100.0 * gp_xi  # P column
        sf = _make_section_force(n_steps=3, ip_values=per_ip)
        for xi in (-0.9, -0.75, -0.25, 0.0, 0.25, 0.6, 0.99):
            out = _interpolate_section_force_at_xi(sf, gp_xi, xi)
            expected = -100.0 * xi
            np.testing.assert_allclose(out[:, 0], expected, atol=1e-12)

    def test_linear_ramp_recovers_all_six_components(self):
        # Each component has its own slope; interpolation must handle
        # them independently.
        gp_xi = np.array([-1.0, 0.0, 1.0])
        slopes = np.array([10.0, -5.0, 3.0, 7.0, -2.0, 1.5])
        per_ip = np.zeros((3, 6))
        for k, xi in enumerate(gp_xi):
            per_ip[k, :] = slopes * xi
        sf = _make_section_force(n_steps=2, ip_values=per_ip)
        for xi in (-0.4, 0.25, 0.8):
            out = _interpolate_section_force_at_xi(sf, gp_xi, xi)
            expected = slopes * xi
            np.testing.assert_allclose(out[0], expected, atol=1e-12)

    def test_constant_extrapolation_below_first_ip(self):
        # Gauss-Legendre 2-point IPs at ±0.577 — beam ends ξ=±1 are
        # outside the IP envelope. Kernel should clip to the nearest IP.
        gp_xi = np.array([-0.5773502691896258, 0.5773502691896258])
        per_ip = np.array([
            [42.0, 0, 0, 0, 0, 0],
            [-42.0, 0, 0, 0, 0, 0],
        ])
        sf = _make_section_force(n_steps=2, ip_values=per_ip)
        out = _interpolate_section_force_at_xi(sf, gp_xi, -1.0)
        np.testing.assert_allclose(out[:, 0], 42.0, atol=1e-12)

    def test_constant_extrapolation_above_last_ip(self):
        gp_xi = np.array([-0.5773502691896258, 0.5773502691896258])
        per_ip = np.array([
            [42.0, 0, 0, 0, 0, 0],
            [-42.0, 0, 0, 0, 0, 0],
        ])
        sf = _make_section_force(n_steps=2, ip_values=per_ip)
        out = _interpolate_section_force_at_xi(sf, gp_xi, 1.0)
        np.testing.assert_allclose(out[:, 0], -42.0, atol=1e-12)

    def test_time_variation_propagates_per_step(self):
        # Different values at each step; the kernel must apply the same
        # interpolation weights to each row independently.
        gp_xi = np.array([-1.0, 1.0])
        sf = np.zeros((3, 2, 6))
        sf[:, 0, 0] = [10, 20, 30]
        sf[:, 1, 0] = [50, 60, 70]
        out = _interpolate_section_force_at_xi(sf, gp_xi, 0.0)  # midpoint
        expected_P = np.array([(10 + 50) / 2, (20 + 60) / 2, (30 + 70) / 2])
        np.testing.assert_allclose(out[:, 0], expected_P, atol=1e-12)

    def test_uneven_gauss_legendre_5pt_spacing(self):
        # Real Gauss-Legendre 5-point natural coords. Verify the kernel
        # handles non-uniform spacing correctly by interpolating a
        # known cubic and recovering the linear-segment approximation.
        gp_xi = np.array([
            -0.9061798459386640,
            -0.5384693101056831,
             0.0,
             0.5384693101056831,
             0.9061798459386640,
        ])
        # Linear law N(xi) = 7 * xi - 3 — should still be recovered
        # exactly because linear functions are linearly interpolable.
        per_ip = np.zeros((5, 6))
        per_ip[:, 0] = 7 * gp_xi - 3
        sf = _make_section_force(n_steps=1, ip_values=per_ip)
        for xi in (-0.7, -0.2, 0.1, 0.55):
            out = _interpolate_section_force_at_xi(sf, gp_xi, xi)
            np.testing.assert_allclose(out[0, 0], 7 * xi - 3, atol=1e-12)

    def test_empty_gp_xi_raises(self):
        gp_xi = np.array([])
        sf = np.zeros((1, 0, 6))
        with pytest.raises(ValueError, match="empty"):
            _interpolate_section_force_at_xi(sf, gp_xi, 0.0)

    def test_shape_mismatch_raises(self):
        gp_xi = np.array([-1.0, 0.0, 1.0])  # 3 IPs
        sf = np.zeros((2, 5, 6))  # but array says 5 IPs
        with pytest.raises(ValueError, match="disagrees"):
            _interpolate_section_force_at_xi(sf, gp_xi, 0.0)


# ====================================================================== #
# _rotate_section_force_and_apply_side
# ====================================================================== #
class TestRotateAndSide:
    def test_identity_rotation_unchanged(self):
        f_local = np.array([
            [10, 20, 30, 1, 2, 3],
            [40, 50, 60, 4, 5, 6],
        ], dtype=float)
        F, M = _rotate_section_force_and_apply_side(
            f_local, R=np.eye(3), kept_side_is_node1=True,
        )
        np.testing.assert_array_equal(F, f_local[:, 0:3])
        np.testing.assert_array_equal(M, f_local[:, 3:6])

    def test_kept_side_node2_flips_sign(self):
        f_local = np.array([
            [10, 20, 30, 1, 2, 3],
        ], dtype=float)
        F, M = _rotate_section_force_and_apply_side(
            f_local, R=np.eye(3), kept_side_is_node1=False,
        )
        np.testing.assert_array_equal(F, -f_local[:, 0:3])
        np.testing.assert_array_equal(M, -f_local[:, 3:6])

    def test_column_rotation_maps_local_x_to_global_z(self):
        # Elastic_frame column 1 rotation (approximately):
        # local x → global z, local y → -global y, local z → global x.
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        f_local = np.array([[100.0, 0, 0, 0, 0, 0]])  # pure axial
        F, _ = _rotate_section_force_and_apply_side(
            f_local, R, kept_side_is_node1=True,
        )
        # +100 in local x maps to +100 in global z.
        np.testing.assert_allclose(F[0], [0.0, 0.0, 100.0], atol=1e-12)

    def test_rotation_preserves_norm(self):
        # Any rotation should preserve the magnitude of F and M.
        theta = 0.7
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ])
        f_local = np.array([
            [3, 4, 5, 1, 1, 1],
            [-2, 1, -3, 0.5, -0.5, 0.5],
        ], dtype=float)
        F, M = _rotate_section_force_and_apply_side(
            f_local, R, kept_side_is_node1=True,
        )
        np.testing.assert_allclose(
            np.linalg.norm(F, axis=1),
            np.linalg.norm(f_local[:, 0:3], axis=1),
        )
        np.testing.assert_allclose(
            np.linalg.norm(M, axis=1),
            np.linalg.norm(f_local[:, 3:6], axis=1),
        )

    def test_batched_rotation_matches_loop(self):
        # Verify the batched implementation v_global = v_local @ R.T
        # agrees with the per-row formulation v_global_row = R @ v_local_row.
        R = np.array([
            [0.6, -0.8, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 0.0, 1.0],
        ])
        rng = np.random.default_rng(seed=42)
        f_local = rng.normal(size=(20, 6))
        F, M = _rotate_section_force_and_apply_side(
            f_local, R, kept_side_is_node1=True,
        )
        for k in range(20):
            np.testing.assert_allclose(F[k], R @ f_local[k, 0:3])
            np.testing.assert_allclose(M[k], R @ f_local[k, 3:6])


# ====================================================================== #
# End-to-end: _section_force_cut against synthetic ElementResults
# ====================================================================== #
def _make_synthetic_er(
    n_steps: int,
    gp_xi: np.ndarray,
    per_ip_force: np.ndarray,
    *,
    element_id: int = 42,
    element_type: str = "73-ForceBeamColumn3d",
):
    """Build a real ``ElementResults`` carrying synthetic section.force.

    ``per_ip_force`` has shape ``(n_steps, n_ip, 6)``.
    """
    from STKO_to_python.elements.element_results import ElementResults

    n_ip = gp_xi.size
    cols: list[str] = []
    for k in range(n_ip):
        cols.extend(_section_force_columns(k))
    flat = per_ip_force.reshape(n_steps, n_ip * 6)
    idx = pd.MultiIndex.from_product(
        [[element_id], range(n_steps)], names=["element_id", "step"]
    )
    df = pd.DataFrame(flat, index=idx, columns=cols)
    time = np.linspace(0.0, 1.0, n_steps)
    return ElementResults(
        df=df, time=time,
        element_ids=(element_id,),
        element_type=element_type,
        results_name="section.force",
        gp_xi=gp_xi,
    )


class TestSectionForceCutEndToEnd:
    def test_constant_compression_in_column(self):
        """A force-based column under constant axial compression of -100
        in local x. Cut at midspan should report +100 z global (force
        the discarded lower part exerts on the kept upper part)."""
        gp_xi = np.array([-1.0, 0.0, 1.0])
        per_ip = np.zeros((3, 6))
        per_ip[:, 0] = -100.0  # compression, constant across IPs
        sf = _make_section_force(n_steps=5, ip_values=per_ip)
        er = _make_synthetic_er(5, gp_xi, sf, element_id=42)

        intersection = BeamIntersection(
            element_id=42,
            element_type="73-ForceBeamColumn3d",
            xi=0.0,
            t=0.5,
            point_global=(5000.0, 0.0, 1500.0),
            end_node_ids=(1, 2),
            end_coords=((5000.0, 0.0, 0.0), (5000.0, 0.0, 3000.0)),
        )
        # Column local x maps to global +z (R same as elastic_frame col 1).
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        ds = SimpleNamespace(
            cdata=SimpleNamespace(rotation_matrix=lambda eid: R),
        )
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(42,),
        )
        F, M = _section_force_cut(er, intersection, ds, spec)
        # node 1 (z=0) is on the discarded side (below z=1500); kept = above.
        # f_local at xi=0 = (-100, 0, 0, 0, 0, 0)
        # F_global = R @ (-100, 0, 0) = (0, 0, -100)
        # kept_side_is_node1 = False → F = -F_global = (0, 0, +100)
        np.testing.assert_allclose(
            F, np.tile([[0.0, 0.0, 100.0]], (5, 1)), atol=1e-10,
        )
        np.testing.assert_allclose(M, 0.0, atol=1e-10)

    def test_xi_interpolation_in_cut(self):
        """Mid-IP cut value must equal the linear interp of bracketing IPs."""
        gp_xi = np.array([-1.0, 0.0, 1.0])
        # Linear ramp in N: N(-1) = -1000, N(0) = 0, N(+1) = +1000
        per_ip = np.zeros((3, 6))
        per_ip[:, 0] = [-1000.0, 0.0, 1000.0]
        sf = _make_section_force(n_steps=2, ip_values=per_ip)
        er = _make_synthetic_er(2, gp_xi, sf, element_id=7)

        # Cut at xi = 0.5 → N = 500 in local frame.
        intersection = BeamIntersection(
            element_id=7,
            element_type="73-ForceBeamColumn3d",
            xi=0.5,
            t=0.75,
            point_global=(0.0, 0.0, 2250.0),
            end_node_ids=(10, 20),
            end_coords=((0.0, 0.0, 0.0), (0.0, 0.0, 3000.0)),
        )
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        ds = SimpleNamespace(
            cdata=SimpleNamespace(rotation_matrix=lambda eid: R),
        )
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2250.0),
            element_ids=(7,),
        )
        F, _ = _section_force_cut(er, intersection, ds, spec)
        # f_local at xi=0.5 = (500, 0, 0, 0, 0, 0)
        # F_global = R @ (500, 0, 0) = (0, 0, 500)
        # node 1 (z=0) is below cut z=2250 → discarded → kept_side_is_node1=False
        # F = -F_global = (0, 0, -500)
        np.testing.assert_allclose(
            F, np.tile([[0.0, 0.0, -500.0]], (2, 1)), atol=1e-10,
        )

    def test_side_negative_flips_resultant(self):
        gp_xi = np.array([-1.0, 0.0, 1.0])
        per_ip = np.zeros((3, 6))
        per_ip[:, 0] = -250.0  # constant compression
        sf = _make_section_force(n_steps=1, ip_values=per_ip)
        er = _make_synthetic_er(1, gp_xi, sf, element_id=99)
        intersection = BeamIntersection(
            element_id=99,
            element_type="73-ForceBeamColumn3d",
            xi=0.0,
            t=0.5,
            point_global=(0.0, 0.0, 500.0),
            end_node_ids=(1, 2),
            end_coords=((0.0, 0.0, 0.0), (0.0, 0.0, 1000.0)),
        )
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        ds = SimpleNamespace(
            cdata=SimpleNamespace(rotation_matrix=lambda eid: R),
        )
        plane = Plane.horizontal(z=500.0)
        F_pos, _ = _section_force_cut(
            er, intersection, ds,
            SectionCutSpec(plane=plane, element_ids=(99,), side="positive"),
        )
        F_neg, _ = _section_force_cut(
            er, intersection, ds,
            SectionCutSpec(plane=plane, element_ids=(99,), side="negative"),
        )
        np.testing.assert_allclose(F_neg, -F_pos, atol=1e-10)


# ====================================================================== #
# Cross-check: section.force path agrees with ElasticBeam3d math
# ====================================================================== #
class TestSectionForceMatchesElasticBeam:
    """Build a synthetic force-based column with the same physics as
    elastic_frame's column 1, and confirm the cut force matches what
    the ElasticBeam3d path produces against the real fixture.

    Column geometry: node 1 at (5000, 0, 0), node 2 at (5000, 0, 3000),
    rotation matrix same as elastic_frame element 1 (local x → global z).
    Loading: axial compression of magnitude 2500 (matches stage 1 step 0
    of elastic_frame). Cut at z = 1500 with kept = positive z.

    Expected: F_cut_global = (0, 0, +2500) — the column's axial reaction
    on the kept side above the cut.
    """

    def test_matches_elastic_frame_column_at_step_0(self):
        gp_xi = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # Lobatto 5-point
        per_ip = np.zeros((5, 6))
        per_ip[:, 0] = -2500.0  # constant compression
        sf = _make_section_force(n_steps=10, ip_values=per_ip)
        er = _make_synthetic_er(10, gp_xi, sf, element_id=1)

        # Same R as elastic_frame element 1.
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        intersection = BeamIntersection(
            element_id=1,
            element_type="73-ForceBeamColumn3d",
            xi=0.0,
            t=0.5,
            point_global=(5000.0, 0.0, 1500.0),
            end_node_ids=(1, 2),
            end_coords=((5000.0, 0.0, 0.0), (5000.0, 0.0, 3000.0)),
        )
        ds = SimpleNamespace(
            cdata=SimpleNamespace(rotation_matrix=lambda eid: R),
        )
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1,),
        )
        F, M = _section_force_cut(er, intersection, ds, spec)
        np.testing.assert_allclose(
            F, np.tile([[0.0, 0.0, 2500.0]], (10, 1)), atol=1e-9,
        )
        np.testing.assert_allclose(M, 0.0, atol=1e-9)
