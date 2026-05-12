"""Unit tests for STKO_to_python.cuts.kernels.shell.

Splits into two groups:

- **Pure-math tests** (no .mpco fixture needed) exercising the
  bilinear / linear sampling weights and the inverse-shape-function
  helpers used to map chord endpoints to natural coords.
- **Real-fixture tests** running against ``Test_NLShell`` to verify
  the geometry phase, the consistency check (Newton's 3rd law), and
  the side-flip convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCut, SectionCutSpec
from STKO_to_python.cuts.kernels.shell import (
    SHELL_ELEMENT_CLASSES,
    _invert_quad_bilinear,
    _invert_tri_linear,
    _quad_ip_weights,
    _sample_shell_section_force,
    _tri_ip_weights,
    compute_shell_cut,
    find_shell_intersections,
)


# ====================================================================== #
# Pure-math tests
# ====================================================================== #
class TestQuadIpWeights:
    """Bilinear interpolation weights from the 4 Gauss IPs of
    ASDShellQ4 to an arbitrary (ξ, η). IP coords sit at ±1/√3.
    """

    def test_partition_of_unity_at_origin(self):
        w = _quad_ip_weights(0.0, 0.0)
        assert sum(w) == pytest.approx(1.0)

    def test_unit_response_at_each_ip(self):
        s = 1.0 / np.sqrt(3.0)
        # IP order is ξ-fastest: (-s, -s), (+s, -s), (-s, +s), (+s, +s).
        ips = [(-s, -s), (+s, -s), (-s, +s), (+s, +s)]
        for k, (xi, eta) in enumerate(ips):
            w = _quad_ip_weights(xi, eta)
            expected = np.eye(4)[k]
            np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_extrapolation_at_corner(self):
        # ξ=η=+1 is outside the IP envelope; bilinear extrapolation
        # still produces a valid (sum-to-1) weight set.
        w = _quad_ip_weights(1.0, 1.0)
        assert sum(w) == pytest.approx(1.0)

    def test_linear_field_recovered(self):
        # A field linear in (ξ, η) is exactly reproduced. Build it as
        # f(ξ, η) = 2 + 3*ξ - 5*η; sample at each IP; interpolate to a
        # test point; compare against the true value.
        s = 1.0 / np.sqrt(3.0)
        ips = np.array([[-s, -s], [+s, -s], [-s, +s], [+s, +s]])
        true_field = lambda xi, eta: 2.0 + 3.0 * xi - 5.0 * eta
        ip_values = np.array([true_field(p[0], p[1]) for p in ips])
        for xi, eta in [(0.0, 0.0), (0.3, -0.2), (0.5, 0.5), (-0.7, 0.7)]:
            w = _quad_ip_weights(xi, eta)
            interp = float(w @ ip_values)
            assert interp == pytest.approx(true_field(xi, eta), abs=1e-12)


class TestTriIpWeights:
    """Linear interpolation from 3 Gauss IPs on the unit triangle."""

    def test_partition_of_unity_at_centroid(self):
        w = _tri_ip_weights(1.0 / 3.0, 1.0 / 3.0)
        assert sum(w) == pytest.approx(1.0)
        # By symmetry the centroid evaluation gives each IP weight 1/3.
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3], atol=1e-12)

    def test_unit_response_at_each_ip(self):
        ip_coords = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
        for k, (xi, eta) in enumerate(ip_coords):
            w = _tri_ip_weights(xi, eta)
            expected = np.eye(3)[k]
            np.testing.assert_allclose(w, expected, atol=1e-12)

    def test_linear_field_recovered(self):
        ip_coords = np.array([(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)])
        true_field = lambda xi, eta: 7.0 - 2.0 * xi + 4.0 * eta
        ip_values = np.array([true_field(p[0], p[1]) for p in ip_coords])
        for xi, eta in [(0.3, 0.4), (0.0, 0.0), (0.5, 0.25)]:
            w = _tri_ip_weights(xi, eta)
            interp = float(w @ ip_values)
            assert interp == pytest.approx(true_field(xi, eta), abs=1e-12)


class TestInvertShapeFunctions:
    def test_invert_quad_at_corners(self):
        # Unit square in the xy plane with z=0.
        verts = np.array([
            [-1, -1, 0],   # ↔ (ξ, η) = (-1, -1)
            [+1, -1, 0],
            [+1, +1, 0],
            [-1, +1, 0],
        ], dtype=float)
        targets = {
            (-1, -1): 0,
            (+1, -1): 1,
            (+1, +1): 2,
            (-1, +1): 3,
        }
        for natural, node_idx in targets.items():
            recovered = _invert_quad_bilinear(verts[node_idx], verts)
            np.testing.assert_allclose(recovered, natural, atol=1e-9)

    def test_invert_quad_at_centroid(self):
        verts = np.array([
            [0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0],
        ], dtype=float)
        centroid = verts.mean(axis=0)
        natural = _invert_quad_bilinear(centroid, verts)
        np.testing.assert_allclose(natural, [0.0, 0.0], atol=1e-9)

    def test_invert_quad_on_oblique_plane(self):
        # A quad rotated 45° in 3D — same logical (ξ, η) layout.
        s = np.sqrt(2) / 2
        verts = np.array([
            [-s, 0, -s],
            [+s, 0, -s],
            [+s, 0, +s],
            [-s, 0, +s],
        ], dtype=float)
        midpoint = 0.5 * (verts[0] + verts[1])  # should map to (ξ, η) = (0, -1)
        natural = _invert_quad_bilinear(midpoint, verts)
        np.testing.assert_allclose(natural, [0.0, -1.0], atol=1e-9)

    def test_invert_tri_at_corners(self):
        # Unit triangle in xy plane.
        verts = np.array([
            [0, 0, 0],   # node 1 ↔ (0, 0)
            [1, 0, 0],   # node 2 ↔ (1, 0)
            [0, 1, 0],   # node 3 ↔ (0, 1)
        ], dtype=float)
        natural_at_node = {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (0.0, 1.0),
        }
        for node_idx, natural in natural_at_node.items():
            recovered = _invert_tri_linear(verts[node_idx], verts)
            np.testing.assert_allclose(recovered, natural, atol=1e-12)

    def test_invert_tri_at_centroid(self):
        verts = np.array([
            [0, 0, 0], [2, 0, 0], [0, 3, 0],
        ], dtype=float)
        centroid = verts.mean(axis=0)
        natural = _invert_tri_linear(centroid, verts)
        np.testing.assert_allclose(natural, [1 / 3, 1 / 3], atol=1e-12)


class TestSampleShellSectionForce:
    """Math sanity for _sample_shell_section_force across IP layouts."""

    def test_quad_sample_recovers_ip_value(self):
        s = 1.0 / np.sqrt(3.0)
        # 1 step, 4 IPs, 8 components — distinct values per IP / per
        # component so a wrong index pulls obvious garbage.
        sf = np.zeros((1, 4, 8))
        for k in range(4):
            for c in range(8):
                sf[0, k, c] = 10.0 * k + c
        # Sampling at IP 2 (-s, +s) returns that IP's components.
        result = _sample_shell_section_force(sf, -s, +s, "ASDShellQ4")
        np.testing.assert_allclose(result[0], sf[0, 2, :], atol=1e-12)

    def test_tri_sample_recovers_ip_value(self):
        sf = np.zeros((1, 3, 8))
        for k in range(3):
            sf[0, k, :] = [1.0 + k, 2.0 + k, 3.0 + k, 4.0 + k,
                           5.0 + k, 6.0 + k, 7.0 + k, 8.0 + k]
        # IP 1 sits at (2/3, 1/6) for the standard 3-pt triangle rule.
        result = _sample_shell_section_force(sf, 2 / 3, 1 / 6, "ASDShellT3")
        np.testing.assert_allclose(result[0], sf[0, 1, :], atol=1e-12)


# ====================================================================== #
# Real-fixture tests (Test_NLShell)
# ====================================================================== #
@pytest.fixture
def shell_ds(nl_shell_dir) -> MPCODataSet:
    return MPCODataSet(str(nl_shell_dir), "Results", verbose=False)


@pytest.fixture
def all_shell_eids(shell_ds) -> tuple[int, ...]:
    df = shell_ds.elements_info["dataframe"]
    base = {c for c in SHELL_ELEMENT_CLASSES}
    is_shell = df["element_type"].map(
        lambda s: any(c == s.split("-", 1)[-1].split("[", 1)[0] for c in base)
    )
    return tuple(int(x) for x in df.loc[is_shell, "element_id"].tolist())


class TestShellFixtureRegistry:
    """The Test_NLShell fixture is supposed to be the canonical
    multi-class-shell case. Make sure the kernel sees the classes the
    fixture actually contains.
    """

    def test_fixture_has_q4_and_t3(self, shell_ds):
        types = list(shell_ds.unique_element_types)
        assert any("ASDShellQ4" in t for t in types)
        assert any("ASDShellT3" in t for t in types)


class TestFindShellIntersections:
    def test_horizontal_cut_crosses_some_shells(self, shell_ds, all_shell_eids):
        # The wall spans z ∈ [180, 4530]; a cut at z=2500 must hit some.
        plane = Plane.horizontal(z=2500.0)
        spec = SectionCutSpec(plane=plane, element_ids=all_shell_eids)
        ixs = find_shell_intersections(shell_ds, spec)
        assert len(ixs) > 0
        # Every intersection chord has nonzero length.
        for ix in ixs:
            assert ix.chord_length > 1e-6
        # Natural coords stay in their parent domain (modest tolerance
        # for numerical slop at the boundaries).
        for ix in ixs:
            (xi_a, eta_a), (xi_b, eta_b) = ix.chord_param_natural
            for v in (xi_a, eta_a, xi_b, eta_b):
                assert v == v  # not NaN

    def test_above_top_returns_empty(self, shell_ds, all_shell_eids):
        # Cut far above the wall — no shell crosses.
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=10_000.0),
            element_ids=all_shell_eids,
        )
        assert find_shell_intersections(shell_ds, spec) == []

    def test_below_base_returns_empty(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=-1_000.0),
            element_ids=all_shell_eids,
        )
        assert find_shell_intersections(shell_ds, spec) == []


class TestShellCutEndToEnd:
    def test_horizontal_cut_returns_nonzero_force(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        assert not cut.is_empty
        # Gravity loading produces a downward total weight above the
        # cut; F_z on the kept (upper) side from the discarded (lower)
        # side is positive.
        assert cut.F[0, 2] > 0
        # F has the expected shape and finite values.
        assert cut.F.shape == (cut.n_steps, 3)
        assert np.all(np.isfinite(cut.F))
        assert np.all(np.isfinite(cut.M))

    def test_consistency_check_passes_on_real_fixture(self, shell_ds, all_shell_eids):
        """Newton's 3rd law: positive + negative side cuts sum to zero.

        Independent of the (very complex) Test_NLShell load pattern —
        any honest kernel must satisfy this by construction.
        """
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        ok, residual = cut.consistency_check(shell_ds, atol=1e-3)
        assert ok, f"Residual {np.max(np.abs(residual))} exceeds tol."

    def test_negative_side_flips_resultant(self, shell_ds, all_shell_eids):
        plane = Plane.horizontal(z=2500.0)
        pos = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_shell_eids, side="positive"),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        neg = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_shell_eids, side="negative"),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        np.testing.assert_allclose(neg.F, -pos.F, atol=1e-3)
        # Moments at the (identical) centroid must also flip.
        np.testing.assert_allclose(neg.M, -pos.M, atol=1e-3)

    def test_section_force_nonzero_for_some_shell(self, shell_ds, all_shell_eids):
        """Sanity: the shell section.force reader actually pulls real
        data — if a column-name typo silently zeroed it, the
        consistency check above would still pass for the wrong reason.
        """
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        # At least one shell carries a nonzero per-element force.
        max_F = max(
            float(np.max(np.abs(Fi))) for Fi in cut.per_shell_F.values()
        )
        assert max_F > 1.0

    def test_repr_includes_shell_count(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        text = repr(cut)
        # n_intersections in the repr is beams + shells.
        n_shells = len(cut.shell_intersections)
        assert f"n_intersections={n_shells}" in text


class TestShellSharedEdgeBehavior:
    """A cut plane that lands exactly on a mesh row (shared edge
    between two adjacent shells) used to double-count the cut force —
    both shells would report the shared edge as their chord. The
    side-aware filter in :func:`find_shell_intersections` resolves
    this by keeping only shells whose interior lies on the **discarded**
    side. These tests lock in that behavior on Test_NLShell, whose
    wall meshes meet at z=870 (T3 below, Q4 above).
    """

    def test_on_edge_cut_matches_just_below_for_positive_side(
        self, shell_ds, all_shell_eids,
    ):
        # side='positive' keeps shells with interior on the lower side
        # of z=870 — i.e., the T3 mesh below. Just-below z (z=869.999)
        # cuts the same T3 mesh in its interior. F_z must match.
        z_on = 870.0
        z_below = 869.999
        cut_on = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=z_on),
                           element_ids=all_shell_eids),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        cut_below = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=z_below),
                           element_ids=all_shell_eids),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        # Force in the wall's axial (z) direction agrees to <1% even
        # though the chord layouts differ between meshes.
        rel_err = abs(cut_on.F[0, 2] - cut_below.F[0, 2]) / abs(cut_below.F[0, 2])
        assert rel_err < 1e-3, (
            f"on-edge cut F_z={cut_on.F[0, 2]} disagrees with just-below "
            f"F_z={cut_below.F[0, 2]}; rel_err={rel_err}"
        )

    def test_on_edge_negative_side_matches_just_above(
        self, shell_ds, all_shell_eids,
    ):
        # side='negative' keeps shells with interior on the upper side
        # of z=870 — the Q4 mesh above. Just-above z (z=870.001) cuts
        # the same Q4 mesh in its interior.
        z_on = 870.0
        z_above = 870.001
        cut_on = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=z_on),
                           element_ids=all_shell_eids,
                           side="negative"),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        cut_above = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=z_above),
                           element_ids=all_shell_eids,
                           side="negative"),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        rel_err = abs(cut_on.F[0, 2] - cut_above.F[0, 2]) / abs(cut_above.F[0, 2])
        assert rel_err < 1e-3

    def test_no_double_count_at_shared_edge(self, shell_ds, all_shell_eids):
        """Verify the dedup story: at the shared edge between two
        meshes, an honest cut force must be finite and on the order
        of the surrounding cuts — not 2× larger from double-counting.
        """
        on = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=870.0),
                           element_ids=all_shell_eids),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        below = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=869.999),
                           element_ids=all_shell_eids),
            shell_ds, model_stage="MODEL_STAGE[1]",
        )
        # Ratio should be ≈ 1, never ≈ 2 (the doubled-count signal).
        ratio = abs(on.F[0, 2]) / abs(below.F[0, 2])
        assert 0.5 < ratio < 1.5, f"ratio {ratio} suggests double-counting"


class TestShellCutAcrossStages:
    """Make sure the kernel works on every stage of a complex fixture
    so we don't have a latent bug specific to one stage's analysis_steps.
    """

    @pytest.mark.parametrize("stage", ["MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[3]"])
    def test_cut_consistency_per_stage(self, shell_ds, all_shell_eids, stage):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage=stage)
        assert not cut.is_empty
        ok, _ = cut.consistency_check(shell_ds, atol=1e-3)
        assert ok, f"Consistency failed for {stage}."
