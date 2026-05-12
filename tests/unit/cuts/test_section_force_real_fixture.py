"""Real-fixture integration test for the section.force path.

Runs against ``elasticFrame_mesh_displacementBased_results``: the same
portal geometry as the regular elastic_frame fixture, but with each
column and beam meshed into multiple ``dispBeamColumn`` elements with
5-point Lobatto integration and ``section.force`` recorded at every IP.

Geometry (after meshing):
    Column at x=5000: elements 1 (z=0→1000), 2 (1000→2000), 3 (2000→3000)
    Column at x=0:    elements 4 (z=0→1000), 5 (1000→2000), 6 (2000→3000)
    Top beam at z=3000: elements 7..11 connecting nodes 4→9→10→11→12→2.

Stage 1 loads (linearly ramping in time, factor = time):
    node 4 : -5000  (top of left column)
    node 9 : -10000 (top beam interior)
    node 10: -10000
    node 11: -10000
    node 12: -10000
    node 2 : -5000  (top of right column)
    Total : -50000 z at full load → symmetric about x=2500.

Stage 2: gravity frozen at full magnitude (loadConst), plus horizontal
load at node 4 = +10000 x using a Linear time series (factor = time).
Stage 2 starts at time=1.0; step 0 lands at time=1.1.

What this fixture exercises that no synthetic test does:
- Real ``section.force`` columns (``P_ip<k>, Mz_ip<k>, My_ip<k>,
  T_ip<k>`` — 4 components, no shear), proving the reader handles the
  common 4-component case.
- Real Lobatto 5-point ``gp_xi`` from the file.
- Real ``cdata.rotation_matrix`` lookup for each element.
- Linear interpolation in xi between bracketing IPs against the real
  data stream.
- The dispatch logic that routes ``DispBeamColumn3d`` elements through
  the section.force path (vs. ``ElasticBeam3d`` through ``force``).
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec
from STKO_to_python.cuts.kernels import compute_beam_cut


@pytest.fixture
def ds(elastic_frame_dispbased_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dispbased_dir), "results", verbose=False)


# ====================================================================== #
# Stage 1 — symmetric gravity loading
# ====================================================================== #
class TestStage1Equilibrium:
    """Cut at z=1500 (inside elements 2 & 5) catches the two columns at
    their middle segments. Symmetric vertical loading means F_x = F_y = 0,
    F_z balances the gravity above the cut.
    """

    @pytest.fixture
    def result(self, ds):
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=all_eids,
        )
        return compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")

    def test_two_columns_contribute(self, result):
        # Only elements 2 and 5 span z=1500 (column mid-segments).
        assert sorted(ix.element_id for ix in result.intersections) == [2, 5]

    def test_step_0_vertical_force(self, result):
        # Step 0: time=0.1, total gravity = -5000 z, F_cut_z = +5000.
        np.testing.assert_allclose(result.F[0], [0.0, 0.0, 5000.0], atol=1e-4)

    def test_force_ramps_linearly(self, result):
        for k in range(result.n_steps):
            expected_fz = (k + 1) * 5000.0
            np.testing.assert_allclose(
                result.F[k], [0.0, 0.0, expected_fz], atol=1e-4,
            )

    def test_no_horizontal_force(self, result):
        # Symmetric loading -> F_x and F_y should be zero (modulo round-off).
        np.testing.assert_allclose(result.F[:, 0], 0.0, atol=1e-4)
        np.testing.assert_allclose(result.F[:, 1], 0.0, atol=1e-4)

    def test_moments_zero_by_symmetry(self, result):
        # Both columns bend in equal and opposite directions; arm
        # contributions from axial forces about the (2500, 0, 1500)
        # centroid also cancel. Total moment should be ~zero.
        np.testing.assert_allclose(result.M, 0.0, atol=1e-3)


# ====================================================================== #
# Stage 1 — interpolation between IPs
# ====================================================================== #
class TestStage1Interpolation:
    """For axial-only loading, the section P along a column is constant
    -2500 at every IP. Cuts at any xi (inside-IP or at an IP) should
    give the same F_z = +5000 — exercising both the interpolation path
    (xi mid-IP) and the at-IP path (xi exact-IP).
    """

    def test_cut_at_middle_xi(self, ds):
        # Cut at z=1300 lands inside element 2 (z=1000→2000) at
        # xi = 2*(1300-1000)/1000 - 1 = -0.4 — between IP-1 (xi=-0.6547)
        # and IP-2 (xi=0). Exercises true linear interpolation.
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1300.0), element_ids=all_eids,
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 5000.0], atol=1e-4)

    def test_cut_at_exact_lobatto_endpoint(self, ds):
        # Cut at z=2000 lands at xi=+1 of element 2 (the top Lobatto IP),
        # equivalently xi=-1 of element 3 (the bottom Lobatto IP) — but
        # only one element will report the cut depending on the segment
        # parameterization tolerance. Either way, F_z must be +5000.
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2000.0), element_ids=all_eids,
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 5000.0], atol=1e-4)

    def test_cut_at_middle_segment_midpoint(self, ds):
        # Cut at z=1500: xi=0 of element 2 (the central Lobatto IP).
        # No interpolation needed at this xi — direct IP read.
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=all_eids,
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 5000.0], atol=1e-4)

    def test_position_invariance(self, ds):
        # Different z values, all inside column segments — should all
        # give the same axial cut force (no distributed transverse load).
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        results_fz = []
        for z in (250.0, 500.0, 750.0, 1300.0, 1500.0, 1700.0, 2500.0, 2900.0):
            spec = SectionCutSpec(
                plane=Plane.horizontal(z=z), element_ids=all_eids,
            )
            r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
            results_fz.append(r.F[0, 2])
        # All step-0 F_z values should equal +5000 within numerical noise.
        np.testing.assert_allclose(results_fz, 5000.0, atol=1e-4)


# ====================================================================== #
# Stage 2 — gravity + lateral asymmetric load
# ====================================================================== #
class TestStage2LateralLoad:
    """Stage 2: gravity locked at -50000 total, plus lateral +10000 at
    node 4 via a Linear time series. Step 0 of stage 2 is at time=1.1,
    so lateral = 11000.
    Cut at z=1500 should balance both: F_x = -11000, F_z = +50000.
    """

    @pytest.fixture
    def result(self, ds):
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=all_eids,
        )
        return compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[2]")

    def test_step_0_combined_load(self, result):
        # At time=1.1: gravity full (-50000) + lateral 11000 → F_cut = (-11000, 0, 50000).
        np.testing.assert_allclose(result.F[0], [-11000.0, 0.0, 50000.0], atol=1e-3)

    def test_horizontal_ramps_vertical_constant(self, result):
        np.testing.assert_allclose(result.F[:, 2], 50000.0, atol=1e-3)
        for k in range(result.n_steps):
            expected_fx = -(11 + k) * 1000.0
            np.testing.assert_allclose(result.F[k, 0], expected_fx, atol=1e-3)


# ====================================================================== #
# Side flip in real data
# ====================================================================== #
class TestRealFixtureSideFlip:
    def test_negative_side_flips_resultant(self, ds):
        plane = Plane.horizontal(z=1500.0)
        all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
        pos = compute_beam_cut(
            ds,
            SectionCutSpec(plane=plane, element_ids=all_eids, side="positive"),
            model_stage="MODEL_STAGE[1]",
        )
        neg = compute_beam_cut(
            ds,
            SectionCutSpec(plane=plane, element_ids=all_eids, side="negative"),
            model_stage="MODEL_STAGE[1]",
        )
        np.testing.assert_allclose(neg.F, -pos.F, atol=1e-4)
        np.testing.assert_allclose(neg.M, -pos.M, atol=1e-4)


# ====================================================================== #
# Sanity: the columns ARE bending in this fixture (verifies the
# section.force reader is picking up the non-zero My values)
# ====================================================================== #
class TestRealFixtureCarriesBending:
    """The dispBeam mesh transfers load through the beam, inducing
    non-zero ``My`` in the columns at every IP. This test confirms the
    raw section.force values are nonzero — if my new reader were
    silently zeroing them out (e.g., because of a column-name typo),
    the cut moment would be artificially zero and the symmetry test
    above would still pass but for the wrong reason.
    """

    def test_my_is_nonzero_at_some_ip(self, ds):
        er = ds.elements.get_element_results(
            results_name="section.force",
            element_type="64-DispBeamColumn3d",
            model_stage="MODEL_STAGE[1]",
            element_ids=[2],
        )
        # At least one My_ip<k> column at step 0 should be far from zero.
        my_cols = [c for c in er.df.columns if c.startswith("My_ip")]
        assert my_cols, "section.force on this fixture should expose My_ip<k> columns."
        my_vals = er.df.loc[2].loc[0, my_cols].to_numpy()
        assert np.max(np.abs(my_vals)) > 100.0, (
            f"Expected non-trivial bending; got {my_vals.tolist()}"
        )
