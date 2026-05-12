"""Resultant tests for STKO_to_python.cuts.kernels.beam_resultant.

ElasticBeam3d (closed-form ``force``) tests run against the small
``elasticFrame`` fixture. The model is a 2D portal with:

    node 4 (0, 3000) ─── el 3 (top beam) ─── node 2 (5000, 3000)
        │                                        │
       el 2 (column)                        el 1 (column)
        │                                        │
    node 3 (0, 0) [fixed]              node 1 (5000, 0) [fixed]

MODEL_STAGE[1]: vertical -25000 at nodes 2 and 4, ramping linearly
over 10 increments. At step k (0-indexed), load factor = (k+1) * 0.1.

MODEL_STAGE[2]: gravity held constant, plus horizontal +10000 at node 4,
ramping linearly over 10 increments.

Section.force tests target the ``force_beam_col`` / ``disp_beam_col``
fixtures when available; otherwise they skip gracefully.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec
from STKO_to_python.cuts.kernels import (
    BeamCutResult,
    compute_beam_cut,
    find_beam_intersections,
)


# ====================================================================== #
# ElasticBeam3d (closed-form) — elastic_frame
# ====================================================================== #
@pytest.fixture
def ds(elastic_frame_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dir), "results", verbose=False)


class TestStage1GravityMidHeightCut:
    """Cut at z=1500 in MODEL_STAGE[1] (vertical-load-only)."""

    @pytest.fixture
    def result(self, ds) -> BeamCutResult:
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
        )
        return compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")

    def test_two_columns_contribute(self, result):
        assert sorted(ix.element_id for ix in result.intersections) == [1, 2]

    def test_force_at_step_0(self, result):
        # Load factor 0.1 → -2500 per top node; kept side (above z=1500)
        # carries -5000 z external. F_cut = -(-5000) = +5000 z.
        np.testing.assert_allclose(result.F[0], [0.0, 0.0, 5000.0], atol=1e-6)

    def test_force_ramp_is_linear(self, result):
        # At step k: load factor = (k+1)*0.1, so F_z = (k+1) * 5000.
        for k in range(result.n_steps):
            expected_fz = (k + 1) * 5000.0
            np.testing.assert_allclose(result.F[k], [0.0, 0.0, expected_fz], atol=1e-6)

    def test_no_horizontal_force(self, result):
        np.testing.assert_allclose(result.F[:, 0], 0.0, atol=1e-6)
        np.testing.assert_allclose(result.F[:, 1], 0.0, atol=1e-6)

    def test_zero_moment_about_symmetric_centroid(self, result):
        # Symmetric vertical loading: total moment about centroid is zero.
        np.testing.assert_allclose(result.M, 0.0, atol=1e-6)

    def test_centroid_midspan(self, result):
        # Average of (5000, 0, 1500) and (0, 0, 1500).
        np.testing.assert_allclose(result.centroid, [2500.0, 0.0, 1500.0])

    def test_per_beam_dict_populated(self, result):
        assert set(result.per_beam_F.keys()) == {1, 2}
        for eid in (1, 2):
            assert result.per_beam_F[eid].shape == (result.n_steps, 3)
            assert result.per_beam_M_at_intersection[eid].shape == (result.n_steps, 3)


class TestStage1CutPositionInvariance:
    """No distributed load -> cut force is identical at any z."""

    def test_cut_force_independent_of_z(self, ds):
        zs = [400.0, 1500.0, 2500.0, 2999.0]
        Fs = []
        for z in zs:
            spec = SectionCutSpec(
                plane=Plane.horizontal(z=z), element_ids=(1, 2, 3),
            )
            r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
            Fs.append(r.F)
        for k in range(1, len(Fs)):
            np.testing.assert_allclose(Fs[k], Fs[0], atol=1e-6)


class TestSideFlipping:
    """side='negative' negates the resultant (Newton 3rd law check)."""

    def test_flip(self, ds):
        plane = Plane.horizontal(z=1500.0)
        pos = compute_beam_cut(
            ds,
            SectionCutSpec(plane=plane, element_ids=(1, 2, 3), side="positive"),
            model_stage="MODEL_STAGE[1]",
        )
        neg = compute_beam_cut(
            ds,
            SectionCutSpec(plane=plane, element_ids=(1, 2, 3), side="negative"),
            model_stage="MODEL_STAGE[1]",
        )
        np.testing.assert_allclose(neg.F, -pos.F, atol=1e-6)
        np.testing.assert_allclose(neg.M, -pos.M, atol=1e-6)


class TestVerticalCutThroughTopBeam:
    """The horizontal top beam carries no load in this elastic gravity case."""

    def test_zero_resultant(self, ds):
        spec = SectionCutSpec(
            plane=Plane.vertical(axis="x", at=2500.0), element_ids=(1, 2, 3),
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        assert len(r.intersections) == 1
        assert r.intersections[0].element_id == 3
        np.testing.assert_allclose(r.F, 0.0, atol=1e-6)
        np.testing.assert_allclose(r.M, 0.0, atol=1e-6)


class TestStage2GravityPlusLateral:
    """MODEL_STAGE[2]: locked gravity + horizontal +10000 at node 4.

    The Plain pattern uses a Linear time series, so load = factor * time.
    Stage 2 starts at time=1.0 (continued from stage 1) and runs 10 more
    increments of dt=0.1, recording at times 1.1, 1.2, ..., 2.0 — i.e.
    lateral load = (11000, 12000, ..., 20000) across the 10 steps.
    Gravity is frozen at the full -25000 per top node via ``loadConst``.
    """

    @pytest.fixture
    def result(self, ds) -> BeamCutResult:
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=(1, 2, 3),
        )
        return compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[2]")

    def test_step_0_combined_load(self, result):
        # Gravity locked: (0, 0, -50000) on kept side.
        # Lateral at t=1.1: (+11000, 0, 0) at node 4.
        # Kept side external: (11000, 0, -50000). F_cut = -external.
        np.testing.assert_allclose(result.F[0], [-11000.0, 0.0, 50000.0], atol=1e-6)

    def test_horizontal_ramps_vertical_constant(self, result):
        # F_z stays at 50000 across the stage (gravity locked). F_x ramps
        # from -11000 at step 0 to -20000 at step 9.
        np.testing.assert_allclose(result.F[:, 2], 50000.0, atol=1e-6)
        for k in range(result.n_steps):
            expected_fx = -(11 + k) * 1000.0
            assert result.F[k, 0] == pytest.approx(expected_fx, abs=1e-6)

    def test_moment_about_centroid_step_0(self, result):
        # Lateral force +11000 at node 4 (0, 0, 3000), centroid (2500, 0, 1500).
        # Lever arm r = (-2500, 0, 1500), F_ext = (11000, 0, 0).
        # r x F = (0*0 - 1500*0, 1500*11000 - (-2500)*0, (-2500)*0 - 0*11000)
        #       = (0, 16500000, 0)
        # F_cut moment = -external = (0, -16500000, 0).
        np.testing.assert_allclose(result.M[0], [0.0, -16500000.0, 0.0], atol=1e-3)

    def test_moment_ramps_linearly(self, result):
        for k in range(result.n_steps):
            expected_my = -(11 + k) * 1500000.0
            assert result.M[k, 1] == pytest.approx(expected_my, rel=1e-9)


class TestEquilibriumAgainstReactions:
    """Cross-check: cut at z=0+ catches both columns; F must equal
    -sum_of_top_node_external_loads = +5000 z at step 0 (stage 1).
    """

    def test_cut_just_above_base(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=10.0), element_ids=(1, 2, 3),
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 5000.0], atol=1e-6)

    def test_cut_just_below_top(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2990.0), element_ids=(1, 2, 3),
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 5000.0], atol=1e-6)


class TestEmptyCut:
    """No beam crosses the plane -> well-formed empty result."""

    def test_above_model(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=10000.0), element_ids=(1, 2, 3),
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        assert r.is_empty
        assert r.intersections == ()
        assert r.F.shape == (0, 3)
        assert r.M.shape == (0, 3)


class TestSingleBeamFilter:
    """Filtering to one beam isolates its contribution."""

    def test_one_column_carries_half(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=(1,),
        )
        r = compute_beam_cut(ds, spec, model_stage="MODEL_STAGE[1]")
        assert len(r.intersections) == 1
        # One column carries one of the two top loads -> 2500 z at step 0.
        np.testing.assert_allclose(r.F[0], [0.0, 0.0, 2500.0], atol=1e-6)


# ====================================================================== #
# section.force path — force_beam_col / disp_beam_col
# (skip-gated; assertions kept minimal because the fixture is developer-
# local and not always present)
# ====================================================================== #
class TestSectionForcePathSmoke:
    """End-to-end run of the section.force path against a force-based-
    beam fixture. Verifies the kernel doesn't crash and produces a
    well-formed BeamCutResult; equilibrium checks live with the
    SectionCut layer once it lands.
    """

    def test_force_beam_col_runs(self, force_beam_col_dir):
        ds = MPCODataSet(str(force_beam_col_dir), "results", verbose=False)
        # Pick the first available beam-type element.
        df = ds.elements_info["dataframe"]
        beams = df[
            df["element_type"].str.contains("BeamColumn", regex=False)
        ]
        if beams.empty:
            pytest.skip(
                "force_beam_col fixture present but contains no BeamColumn elements."
            )
        eids = tuple(int(x) for x in beams["element_id"].tolist())
        # Pick a cut plane through the model's centroid so we definitely
        # cross at least one beam.
        zc = float(df["centroid_z"].mean())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=zc), element_ids=eids,
        )
        stage = ds.model_stages[0]
        r = compute_beam_cut(ds, spec, model_stage=stage)
        # The cut should be non-empty and produce sane shapes.
        if r.is_empty:
            pytest.skip("Horizontal mid-height cut produced no intersections.")
        assert r.F.shape[1] == 3
        assert r.M.shape[1] == 3
        assert r.F.shape[0] == r.n_steps
        assert r.M.shape[0] == r.n_steps
        # Per-beam dict has one entry per intersecting element.
        assert set(r.per_beam_F.keys()) == {ix.element_id for ix in r.intersections}

    def test_disp_beam_col_runs(self, disp_beam_col_dir):
        ds = MPCODataSet(str(disp_beam_col_dir), "results", verbose=False)
        df = ds.elements_info["dataframe"]
        beams = df[
            df["element_type"].str.contains("BeamColumn", regex=False)
        ]
        if beams.empty:
            pytest.skip(
                "disp_beam_col fixture present but contains no BeamColumn elements."
            )
        eids = tuple(int(x) for x in beams["element_id"].tolist())
        zc = float(df["centroid_z"].mean())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=zc), element_ids=eids,
        )
        stage = ds.model_stages[0]
        r = compute_beam_cut(ds, spec, model_stage=stage)
        if r.is_empty:
            pytest.skip("Horizontal mid-height cut produced no intersections.")
        assert r.F.shape[1] == 3
        assert r.M.shape[1] == 3
