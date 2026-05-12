"""Tests for the user-facing SectionCut dataclass.

Covers:
- Construction via ``ds.section_cut(plane=..., ...)`` (inline form),
  ``ds.section_cut(spec=...)`` (reusable form), and direct
  ``SectionCut.compute(spec, ds, model_stage=...)``.
- Broker-style accessors: ``resultant``, ``at_step``, ``at_time``,
  ``envelope``, ``to_dataframe``, ``moment_about``.
- Validators: ``consistency_check`` (Newton 3rd) and ``compare_to``
  (position invariance for no-load bands).
- Pickle round-trip — payload is the spec + arrays, no live dataset.
- Edge cases: empty cut, bad arguments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet, Plane, SectionCut, SectionCutSpec


@pytest.fixture
def ds(elastic_frame_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dir), "results", verbose=False)


# ====================================================================== #
# Construction
# ====================================================================== #
class TestConstruction:
    def test_inline_form(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        assert isinstance(cut, SectionCut)
        assert cut.spec.side == "positive"
        assert cut.model_stage == "MODEL_STAGE[1]"
        assert cut.n_steps > 0

    def test_spec_form(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            label="Base shear",
        )
        cut = ds.section_cut(spec=spec, model_stage="MODEL_STAGE[1]")
        assert cut.spec.label == "Base shear"

    def test_compute_classmethod_form(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=(1, 2, 3),
        )
        cut = SectionCut.compute(spec, ds, model_stage="MODEL_STAGE[1]")
        assert cut.spec is spec

    def test_must_pass_plane_or_spec(self, ds):
        with pytest.raises(ValueError, match="Provide either"):
            ds.section_cut(model_stage="MODEL_STAGE[1]")

    def test_spec_and_plane_together_raise(self, ds):
        spec = SectionCutSpec(plane=Plane.horizontal(z=0.0), element_ids=(1,))
        with pytest.raises(ValueError, match="alone"):
            ds.section_cut(
                spec=spec, plane=Plane.horizontal(z=1.0),
                model_stage="MODEL_STAGE[1]",
            )

    def test_spec_with_filter_kwargs_raises(self, ds):
        spec = SectionCutSpec(plane=Plane.horizontal(z=0.0), element_ids=(1,))
        with pytest.raises(ValueError, match="alone"):
            ds.section_cut(
                spec=spec, element_ids=(2,),
                model_stage="MODEL_STAGE[1]",
            )

    def test_repr_is_informative(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3), label="Test cut",
            model_stage="MODEL_STAGE[1]",
        )
        r = repr(cut)
        assert "SectionCut" in r
        assert "MODEL_STAGE[1]" in r
        assert "Test cut" in r


# ====================================================================== #
# Resultant accessors
# ====================================================================== #
class TestResultantAccessors:
    @pytest.fixture
    def cut(self, ds) -> SectionCut:
        return ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )

    def test_resultant_shapes(self, cut):
        F, M = cut.resultant()
        assert F.shape == (cut.n_steps, 3)
        assert M.shape == (cut.n_steps, 3)

    def test_resultant_returns_copies(self, cut):
        F1, M1 = cut.resultant()
        F1[0, 0] = 99999.0
        F2, _ = cut.resultant()
        # The mutation should not propagate back through the cut.
        assert F2[0, 0] != 99999.0

    def test_at_step_returns_series(self, cut):
        s = cut.at_step(0)
        assert isinstance(s, pd.Series)
        assert list(s.index) == ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        # At step 0, F_z = +5000 (verified by the beam kernel tests).
        assert s["Fz"] == pytest.approx(5000.0, abs=1e-6)

    def test_at_step_out_of_range(self, cut):
        with pytest.raises(IndexError):
            cut.at_step(-1)
        with pytest.raises(IndexError):
            cut.at_step(cut.n_steps)

    def test_at_time_finds_nearest_step(self, cut):
        t0 = float(cut.time[3])
        s = cut.at_time(t0)
        assert s["Fz"] == cut.F[3, 2]

    def test_at_time_with_tolerance(self, cut):
        # tol smaller than step spacing should reject far-from-step times.
        with pytest.raises(ValueError, match="No step within tol"):
            cut.at_time(-1.0, tol=1e-6)

    def test_envelope_columns(self, cut):
        env = cut.envelope()
        assert list(env.columns) == [
            "max", "min", "peak_abs", "peak_abs_step", "peak_abs_time"
        ]
        assert list(env.index) == ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    def test_envelope_fz_peak(self, cut):
        env = cut.envelope()
        # Last step: load factor 1.0 -> F_z = 50000.
        assert env.loc["Fz", "peak_abs"] == pytest.approx(50000.0, abs=1e-6)
        assert env.loc["Fz", "max"] == pytest.approx(50000.0, abs=1e-6)

    def test_to_dataframe_shape(self, cut):
        df = cut.to_dataframe()
        assert df.shape == (cut.n_steps, 6)
        assert list(df.columns) == ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        assert df.index.name == "time"


# ====================================================================== #
# moment_about — moment-arm transfer
# ====================================================================== #
class TestMomentAbout:
    def test_self_centroid_returns_unchanged(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        M_again = cut.moment_about(cut.centroid)
        np.testing.assert_allclose(M_again, cut.M, atol=1e-10)

    def test_transfer_uses_arm_cross_F(self, ds):
        # Symmetric vertical loading -> F = (0, 0, F_z). Transferring to
        # a reference point with x-offset Δx changes M_y by Δx * F_z.
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        # Cut centroid at (2500, 0, 1500). Transfer to (0, 0, 1500).
        new_ref = np.array([0.0, 0.0, 1500.0])
        M_new = cut.moment_about(new_ref)
        # arm = centroid - ref = (2500, 0, 0); F = (0, 0, F_z)
        # arm x F = (0*F_z - 0*0, 0*0 - 2500*F_z, 2500*0 - 0*0) = (0, -2500*F_z, 0)
        # M_new = M + arm x F
        expected_my = cut.M[:, 1] - 2500.0 * cut.F[:, 2]
        np.testing.assert_allclose(M_new[:, 1], expected_my, atol=1e-6)

    def test_bad_shape_raises(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0), element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        with pytest.raises(ValueError, match="length-3"):
            cut.moment_about([1.0, 2.0])


# ====================================================================== #
# Validators — Newton 3rd
# ====================================================================== #
class TestConsistencyCheck:
    def test_passes_against_elastic_frame_stage_1(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        ok, residual = cut.consistency_check(ds)
        assert ok, f"Residual was {residual}"
        # Quantitative: per-step sum should be effectively zero.
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_passes_against_elastic_frame_stage_2(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[2]",
        )
        ok, _ = cut.consistency_check(ds)
        assert ok

    def test_passes_when_starting_with_negative_side(self, ds):
        # The check must work regardless of which side we started from.
        cut_neg = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            side="negative",
            model_stage="MODEL_STAGE[1]",
        )
        ok, _ = cut_neg.consistency_check(ds)
        assert ok


# ====================================================================== #
# Validators — compare_to (no-load band)
# ====================================================================== #
class TestCompareTo:
    def test_parallel_cuts_in_unloaded_column_band_agree(self, ds):
        # Two cuts at z=500 and z=2500, both catching the same columns
        # in a region with no external load between them. Resultants
        # must match.
        cut_lo = ds.section_cut(
            plane=Plane.horizontal(z=500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        cut_hi = ds.section_cut(
            plane=Plane.horizontal(z=2500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        ok, residual = cut_lo.compare_to(cut_hi)
        assert ok, f"Residual was {residual.max()}"

    def test_step_count_mismatch_raises(self, ds):
        cut_s1 = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        cut_s2 = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[2]",
        )
        if cut_s1.n_steps != cut_s2.n_steps:
            with pytest.raises(ValueError, match="step counts"):
                cut_s1.compare_to(cut_s2)


# ====================================================================== #
# Pickle round-trip
# ====================================================================== #
class TestPickle:
    def test_roundtrip_preserves_results(self, ds, tmp_path):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3), label="Story 2",
            model_stage="MODEL_STAGE[1]",
        )
        path = cut.save_pickle(tmp_path / "cut.pkl")
        loaded = SectionCut.load_pickle(path)
        np.testing.assert_array_equal(loaded.F, cut.F)
        np.testing.assert_array_equal(loaded.M, cut.M)
        np.testing.assert_array_equal(loaded.time, cut.time)
        assert loaded.spec == cut.spec
        assert loaded.model_stage == cut.model_stage
        assert loaded.contributing_element_ids == cut.contributing_element_ids

    def test_gzipped_roundtrip(self, ds, tmp_path):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        path = cut.save_pickle(tmp_path / "cut.pkl.gz")
        loaded = SectionCut.load_pickle(path)
        np.testing.assert_array_equal(loaded.F, cut.F)

    def test_load_wrong_type_raises(self, ds, tmp_path):
        import pickle
        # Drop a non-SectionCut payload into a file and verify rejection.
        path = tmp_path / "wrong.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a cut"}, f)
        with pytest.raises(TypeError, match="expected SectionCut"):
            SectionCut.load_pickle(path)


# ====================================================================== #
# Empty cut
# ====================================================================== #
class TestEmptyCut:
    def test_above_model(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=10000.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        assert cut.is_empty
        assert cut.F.shape == (0, 3)
        assert cut.contributing_element_ids == ()

    def test_envelope_of_empty(self, ds):
        cut = ds.section_cut(
            plane=Plane.horizontal(z=10000.0),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        env = cut.envelope()
        # Should be a well-formed DataFrame, even if empty.
        assert list(env.index) == ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
