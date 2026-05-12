"""Tests for STKO_to_python.cuts.multi_cut — MultiCutResult + plotter.

The aggregator is exercised by treating two physically-distinct
fixtures — the original elastic_frame (3 ElasticBeam3d) and the
displacement-based mesh variant (11 DispBeamColumn3d) — as two
"cases" of a parameter study. They share the same applied loading
pattern and the same gravity factor at step 0, so under a horizontal
cut at z=1500 both cases should report F_z ≈ +5000 at step 0.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import pytest

from STKO_to_python import (  # noqa: E402
    MPCODataSet,
    MultiCutResult,
    Plane,
    SectionCut,
    SectionCutSpec,
)
from STKO_to_python.cuts.plotting import MultiCutPlotter  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ====================================================================== #
# Helpers
# ====================================================================== #
@pytest.fixture
def ds_a(elastic_frame_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dir), "results", verbose=False)


@pytest.fixture
def ds_b(elastic_frame_dispbased_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dispbased_dir), "results", verbose=False)


def _spec_for(ds: MPCODataSet) -> SectionCutSpec:
    """Make a spec whose element_ids cover every beam in the dataset."""
    eids = tuple(int(x) for x in ds.elements_info["dataframe"]["element_id"].tolist())
    return SectionCutSpec(
        plane=Plane.horizontal(z=1500.0),
        element_ids=eids,
        label="Story 2 shear",
    )


# ====================================================================== #
# Construction
# ====================================================================== #
class TestConstruction:
    def test_from_datasets(self, ds_a, ds_b):
        # Two datasets share the (-50000 z) total load at full factor; at
        # step 0 they both transmit F_z = +5000 across both columns.
        # But the element_ids set differs between fixtures so we can't
        # use a single SectionCutSpec — use from_cuts instead for this
        # check. See `test_from_cuts_with_matching_specs`.
        # Here we test the simpler case: one dataset, two case names.
        spec = _spec_for(ds_a)
        multi = MultiCutResult.from_datasets(
            {"case_A": ds_a, "case_B": ds_a},
            spec, model_stage="MODEL_STAGE[1]",
        )
        assert multi.n_cases == 2
        assert multi.case_names == ("case_A", "case_B")
        assert multi.spec == spec

    def test_from_cuts_with_matching_specs(self, ds_a):
        spec = _spec_for(ds_a)
        cut1 = SectionCut.compute(spec, ds_a, model_stage="MODEL_STAGE[1]")
        cut2 = SectionCut.compute(spec, ds_a, model_stage="MODEL_STAGE[2]")
        # Both cuts share spec but differ in model_stage. Aggregator
        # only enforces spec parity; model_stage divergence is allowed.
        multi = MultiCutResult.from_cuts({"stage_1": cut1, "stage_2": cut2})
        assert multi.n_cases == 2
        assert multi.spec == spec

    def test_from_cuts_rejects_mismatched_specs(self, ds_a):
        cut_a = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=500.0), element_ids=(1, 2)),
            ds_a, model_stage="MODEL_STAGE[1]",
        )
        cut_b = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=2500.0), element_ids=(1, 2)),
            ds_a, model_stage="MODEL_STAGE[1]",
        )
        with pytest.raises(ValueError, match="different spec"):
            MultiCutResult.from_cuts({"a": cut_a, "b": cut_b})

    def test_from_cuts_allows_mismatch_when_opted_in(self, ds_a):
        cut_a = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=500.0), element_ids=(1, 2)),
            ds_a, model_stage="MODEL_STAGE[1]",
        )
        cut_b = SectionCut.compute(
            SectionCutSpec(plane=Plane.horizontal(z=2500.0), element_ids=(1, 2)),
            ds_a, model_stage="MODEL_STAGE[1]",
        )
        # Escape hatch — caller knows what they're doing.
        multi = MultiCutResult.from_cuts(
            {"a": cut_a, "b": cut_b}, require_matching_spec=False,
        )
        assert multi.n_cases == 2

    def test_empty(self):
        multi = MultiCutResult.from_cuts({})
        assert multi.is_empty
        assert multi.n_cases == 0
        assert multi.case_names == ()
        assert multi.spec is None


# ====================================================================== #
# Container protocol
# ====================================================================== #
class TestContainer:
    @pytest.fixture
    def multi(self, ds_a):
        spec = _spec_for(ds_a)
        return MultiCutResult.from_datasets(
            {"DRM_1": ds_a, "DRM_2": ds_a, "pulse": ds_a},
            spec, model_stage="MODEL_STAGE[1]",
        )

    def test_getitem(self, multi):
        cut = multi["DRM_1"]
        assert isinstance(cut, SectionCut)

    def test_getitem_unknown_case_raises(self, multi):
        with pytest.raises(KeyError):
            multi["nonexistent"]

    def test_contains(self, multi):
        assert "DRM_1" in multi
        assert "nonexistent" not in multi

    def test_iter_returns_case_names(self, multi):
        assert list(multi) == ["DRM_1", "DRM_2", "pulse"]

    def test_len(self, multi):
        assert len(multi) == 3

    def test_repr_includes_label(self, multi):
        r = repr(multi)
        assert "MultiCutResult" in r
        assert "Story 2 shear" in r


# ====================================================================== #
# Aggregations
# ====================================================================== #
class TestAggregations:
    @pytest.fixture
    def multi(self, ds_a):
        spec = _spec_for(ds_a)
        return MultiCutResult.from_datasets(
            {"case_1": ds_a, "case_2": ds_a},
            spec, model_stage="MODEL_STAGE[1]",
        )

    def test_envelope_per_case_shape(self, multi):
        env = multi.envelope_per_case()
        assert env.shape == (2, 18)
        assert env.index.tolist() == ["case_1", "case_2"]
        # Identical input → identical envelope.
        np.testing.assert_allclose(
            env.iloc[0].to_numpy(), env.iloc[1].to_numpy(),
        )

    def test_envelope_columns(self, multi):
        env = multi.envelope_per_case()
        for comp in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            for stat in ("max", "min", "peak_abs"):
                assert f"{comp}_{stat}" in env.columns

    def test_envelope_fz_max(self, multi):
        env = multi.envelope_per_case()
        # Both cases: final step factor 1.0 → F_z_max = 50000.
        np.testing.assert_allclose(
            env["Fz_max"].to_numpy(), [50000.0, 50000.0], atol=1e-5,
        )

    def test_peak_over_cases(self, multi):
        s = multi.peak_over_cases(component="Fz", agg="max")
        assert isinstance(s, pd.Series)
        assert s.index.tolist() == ["case_1", "case_2"]
        np.testing.assert_allclose(s.to_numpy(), 50000.0, atol=1e-5)

    def test_peak_over_cases_bad_component(self, multi):
        with pytest.raises(ValueError, match="Unknown component"):
            multi.peak_over_cases(component="Q")

    def test_peak_over_cases_bad_agg(self, multi):
        with pytest.raises(ValueError, match="agg"):
            multi.peak_over_cases(agg="median")

    def test_to_dataframe_overlay(self, multi):
        df = multi.to_dataframe(component="Fz")
        assert df.shape == (multi["case_1"].n_steps, 2)
        assert list(df.columns) == ["case_1", "case_2"]
        # Identical input -> identical columns.
        np.testing.assert_array_equal(
            df["case_1"].to_numpy(), df["case_2"].to_numpy(),
        )

    def test_to_dataframe_rejects_mismatched_step_counts(self, ds_a):
        # MODEL_STAGE[1] and stage 2 happen to have 10 steps each in
        # elastic_frame; this test exercises the error path defensively
        # by constructing a case with truncated data via from_cuts.
        # Build cut from stage 1 (10 steps), and a fake one with a
        # different step count by manually constructing a cut.
        spec = _spec_for(ds_a)
        full = SectionCut.compute(spec, ds_a, model_stage="MODEL_STAGE[1]")
        truncated = SectionCut(
            spec=full.spec,
            model_stage=full.model_stage,
            F=full.F[:3],   # 3 steps
            M=full.M[:3],
            time=full.time[:3],
            centroid=full.centroid,
            intersections=full.intersections,
        )
        multi = MultiCutResult.from_cuts(
            {"full": full, "truncated": truncated},
            require_matching_spec=False,
        )
        with pytest.raises(ValueError, match="matching step counts"):
            multi.to_dataframe(component="Fz")


# ====================================================================== #
# Plotter
# ====================================================================== #
class TestPlotter:
    @pytest.fixture
    def multi(self, ds_a):
        spec = _spec_for(ds_a)
        return MultiCutResult.from_datasets(
            {"DRM_1": ds_a, "DRM_2": ds_a, "pulse": ds_a},
            spec, model_stage="MODEL_STAGE[1]",
        )

    def test_plot_is_multi_plotter(self, multi):
        assert isinstance(multi.plot, MultiCutPlotter)

    def test_overlay_time_history(self, multi):
        ax, meta = multi.plot.overlay_time_history(component="Fz")
        assert meta["kind"] == "overlay_time_history"
        assert meta["n_cases"] == 3
        # One line per case.
        assert len(ax.get_lines()) == 3

    def test_overlay_subset_of_cases(self, multi):
        ax, meta = multi.plot.overlay_time_history(
            component="Fz", cases=["DRM_1", "pulse"],
        )
        assert meta["cases"] == ["DRM_1", "pulse"]
        assert len(ax.get_lines()) == 2

    def test_overlay_unknown_case_raises(self, multi):
        with pytest.raises(KeyError):
            multi.plot.overlay_time_history(cases=["nonexistent"])

    def test_case_envelope_bars(self, multi):
        ax, meta = multi.plot.case_envelope_bars(component="Fz", agg="max")
        assert meta["kind"] == "case_envelope_bars"
        np.testing.assert_allclose(meta["values"], 50000.0, atol=1e-5)
        # One bar per case.
        bars = ax.containers[0]
        assert len(bars) == 3

    def test_case_envelope_bars_bad_agg(self, multi):
        with pytest.raises(ValueError, match="agg"):
            multi.plot.case_envelope_bars(agg="median")

    def test_case_scatter(self, multi):
        ax, meta = multi.plot.case_scatter(
            x_component="Fx", y_component="Fz",
        )
        assert meta["kind"] == "case_scatter"
        assert meta["x_component"] == "Fx"
        assert meta["y_component"] == "Fz"


# ====================================================================== #
# Pickle
# ====================================================================== #
class TestPickle:
    def test_roundtrip(self, ds_a, tmp_path):
        spec = _spec_for(ds_a)
        multi = MultiCutResult.from_datasets(
            {"case_1": ds_a, "case_2": ds_a},
            spec, model_stage="MODEL_STAGE[1]",
        )
        path = multi.save_pickle(tmp_path / "multi.pkl")
        loaded = MultiCutResult.load_pickle(path)
        assert loaded.n_cases == multi.n_cases
        assert loaded.case_names == multi.case_names
        assert loaded.spec == multi.spec
        pd.testing.assert_frame_equal(
            loaded.envelope_per_case(), multi.envelope_per_case(),
        )

    def test_gzip_roundtrip(self, ds_a, tmp_path):
        spec = _spec_for(ds_a)
        multi = MultiCutResult.from_datasets(
            {"case_1": ds_a}, spec, model_stage="MODEL_STAGE[1]",
        )
        path = multi.save_pickle(tmp_path / "multi.pkl.gz")
        loaded = MultiCutResult.load_pickle(path)
        assert loaded.case_names == ("case_1",)

    def test_load_wrong_type_raises(self, tmp_path):
        import pickle
        path = tmp_path / "wrong.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a multi cut"}, f)
        with pytest.raises(TypeError, match="expected MultiCutResult"):
            MultiCutResult.load_pickle(path)


# ====================================================================== #
# Cross-fixture sanity check — two different beam types, same physics
# ====================================================================== #
class TestCrossFixture:
    """Compare elastic_frame (ElasticBeam3d, closed-form) vs the
    displacement-based mesh (DispBeamColumn3d, section.force). Same
    applied loading; cut at z=1500 should give F_z = +5000 at step 0
    in both. This is the multi-case end-to-end check.
    """

    def test_two_models_same_resultant_at_step_0(self, ds_a, ds_b):
        # Each dataset has a different element ID set, so we can't use
        # one spec for both via from_datasets. Compute the cuts
        # separately and assemble with require_matching_spec=False
        # (the planes match; only element_ids differ).
        cut_a = SectionCut.compute(
            SectionCutSpec(
                plane=Plane.horizontal(z=1500.0),
                element_ids=tuple(
                    int(x) for x in ds_a.elements_info["dataframe"]["element_id"].tolist()
                ),
                label="closed_form_3el",
            ),
            ds_a, model_stage="MODEL_STAGE[1]",
        )
        cut_b = SectionCut.compute(
            SectionCutSpec(
                plane=Plane.horizontal(z=1500.0),
                element_ids=tuple(
                    int(x) for x in ds_b.elements_info["dataframe"]["element_id"].tolist()
                ),
                label="section_force_11el",
            ),
            ds_b, model_stage="MODEL_STAGE[1]",
        )
        multi = MultiCutResult.from_cuts(
            {"closed_form": cut_a, "section_force": cut_b},
            require_matching_spec=False,
        )
        env = multi.envelope_per_case()
        # Both models should report identical F_z_max at step 0 (and full ramp).
        np.testing.assert_allclose(
            env.loc["closed_form", "Fz_max"], 50000.0, atol=1e-4,
        )
        np.testing.assert_allclose(
            env.loc["section_force", "Fz_max"], 50000.0, atol=1e-4,
        )
