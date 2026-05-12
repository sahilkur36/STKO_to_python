"""Tests for STKO_to_python.cuts.sweep — SectionSweep + plotter.

Verifies the multi-plane wrapper against the small elastic_frame
fixture (portal frame with 2 columns + 1 top beam, gravity ramp +
lateral ramp). All horizontal cuts that intersect both columns under
gravity must give the same F_z (no distributed load between cuts),
so :meth:`SectionSweep.envelope` should report a flat profile.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (after .use())
import numpy as np
import pandas as pd
import pytest

from STKO_to_python import (  # noqa: E402
    MPCODataSet,
    Plane,
    SectionCutSpec,
    SectionSweep,
)
from STKO_to_python.cuts.plotting import SectionSweepPlotter  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ds(elastic_frame_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dir), "results", verbose=False)


# ====================================================================== #
# Construction & container protocol
# ====================================================================== #
class TestConstruction:
    def test_compute_via_ds_section_sweep(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0, 2500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        assert isinstance(sweep, SectionSweep)
        assert len(sweep) == 3
        assert sweep.n_planes == 3
        assert sweep.model_stage == "MODEL_STAGE[1]"

    def test_iterable_and_indexable(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0, 2500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        first = sweep[0]
        assert first.spec.plane.point == (0.0, 0.0, 500.0)
        # Iteration order matches plane order.
        zs = [cut.spec.plane.point[2] for cut in sweep]
        assert zs == [500.0, 1500.0, 2500.0]

    def test_from_specs(self, ds):
        specs = [
            SectionCutSpec(plane=Plane.horizontal(z=500.0), element_ids=(1, 2, 3)),
            SectionCutSpec(plane=Plane.horizontal(z=1500.0), element_ids=(1, 2, 3)),
        ]
        sweep = SectionSweep.from_specs(specs, ds, model_stage="MODEL_STAGE[1]")
        assert len(sweep) == 2

    def test_empty_planes(self, ds):
        sweep = ds.section_sweep(
            planes=[], element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        assert sweep.is_empty
        assert sweep.n_planes == 0
        assert sweep.n_steps == 0
        assert sweep.time.size == 0

    def test_repr(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        assert "SectionSweep" in repr(sweep)
        assert "n_planes=2" in repr(sweep)


# ====================================================================== #
# Aggregations
# ====================================================================== #
class TestEnvelope:
    @pytest.fixture
    def sweep(self, ds):
        return ds.section_sweep(
            planes=Plane.horizontal_grid([400.0, 800.0, 1500.0, 2200.0, 2800.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )

    def test_envelope_shape_and_columns(self, sweep):
        env = sweep.envelope()
        assert env.shape == (5, 18)
        # Columns include the cross of components × {max, min, peak_abs}.
        for comp in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"):
            for stat in ("max", "min", "peak_abs"):
                assert f"{comp}_{stat}" in env.columns
        assert env.index.name == "plane_index"

    def test_envelope_flat_fz_under_no_distributed_load(self, sweep):
        # All 5 horizontal cuts catch both columns. Without distributed
        # load between them, max F_z should be the same at every plane.
        env = sweep.envelope()
        fz_max = env["Fz_max"].to_numpy()
        # Last step has factor 1.0 -> total gravity 50000 N -> F_z_cut = +50000.
        np.testing.assert_allclose(fz_max, 50000.0, atol=1e-5)

    def test_envelope_handles_empty_cut(self, ds):
        # A plane above the frame produces an empty cut. Envelope row
        # must still exist with NaN-filled stats.
        sweep = ds.section_sweep(
            planes=[Plane.horizontal(z=1500.0), Plane.horizontal(z=10000.0)],
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        env = sweep.envelope()
        assert env.shape == (2, 18)
        # Row 0 has finite values; row 1 is all NaN (empty cut).
        assert np.isfinite(env.loc[0, "Fz_max"])
        assert np.isnan(env.loc[1, "Fz_max"])


class TestToDataFrame:
    @pytest.fixture
    def sweep(self, ds):
        return ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0, 2500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )

    def test_to_dataframe_shape(self, sweep):
        df = sweep.to_dataframe(component="Fz")
        assert df.shape == (sweep.n_steps, 3)
        assert df.index.name == "time"
        assert df.columns.name == "plane_index"

    def test_to_dataframe_constant_fz_columns(self, sweep):
        # All three column-vs-time series should match within numerical noise.
        df = sweep.to_dataframe(component="Fz")
        col0 = df.iloc[:, 0].to_numpy()
        col1 = df.iloc[:, 1].to_numpy()
        col2 = df.iloc[:, 2].to_numpy()
        np.testing.assert_allclose(col0, col1, atol=1e-5)
        np.testing.assert_allclose(col0, col2, atol=1e-5)

    def test_to_dataframe_bad_component(self, sweep):
        with pytest.raises(ValueError, match="Unknown component"):
            sweep.to_dataframe(component="Q")

    def test_to_dataframe_empty_cut_yields_nan_column(self, ds):
        sweep = ds.section_sweep(
            planes=[Plane.horizontal(z=1500.0), Plane.horizontal(z=10000.0)],
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        df = sweep.to_dataframe(component="Fz")
        assert df.shape[1] == 2
        # Column 1 (empty cut) is all NaN.
        assert df.iloc[:, 1].isna().all()


class TestPlaneLocators:
    def test_inferred_z_for_horizontal_grid(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([100.0, 500.0, 2000.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        locs = sweep.plane_locators()
        np.testing.assert_allclose(locs, [100.0, 500.0, 2000.0])

    def test_inferred_x_for_vertical_grid(self, ds):
        planes = [
            Plane.vertical(axis="x", at=at) for at in (1000.0, 2500.0, 4000.0)
        ]
        sweep = ds.section_sweep(
            planes=planes,
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        locs = sweep.plane_locators()
        np.testing.assert_allclose(locs, [1000.0, 2500.0, 4000.0])

    def test_explicit_axis(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([0.0, 500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        # Explicitly asking for x — point is at (0, 0, z) so x = 0.
        np.testing.assert_allclose(sweep.plane_locators(axis="x"), [0.0, 0.0])

    def test_oblique_normal_requires_explicit_axis(self, ds):
        sweep = ds.section_sweep(
            planes=[Plane(point=(0.0, 0.0, 0.0), normal=(1.0, 1.0, 1.0))],
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        with pytest.raises(ValueError, match="oblique"):
            sweep.plane_locators()
        # But explicit axis works.
        np.testing.assert_allclose(sweep.plane_locators(axis="z"), [0.0])

    def test_bad_axis_raises(self, ds):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([0.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        with pytest.raises(ValueError, match="must be"):
            sweep.plane_locators(axis="w")


# ====================================================================== #
# Plotter
# ====================================================================== #
class TestPlotter:
    @pytest.fixture
    def sweep(self, ds):
        return ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0, 2500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )

    def test_plot_attribute_is_plotter(self, sweep):
        assert isinstance(sweep.plot, SectionSweepPlotter)

    def test_profile_returns_meta(self, sweep):
        ax, meta = sweep.plot.profile(component="Fz", agg="max")
        assert meta["kind"] == "profile"
        assert meta["component"] == "Fz"
        assert meta["agg"] == "max"
        np.testing.assert_allclose(meta["locators"], [500.0, 1500.0, 2500.0])
        # All three planes report the same max F_z (no distributed load
        # between them) — the values column should be flat.
        np.testing.assert_allclose(meta["values"], 50000.0, atol=1e-5)

    def test_profile_vertical_default(self, sweep):
        ax, _ = sweep.plot.profile(component="Fz", agg="max")
        # Vertical default puts locator on y-axis.
        assert "locator" in ax.get_ylabel()
        assert "Fz" in ax.get_xlabel()

    def test_profile_horizontal_orientation(self, sweep):
        ax, _ = sweep.plot.profile(component="Fz", agg="max", vertical=False)
        assert "locator" in ax.get_xlabel()
        assert "Fz" in ax.get_ylabel()

    def test_profile_bad_agg_raises(self, sweep):
        with pytest.raises(ValueError, match="agg"):
            sweep.plot.profile(agg="median")

    def test_heatmap_returns_meta(self, sweep):
        ax, meta = sweep.plot.heatmap(component="Fz")
        assert meta["kind"] == "heatmap"
        assert meta["component"] == "Fz"
        # Heatmap data should be shape (n_planes, n_steps).
        assert meta["values"].shape == (3, sweep.n_steps)

    def test_heatmap_empty_sweep(self, ds):
        sweep = ds.section_sweep(
            planes=[], element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        ax, meta = sweep.plot.heatmap(component="Fx")
        assert meta.get("empty") is True


# ====================================================================== #
# Pickle
# ====================================================================== #
class TestPickle:
    def test_roundtrip(self, ds, tmp_path):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([500.0, 1500.0, 2500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        path = sweep.save_pickle(tmp_path / "sweep.pkl")
        loaded = SectionSweep.load_pickle(path)
        assert loaded.n_planes == sweep.n_planes
        assert loaded.model_stage == sweep.model_stage
        # Envelope contents survive.
        pd.testing.assert_frame_equal(loaded.envelope(), sweep.envelope())

    def test_gzip_roundtrip(self, ds, tmp_path):
        sweep = ds.section_sweep(
            planes=Plane.horizontal_grid([1500.0]),
            element_ids=(1, 2, 3),
            model_stage="MODEL_STAGE[1]",
        )
        path = sweep.save_pickle(tmp_path / "sweep.pkl.gz")
        loaded = SectionSweep.load_pickle(path)
        assert loaded.n_planes == 1

    def test_load_wrong_type_raises(self, tmp_path):
        import pickle
        path = tmp_path / "wrong.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a sweep"}, f)
        with pytest.raises(TypeError, match="expected SectionSweep"):
            SectionSweep.load_pickle(path)
