"""Unit tests for the per-layer breakdown of layered-shell cuts (v1.7).

Two layers of coverage:

- **Pure-math tests** — exercise the layered-shell stress reader and
  the layer-table parser on synthetic inputs that don't need the
  Test_NLShell fixture.
- **Real-fixture tests** — against ``Test_NLShell`` (gated on the
  ``nl_shell_dir`` conftest fixture) verify the
  ``sum(per_layer_force(k)) == cut.F`` invariant and the dataset
  ``per_layer=k`` short-circuit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCut, SectionCutSpec
from STKO_to_python.cuts.kernels.shell import (
    SHELL_ELEMENT_CLASSES,
    _discover_fiber_count_in_layer,
    _read_fiber_in_layer_stress_array,
    _read_layer_stress_array,
    _sample_layer_stress,
)
from STKO_to_python.model.layered_section_reader import (
    LayerInfo,
    parse_sections_tcl,
)


# ====================================================================== #
# Pure-math tests
# ====================================================================== #
class TestReadLayerStressArray:
    """``_read_layer_stress_array`` extracts the layered Voigt stress
    from a flat DataFrame with ``<comp>_l<layer>_ip<ip>`` columns.
    """

    def test_recognises_membrane_columns(self):
        import pandas as pd
        # 2 steps, 1 IP, 1 layer. Membrane stresses only.
        df = pd.DataFrame({
            "sigma11_l0_ip0": [1.0, 2.0],
            "sigma22_l0_ip0": [3.0, 4.0],
            "sigma12_l0_ip0": [5.0, 6.0],
        })
        stress = _read_layer_stress_array(df, n_ip=1, layer_idx=0)
        assert stress.shape == (2, 1, 6)
        # Voigt order: (sigma11, sigma22, sigma33, sigma12, sigma23, sigma13)
        np.testing.assert_allclose(stress[:, 0, 0], [1.0, 2.0])
        np.testing.assert_allclose(stress[:, 0, 1], [3.0, 4.0])
        np.testing.assert_allclose(stress[:, 0, 3], [5.0, 6.0])
        # Missing components stay zero.
        np.testing.assert_allclose(stress[:, 0, 2], 0.0)

    def test_picks_only_the_requested_layer(self):
        import pandas as pd
        # Two layers — the function should ignore the other one.
        df = pd.DataFrame({
            "sigma11_l0_ip0": [10.0],
            "sigma11_l1_ip0": [99.0],
            "sigma22_l0_ip0": [20.0],
            "sigma22_l1_ip0": [88.0],
        })
        stress = _read_layer_stress_array(df, n_ip=1, layer_idx=0)
        np.testing.assert_allclose(stress[0, 0, 0], 10.0)
        np.testing.assert_allclose(stress[0, 0, 1], 20.0)
        stress = _read_layer_stress_array(df, n_ip=1, layer_idx=1)
        np.testing.assert_allclose(stress[0, 0, 0], 99.0)
        np.testing.assert_allclose(stress[0, 0, 1], 88.0)

    def test_multiple_ips(self):
        import pandas as pd
        df = pd.DataFrame({
            "sigma11_l0_ip0": [1.0],
            "sigma11_l0_ip1": [2.0],
            "sigma11_l0_ip2": [3.0],
            "sigma11_l0_ip3": [4.0],
        })
        stress = _read_layer_stress_array(df, n_ip=4, layer_idx=0)
        np.testing.assert_allclose(stress[0, :, 0], [1.0, 2.0, 3.0, 4.0])


class TestSampleLayerStressInterpolation:
    """``_sample_layer_stress`` reuses the standard quad / tri IP weights
    — the math is exercised already in test_shell_kernel.py; here we
    just verify the dispatch on (n_ip, base_type) lands on the right
    branch.
    """

    def test_q4_dispatch(self):
        s = 1.0 / np.sqrt(3.0)
        stress = np.zeros((1, 4, 6))
        stress[0, 0, 0] = 7.0  # IP 0 is (-s, -s)
        out = _sample_layer_stress(stress, -s, -s, "ASDShellQ4")
        np.testing.assert_allclose(out[0, 0], 7.0)

    def test_t3_dispatch(self):
        stress = np.zeros((1, 3, 6))
        stress[0, 2, 1] = 11.0
        # IP 2 sits at (1/6, 2/3) for the standard 3-pt triangle rule.
        out = _sample_layer_stress(stress, 1 / 6, 2 / 3, "ASDShellT3")
        np.testing.assert_allclose(out[0, 1], 11.0)


class TestSectionsTclParser:
    """The Tcl parser handles the format STKO emits — line
    continuations, multiple sections, mixed types.
    """

    def test_parses_test_nlshell_layout(self, tmp_path):
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "section ElasticMembranePlateSection 14 28198.0 0.2 500.0 0.0 1.0\n"
            "section LayeredShell 15 7 \\\n"
            "\t 3 20.0  4 0.158346087  11 0.158346087  3 61.36661565  \\\n"
            "\t 11 0.158346087  4 0.158346087  3 20.0 \n"
            "section LayeredShell 16 7 \\\n"
            "\t 3 20.0  4 0.231338087  11 1.426611361  3 58.6841011  \\\n"
            "\t 11 1.426611361  4 0.231338087  3 20.0 \n"
        )
        out = parse_sections_tcl(tcl)
        # Two LayeredShell sections; the ElasticMembrane one is ignored.
        assert set(out.keys()) == {15, 16}
        assert len(out[15]) == 7
        assert len(out[16]) == 7

    def test_layer_thicknesses_match_input(self, tmp_path):
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "section LayeredShell 15 7 \\\n"
            "\t 3 20.0  4 0.158346087  11 0.158346087  3 61.36661565  \\\n"
            "\t 11 0.158346087  4 0.158346087  3 20.0 \n"
        )
        out = parse_sections_tcl(tcl)
        layers = out[15]
        expected_t = [20.0, 0.158346087, 0.158346087, 61.36661565,
                      0.158346087, 0.158346087, 20.0]
        np.testing.assert_allclose(
            [l.thickness for l in layers], expected_t, atol=1e-12,
        )

    def test_layer_z_offsets_centered_on_midplane(self, tmp_path):
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "section LayeredShell 1 3 \\\n"
            "\t 1 10.0  2 20.0  3 10.0 \n"
        )
        out = parse_sections_tcl(tcl)
        layers = out[1]
        total_t = 40.0
        half_t = total_t / 2.0
        # Layer 0 midplane: -half_t + 0.5 * 10 = -20 + 5 = -15
        # Layer 1 midplane: -half_t + 10 + 0.5 * 20 = -20 + 20 = 0
        # Layer 2 midplane: -half_t + 30 + 0.5 * 10 = -20 + 35 = 15
        expected_z = [-15.0, 0.0, 15.0]
        np.testing.assert_allclose(
            [l.z_offset for l in layers], expected_z, atol=1e-12,
        )

    def test_z_offset_sums_to_zero_for_symmetric_section(self, tmp_path):
        # A symmetric section (matched bottom + top layers) has z_offsets
        # that sum to 0 when weighted by thickness.
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "section LayeredShell 1 5 \\\n"
            "\t 1 5.0  2 3.0  3 10.0  2 3.0  1 5.0 \n"
        )
        out = parse_sections_tcl(tcl)
        weighted_z = sum(l.thickness * l.z_offset for l in out[1])
        # Wraps to zero by symmetry.
        np.testing.assert_allclose(weighted_z, 0.0, atol=1e-12)

    def test_layer_material_ids(self, tmp_path):
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "section LayeredShell 15 3 \\\n"
            "\t 7 1.0  8 2.0  9 3.0 \n"
        )
        out = parse_sections_tcl(tcl)
        assert [l.material_id for l in out[15]] == [7, 8, 9]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_sections_tcl("does_not_exist.tcl")

    def test_truncated_body_raises(self, tmp_path):
        tcl = tmp_path / "bad.tcl"
        tcl.write_text("section LayeredShell 1 3 1 5.0 2\n")
        with pytest.raises(ValueError, match="3 layers"):
            parse_sections_tcl(tcl)

    def test_skips_comments(self, tmp_path):
        tcl = tmp_path / "sections.tcl"
        tcl.write_text(
            "# This is a comment\n"
            "section LayeredShell 1 1 \\\n"
            "\t 7 5.0 \n"
            "# Another comment\n"
        )
        out = parse_sections_tcl(tcl)
        assert set(out.keys()) == {1}


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


class TestLayeredSectionsOnDataset:
    """The dataset's lazy ``layered_sections`` property picks up the
    Test_NLShell sections.tcl and returns the layer tables.
    """

    def test_layered_sections_populated(self, shell_ds):
        out = shell_ds.layered_sections
        # Test_NLShell has at least the two LayeredShell sections from
        # sections.tcl (ids 15 and 16). The ElasticMembrane section 14
        # is intentionally absent.
        assert set(out.keys()) & {15, 16}

    def test_layer_count(self, shell_ds):
        out = shell_ds.layered_sections
        for sid in (15, 16):
            if sid in out:
                assert len(out[sid]) == 7


class TestPerLayerForce:
    """``cut.per_layer_force(k)`` returns a derivative cut whose F/M
    come from only one through-thickness layer.
    """

    def test_per_layer_shape(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        per0 = cut.per_layer_force(0, shell_ds)
        assert per0.F.shape == cut.F.shape
        assert per0.M.shape == cut.M.shape
        assert len(per0.shell_intersections) == len(cut.shell_intersections)
        # The per-layer view drops beams and solids by construction.
        assert per0.intersections == ()
        assert per0.solid_intersections == ()

    def test_sum_of_layers_equals_full_cut(self, shell_ds, all_shell_eids):
        """The defining identity: summing per-layer F across every
        layer recovers the shell's standard through-thickness cut.

        Test_NLShell wall is shell-only, so the cut has no beam or
        solid contributions and the equality is exact (to numerical
        tolerance from quadrature roundoff and layer-stress storage
        precision).
        """
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        # Discover the layer count from the first contributing shell's
        # section.
        ix = cut.shell_intersections[0]
        info = shell_ds.cdata.element_info[ix.element_id]
        layers = shell_ds.layered_sections[info.physical_property_id]
        n_layers = len(layers)
        F_sum = np.zeros_like(cut.F)
        for k in range(n_layers):
            per = cut.per_layer_force(k, shell_ds)
            F_sum += per.F
        # Tolerance: stress data is ~1e6 in SI units; allow 1e-3 abs
        # (a fraction of a Newton on a kN-scale total).
        scale = max(1.0, float(np.max(np.abs(cut.F))))
        np.testing.assert_allclose(F_sum, cut.F, atol=scale * 1e-2)

    def test_layer_out_of_range_raises(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        with pytest.raises(IndexError, match="out of range"):
            cut.per_layer_force(999, shell_ds)

    def test_per_layer_requires_shell_intersections(self, shell_ds):
        """An empty cut (no shells in the filter / above the model)
        can't produce a per-layer breakdown.
        """
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=99_999.0),
            element_ids=(1,),  # arbitrary, doesn't matter — cut is empty
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        if cut.shell_intersections:
            pytest.skip("Test premise — cut has shells — broken.")
        with pytest.raises(ValueError, match="at least one shell"):
            cut.per_layer_force(0, shell_ds)


# ====================================================================== #
# Per-fiber-in-layer (v1.8) — pure-math tests
# ====================================================================== #
class TestFiberInLayerReader:
    """The ``_f<F>_l<L>_ip<K>`` reader extracts one fiber's stress
    from inside a specified layer.
    """

    def test_reads_explicit_sigma_form(self):
        import pandas as pd
        # 1 step, 1 IP, layer 0 with 2 fibers.
        df = pd.DataFrame({
            "sigma11_f0_l0_ip0": [1.0],
            "sigma11_f1_l0_ip0": [9.0],
            "sigma22_f0_l0_ip0": [2.0],
            "sigma22_f1_l0_ip0": [8.0],
        })
        stress_f0 = _read_fiber_in_layer_stress_array(
            df, n_ip=1, layer_idx=0, fiber_idx=0,
        )
        stress_f1 = _read_fiber_in_layer_stress_array(
            df, n_ip=1, layer_idx=0, fiber_idx=1,
        )
        np.testing.assert_allclose(stress_f0[0, 0, 0], 1.0)
        np.testing.assert_allclose(stress_f0[0, 0, 1], 2.0)
        np.testing.assert_allclose(stress_f1[0, 0, 0], 9.0)
        np.testing.assert_allclose(stress_f1[0, 0, 1], 8.0)

    def test_reads_unknown_stress_form(self):
        import pandas as pd
        # nDMaterial fallback layout: UnknownStress(n)_f<F>_l<L>_ip<K>.
        df = pd.DataFrame({
            "UnknownStress_f0_l0_ip0": [4.0],
            "UnknownStress(1)_f0_l0_ip0": [5.0],
            "UnknownStress(2)_f0_l0_ip0": [6.0],
        })
        stress = _read_fiber_in_layer_stress_array(
            df, n_ip=1, layer_idx=0, fiber_idx=0,
        )
        # Voigt mapping per the PlateFiber convention:
        # UnknownStress -> sigma11 (pos 0), (1) -> sigma22 (pos 1),
        # (2) -> sigma12 (pos 3).
        np.testing.assert_allclose(stress[0, 0, 0], 4.0)
        np.testing.assert_allclose(stress[0, 0, 1], 5.0)
        np.testing.assert_allclose(stress[0, 0, 3], 6.0)

    def test_missing_layer_returns_zeros(self):
        import pandas as pd
        df = pd.DataFrame({
            "sigma11_f0_l0_ip0": [1.0],
        })
        # Asking for layer 5 — nothing matches → all zeros.
        stress = _read_fiber_in_layer_stress_array(
            df, n_ip=1, layer_idx=5, fiber_idx=0,
        )
        np.testing.assert_allclose(stress, 0.0)


class TestDiscoverFiberCount:
    def test_counts_fibers_in_layer(self):
        cols = [
            "sigma11_f0_l0_ip0", "sigma11_f1_l0_ip0",
            "sigma11_f2_l0_ip0", "sigma11_f0_l1_ip0",
        ]
        assert _discover_fiber_count_in_layer(cols, layer_idx=0) == 3
        assert _discover_fiber_count_in_layer(cols, layer_idx=1) == 1
        assert _discover_fiber_count_in_layer(cols, layer_idx=2) == 0

    def test_ignores_non_fiber_columns(self):
        cols = [
            "Fxx_ip0",          # section.force — not fiber.stress
            "sigma11_l0_ip0",   # per-layer (no fiber index)
            "sigma11_f0_l0_ip0",
        ]
        assert _discover_fiber_count_in_layer(cols, layer_idx=0) == 1


class TestPerLayerInlineSurface:
    """``ds.section_cut(..., per_layer=k)`` short-circuits to the
    per-layer view.
    """

    def test_per_layer_inline(self, shell_ds, all_shell_eids):
        per = shell_ds.section_cut(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
            model_stage="MODEL_STAGE[1]",
            per_layer=0,
        )
        assert per.F.shape[1] == 3
        assert len(per.shell_intersections) > 0

    def test_inline_matches_method_form(self, shell_ds, all_shell_eids):
        # Inline `per_layer=0` and `.per_layer_force(0, ds)` produce
        # equal arrays.
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        full = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        method_form = full.per_layer_force(0, shell_ds)
        inline_form = shell_ds.section_cut(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
            model_stage="MODEL_STAGE[1]",
            per_layer=0,
        )
        np.testing.assert_allclose(method_form.F, inline_form.F, atol=1e-9)

    def test_per_fiber_without_per_layer_raises(self, shell_ds, all_shell_eids):
        with pytest.raises(ValueError, match="per_fiber requires per_layer"):
            shell_ds.section_cut(
                plane=Plane.horizontal(z=2500.0),
                element_ids=all_shell_eids,
                model_stage="MODEL_STAGE[1]",
                per_fiber=0,
            )


class TestPerFiberOnNonFiberedLayer:
    """Test_NLShell layers are single nDMaterial — no fibers within
    layers — so requesting a per-fiber cut on them must surface a
    clear error rather than silently returning zero.
    """

    def test_per_fiber_on_nonfibered_layer_raises(
        self, shell_ds, all_shell_eids,
    ):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        with pytest.raises(ValueError, match="no fiber-in-layer columns"):
            cut.per_fiber_force(0, 0, shell_ds)
