"""Integration tests for the ``bounding_polygon`` feature added in v1.6.

The polygon validation contract lives in ``test_specs.py``; the
plane-projection + Cyrus-Beck math lives in ``test_geometry.py``. This
file glues both together against real cut kernels:

- Beam kernel: an intersection point outside the polygon is dropped.
- Shell kernel: a chord crossing the polygon boundary is clipped.
- End-to-end: ``dataset.section_cut(plane=..., bounding_polygon=...)``
  produces the polygon-restricted cut.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCut, SectionCutSpec
from STKO_to_python.cuts.kernels import (
    SHELL_ELEMENT_CLASSES,
    compute_beam_cut,
    compute_shell_cut,
    find_beam_intersections,
    find_shell_intersections,
)


# ---------------------------------------------------------------------- #
# Beam side — polygon drops out-of-region intersections
# ---------------------------------------------------------------------- #
@pytest.fixture
def beam_ds(elastic_frame_dispbased_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dispbased_dir), "results", verbose=False)


class TestBeamBoundingPolygon:
    """``elasticFrame_mesh_displacementBased_results`` has two columns:
    one at x=0 and one at x=5000. A bounding polygon over only the
    left column should keep exactly one intersection at a horizontal
    cut through both.
    """

    def test_unbounded_cut_has_two_intersections(self, beam_ds):
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=all_eids,
        )
        ixs = find_beam_intersections(beam_ds, spec)
        # Two columns crossing.
        assert len(ixs) == 2
        xs = sorted(round(ix.point_global[0]) for ix in ixs)
        assert xs == [0, 5000]

    def test_polygon_keeps_only_left_column(self, beam_ds):
        # Square polygon covering x ∈ [-500, 500] on the z=1500 plane.
        # Column at x=5000 falls outside.
        poly = ((-500, -500, 1500), (500, -500, 1500),
                (500, 500, 1500), (-500, 500, 1500))
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=all_eids,
            bounding_polygon=poly,
        )
        cut = SectionCut.compute(spec, beam_ds, model_stage="MODEL_STAGE[1]")
        # Only the left column survives.
        assert len(cut.intersections) == 1
        ix = cut.intersections[0]
        assert abs(ix.point_global[0]) < 1.0  # x ≈ 0 (left column)

    def test_polygon_outside_both_columns_returns_empty(self, beam_ds):
        # Polygon at x ∈ [10000, 11000] — neither column is inside.
        poly = ((10000, -500, 1500), (11000, -500, 1500),
                (11000, 500, 1500), (10000, 500, 1500))
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=all_eids,
            bounding_polygon=poly,
        )
        cut = SectionCut.compute(spec, beam_ds, model_stage="MODEL_STAGE[1]")
        assert cut.is_empty
        # F and M have correct shape (0, 3).
        assert cut.F.shape == (0, 3)
        assert cut.M.shape == (0, 3)

    def test_polygon_halves_the_cut_force(self, beam_ds):
        """The unbounded cut carries F_z = +5000 (gravity) split
        equally between the two columns; the half-plate polygon picks
        up just one — F_z ≈ +2500.
        """
        plane = Plane.horizontal(z=1500.0)
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        full = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_eids),
            beam_ds, model_stage="MODEL_STAGE[1]",
        )
        poly = ((-500, -500, 1500), (500, -500, 1500),
                (500, 500, 1500), (-500, 500, 1500))
        half = SectionCut.compute(
            SectionCutSpec(plane=plane, element_ids=all_eids, bounding_polygon=poly),
            beam_ds, model_stage="MODEL_STAGE[1]",
        )
        # Step 0 (time=0.1): total F_z = +5000. Each column carries
        # half by symmetry (top beam mid-load splits equally) →
        # half cut F_z ≈ +2500.
        assert full.F[0, 2] == pytest.approx(5000.0, abs=1e-3)
        assert half.F[0, 2] == pytest.approx(2500.0, abs=1e-3)


# ---------------------------------------------------------------------- #
# Shell side — polygon clips chords
# ---------------------------------------------------------------------- #
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


class TestShellBoundingPolygon:
    """Cut at z=2500 through the Test_NLShell wall: the wall spans
    x ∈ [-485, 735]. A polygon over x ∈ [-1000, 0] should clip the
    chord of each crossing shell to its left portion.
    """

    def test_unbounded_cut_picks_up_shells(self, shell_ds, all_shell_eids):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0), element_ids=all_shell_eids,
        )
        ixs = find_shell_intersections(shell_ds, spec)
        assert len(ixs) > 0

    def test_polygon_clips_chords(self, shell_ds, all_shell_eids):
        # Bounding polygon: large square covering only x in [-1000, 0].
        poly = ((-1000, -1000, 2500), (0, -1000, 2500),
                (0, 1000, 2500), (-1000, 1000, 2500))
        full_spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0), element_ids=all_shell_eids,
        )
        clipped_spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
            bounding_polygon=poly,
        )
        full = SectionCut.compute(full_spec, shell_ds, model_stage="MODEL_STAGE[1]")
        clipped = SectionCut.compute(clipped_spec, shell_ds, model_stage="MODEL_STAGE[1]")
        # Every clipped chord's endpoints must have x ≤ 0.
        for ix in clipped.shell_intersections:
            for ep in ix.chord_endpoints_global:
                assert ep[0] <= 1e-6
        # Reduced extent ⇒ |F| of the clipped cut is no larger than
        # the full cut — strictly smaller in any meaningful case.
        assert np.linalg.norm(clipped.F[0]) <= np.linalg.norm(full.F[0]) + 1e-6

    def test_polygon_outside_returns_empty(self, shell_ds, all_shell_eids):
        # The Test_NLShell wall is at y=0; a polygon at y ∈ [100, 200]
        # misses every shell.
        poly = ((-1000, 100, 2500), (1000, 100, 2500),
                (1000, 200, 2500), (-1000, 200, 2500))
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=2500.0),
            element_ids=all_shell_eids,
            bounding_polygon=poly,
        )
        cut = SectionCut.compute(spec, shell_ds, model_stage="MODEL_STAGE[1]")
        assert cut.is_empty


# ---------------------------------------------------------------------- #
# Dataset API surface — inline kwarg threads through
# ---------------------------------------------------------------------- #
class TestDatasetAPIBoundingPolygon:
    def test_inline_kwarg(self, beam_ds):
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        cut = beam_ds.section_cut(
            plane=Plane.horizontal(z=1500.0),
            element_ids=all_eids,
            bounding_polygon=(
                (-500, -500, 1500), (500, -500, 1500),
                (500, 500, 1500), (-500, 500, 1500),
            ),
            model_stage="MODEL_STAGE[1]",
        )
        assert len(cut.intersections) == 1

    def test_spec_form_with_polygon(self, beam_ds):
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=all_eids,
            bounding_polygon=(
                (-500, -500, 1500), (500, -500, 1500),
                (500, 500, 1500), (-500, 500, 1500),
            ),
        )
        cut = beam_ds.section_cut(spec=spec, model_stage="MODEL_STAGE[1]")
        assert len(cut.intersections) == 1

    def test_spec_and_inline_polygon_raises(self, beam_ds):
        all_eids = tuple(beam_ds.elements_info["dataframe"]["element_id"].tolist())
        spec = SectionCutSpec(plane=Plane.horizontal(z=1500.0), element_ids=all_eids)
        with pytest.raises(ValueError, match="spec.*alone"):
            beam_ds.section_cut(
                spec=spec,
                bounding_polygon=((0, 0, 1500), (1, 0, 1500), (0, 1, 1500)),
                model_stage="MODEL_STAGE[1]",
            )
