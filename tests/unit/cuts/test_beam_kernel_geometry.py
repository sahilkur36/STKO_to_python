"""Geometry tests for STKO_to_python.cuts.kernels.beam.

Exercises ``find_beam_intersections`` against the small ``elasticFrame``
fixture: a 2D portal with two vertical columns and one horizontal top
beam, all ``5-ElasticBeam3d`` (closed-form, no section.force). That's
fine for the geometry phase — we only need connectivity and node
coordinates here, not internal forces.

Model layout from inspect (z up, x lateral):

    node 4 (0, 3000) ─── el 3 ─── node 2 (5000, 3000)
        │                                │
       el 2                              el 1
        │                                │
    node 3 (0, 0)                    node 1 (5000, 0)
"""
from __future__ import annotations

import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec
from STKO_to_python.cuts.kernels import (
    BEAM_ELEMENT_CLASSES,
    BeamIntersection,
    find_beam_intersections,
)


@pytest.fixture
def ds(elastic_frame_dir) -> MPCODataSet:
    return MPCODataSet(str(elastic_frame_dir), "results", verbose=False)


# ---------------------------------------------------------------------- #
# Basic geometric cases
# ---------------------------------------------------------------------- #
class TestMidHeightHorizontalCut:
    """Plane z=1500 cuts both columns at their midpoint, misses the top beam."""

    @pytest.fixture
    def hits(self, ds) -> list[BeamIntersection]:
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0),
            element_ids=(1, 2, 3),
        )
        return find_beam_intersections(ds, spec)

    def test_two_columns_hit(self, hits):
        assert len(hits) == 2

    def test_element_ids(self, hits):
        # Sorted by element_id in the kernel.
        assert [h.element_id for h in hits] == [1, 2]

    def test_xi_at_midpoint(self, hits):
        for h in hits:
            assert h.xi == pytest.approx(0.0)
            assert h.t == pytest.approx(0.5)

    def test_intersection_point_z(self, hits):
        for h in hits:
            assert h.point_global[2] == pytest.approx(1500.0)

    def test_intersection_point_x(self, hits):
        x_values = sorted(h.point_global[0] for h in hits)
        assert x_values == pytest.approx([0.0, 5000.0])

    def test_top_beam_skipped_when_parallel(self, hits):
        # Element 3 lies in the plane z=3000, parallel to z=1500 cut.
        # When the cut height is 1500 it's fully above the plane; here
        # we just confirm element_id 3 is absent.
        assert 3 not in [h.element_id for h in hits]

    def test_end_nodes_consistent(self, hits):
        for h in hits:
            n1, n2 = h.end_node_ids
            c1, c2 = h.end_coords
            assert c1[2] == 0.0
            assert c2[2] == 3000.0


# ---------------------------------------------------------------------- #
# Vertical cut hits the top beam
# ---------------------------------------------------------------------- #
class TestVerticalCutThroughTopBeam:
    @pytest.fixture
    def hits(self, ds) -> list[BeamIntersection]:
        spec = SectionCutSpec(
            plane=Plane.vertical(axis="x", at=2500.0),
            element_ids=(1, 2, 3),
        )
        return find_beam_intersections(ds, spec)

    def test_only_top_beam_hit(self, hits):
        assert len(hits) == 1
        assert hits[0].element_id == 3

    def test_midpoint(self, hits):
        h = hits[0]
        assert h.xi == pytest.approx(0.0)
        assert h.point_global == pytest.approx((2500.0, 0.0, 3000.0))


# ---------------------------------------------------------------------- #
# Cut exactly at a node
# ---------------------------------------------------------------------- #
class TestCutAtBaseNodes:
    """Plane z=0 should report the two columns as crossing at their base."""

    @pytest.fixture
    def hits(self, ds) -> list[BeamIntersection]:
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=0.0),
            element_ids=(1, 2, 3),
        )
        return find_beam_intersections(ds, spec)

    def test_two_columns_hit(self, hits):
        assert len(hits) == 2
        assert sorted(h.element_id for h in hits) == [1, 2]

    def test_xi_at_first_node(self, hits):
        for h in hits:
            assert h.xi == pytest.approx(-1.0)
            assert h.t == pytest.approx(0.0)


# ---------------------------------------------------------------------- #
# Top beam parallel to a cut at its own elevation -> skip
# ---------------------------------------------------------------------- #
class TestCutThroughTopBeamPlane:
    """Plane z=3000 contains the top beam and the top end-nodes of the columns.

    Expected behavior:
    - Top beam (element 3) lies in the plane -> intersect_segment returns
      None (parallel), so it's skipped.
    - Columns 1 and 2 end on the plane -> reported at xi = +1.
    """

    @pytest.fixture
    def hits(self, ds) -> list[BeamIntersection]:
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=3000.0),
            element_ids=(1, 2, 3),
        )
        return find_beam_intersections(ds, spec)

    def test_top_beam_skipped(self, hits):
        assert 3 not in [h.element_id for h in hits]

    def test_columns_at_top_station(self, hits):
        column_hits = [h for h in hits if h.element_id in (1, 2)]
        assert len(column_hits) == 2
        for h in column_hits:
            assert h.xi == pytest.approx(1.0)
            assert h.t == pytest.approx(1.0)
            assert h.point_global[2] == pytest.approx(3000.0)


# ---------------------------------------------------------------------- #
# Cut outside the model -> no hits
# ---------------------------------------------------------------------- #
class TestCutOutsideModel:
    def test_above(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=4000.0), element_ids=(1, 2, 3),
        )
        assert find_beam_intersections(ds, spec) == []

    def test_below(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=-1.0), element_ids=(1, 2, 3),
        )
        assert find_beam_intersections(ds, spec) == []


# ---------------------------------------------------------------------- #
# Filter behavior
# ---------------------------------------------------------------------- #
class TestFiltering:
    def test_subset_filter_returns_only_listed(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=(1,),
        )
        hits = find_beam_intersections(ds, spec)
        assert [h.element_id for h in hits] == [1]

    def test_unrelated_explicit_id_silently_ignored(self, ds):
        # ID 999 doesn't exist; resolver returns the union, the kernel
        # should just not find it among beams (it's neither a beam nor
        # in elements_info), and report only the real beams.
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=1500.0), element_ids=(1, 999),
        )
        hits = find_beam_intersections(ds, spec)
        assert [h.element_id for h in hits] == [1]


# ---------------------------------------------------------------------- #
# Structural invariants of BeamIntersection
# ---------------------------------------------------------------------- #
class TestBeamIntersectionInvariants:
    def test_xi_t_consistency(self, ds):
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=750.0), element_ids=(1, 2, 3),
        )
        for h in find_beam_intersections(ds, spec):
            assert h.xi == pytest.approx(2.0 * h.t - 1.0)

    def test_point_on_segment(self, ds):
        # Intersection point must be a convex combination of the end
        # coords with weight t (and 1 - t).
        import numpy as np
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=750.0), element_ids=(1, 2, 3),
        )
        for h in find_beam_intersections(ds, spec):
            c1, c2 = h.end_coords_arr
            expected = (1 - h.t) * c1 + h.t * c2
            np.testing.assert_allclose(h.point_arr, expected, atol=1e-9)

    def test_axis_length_matches_node_distance(self, ds):
        import numpy as np
        spec = SectionCutSpec(
            plane=Plane.horizontal(z=750.0), element_ids=(1, 2, 3),
        )
        for h in find_beam_intersections(ds, spec):
            c1, c2 = h.end_coords_arr
            expected = float(np.linalg.norm(c2 - c1))
            assert h.axis_length == pytest.approx(expected)


# ---------------------------------------------------------------------- #
# BEAM_ELEMENT_CLASSES coverage check
# ---------------------------------------------------------------------- #
class TestBeamClassRegistry:
    def test_registry_is_nonempty(self):
        assert len(BEAM_ELEMENT_CLASSES) >= 1

    def test_registry_does_not_contain_class_tags(self):
        # Entries must be stripped class names ("ElasticBeam3d"), not
        # decorated tag prefixes ("5-ElasticBeam3d").
        for name in BEAM_ELEMENT_CLASSES:
            assert "-" not in name or not name.split("-")[0].isdigit()
