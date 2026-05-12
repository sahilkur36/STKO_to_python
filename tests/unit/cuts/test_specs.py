"""Unit tests for STKO_to_python.cuts.specs."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.cuts import DriftSpec, Plane, SectionCutSpec


PLANE = Plane.horizontal(z=3.0)


# ---------------------------------------------------------------------- #
# SectionCutSpec
# ---------------------------------------------------------------------- #
class TestSectionCutSpecConstruction:
    def test_by_selection_set_name(self):
        spec = SectionCutSpec(plane=PLANE, selection_set_name="story_2")
        assert spec.plane is PLANE
        assert spec.selection_set_name == "story_2"
        assert spec.side == "positive"

    def test_by_selection_set_id(self):
        spec = SectionCutSpec(plane=PLANE, selection_set_id=3)
        assert spec.selection_set_id == 3
        assert isinstance(spec.selection_set_id, int)

    def test_selection_set_id_coerced_from_numpy(self):
        spec = SectionCutSpec(plane=PLANE, selection_set_id=np.int64(7))
        assert isinstance(spec.selection_set_id, int)
        assert spec.selection_set_id == 7

    def test_by_element_ids(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1, 2, 3))
        assert spec.element_ids == (1, 2, 3)

    def test_element_ids_coerced_from_list(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=[1, 2, 3])
        assert spec.element_ids == (1, 2, 3)
        assert isinstance(spec.element_ids, tuple)

    def test_element_ids_coerced_from_ndarray(self):
        arr = np.array([4, 5, 6], dtype=np.int64)
        spec = SectionCutSpec(plane=PLANE, element_ids=arr)
        assert spec.element_ids == (4, 5, 6)

    def test_multiple_filters_allowed(self):
        # Resolver takes the union; spec should permit specifying all.
        spec = SectionCutSpec(
            plane=PLANE,
            selection_set_name="story_2",
            element_ids=(10, 11),
        )
        assert spec.selection_set_name == "story_2"
        assert spec.element_ids == (10, 11)

    def test_negative_side(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,), side="negative")
        assert spec.side == "negative"

    def test_label_and_name(self):
        spec = SectionCutSpec(
            plane=PLANE, element_ids=(1,),
            label="Story 2 shear", name="story_2_cut",
        )
        assert spec.label == "Story 2 shear"
        assert spec.name == "story_2_cut"


class TestSectionCutSpecValidation:
    def test_missing_all_filters_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SectionCutSpec(plane=PLANE)

    def test_bad_side_raises(self):
        with pytest.raises(ValueError, match="positive.*negative"):
            SectionCutSpec(plane=PLANE, element_ids=(1,), side="up")  # type: ignore[arg-type]

    def test_plane_type_check(self):
        with pytest.raises(TypeError, match="Plane"):
            SectionCutSpec(plane="not a plane", element_ids=(1,))  # type: ignore[arg-type]

    def test_empty_element_ids_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SectionCutSpec(plane=PLANE, element_ids=())

    def test_non_int_element_ids_raises(self):
        with pytest.raises(ValueError, match="integers"):
            SectionCutSpec(plane=PLANE, element_ids=("a", "b"))  # type: ignore[arg-type]


class TestSectionCutSpecSignedNormal:
    def test_positive_side_returns_plane_normal(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,), side="positive")
        np.testing.assert_allclose(spec.signed_normal, PLANE.normal_arr)

    def test_negative_side_flips_normal(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,), side="negative")
        np.testing.assert_allclose(spec.signed_normal, -PLANE.normal_arr)


class TestSectionCutSpecImmutability:
    def test_frozen(self):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,))
        with pytest.raises(Exception):
            spec.side = "negative"  # type: ignore[misc]

    def test_hashable(self):
        a = SectionCutSpec(plane=PLANE, selection_set_name="story_2")
        b = SectionCutSpec(plane=PLANE, selection_set_name="story_2")
        assert hash(a) == hash(b)
        assert {a, b} == {a}

    def test_equality_value_based(self):
        a = SectionCutSpec(plane=PLANE, element_ids=(1, 2, 3))
        b = SectionCutSpec(plane=PLANE, element_ids=(1, 2, 3))
        assert a == b

    def test_inequality_on_side(self):
        a = SectionCutSpec(plane=PLANE, element_ids=(1,), side="positive")
        b = SectionCutSpec(plane=PLANE, element_ids=(1,), side="negative")
        assert a != b


class TestSectionCutSpecPickle:
    def test_roundtrip_plain(self, tmp_path):
        original = SectionCutSpec(
            plane=PLANE, selection_set_name="story_2",
            side="negative", label="Lvl 2",
        )
        path = original.save_pickle(tmp_path / "spec.pkl")
        restored = SectionCutSpec.load_pickle(path)
        assert restored == original

    def test_roundtrip_gzipped(self, tmp_path):
        original = SectionCutSpec(plane=PLANE, element_ids=(1, 2, 3))
        path = original.save_pickle(tmp_path / "spec.pkl.gz")
        assert path.suffix == ".gz"
        restored = SectionCutSpec.load_pickle(path)
        assert restored == original

    def test_load_wrong_type_raises(self, tmp_path):
        # Pickle a DriftSpec and try to load as SectionCutSpec.
        drift = DriftSpec(top_node=4, bottom_node=1, component=1)
        path = drift.save_pickle(tmp_path / "drift.pkl")
        with pytest.raises(TypeError, match="expected SectionCutSpec"):
            SectionCutSpec.load_pickle(path)

    def test_plain_pickle_module_works(self):
        # Spec should pickle via the stdlib too.
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,))
        restored = pickle.loads(pickle.dumps(spec))
        assert restored == spec


# ---------------------------------------------------------------------- #
# DriftSpec
# ---------------------------------------------------------------------- #
class TestDriftSpecConstruction:
    def test_basic(self):
        spec = DriftSpec(top_node=4, bottom_node=1, component=1)
        assert spec.top_node == 4
        assert spec.bottom_node == 1
        assert spec.component == 1
        assert spec.normalize_by is None

    def test_string_component(self):
        spec = DriftSpec(top_node=4, bottom_node=1, component="X")
        assert spec.component == "X"

    def test_normalize_by(self):
        spec = DriftSpec(top_node=4, bottom_node=1, component=1, normalize_by=3.0)
        assert spec.normalize_by == 3.0

    def test_numpy_node_ids_coerced(self):
        spec = DriftSpec(top_node=np.int64(4), bottom_node=np.int64(1), component=1)
        assert isinstance(spec.top_node, int)
        assert spec.top_node == 4

    def test_label(self):
        spec = DriftSpec(
            top_node=4, bottom_node=1, component=1, label="Roof drift",
        )
        assert spec.label == "Roof drift"


class TestDriftSpecValidation:
    def test_same_nodes_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            DriftSpec(top_node=1, bottom_node=1, component=1)

    def test_non_int_nodes_raises(self):
        with pytest.raises(ValueError, match="integers"):
            DriftSpec(top_node="a", bottom_node=1, component=1)  # type: ignore[arg-type]

    def test_zero_normalize_raises(self):
        with pytest.raises(ValueError, match="nonzero"):
            DriftSpec(top_node=4, bottom_node=1, component=1, normalize_by=0.0)


class TestDriftSpecImmutability:
    def test_frozen(self):
        spec = DriftSpec(top_node=4, bottom_node=1, component=1)
        with pytest.raises(Exception):
            spec.top_node = 5  # type: ignore[misc]

    def test_hashable(self):
        a = DriftSpec(top_node=4, bottom_node=1, component=1)
        b = DriftSpec(top_node=4, bottom_node=1, component=1)
        assert {a, b} == {a}

    def test_value_equality(self):
        a = DriftSpec(top_node=4, bottom_node=1, component=1, normalize_by=3.0)
        b = DriftSpec(top_node=4, bottom_node=1, component=1, normalize_by=3.0)
        assert a == b


class TestDriftSpecPickle:
    def test_roundtrip(self, tmp_path):
        original = DriftSpec(
            top_node=4, bottom_node=1, component=1,
            normalize_by=3.0, label="Roof drift",
        )
        path = original.save_pickle(tmp_path / "drift.pkl")
        restored = DriftSpec.load_pickle(path)
        assert restored == original

    def test_load_wrong_type_raises(self, tmp_path):
        spec = SectionCutSpec(plane=PLANE, element_ids=(1,))
        path = spec.save_pickle(tmp_path / "spec.pkl")
        with pytest.raises(TypeError, match="expected DriftSpec"):
            DriftSpec.load_pickle(path)


# ---------------------------------------------------------------------- #
# DriftSpec.apply — integration against the elasticFrame fixture
# ---------------------------------------------------------------------- #
class TestDriftSpecApply:
    """Exercise apply() end-to-end with a real .mpco fixture."""

    def test_returns_series_against_elastic_frame(self, elastic_frame_dir):
        from STKO_to_python import MPCODataSet

        ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
        spec = DriftSpec(top_node=4, bottom_node=1, component=1)
        s = spec.apply(ds, model_stage="MODEL_STAGE[1]")
        assert isinstance(s, pd.Series)
        assert s.size > 0

    def test_normalize_by_divides_series(self, elastic_frame_dir):
        from STKO_to_python import MPCODataSet

        ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
        raw = DriftSpec(top_node=4, bottom_node=1, component=1)
        ratio = DriftSpec(top_node=4, bottom_node=1, component=1, normalize_by=3.0)
        s_raw = raw.apply(ds, model_stage="MODEL_STAGE[1]")
        s_ratio = ratio.apply(ds, model_stage="MODEL_STAGE[1]")
        np.testing.assert_allclose(s_ratio.to_numpy(), s_raw.to_numpy() / 3.0)

    def test_label_sets_series_name(self, elastic_frame_dir):
        from STKO_to_python import MPCODataSet

        ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
        spec = DriftSpec(
            top_node=4, bottom_node=1, component=1, label="Roof drift",
        )
        s = spec.apply(ds, model_stage="MODEL_STAGE[1]")
        assert s.name == "Roof drift"
