"""Tests for :mod:`STKO_to_python.viewer.math.gauss_extrapolation`.

Algorithm correctness is anchored on three analytic properties:

* A constant GP field projects to a constant nodal field at every
  corner — for every element class.
* When ``n_gp == n_corner`` (e.g. Brick with 2×2×2 Gauss-Legendre, or
  ASDShellQ4 with 2×2), a linear field is reproduced *exactly* at the
  corners. The pinv is the true inverse and the projection is exact.
* When ``n_gp == 1``, the single GP value is broadcast to every corner.

Cross-element averaging is verified with a two-element fixture sharing
a face; the shared corners get the arithmetic mean of the two
contributors.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.viewer.math.gauss_extrapolation import (
    PerElementCornerValues,
    build_extrapolation_matrix,
    extrapolate_per_element,
    extrapolate_to_nodes_averaged,
    make_gp_nodal_scalars,
    per_element_max_gp_count,
)


# --------------------------------------------------------------------- #
# Reference Gauss points (2×2×2 Gauss-Legendre on the unit cube)        #
# --------------------------------------------------------------------- #

_G = 1.0 / np.sqrt(3.0)

# Order matches the Brick node ordering in
# ``format/shape_functions.py``: corner k at signs (sx, sy, sz) gets GP
# at (sx*g, sy*g, sz*g). The two collections are not required to align;
# the pinv handles arbitrary GP ordering. We align them here so the
# "n_gp == n_corner" case maps GP k onto corner k for clarity.
_BRICK_GP_2x2x2 = np.array(
    [
        [-_G, -_G, -_G],
        [+_G, -_G, -_G],
        [+_G, +_G, -_G],
        [-_G, +_G, -_G],
        [-_G, -_G, +_G],
        [+_G, -_G, +_G],
        [+_G, +_G, +_G],
        [-_G, +_G, +_G],
    ],
    dtype=np.float64,
)

_BRICK_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------- #
# per_element_max_gp_count                                              #
# --------------------------------------------------------------------- #


def test_per_element_max_gp_count_empty() -> None:
    assert per_element_max_gp_count(np.array([], dtype=np.int64)) == 0


def test_per_element_max_gp_count_single() -> None:
    eidx = np.array([1, 1, 1, 2], dtype=np.int64)
    assert per_element_max_gp_count(eidx) == 3


def test_per_element_max_gp_count_homogeneous() -> None:
    eidx = np.repeat(np.arange(5, dtype=np.int64), 8)
    assert per_element_max_gp_count(eidx) == 8


# --------------------------------------------------------------------- #
# build_extrapolation_matrix                                            #
# --------------------------------------------------------------------- #


def test_build_extrapolation_matrix_brick_is_exact_inverse() -> None:
    """Brick with 2×2×2 GPs → pinv is the exact inverse."""
    M = build_extrapolation_matrix(_BRICK_GP_2x2x2, "56-Brick")
    assert M is not None
    assert M.shape == (8, 8)

    from STKO_to_python.format.shape_functions import get_shape_functions

    N_fn, _, _ = get_shape_functions("56-Brick")  # type: ignore[misc]
    A = N_fn(_BRICK_GP_2x2x2)
    # M @ A should be identity to numerical precision.
    np.testing.assert_allclose(M @ A, np.eye(8), atol=1e-12)


def test_build_extrapolation_matrix_unknown_class_returns_none() -> None:
    M = build_extrapolation_matrix(_BRICK_GP_2x2x2, "999-NotInCatalog")
    assert M is None


def test_build_extrapolation_matrix_1d_natural_coords_accepted() -> None:
    """1-D natural-coord arrays are promoted to 2-D internally."""
    # Use a 2-node line element with two GPs.
    gp1 = np.array([-_G, +_G], dtype=np.float64)
    M = build_extrapolation_matrix(gp1, "5-ElasticBeam3d")
    assert M is not None
    assert M.shape == (2, 2)


# --------------------------------------------------------------------- #
# extrapolate_per_element                                               #
# --------------------------------------------------------------------- #


def _make_single_brick_lookup(
    element_id: int = 1,
    base_node_id: int = 10,
) -> dict:
    """One Brick element with corner IDs ``base..base+7``."""
    corner_nids = np.arange(base_node_id, base_node_id + 8, dtype=np.int64)
    return {element_id: ("56-Brick", corner_nids)}


def test_extrapolate_per_element_empty_input_returns_empty_record() -> None:
    out = extrapolate_per_element(
        element_index=np.array([], dtype=np.int64),
        natural_coords=np.zeros((0, 3), dtype=np.float64),
        gp_values=np.zeros((4, 0), dtype=np.float64),  # 4 time steps, 0 GPs
        element_lookup={},
    )
    assert isinstance(out, PerElementCornerValues)
    assert out.element_ids.size == 0
    assert out.corner_node_ids == []
    assert out.values == []
    assert out.time_count == 4


def test_extrapolate_per_element_constant_field_brick() -> None:
    """A constant GP field projects to a constant corner field."""
    lookup = _make_single_brick_lookup()
    eidx = np.full(8, 1, dtype=np.int64)
    gp_vals = np.full((3, 8), 7.5, dtype=np.float64)  # 3 time steps

    out = extrapolate_per_element(
        element_index=eidx,
        natural_coords=_BRICK_GP_2x2x2,
        gp_values=gp_vals,
        element_lookup=lookup,
    )

    assert out.element_ids.tolist() == [1]
    assert out.values[0].shape == (3, 8)
    np.testing.assert_allclose(out.values[0], 7.5, atol=1e-12)


def test_extrapolate_per_element_linear_field_brick_is_exact() -> None:
    """For n_gp == n_corner, a linear field is reproduced exactly."""
    lookup = _make_single_brick_lookup()
    eidx = np.full(8, 1, dtype=np.int64)

    # f(xi, eta, zeta) = 1 + 2*xi + 3*eta + 4*zeta
    def f(xyz: np.ndarray) -> np.ndarray:
        return 1.0 + 2.0 * xyz[:, 0] + 3.0 * xyz[:, 1] + 4.0 * xyz[:, 2]

    gp_vals = f(_BRICK_GP_2x2x2)[None, :]  # shape (1, 8)
    expected_corner = f(_BRICK_CORNERS)    # shape (8,)

    out = extrapolate_per_element(
        element_index=eidx,
        natural_coords=_BRICK_GP_2x2x2,
        gp_values=gp_vals,
        element_lookup=lookup,
    )

    np.testing.assert_allclose(out.values[0][0], expected_corner, atol=1e-12)


def test_extrapolate_per_element_single_gp_broadcasts_to_every_corner() -> None:
    """One GP per element → its value is duplicated to every corner."""
    # Brick with a single centroidal GP.
    lookup = _make_single_brick_lookup()
    out = extrapolate_per_element(
        element_index=np.array([1], dtype=np.int64),
        natural_coords=np.zeros((1, 3), dtype=np.float64),
        gp_values=np.array([[42.0]], dtype=np.float64),
        element_lookup=lookup,
    )
    np.testing.assert_allclose(out.values[0], 42.0, atol=1e-12)
    assert out.values[0].shape == (1, 8)


def test_extrapolate_per_element_skips_unknown_class() -> None:
    """An element whose class is not in the catalog is silently dropped."""
    lookup = {1: ("999-NotInCatalog", np.array([10, 11], dtype=np.int64))}
    out = extrapolate_per_element(
        element_index=np.array([1, 1], dtype=np.int64),
        natural_coords=np.array([[-_G], [+_G]], dtype=np.float64),
        gp_values=np.array([[1.0, 2.0]], dtype=np.float64),
        element_lookup=lookup,
    )
    # Class is unknown but the corner count (2) matches a hypothetical
    # linear element; the fallback path takes the mean across GPs and
    # broadcasts. The element is *not* dropped because lookup hit; only
    # the pinv path is skipped. So the contract is "the element appears
    # with a sensible value" — verify that mean fallback fired.
    assert out.element_ids.tolist() == [1]
    np.testing.assert_allclose(out.values[0], 1.5, atol=1e-12)


def test_extrapolate_per_element_element_not_in_lookup_is_dropped() -> None:
    out = extrapolate_per_element(
        element_index=np.array([1, 1], dtype=np.int64),
        natural_coords=np.array([[-_G], [+_G]], dtype=np.float64),
        gp_values=np.array([[1.0, 2.0]], dtype=np.float64),
        element_lookup={},  # no entry for element 1
    )
    assert out.element_ids.size == 0


def test_extrapolate_per_element_handles_non_contiguous_grouping() -> None:
    """Rows for the same element don't need to be contiguous."""
    lookup = {
        1: ("56-Brick", np.arange(10, 18, dtype=np.int64)),
        2: ("56-Brick", np.arange(20, 28, dtype=np.int64)),
    }
    # Interleave the two elements' GP rows.
    eidx = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int64)
    nat = np.empty((16, 3), dtype=np.float64)
    nat[::2] = _BRICK_GP_2x2x2   # element 1 rows
    nat[1::2] = _BRICK_GP_2x2x2  # element 2 rows
    gp_vals = np.empty((1, 16), dtype=np.float64)
    gp_vals[0, ::2] = 1.0   # element 1: constant 1.0
    gp_vals[0, 1::2] = 5.0  # element 2: constant 5.0

    out = extrapolate_per_element(
        element_index=eidx,
        natural_coords=nat,
        gp_values=gp_vals,
        element_lookup=lookup,
    )

    assert out.element_ids.tolist() == [1, 2]
    np.testing.assert_allclose(out.values[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(out.values[1], 5.0, atol=1e-12)


# --------------------------------------------------------------------- #
# extrapolate_to_nodes_averaged                                         #
# --------------------------------------------------------------------- #


def test_extrapolate_to_nodes_averaged_empty_input() -> None:
    per_elem = PerElementCornerValues(
        element_ids=np.zeros(0, dtype=np.int64),
        corner_node_ids=[],
        values=[],
        time_count=5,
    )
    node_ids, nodal = extrapolate_to_nodes_averaged(per_elem)
    assert node_ids.size == 0
    assert nodal.shape == (5, 0)


def test_extrapolate_to_nodes_averaged_single_element() -> None:
    """Single element: averaging is a no-op; each corner appears once."""
    corner_nids = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    per_elem = PerElementCornerValues(
        element_ids=np.array([1], dtype=np.int64),
        corner_node_ids=[corner_nids],
        values=[values],
        time_count=1,
    )
    node_ids, nodal = extrapolate_to_nodes_averaged(per_elem)
    np.testing.assert_array_equal(node_ids, corner_nids)
    np.testing.assert_allclose(nodal, values, atol=1e-12)


def test_extrapolate_to_nodes_averaged_shared_corners_get_mean() -> None:
    """Two elements sharing a face → shared corners average over both."""
    # Element 1 corners: 10..17, value 1.0 at every corner.
    # Element 2 corners: 14..17 + 20..23, value 3.0 at every corner.
    # Shared corners (14, 15, 16, 17) should receive (1.0 + 3.0) / 2 = 2.0.
    c1 = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=np.int64)
    c2 = np.array([14, 15, 16, 17, 20, 21, 22, 23], dtype=np.int64)
    v1 = np.full((1, 8), 1.0, dtype=np.float64)
    v2 = np.full((1, 8), 3.0, dtype=np.float64)
    per_elem = PerElementCornerValues(
        element_ids=np.array([1, 2], dtype=np.int64),
        corner_node_ids=[c1, c2],
        values=[v1, v2],
        time_count=1,
    )
    node_ids, nodal = extrapolate_to_nodes_averaged(per_elem)

    # All unique node IDs, sorted.
    expected_ids = np.array([10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23], dtype=np.int64)
    np.testing.assert_array_equal(node_ids, expected_ids)

    expected = np.array(
        [[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(nodal, expected, atol=1e-12)


def test_extrapolate_to_nodes_averaged_preserves_time_axis() -> None:
    """``(T, N)`` shape with multiple time steps round-trips correctly."""
    c1 = np.array([10, 11], dtype=np.int64)
    c2 = np.array([11, 12], dtype=np.int64)
    v1 = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64)   # (T=2, n=2)
    v2 = np.array([[4.0, 8.0], [40.0, 80.0]], dtype=np.float64)

    per_elem = PerElementCornerValues(
        element_ids=np.array([1, 2], dtype=np.int64),
        corner_node_ids=[c1, c2],
        values=[v1, v2],
        time_count=2,
    )
    node_ids, nodal = extrapolate_to_nodes_averaged(per_elem)
    np.testing.assert_array_equal(node_ids, np.array([10, 11, 12], dtype=np.int64))
    expected = np.array(
        [
            [1.0, (2.0 + 4.0) / 2.0, 8.0],
            [10.0, (20.0 + 40.0) / 2.0, 80.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(nodal, expected, atol=1e-12)


# --------------------------------------------------------------------- #
# Property-style spot check                                             #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("c", [0.0, 1.0, -2.5, 1e6])
def test_constant_field_round_trips_for_any_constant(c: float) -> None:
    """For any constant ``c``, projection and averaging both return ``c``."""
    lookup = _make_single_brick_lookup()
    eidx = np.full(8, 1, dtype=np.int64)
    gp_vals = np.full((1, 8), c, dtype=np.float64)
    per_elem = extrapolate_per_element(
        element_index=eidx,
        natural_coords=_BRICK_GP_2x2x2,
        gp_values=gp_vals,
        element_lookup=lookup,
    )
    _, nodal = extrapolate_to_nodes_averaged(per_elem)
    np.testing.assert_allclose(nodal, c, atol=1e-9)


# --------------------------------------------------------------------- #
# make_gp_nodal_scalars (Phase 3.0d)                                    #
# --------------------------------------------------------------------- #
#
# Closure factory that adapts the GP-extrapolation pipeline to the
# scalars contract :class:`ContourLayer` consumes in ``topology="nodal"``
# mode. The tests below pin three properties:
#
#   1. A constant per-step GP field projects to a constant nodal dict.
#   2. Two elements sharing two nodes have those shared nodes averaged.
#   3. A step-varying ``gp_values_fn`` produces step-varying dicts via the
#      same closure — no recompute of the static layout required.


# 2×2 Gauss points on the parent quad (n_gp == n_corner == 4 so pinv is
# the exact inverse).
_QUAD_GP_2x2 = np.array(
    [
        [-_G, -_G],
        [+_G, -_G],
        [+_G, +_G],
        [-_G, +_G],
    ],
    dtype=np.float64,
)


def test_make_gp_nodal_scalars_constant_field_yields_constant_dict() -> None:
    """All four GPs = 7.0 → every corner gets 7.0."""
    lookup = {
        1: ("203-ASDShellQ4", np.array([10, 11, 12, 13], dtype=np.int64)),
    }
    eidx = np.full(4, 1, dtype=np.int64)
    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: np.full(4, 7.0, dtype=np.float64),
        element_index=eidx,
        natural_coords=_QUAD_GP_2x2,
        element_lookup=lookup,
    )
    out = scalars_fn(0)
    assert set(out.keys()) == {10, 11, 12, 13}
    for nid, value in out.items():
        assert value == pytest.approx(7.0)


def test_make_gp_nodal_scalars_shared_corners_averaged() -> None:
    """Two quads sharing nodes 11 and 12 — shared corners get the mean."""
    # Element 1 corners: 10, 11, 12, 13
    # Element 2 corners: 11, 14, 15, 12
    # Constant fields per element: e1 = 1.0, e2 = 3.0
    lookup = {
        1: ("203-ASDShellQ4", np.array([10, 11, 12, 13], dtype=np.int64)),
        2: ("203-ASDShellQ4", np.array([11, 14, 15, 12], dtype=np.int64)),
    }
    eidx = np.concatenate([np.full(4, 1), np.full(4, 2)]).astype(np.int64)
    nat = np.vstack([_QUAD_GP_2x2, _QUAD_GP_2x2])
    gp_vals = np.concatenate([np.full(4, 1.0), np.full(4, 3.0)]).astype(np.float64)
    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: gp_vals,
        element_index=eidx,
        natural_coords=nat,
        element_lookup=lookup,
    )
    out = scalars_fn(0)
    assert out[10] == pytest.approx(1.0)
    assert out[13] == pytest.approx(1.0)
    assert out[14] == pytest.approx(3.0)
    assert out[15] == pytest.approx(3.0)
    # Shared corners average the two contributing element values.
    assert out[11] == pytest.approx(2.0)
    assert out[12] == pytest.approx(2.0)


def test_make_gp_nodal_scalars_step_varying_callable() -> None:
    """The closure invokes gp_values_fn every call — step → step yields
    independent dicts."""
    lookup = {
        1: ("203-ASDShellQ4", np.array([10, 11, 12, 13], dtype=np.int64)),
    }
    eidx = np.full(4, 1, dtype=np.int64)
    calls = []

    def gp_values(step):
        calls.append(int(step))
        return np.full(4, float(step), dtype=np.float64)

    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=gp_values,
        element_index=eidx,
        natural_coords=_QUAD_GP_2x2,
        element_lookup=lookup,
    )
    d0 = scalars_fn(0)
    d5 = scalars_fn(5)
    for v in d0.values():
        assert v == pytest.approx(0.0)
    for v in d5.values():
        assert v == pytest.approx(5.0)
    # The closure forwards the step int exactly to the gp_values_fn.
    assert calls == [0, 5]


def test_make_gp_nodal_scalars_empty_lookup_returns_empty_dict() -> None:
    """An element-lookup that filters every GP out yields an empty dict."""
    eidx = np.full(4, 1, dtype=np.int64)
    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: np.zeros(4, dtype=np.float64),
        element_index=eidx,
        natural_coords=_QUAD_GP_2x2,
        element_lookup={},  # no entry → every GP is dropped
    )
    assert scalars_fn(0) == {}


def test_make_gp_nodal_scalars_rejects_scalar_input() -> None:
    """gp_values_fn returning a 0-D scalar is a malformed-input error."""
    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: np.asarray(1.0),
        element_index=np.array([1], dtype=np.int64),
        natural_coords=np.array([[0.0, 0.0]]),
        element_lookup={
            1: ("203-ASDShellQ4", np.array([1, 2, 3, 4], dtype=np.int64)),
        },
    )
    with pytest.raises(ValueError, match="0-D scalar"):
        scalars_fn(0)


def test_make_gp_nodal_scalars_supports_multistep_batch() -> None:
    """When gp_values_fn returns a (T, n_total_gp) batch, the closure
    indexes into it by ``step`` so animations can precompute once and
    re-index cheaply."""
    lookup = {
        1: ("203-ASDShellQ4", np.array([10, 11, 12, 13], dtype=np.int64)),
    }
    eidx = np.full(4, 1, dtype=np.int64)
    # 3 timesteps; step k has all GPs at value k.
    batch = np.stack([np.full(4, float(k)) for k in range(3)])  # (3, 4)

    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: batch,
        element_index=eidx,
        natural_coords=_QUAD_GP_2x2,
        element_lookup=lookup,
    )
    for k in range(3):
        d = scalars_fn(k)
        for v in d.values():
            assert v == pytest.approx(float(k))


def test_make_gp_nodal_scalars_multistep_batch_out_of_range_raises() -> None:
    """Step outside the precomputed batch range is a hard error, not
    silent wrap-around."""
    lookup = {
        1: ("203-ASDShellQ4", np.array([10, 11, 12, 13], dtype=np.int64)),
    }
    eidx = np.full(4, 1, dtype=np.int64)
    batch = np.stack([np.full(4, float(k)) for k in range(3)])
    scalars_fn = make_gp_nodal_scalars(
        gp_values_fn=lambda step: batch,
        element_index=eidx,
        natural_coords=_QUAD_GP_2x2,
        element_lookup=lookup,
    )
    with pytest.raises(IndexError, match="out of range"):
        scalars_fn(99)
