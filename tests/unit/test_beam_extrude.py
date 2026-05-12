"""Unit tests for :func:`STKO_to_python.plotting.beam_solid.extrude_beam_geometry`.

Pure-geometry tests — no matplotlib, no .mpco fixture. Builds synthetic
``BeamProfile`` instances with known shapes (single triangle, unit
square) and verifies vertex positions, face winding, dtypes, and the
identities ``vertices[k+n_pts] - vertices[k] == axis_end - axis_start``.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.model.cdata_reader import BeamProfile
from STKO_to_python.plotting.beam_solid import extrude_beam_geometry


# ---------------------------------------------------------------------- #
# Profile factories
# ---------------------------------------------------------------------- #
def _triangle_profile() -> BeamProfile:
    """Right triangle in the local (y, z) plane with one outline."""
    return BeamProfile(
        profile_id=1,
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        triangles=np.array([[0, 1, 2]], dtype=int),
        edges=[np.array([0, 1, 2, 0], dtype=int)],
        sweeps=np.array([0, 1, 2], dtype=int),
    )


def _unit_square_profile() -> BeamProfile:
    """Unit square in local (y, z), CCW from outside, two triangles, closed sweep."""
    return BeamProfile(
        profile_id=2,
        points=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=float,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        edges=[np.array([0, 1, 2, 3, 0], dtype=int)],
        sweeps=np.array([0, 1, 2, 3], dtype=int),
    )


def _no_caps_profile() -> BeamProfile:
    """Sweeps-only profile (open shell) — no fill triangles."""
    return BeamProfile(
        profile_id=3,
        points=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        triangles=np.empty((0, 3), dtype=int),
        edges=[np.array([0, 1], dtype=int)],
        sweeps=np.array([0, 1], dtype=int),
    )


# ---------------------------------------------------------------------- #
# Shape / dtype contract
# ---------------------------------------------------------------------- #
def test_vertex_count_is_twice_npts():
    p = _unit_square_profile()
    v, _ = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    assert v.shape == (8, 3)
    assert v.dtype == np.float64


def test_face_array_is_int64():
    p = _unit_square_profile()
    _, f = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    assert f.dtype == np.int64
    assert f.shape[1] == 3
    # 2 cap triangles per end + 4 sweep segments * 2 triangles
    assert f.shape[0] == 2 + 2 + 4 * 2


def test_face_indices_within_vertex_range():
    p = _unit_square_profile()
    v, f = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([5.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    assert f.min() >= 0
    assert f.max() < v.shape[0]


# ---------------------------------------------------------------------- #
# Vertex placement
# ---------------------------------------------------------------------- #
def test_identity_rotation_places_section_in_yz_plane():
    """With R=I and axis along global x, the cross-section's (y, z) coords
    map straight to global (y, z) at both endpoints.
    """
    p = _triangle_profile()
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([10.0, 0.0, 0.0])
    v, _ = extrude_beam_geometry(p, axis_start=start, axis_end=end, R=np.eye(3))

    # End-1 ring: (0, y, z) at axis_start
    expected_end1 = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    np.testing.assert_allclose(v[:3], expected_end1)

    # End-2 ring: same (0, y, z) added to axis_end
    expected_end2 = expected_end1 + np.array([10.0, 0.0, 0.0])
    np.testing.assert_allclose(v[3:], expected_end2)


def test_end_pair_differs_by_axis_delta():
    """For every k in [0, n_pts), vertices[k+n_pts] - vertices[k] must
    equal axis_end - axis_start exactly.
    """
    p = _unit_square_profile()
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([4.0, 8.0, -1.0])
    R = np.eye(3)
    v, _ = extrude_beam_geometry(p, axis_start=start, axis_end=end, R=R)
    n_pts = p.points.shape[0]
    delta = v[n_pts:] - v[:n_pts]
    expected = np.broadcast_to(end - start, (n_pts, 3))
    np.testing.assert_allclose(delta, expected)


def test_rotation_maps_local_y_to_global_x():
    """A 90-deg rotation about local-z that takes local-y to global-x.

    Equivalent matrix:
        R = [[0, -1, 0],
             [1,  0, 0],
             [0,  0, 1]]
    so R @ (0, 1, 0) = (-1, 0, 0). Wait — we need R @ local_y to be
    global +x. R @ (0, 1, 0) reads column-1 of R, so we want
    R[:, 1] == (1, 0, 0). Build R accordingly.
    """
    R = np.array(
        [[0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    # Profile with one point at local (y, z) = (1, 0)
    profile = BeamProfile(
        profile_id=99,
        points=np.array([[1.0, 0.0]], dtype=float),
        triangles=np.empty((0, 3), dtype=int),
        edges=[],
        sweeps=np.array([0], dtype=int),
    )
    v, _ = extrude_beam_geometry(
        profile, axis_start=np.zeros(3), axis_end=np.zeros(3), R=R,
    )
    # local (0, 1, 0) → global (1, 0, 0)
    np.testing.assert_allclose(v[0], [1.0, 0.0, 0.0])


def test_section_offset_translates_in_local_frame():
    """A section offset of (yOff, zOff) must shift every vertex by
    R @ (0, yOff, zOff) — the local-frame translation, then rotated.
    """
    p = _triangle_profile()
    R = np.eye(3)
    offset = np.array([0.5, -0.25])
    v_no_offset, _ = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]), R=R,
    )
    v_offset, _ = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]), R=R,
        section_offset=offset,
    )
    # With R = I, the global delta equals (0, 0.5, -0.25) per vertex.
    expected_delta = np.tile([0.0, 0.5, -0.25], (v_no_offset.shape[0], 1))
    np.testing.assert_allclose(v_offset - v_no_offset, expected_delta)


# ---------------------------------------------------------------------- #
# Face structure
# ---------------------------------------------------------------------- #
def test_end1_cap_winding_reversed():
    """End-1 cap reverses ``profile.triangles`` winding so the outward
    normal points along local -x (away from axis_end).
    """
    p = _triangle_profile()
    _, faces = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    # First face is end-1 cap, reversed: (0, 2, 1) from (0, 1, 2)
    assert faces[0].tolist() == [0, 2, 1]


def test_end2_cap_offset_by_npts():
    p = _triangle_profile()
    n_pts = p.points.shape[0]
    _, faces = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    # Second face is end-2 cap, same winding as profile.triangles[0]
    # shifted by n_pts: (0+3, 1+3, 2+3) = (3, 4, 5).
    assert faces[1].tolist() == [0 + n_pts, 1 + n_pts, 2 + n_pts]


def test_side_surface_uses_sweep_loop():
    """Square sweep [0, 1, 2, 3] should produce 4 quads = 8 triangles
    on the side surface, with indices wrapping around the loop.
    """
    p = _unit_square_profile()
    n_pts = p.points.shape[0]
    _, faces = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    # Skip the 2 + 2 cap faces; remaining are 8 side triangles.
    side = faces[4:]
    assert side.shape == (8, 3)
    # First side quad: a=0, b=1, c=1+n_pts, d=0+n_pts; triangulated
    # as (a, b, c) and (a, c, d).
    assert side[0].tolist() == [0, 1, 1 + n_pts]
    assert side[4].tolist() == [0, 1 + n_pts, 0 + n_pts]
    # Wrap-around quad: a=3, b=0, c=0+n_pts, d=3+n_pts.
    assert side[3].tolist() == [3, 0, 0 + n_pts]
    assert side[7].tolist() == [3, 0 + n_pts, 3 + n_pts]


def test_no_sweeps_emits_caps_only():
    """A profile with sweeps.size < 2 produces only the two end caps."""
    profile = BeamProfile(
        profile_id=4,
        points=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        triangles=np.array([[0, 1, 2]], dtype=int),
        edges=[],
        sweeps=np.array([0], dtype=int),  # single sweep — no segment
    )
    _, faces = extrude_beam_geometry(
        profile, axis_start=np.zeros(3), axis_end=np.array([1.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    # Just 2 caps (1 triangle each).
    assert faces.shape[0] == 2


def test_no_triangles_emits_side_surface_only():
    """A profile without fill triangles still produces side faces."""
    p = _no_caps_profile()
    _, faces = extrude_beam_geometry(
        p, axis_start=np.zeros(3), axis_end=np.array([2.0, 0.0, 0.0]),
        R=np.eye(3),
    )
    # No caps. Sweeps [0, 1] = 2 points; closed loop gives 2 segments,
    # each split into 2 triangles → 4 side triangles.
    assert faces.shape[0] == 4
    assert faces.min() >= 0 and faces.max() < 4  # 2*n_pts = 4 vertices


# ---------------------------------------------------------------------- #
# Real-fixture smoke test
# ---------------------------------------------------------------------- #
def test_extrude_runs_on_elastic_frame_first_element(elastic_frame_dir):
    """End-to-end smoke: real beam profile + real local axes + real
    end-node coords. Does not assert geometry correctness — just that
    the function returns sane shapes on a real fixture.
    """
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    profiles = ds.cdata.beam_profiles
    assignments = ds.cdata.beam_profile_assignments
    assert profiles and assignments

    eid = next(iter(assignments))
    pid, _weight = assignments[eid][0]
    profile = profiles[pid]
    R = ds.cdata.rotation_matrix(eid)

    # Pull the two end-node coords from the element index dataframe.
    df = ds.elements_info["dataframe"]
    row = df.loc[df["element_id"] == eid].iloc[0]
    node_ids = row["node_list"]
    nodes_df = ds.nodes_info["dataframe"]
    n0 = nodes_df.loc[nodes_df["node_id"] == int(node_ids[0])].iloc[0]
    n1 = nodes_df.loc[nodes_df["node_id"] == int(node_ids[1])].iloc[0]
    axis_start = np.array([n0["x"], n0["y"], n0["z"]], dtype=float)
    axis_end = np.array([n1["x"], n1["y"], n1["z"]], dtype=float)

    v, f = extrude_beam_geometry(
        profile, axis_start=axis_start, axis_end=axis_end, R=R,
    )
    n_pts = profile.points.shape[0]
    assert v.shape == (2 * n_pts, 3)
    assert f.shape[1] == 3
    assert f.dtype == np.int64
    # End-pair invariant on a real element too.
    delta = v[n_pts:] - v[:n_pts]
    np.testing.assert_allclose(delta, np.tile(axis_end - axis_start, (n_pts, 1)))
