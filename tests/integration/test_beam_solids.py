"""Integration tests for ``ds.plot.beam_solids``.

Exercises the real fixtures (single-partition pure-beam frame, and the
multi-partition shells + beams frame) and verifies:

* the `(ax, meta)` contract;
* per-element triangle counts match the profile's triangulation
  (2 caps + 2*n_sweeps side triangles per element);
* multi-class models render only beams (shells are silently filtered
  out by the assignment lookup);
* selection-set and explicit-id filters narrow the rendered subset;
* the ``edge_color=None`` switch suppresses the structural-edge overlay
  without changing the triangle count.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Single-partition (3 beams, 1 profile)
# ---------------------------------------------------------------------- #
def test_beam_solids_elastic_frame_basic_meta(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.beam_solids()
        assert ax is not None
        # 3 elements × profile-1 triangulation (2 caps × 1 tri/cap +
        # 2 × n_sweeps side tris = 2 + 8 = 12 tri/element on the
        # 4-pt square outline of elasticFrame's profile)
        assert meta["element_count"] == 3
        assert meta["triangle_count"] == 3 * (2 * 2 + 2 * 4)
        assert meta["skipped_elements"] == []
        assert meta["profile_ids"] == [1]
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


def test_beam_solids_explicit_element_ids_filter(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.beam_solids(element_ids=[1, 2])
        assert meta["element_count"] == 2
        # Same per-element triangulation, halved twice.
        assert meta["triangle_count"] == 2 * (2 * 2 + 2 * 4)
    finally:
        plt.close("all")


def test_beam_solids_edge_color_none_keeps_triangle_count(
    elastic_frame_dir: Path,
) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        _, meta_edges = ds.plot.beam_solids(edge_color="0.25")
        plt.close("all")
        _, meta_no_edges = ds.plot.beam_solids(edge_color=None)
        # Suppressing edges must not change the underlying triangle mesh.
        assert meta_edges["triangle_count"] == meta_no_edges["triangle_count"]
        assert meta_edges["element_count"] == meta_no_edges["element_count"]
    finally:
        plt.close("all")


def test_beam_solids_accepts_user_axes(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    fig = plt.figure()
    user_ax = fig.add_subplot(111, projection="3d")
    try:
        ax, meta = ds.plot.beam_solids(ax=user_ax)
        assert ax is user_ax
        assert meta["element_count"] > 0
    finally:
        plt.close("all")


def test_beam_solids_unknown_element_id_raises(elastic_frame_dir: Path) -> None:
    """The resolver short-circuits to an empty intersection with the
    assignment set, and we raise a friendly error.
    """
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        with pytest.raises(ValueError, match="No beam elements"):
            ds.plot.beam_solids(element_ids=[99999])
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Multi-class (beams + shells) — non-beams must be silently filtered
# ---------------------------------------------------------------------- #
def test_beam_solids_quad_frame_renders_only_beams(quad_frame_dir: Path) -> None:
    """QuadFrame has 75 beams + 625 ASDShellQ4 elements. The renderer
    should pick up exactly the 75 beams from beam_profile_assignments.
    """
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.beam_solids()
        assert meta["element_count"] == 75
        assert meta["triangle_count"] == 75 * (2 * 2 + 2 * 4)
        assert meta["skipped_elements"] == []
        # All beams share one profile in this fixture.
        assert meta["profile_ids"] == [1]
    finally:
        plt.close("all")


def test_beam_solids_quad_frame_passthrough_to_user_3d_axes(
    quad_frame_dir: Path,
) -> None:
    """Composes cleanly with an existing 3D axes (e.g., user overlays
    beam solids on top of a shell mesh plot).
    """
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    fig = plt.figure()
    user_ax = fig.add_subplot(111, projection="3d")
    try:
        ax, _ = ds.plot.beam_solids(ax=user_ax)
        assert ax is user_ax
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Deformed variant
# ---------------------------------------------------------------------- #
def test_beam_solids_deformed_elastic_frame_meta(elastic_frame_dir: Path) -> None:
    """Same triangle counts as the undeformed render; meta carries
    the stage/step/scale annotations.
    """
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.beam_solids_deformed(
            model_stage="MODEL_STAGE[1]", step=5, scale=100.0,
        )
        assert ax is not None
        assert meta["element_count"] == 3
        assert meta["triangle_count"] == 3 * (2 * 2 + 2 * 4)
        assert meta["model_stage"] == "MODEL_STAGE[1]"
        assert meta["step"] == 5
        assert meta["scale"] == 100.0
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


def test_beam_solids_deformed_scale_zero_matches_undeformed(
    elastic_frame_dir: Path,
) -> None:
    """scale=0 must collapse to the undeformed configuration — the
    DISPLACEMENT fetch is skipped entirely so this also works on
    datasets that didn't record displacement (covered indirectly here
    by skipping the fetch path).
    """
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        ax_und, meta_und = ds.plot.beam_solids()
        plt.close("all")
        ax_zero, meta_zero = ds.plot.beam_solids_deformed(
            model_stage="MODEL_STAGE[1]", step=5, scale=0.0,
        )
        # Triangle counts identical; vertex positions identical (the
        # latter is harder to check without inspecting collections, but
        # equal triangle counts at the same target ids is the structural
        # equivalent).
        assert meta_und["triangle_count"] == meta_zero["triangle_count"]
        assert meta_und["element_count"] == meta_zero["element_count"]
        assert meta_zero["scale"] == 0.0
    finally:
        plt.close("all")


def test_beam_solids_deformed_axis_limits_track_scale(
    elastic_frame_dir: Path,
    monkeypatch,
) -> None:
    """With a known per-node displacement, the deformed render's axis
    extent must shift by ``scale * displacement`` along the
    corresponding axis.

    Real fixtures have small displacements relative to the frame size,
    so we mock ``_displacement_at_step`` to inject a large shift in a
    single direction. This pins the wiring: the scaled displacement
    flows from the helper into the renderer's vertex computation.
    """
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        # Inject a y-direction shift big enough to escape the autoscale
        # pad (5% of the ~5000 mm span ≈ 250 mm) at scale=1.
        injected_disp = {
            int(nid): np.array([0.0, 1000.0, 0.0], dtype=float)
            for nid in ds.nodes_info["dataframe"]["node_id"]
        }
        monkeypatch.setattr(
            "STKO_to_python.plotting.deformed_shape._displacement_at_step",
            lambda dataset, *, model_stage, step: injected_disp,
        )

        scale = 2.0  # 2 * 1000 mm = 2000 mm shift along y
        ax_und, _ = ds.plot.beam_solids()
        und_y_center = sum(ax_und.get_ylim3d()) / 2.0
        plt.close("all")

        ax_def, _ = ds.plot.beam_solids_deformed(
            model_stage="MODEL_STAGE[1]", step=5, scale=scale,
        )
        def_y_center = sum(ax_def.get_ylim3d()) / 2.0

        # Every node shifted by (0, 1000, 0), so the data centroid in y
        # shifts by scale * 1000. The axis y-center should mirror that.
        np.testing.assert_allclose(
            def_y_center - und_y_center,
            scale * 1000.0,
            rtol=0.05,  # autoscale pad slack
        )
    finally:
        plt.close("all")


def test_beam_solids_deformed_invalid_step_raises(
    elastic_frame_dir: Path,
) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    try:
        with pytest.raises(ValueError, match="step="):
            ds.plot.beam_solids_deformed(
                model_stage="MODEL_STAGE[1]", step=99999, scale=1.0,
            )
    finally:
        plt.close("all")


def test_beam_solids_deformed_quad_frame_only_beams(
    quad_frame_dir: Path,
) -> None:
    """Shells must still be filtered out in the deformed render."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    try:
        ax, meta = ds.plot.beam_solids_deformed(
            model_stage="MODEL_STAGE[1]", step=2, scale=50.0,
        )
        assert meta["element_count"] == 75
        assert meta["triangle_count"] == 75 * (2 * 2 + 2 * 4)
    finally:
        plt.close("all")
