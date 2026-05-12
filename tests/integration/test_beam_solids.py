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
