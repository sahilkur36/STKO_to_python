"""Section cuts for solids and layered shells (v1.7 + v1.8 surfaces).

Demonstrates the section-cut subpackage's continuum + layered-shell
features end-to-end:

    Layered shells (against ``stko_results_examples/Test_NLShell``):
      1. Standard `SectionCut` through the wall — recap of v1.6.
      2. `MPCODataSet.layered_sections` — parsed LayeredShell geometry.
      3. `cut.per_layer_force(k, ds)` — through-thickness decomposition.
      4. Sum-of-layers identity — Sum_k per_layer.F ≈ cut.F.
      5. `ds.section_cut(..., per_layer=k)` inline shortcut.
      6. `per_fiber_force` error path on non-fibered layers.

    Solids (against ``stko_results_examples/solid_partition_example``):
      7. Brick-only cut + per-element diagnostics.
      8. Composed (beam + solid) cut + consistency_check.
      9. SectionSweep with brick + beam contributions.
     10. `bounding_polygon` clipping over a half-slab region.

The solid sections (7-10) short-circuit gracefully if the heavy
fixture is absent locally. The layered-shell sections (1-6) run
against ``Test_NLShell`` which is also gitignored, so the whole
script silently exits if neither fixture is present.

Run with::

    python examples/section_cut_solid_and_layered_example.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import LayerInfo, MPCODataSet
from STKO_to_python.cuts import Plane, SectionCut, SectionCutSpec
from STKO_to_python.cuts.kernels import SOLID_ELEMENT_CLASSES


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_examples_dir(repo_root: Path) -> Path:
    """Locate the ``stko_results_examples`` dir.

    When this script is launched from a ``.claude/worktrees/<name>``
    worktree (developer workflow), the heavy fixtures live in the
    main checkout rather than the worktree-local copy. Fall back to
    the main checkout in that case so the example still runs.
    """
    local = repo_root / "stko_results_examples"
    has_heavy = (local / "Test_NLShell" / "Results.part-0.mpco").exists() or (
        local / "solid_partition_example" / "Recorder.part-0.mpco"
    ).exists()
    if has_heavy:
        return local

    parts = repo_root.parts
    try:
        idx = parts.index(".claude")
    except ValueError:
        return local
    if idx + 2 < len(parts) and parts[idx + 1] == "worktrees":
        main_root = Path(*parts[:idx])
        candidate = main_root / "stko_results_examples"
        if candidate.exists():
            return candidate
    return local


EXAMPLES_DIR = _resolve_examples_dir(REPO_ROOT)
NL_SHELL_DIR = EXAMPLES_DIR / "Test_NLShell"
SOLID_DIR = EXAMPLES_DIR / "solid_partition_example"


def main() -> None:
    print(f"NL_SHELL_DIR : {NL_SHELL_DIR}")
    print(f"SOLID_DIR    : {SOLID_DIR}")

    layered_present = (NL_SHELL_DIR / "Results.part-0.mpco").exists()
    solid_present = (SOLID_DIR / "Recorder.part-0.mpco").exists()

    if not layered_present and not solid_present:
        print("Neither fixture is present — nothing to do.")
        return

    if layered_present:
        demonstrate_layered_shells()
    else:
        print("Skipping layered-shell demo (Test_NLShell absent).")

    if solid_present:
        demonstrate_solids()
    else:
        print("Skipping solid demo (solid_partition_example absent).")


# ---------------------------------------------------------------------------
# Layered-shell sections (v1.7+v1.8)
# ---------------------------------------------------------------------------
def demonstrate_layered_shells() -> None:
    print("\n" + "=" * 72)
    print(" Layered-shell decomposition (v1.7 per-layer + v1.8 per-fiber)")
    print("=" * 72)

    ds = MPCODataSet(str(NL_SHELL_DIR), "Results", verbose=False)
    shell_eids = tuple(int(x) for x in ds.elements_info["dataframe"]["element_id"])
    stage = "MODEL_STAGE[1]"

    # ---- 1. Standard cut through the wall (re-cap from v1.6) -------------
    cut = ds.section_cut(
        plane=Plane.horizontal(z=2500.0),
        element_ids=shell_eids,
        model_stage=stage,
    )
    print(f"\n[1] Standard cut: {cut}")
    print(f"    F[0] = {cut.F[0]}")
    print(f"    shells crossing the plane: {len(cut.shell_intersections)}")

    # ---- 2. Parsed LayeredShell table -----------------------------------
    sections = ds.layered_sections
    print(f"\n[2] dataset.layered_sections keys: {sorted(sections)}")
    # Pick the section the contributing shells use to print a layer table.
    ix0 = cut.shell_intersections[0]
    section_id = ds.cdata.element_info[ix0.element_id].physical_property_id
    layers = sections[section_id]
    print(f"    element {ix0.element_id} uses LayeredShell section {section_id} "
          f"({len(layers)} layers):")
    for k, layer in enumerate(layers):
        print(f"      layer {k}: mat={layer.material_id:>2}  "
              f"t={layer.thickness:>9.4f}  z_offset={layer.z_offset:>+9.4f}")

    # ---- 3. Per-layer breakdown ----------------------------------------
    n_layers = len(layers)
    per_layer = [cut.per_layer_force(k, ds) for k in range(n_layers)]
    print(f"\n[3] per_layer_force per layer at step 0 (F_x, F_y, F_z):")
    for k, p in enumerate(per_layer):
        print(f"      layer {k}: {p.F[0]}")

    # ---- 4. Sum-of-layers identity --------------------------------------
    F_sum = sum(p.F for p in per_layer)
    abs_diff = float(np.max(np.abs(F_sum - cut.F)))
    rel_scale = max(1.0, float(np.max(np.abs(cut.F))))
    print(f"\n[4] sum(per_layer) F max diff vs full cut.F: "
          f"{abs_diff:.6g}  (~{abs_diff / rel_scale * 100:.3f}%)")

    # ---- 5. Inline per_layer shortcut -----------------------------------
    inline_top = ds.section_cut(
        plane=Plane.horizontal(z=2500.0),
        element_ids=shell_eids,
        model_stage=stage,
        per_layer=n_layers - 1,
    )
    np.testing.assert_allclose(per_layer[-1].F, inline_top.F, atol=1e-9)
    print(f"\n[5] ds.section_cut(..., per_layer={n_layers - 1}) matches the "
          f"method-form per_layer_force to numerical tolerance.")

    # ---- 6. Per-fiber error path on a non-fibered layer -----------------
    print(f"\n[6] Test_NLShell's layers are single nDMaterial each (no fibers).")
    print(f"    cut.per_fiber_force(0, 0, ds) on a non-fibered layer raises:")
    try:
        cut.per_fiber_force(0, 0, ds)
    except ValueError as exc:
        print(f"      ValueError: {exc}")

    # ---- Plot: through-thickness force distribution ---------------------
    z_offsets = [layer.z_offset for layer in layers]
    F_x_per_layer = [p.F[0, 0] for p in per_layer]
    heights = [layer.thickness * 0.9 for layer in layers]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.barh(z_offsets, F_x_per_layer, height=heights, edgecolor="black", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5)
    ax.set_xlabel("F_x carried by layer  (step 0, N)")
    ax.set_ylabel("Layer midplane z (mm)")
    ax.set_title("Through-thickness force distribution at z=2500")
    plt.tight_layout()
    out = REPO_ROOT / "examples" / "_out_layered_shell_per_layer.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n    Plot saved: {out}")


# ---------------------------------------------------------------------------
# Solid sections (v1.7)
# ---------------------------------------------------------------------------
def demonstrate_solids() -> None:
    print("\n" + "=" * 72)
    print(" Solid (continuum) section cuts (v1.7)")
    print("=" * 72)

    ds = MPCODataSet(str(SOLID_DIR), "Recorder", verbose=False)
    df = ds.elements_info["dataframe"]
    stage = "MODEL_STAGE[1]"

    def _strip(t: str) -> str:
        return t.split("-", 1)[-1].split("[", 1)[0]

    is_solid = df["element_type"].map(lambda t: _strip(t) in SOLID_ELEMENT_CLASSES)
    brick_ids = tuple(int(x) for x in df.loc[is_solid, "element_id"])
    all_ids = tuple(int(x) for x in df["element_id"])
    print(f"\nBricks in fixture: {len(brick_ids)}   total elements: {len(all_ids)}")

    # Pick a cut elevation interior to the brick subdomain
    z_bricks = []
    nodes = ds.nodes_info["dataframe"]
    node_z = dict(zip(nodes["node_id"].tolist(), nodes["z"].tolist()))
    eid_set = set(brick_ids)
    for r in df.itertuples(index=False):
        if int(r.element_id) in eid_set:
            z_bricks.extend(node_z[int(nid)] for nid in r.node_list if int(nid) in node_z)
    z_min, z_max = float(min(z_bricks)), float(max(z_bricks))
    z_mid = 0.5 * (z_min + z_max)
    print(f"Brick mesh z range: [{z_min:.3f}, {z_max:.3f}]; cutting at z={z_mid:.3f}")

    # ---- 7. Brick-only cut + per-element diagnostics --------------------
    brick_only = ds.section_cut(
        plane=Plane.horizontal(z=z_mid),
        element_ids=brick_ids,
        model_stage=stage,
    )
    print(f"\n[7] Brick-only cut: {brick_only}")
    print(f"    solids crossing: {len(brick_only.solid_intersections)}")
    print(f"    F[0] = {brick_only.F[0]}")
    if brick_only.solid_intersections:
        heaviest = max(
            brick_only.solid_intersections,
            key=lambda ix: float(np.max(np.abs(brick_only.per_solid_F[ix.element_id]))),
        )
        print(f"    heaviest contributor: element {heaviest.element_id} "
              f"(polygon area = {heaviest.polygon_area:.3f})")

    # ---- 8. Composed (beam + solid) cut --------------------------------
    composed = ds.section_cut(
        plane=Plane.horizontal(z=z_mid),
        element_ids=all_ids,
        model_stage=stage,
    )
    print(f"\n[8] Composed cut: {composed}")
    print(f"    beams  : {len(composed.intersections)}")
    print(f"    shells : {len(composed.shell_intersections)}")
    print(f"    solids : {len(composed.solid_intersections)}")
    scale = max(1.0, float(np.max(np.abs(composed.F))), float(np.max(np.abs(composed.M))))
    ok, residual = composed.consistency_check(ds, atol=scale * 1e-3, rtol=1e-6)
    print(f"    consistency_check: ok={ok}  max residual={np.max(np.abs(residual)):.3g}")

    # ---- 9. SectionSweep over elevations --------------------------------
    zs = np.linspace(z_min + 0.1 * (z_max - z_min), z_max - 0.1 * (z_max - z_min), 6)
    planes = [Plane.horizontal(z=float(z)) for z in zs]
    sweep = ds.section_sweep(
        planes=planes,
        element_ids=all_ids,
        model_stage=stage,
    )
    print(f"\n[9] SectionSweep: {sweep}")
    env = sweep.envelope()
    elevs = sweep.plane_locators("z")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(env["Fz_peak_abs"], elevs, "o-", lw=1.5)
    ax.set_xlabel("|F_z| envelope")
    ax.set_ylabel("Elevation z")
    ax.set_title("Brick + beam composed cut sweep")
    plt.tight_layout()
    out = REPO_ROOT / "examples" / "_out_solid_sweep.png"
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    Plot saved: {out}")

    # ---- 10. bounding_polygon -----------------------------------------
    # A polygon covering the half of the slab where x >= mid_x.
    x_vals = [
        node_z[int(nid)] for r in df.itertuples(index=False)
        if int(r.element_id) in eid_set
        for nid in r.node_list if int(nid) in node_z
    ]  # repurposed to z; for x we need a similar dict
    node_x = dict(zip(nodes["node_id"].tolist(), nodes["x"].tolist()))
    x_min = float(min(node_x.values()))
    x_max = float(max(node_x.values()))
    mid_x = 0.5 * (x_min + x_max)
    big = max(abs(x_min), abs(x_max)) * 2.0
    right_poly = (
        (mid_x, -big, z_mid),
        (   big, -big, z_mid),
        (   big,  big, z_mid),
        (mid_x,  big, z_mid),
    )
    right = ds.section_cut(
        plane=Plane.horizontal(z=z_mid),
        element_ids=all_ids,
        model_stage=stage,
        bounding_polygon=right_poly,
    )
    print(f"\n[10] Right-half cut (x >= {mid_x:.2f}):")
    print(f"     solids in full cut  : {len(composed.solid_intersections)}")
    print(f"     solids in right half: {len(right.solid_intersections)}")
    print(f"     F_z[0] full  = {composed.F[0, 2]:+12.3f}")
    print(f"     F_z[0] right = {right.F[0, 2]:+12.3f}")


if __name__ == "__main__":
    main()
