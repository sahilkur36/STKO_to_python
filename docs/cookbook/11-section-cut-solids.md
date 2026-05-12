# Section cuts through brick continua

> Compute the cut force `(F, M)` across a plane that slices through a
> 3D continuum mesh — combining brick contributions with beams and
> shells in one resultant.

The engineering question is: *for the part of my model above some
elevation, what force does the discarded (below) part exert on the
kept (above) part — counting every element type the cut crosses?*
For continuum bricks, OpenSees writes the Cauchy stress at each Gauss
point in the `material.stress` recorder; v1.7.0 adds the solid kernel
that integrates the traction `t = σ · n_cut` over each crossing
element's planar polygon, then composes the contribution with the
beam and shell kernels in one `SectionCut`.

The fixture used here is
`stko_results_examples/solid_partition_example` — a two-partition
mesh combining `56-Brick` continuum with `64-DispBeamColumn3d` beams,
one model stage of analysis. **It is gitignored**, so the script
below short-circuits gracefully if the fixture is absent locally —
drop your own brick fixture in to run it.

```python
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec, SectionSweep
from STKO_to_python.cuts.kernels import SOLID_ELEMENT_CLASSES

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "solid_partition_example"
STAGE = "MODEL_STAGE[1]"

if not (DATASET / "Recorder.part-0.mpco").exists():
    sys.exit(f"Fixture not present at {DATASET}; skipping.")

ds = MPCODataSet(str(DATASET), "Recorder", verbose=False)
print(ds.unique_element_types)
# ['56-Brick[...]', '64-DispBeamColumn3d[...]']

all_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
```

---

## 1. Cut at a single elevation

The simplest form: one `Plane`, one inline call. With a filter that
includes both bricks and beams, the returned `SectionCut` carries
contributions from *every* element family the kernel knows how to
handle.

```python
z_cut = 5.0  # somewhere interior to the brick column
cut = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=all_eids,
    model_stage=STAGE,
)
print(cut)
# SectionCut(stage='MODEL_STAGE[1]', n_steps=1667, n_intersections=84,
#            side='positive')

print(f"beams crossing  : {len(cut.intersections)}")
print(f"shells crossing : {len(cut.shell_intersections)}")
print(f"solids crossing : {len(cut.solid_intersections)}")
print(f"F[0] = {cut.F[0]}")
```

The composed `SectionCut` carries three blocks of contributions; all
sum into `cut.F` / `cut.M` about a single centroid:

| Field | Source | Shape |
|---|---|---|
| `cut.intersections` | Beam kernel | `tuple[BeamIntersection, ...]` |
| `cut.shell_intersections` | Shell kernel | `tuple[ShellIntersection, ...]` |
| `cut.solid_intersections` | Solid kernel (v1.7+) | `tuple[SolidIntersection, ...]` |
| `cut.per_solid_F` | Per-element contribution | `dict[int, ndarray]` |
| `cut.per_solid_M_at_centroid` | Per-element moment (at the element's polygon centroid) | `dict[int, ndarray]` |
| `cut.contributing_element_ids` | All element ids across all kernels | `tuple[int, ...]` |

A `SolidIntersection` records the planar polygon clipped from the
element's volume in both global and natural coords:

```python
ix = cut.solid_intersections[0]
print(f"element_id     : {ix.element_id}")
print(f"element_type   : {ix.element_type}")
print(f"n_vertices     : {ix.n_vertices}")        # 3..6 for a hex, 3..4 for a tet
print(f"polygon_area   : {ix.polygon_area:.3f}")
print(f"polygon centroid (global) : {ix.polygon_centroid}")
```

---

## 2. Validate the cut

Newton's third law: the cut on the positive side plus the cut on the
negative side must sum to zero (moments transferred to a common
reference). This holds for any analysis — gravity-static, dynamic,
DRM, explicit — so it's a clean per-kernel sanity check that also
catches sign / rotation bugs in the *composition* (beam + shell +
solid):

```python
ok, residual = cut.consistency_check(ds, atol=1e-3, rtol=1e-6)
assert ok, f"Composed cut failed consistency, max residual {np.max(np.abs(residual))}"
```

A second validator compares two parallel cuts in a free-body band:

```python
cut_a = ds.section_cut(plane=Plane.horizontal(z=4.5), element_ids=all_eids, model_stage=STAGE)
cut_b = ds.section_cut(plane=Plane.horizontal(z=5.5), element_ids=all_eids, model_stage=STAGE)
ok, _ = cut_a.compare_to(cut_b, atol=1.0)  # tolerance scaled to the cut magnitude
```

Between the two planes the only loads are gravity body force on the
bricks in that band, so the resultant difference equals
`ρ g · V_band` (or for explicit dynamics, the inertial impulse).

---

## 3. Story-shear profile via `SectionSweep`

A profile of cut forces vs elevation is one `SectionSweep` call.
Bricks, beams, and shells all contribute at each plane automatically.

```python
elevations = np.linspace(1.0, 9.0, 9)
planes = [Plane.horizontal(z=z) for z in elevations]

sweep = ds.section_sweep(
    planes=planes,
    element_ids=all_eids,
    model_stage=STAGE,
)

env = sweep.envelope()
elevs = sweep.plane_locators("z")

fig, ax = plt.subplots()
ax.plot(env["Fz_peak_abs"], elevs, "o-", lw=1.5)
ax.set_xlabel("|F_z|  (axial-force envelope)")
ax.set_ylabel("Elevation z")
plt.show()
```

---

## 4. Restrict to a sub-region with `bounding_polygon`

A convex polygon on the cut plane clips both shell chords and solid
polygons against the polygon edge. Useful when the recorded selection
sets don't pre-filter to the region of interest — e.g. compute the
force carried by just the right half of a wide brick slab:

```python
z_cut = 5.0
right_half = (
    ( 0.0, -10.0, z_cut),
    (10.0, -10.0, z_cut),
    (10.0,  10.0, z_cut),
    ( 0.0,  10.0, z_cut),
)

full = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=all_eids,
    model_stage=STAGE,
)
right = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=all_eids,
    bounding_polygon=right_half,
    model_stage=STAGE,
)

print(f"Full   n_solids={len(full.solid_intersections):>3}, F_z[0]={full.F[0, 2]:>10.1f}")
print(f"Right  n_solids={len(right.solid_intersections):>3}, F_z[0]={right.F[0, 2]:>10.1f}")
```

For solids the kernel runs a Sutherland-Hodgman clip of the
intersection polygon against the bounding polygon, so the per-element
contribution comes from only the portion of the cut polygon inside the
region of interest.

---

## 5. Selecting only solids (or only beams)

`SOLID_ELEMENT_CLASSES` enumerates the classes the solid kernel
handles; combine it with the element-type column on `elements_info`
to filter for solid-only cuts:

```python
from STKO_to_python.cuts.kernels import SOLID_ELEMENT_CLASSES

df = ds.elements_info["dataframe"]
def _strip(t: str) -> str:
    return t.split("-", 1)[-1].split("[", 1)[0]
is_solid = df["element_type"].map(lambda t: _strip(t) in SOLID_ELEMENT_CLASSES)
brick_ids = tuple(int(x) for x in df.loc[is_solid, "element_id"])

solid_only = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=brick_ids,
    model_stage=STAGE,
)
assert len(solid_only.intersections) == 0      # no beams
assert len(solid_only.shell_intersections) == 0  # no shells
assert len(solid_only.solid_intersections) > 0
```

The current registry:

| Class | Topology | Default IP count |
|---|---|---|
| `Brick`, `BbarBrick`, `SSPbrick` | 8-node trilinear hex | 8 (2×2×2) |
| `Brick20`, `TwentyNodeBrick` | 20-node serendipity hex | 8 or 27 |
| `Brick27`, `TwentySevenNodeBrick` | 27-node Lagrange hex | 27 (3×3×3) |
| `FourNodeTetrahedron` | 4-node linear tet | 1 or 4 |

For 27-IP buckets the solid kernel uses **triquadratic Lagrange**
interpolation between the IPs; 8-IP buckets use trilinear
interpolation. Both are automatic — the kernel dispatches on
`er.n_ip`.

---

## 6. Higher-order hexes — what to expect

`Brick20` / `Brick27` carry midpoint / face / center nodes after the
8 corners. The geometry phase runs on the corners only (sound because
they define the convex hull; the cut polygon is on convex
polyhedra), and stress sampling uses the appropriate
trilinear-or-triquadratic interpolation between the volume IPs.

The corner-only geometry is an O(midpoint-deviation²)
approximation — for any well-conditioned higher-order hex the
midpoint nodes lie within a small fraction of the edge length of the
straight-edge midpoints, and the cut force is essentially identical
to the straight-edge approximation. For badly-shaped higher-order
elements (e.g. those with deliberately curved boundary surfaces) the
straight-edge approximation is a known limitation — see the v1.9
candidates in the project [`CHANGELOG.md`](https://github.com/nmorabowen/STKO_to_python/blob/main/CHANGELOG.md).

---

## Variations

- **Combine spec + bounding polygon for batch jobs**: build the
  `SectionCutSpec` once with `bounding_polygon=...` and pass it
  across model stages, datasets, or ground motions — same workflow
  shown for shells in [recipe 10](10-section-cut-shells.md).
- **Per-element diagnostics**: `cut.per_solid_F[eid]` carries the
  force contribution of one brick; useful for finding the heaviest
  contributor along a cut.
- **Polygon visualisation**: the planar polygon a brick contributes
  is `ix.polygon_global`. Plot in 2D after projecting to the plane
  basis — `STKO_to_python.cuts.geometry._project_to_plane_basis` is
  the helper used internally.
- **For an analysis with explicit dynamics**, the
  `cut_a.compare_to(cut_b)` residual carries the inertial impulse
  between the two cut planes — useful for verifying momentum balance.

For the layered-shell decomposition (per-layer / per-fiber views) see
[recipe 12](12-section-cut-layered-shells.md); for the full API
reference of the cuts subpackage see [Section cuts
API](../api/section-cuts.md).
