# Section cuts through frames with shells

> Compute the cut force `(F, M)` across a horizontal plane that slices
> through a structural wall meshed with shells — and use a bounding
> polygon to restrict the cut to just the left half.

The engineering question is: *I have a model whose lateral system
involves shell-meshed walls; for each step, what force does each story
diaphragm transmit?* In OpenSees the answer lives in
`section.force` on each shell, which records the through-thickness-
integrated membrane / bending / shear tensors at every integration
point. v1.6.0 wires this into the same `section_cut` surface used for
beams.

The fixture used here is `stko_results_examples/Test_NLShell` — a
multi-partition (4 ranks) wall meshed with `ASDShellT3` below
`z=870` and `ASDShellQ4` above, with three model stages of analysis.

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec, SectionSweep

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "Test_NLShell"
STAGE = "MODEL_STAGE[1]"

ds = MPCODataSet(str(DATASET), "Results", verbose=False)
shell_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
```

---

## 1. Cut at a single elevation

The simplest form: one `Plane`, one inline call.

```python
cut = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage=STAGE,
)
print(cut)
# SectionCut(stage='MODEL_STAGE[1]', n_steps=10, n_intersections=6,
#            side='positive')

print(cut.F[0])              # [F_x, F_y, F_z] at step 0
print(cut.envelope())        # peak per component
```

Returned `SectionCut` fields:

| Field | Shape | Meaning |
|---|---|---|
| `cut.F` | `(n_steps, 3)` | Force in **global** axes the discarded side exerts on the kept side. |
| `cut.M` | `(n_steps, 3)` | Moment about `cut.centroid` in global axes. |
| `cut.time` | `(n_steps,)` | Time axis. |
| `cut.centroid` | `(3,)` | Reference point — mean of beam intersection points + shell chord midpoints. |
| `cut.intersections` | `tuple[BeamIntersection, ...]` | Beam contributions. |
| `cut.shell_intersections` | `tuple[ShellIntersection, ...]` | Shell contributions. |

For a wall-only model `cut.intersections` is empty and
`cut.shell_intersections` carries the 6 shells the plane crosses.

---

## 2. Validate the cut

The first thing to do with any new cut is to **check that it's
internally consistent**. Newton's third law: the cut force on the
positive side plus the cut force on the negative side must sum to
zero (with the moments transferred to a common reference). This holds
for any analysis — gravity-static, dynamic, DRM, even non-classical
boundary conditions — so it's a clean sanity check.

```python
ok, residual = cut.consistency_check(ds, atol=1e-3)
assert ok, f"Cut failed consistency, max residual {residual.max()}"
```

`residual` is shape `(n_steps, 6)` for the `F + F_neg` and
`M + M_neg_at_self_centroid` deltas. Use it to debug a failing check.

A second, complementary validator compares two parallel cuts at
planes between which there's no external load — they must give the
same resultant (transferred to a common point):

```python
cut_a = ds.section_cut(plane=Plane.horizontal(z=2500.0), element_ids=shell_eids, model_stage=STAGE)
cut_b = ds.section_cut(plane=Plane.horizontal(z=2700.0), element_ids=shell_eids, model_stage=STAGE)
ok, _ = cut_a.compare_to(cut_b, atol=1e-3)
```

This catches sign / rotation bugs that the per-side check might miss
when the shells happen to share the same orientation.

---

## 3. Story-shear profile via `SectionSweep`

For "shear vs elevation" the natural object is `SectionSweep` — a
list of cuts at parallel planes sharing one filter and one model
stage.

```python
elevations = np.linspace(500.0, 4000.0, 8)
planes = [Plane.horizontal(z=z) for z in elevations]

sweep = ds.section_sweep(
    planes=planes,
    element_ids=shell_eids,
    model_stage=STAGE,
)
print(sweep)   # SectionSweep(n_planes=8, model_stage='MODEL_STAGE[1]')
```

A `SectionSweep` is iterable (`for cut in sweep`) and indexable
(`sweep[3]`). Aggregations:

| Call | Shape |
|---|---|
| `sweep.envelope()` | `(n_planes, 18)` — peak `max/min/peak_abs` per component, per plane. |
| `sweep.to_dataframe("Fx")` | `(n_steps, n_planes)` — wide DataFrame for one component, rows=time. |
| `sweep.plane_locators("z")` | `(n_planes,)` — z-coords for an elevation axis. |

Plot the peak F_x vs elevation:

```python
env = sweep.envelope()
elevs = sweep.plane_locators("z")

fig, ax = plt.subplots()
ax.plot(env["Fx_peak_abs"], elevs, "o-", lw=1.5)
ax.set_xlabel("|F_x|  (story shear envelope)")
ax.set_ylabel("Elevation z")
plt.show()
```

---

## 4. Cut only the left half of the wall via `bounding_polygon`

The Test_NLShell wall spans roughly `x ∈ [-485, 735]`. To compute the
shear carried by only the **left half** without rebuilding selection
sets in STKO, drop a convex polygon on the cut plane covering the
left region.

```python
z_cut = 2500.0
left_polygon = (
    (-1000.0, -1000.0, z_cut),
    (   125.0, -1000.0, z_cut),
    (   125.0,  1000.0, z_cut),
    (-1000.0,  1000.0, z_cut),
)

full = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=shell_eids,
    model_stage=STAGE,
)
left = ds.section_cut(
    plane=Plane.horizontal(z=z_cut),
    element_ids=shell_eids,
    bounding_polygon=left_polygon,
    model_stage=STAGE,
)

print(f"Full  n_shells = {len(full.shell_intersections):>2}, F_z[0] = {full.F[0, 2]:.1f}")
print(f"Left  n_shells = {len(left.shell_intersections):>2}, F_z[0] = {left.F[0, 2]:.1f}")
```

A few things to note:

- Every vertex of `bounding_polygon` must lie on the cut plane within
  `1e-6` (validated at `SectionCutSpec` construction).
- The polygon must be **convex** — Cyrus-Beck clipping assumes
  convexity. Non-convex regions can be split into convex sub-polygons
  and the cuts added.
- Polygons are stored as `tuple[tuple[float, float, float], ...]` so
  the spec stays hashable and pickle-stable.
- For shells the bounding polygon clips the chord; for beams it drops
  any intersection point outside the polygon.

---

## 5. Spec-driven cuts (re-use the same definition across cases)

For multi-case studies — same cut against different ground motions,
different stages, different parameter values — define the cut once
as a `SectionCutSpec` and apply it everywhere.

```python
spec = SectionCutSpec(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    bounding_polygon=left_polygon,
    label="Story 6 — left half",
)

# Reuse across stages:
cuts_per_stage = {
    stage: ds.section_cut(spec=spec, model_stage=stage)
    for stage in ds.model_stages
}

# Or persist for a separate batch job:
spec_path = spec.save_pickle("/tmp/story6_left_cut.pkl")
restored = SectionCutSpec.load_pickle(spec_path)
assert restored == spec
```

For multi-case workflows the companion `MultiCutResult` aggregates one
spec across many datasets / cases:

```python
from STKO_to_python.cuts import MultiCutResult

datasets = {"GM1": ds1, "GM2": ds2, ...}
multi = MultiCutResult.from_datasets(datasets, spec, model_stage=STAGE)
print(multi.peak_over_cases("Fx"))      # Series indexed by case name
print(multi.envelope_per_case())        # DataFrame: one row per case
```

---

## 6. Cuts at exact mesh-row elevations

A subtle case worth knowing about: when the cut plane lands **exactly
on a shared edge** between two adjacent shells (e.g. `z=870` in
Test_NLShell where the T3 mesh meets the Q4 mesh), both shells would
report that edge as their chord — the naive sum double-counts.

The kernel resolves this with a **side-aware geometric filter**:
only shells whose interior lies on the *discarded* side contribute.
With `side="positive"` (the default; kept = upper) you get the
section.force from the lower mesh; with `side="negative"` you get
the section.force from the upper mesh. Both correctly represent the
cut force from one side or the other.

```python
on_edge = ds.section_cut(
    plane=Plane.horizontal(z=870.0),
    element_ids=shell_eids,
    model_stage=STAGE,
)
just_below = ds.section_cut(
    plane=Plane.horizontal(z=869.999),
    element_ids=shell_eids,
    model_stage=STAGE,
)

print(on_edge.F[0, 2], just_below.F[0, 2])
# Match within numerical noise — the on-edge cut picks up the same
# section.force as just-below for side='positive'.
```

When the geometry has a **concentrated load at the on-edge elevation**
(common in real models — node loads applied to the diaphragm row),
the `consistency_check` *may* fail at that exact elevation. The cut
just below and just above honestly differ by the magnitude of that
load. This is correct physical behavior, not a kernel bug — but if
you need strict per-side cancellation, cut at an elevation `±ε`
away from the discontinuity.

---

## Variations

- **Multi-stage `(F, M)` series**: drop the `model_stage` kwarg in
  a loop over `ds.model_stages` and concatenate. Time axes are
  per-stage; align them by step index, not by absolute time.
- **Custom moment reference**: `cut.moment_about(point)` transfers
  the moment to any reference point. Useful when comparing cuts
  against support-reaction sums from `REACTION`.
- **Plotting**: `cut.plot.history("Fx")` returns a matplotlib axes
  with the F_x history; the sweep has `sweep.plot.profile(...)` and
  `sweep.plot.heatmap(...)`.
- **Beam + shell mixed**: with no code changes, a model carrying both
  beams and shells in the same filter produces a single combined
  `SectionCut`. Beam intersection points and shell chord midpoints
  go into the same centroid.
- **Save cuts for downstream tools**: `cut.save_pickle("cut.pkl")`
  serializes the whole result (no dataset reference, no HDF5 handle)
  — handy when batch-computing cuts on a cluster and post-processing
  on a workstation.

For the full API reference of the cuts subpackage see
[Section cuts API](../api/section-cuts.md).
