# Section cuts

Compute the resultant `(F, M)` force/moment carried across a plane sliced
through the discretized model. Beams contribute through the line-element
internal force at the intersection; shells contribute via a chord
integral of `section.force` along the intersection chord. The two are
composed in one pass and aggregated about a shared centroid.

A v1.5.0 surface that v1.6.0 extends with:

- the **shell kernel** (`ASDShellQ4`, `ASDShellT3`; layered variants
  transparent),
- an optional **convex bounding polygon** on the cut plane that
  restricts the cut to a structural sub-region.

For the engineering walkthrough — story shears with shells + a
bounding-polygon-bounded "left half of the wall" cut — see the
cookbook recipe [Section cuts through frames with
shells](../cookbook/10-section-cut-shells.md).

```python
from STKO_to_python import MPCODataSet
from STKO_to_python.cuts import Plane

ds = MPCODataSet(r"C:\path\to\Results", "Results", verbose=False)
cut = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=tuple(ds.elements_info["dataframe"]["element_id"].tolist()),
    model_stage="MODEL_STAGE[1]",
)
print(cut)                       # SectionCut(stage=..., n_intersections=...)
print(cut.F[0])                  # force at step 0
print(cut.envelope())            # peak per component
ok, residual = cut.consistency_check(ds)
```

---

## Public surface

| Symbol | Module | Purpose |
|---|---|---|
| `Plane` | `STKO_to_python.cuts.plane` | Geometric primitive (point + outward unit normal). Constructors: `horizontal`, `vertical`, `from_three_points`, `horizontal_grid`. |
| `SectionCutSpec` | `STKO_to_python.cuts.specs` | Picklable definition of a cut (plane + filter + side + optional `bounding_polygon`). Hashable by value. |
| `SectionCut` | `STKO_to_python.cuts.section_cut` | Dataset-bound result with `F`, `M`, `time`, validators, and a `.plot` namespace. |
| `SectionSweep` | `STKO_to_python.cuts.sweep` | Many planes against one dataset — story-shear-vs-elevation profiles. |
| `MultiCutResult` | `STKO_to_python.cuts.multi_cut` | One spec, many cases (ground-motion ensembles). |
| `DriftSpec` | `STKO_to_python.cuts.specs` | Node-pair drift spec — paired here because it's the second "saved cut definition" type. |

The two **dataset entry points** are:

| Call | Returns |
|---|---|
| `ds.section_cut(plane=..., ..., model_stage=...)` | `SectionCut` |
| `ds.section_cut(spec=spec, model_stage=...)` | `SectionCut` |
| `ds.section_sweep(planes=[...], ..., model_stage=...)` | `SectionSweep` |

The inline form accepts `bounding_polygon=` directly. The spec form
carries the polygon on the spec. `section_sweep` does **not** accept
`bounding_polygon` because a single polygon can only lie on a single
plane — for per-plane polygons, build a list of specs and pass them
to `SectionSweep.from_specs(...)`.

---

## What ships in v1.6.0

### Shell kernel

For each shell whose midsurface crosses the cut plane, the kernel:

1. computes the chord at which the cut plane crosses the midsurface
   polygon (3- or 4-node shells in v1.6),
2. maps each chord endpoint to element natural coords `(ξ, η)`
   (bilinear inversion for quads, linear for triangles),
3. integrates `section.force` (8 components per IP — `Fxx, Fyy,
   Fxy, Mxx, Myy, Mxy, Vxz, Vyz`) along the chord with 2-point
   Gauss-Legendre quadrature, sampling between IPs via bilinear
   (Q4) or linear (T3) interpolation,
4. assembles a per-unit-length traction `(F_local, M_local)` from
   the section tensors and the in-plane cut normal, rotates to
   global via `cdata.rotation_matrix(eid)`, and sums.

Supported shell classes (`SHELL_ELEMENT_CLASSES`):

- `ASDShellQ4`, `ASDShellT3` (4-IP and 3-IP standard quadrature)
- `ShellMITC4`, `ShellNLDKGQ`, `ShellNLDKGT`, `ShellDKGQ`,
  `ShellDKGT` (decorated names match after stripping the
  `<classTag>-` prefix)

Layered-shell variants use the same code path — `section.force` is
already through-thickness-integrated regardless of layer count.

### Shared-edge resolution

When a cut plane lands exactly on a shared edge between two adjacent
shells (e.g. `z=870` in a stacked wall mesh where T3 elements sit
below and Q4 sit above), both shells would report that shared edge
as their chord. The naive sum would double-count.

`find_shell_intersections` resolves this with a side-aware geometric
filter: **only shells whose interior lies on the discarded side**
contribute. With `side="positive"` (kept = positive normal side),
the shell on the negative side reports the chord; with
`side="negative"`, the shell on the positive side does. The
``consistency_check`` (positive + negative ≈ 0) holds on either
side of the load-discontinuity; at the exact discontinuity
elevation, the two cuts pick up the section.force just above vs
just below the loads and the check intentionally reflects the load
jump.

### Bounding polygon

`SectionCutSpec.bounding_polygon` is an optional tuple of `(x, y, z)`
triples defining a convex polygon **on the cut plane**. The kernel
clips beam intersection points and shell chords against this polygon
before computing the resultant. Validation at construction:

- at least 3 vertices,
- all vertices within `1e-6` of the plane,
- non-degenerate planar area,
- convex (Cyrus-Beck clipping assumes convexity).

Useful when the recorded selection sets don't pre-filter to a
structural sub-region — e.g. cut just the left half of a wall by
passing a rectangular polygon over that half.

---

## Validators

| Validator | What it checks |
|---|---|
| `cut.consistency_check(ds)` | Newton 3rd law — `F_positive + F_negative ≈ 0`. Independent of boundary conditions; works for DRM / PML / explicit dynamics. |
| `cut.compare_to(other_cut)` | Two parallel cuts at planes between which there is no external load must give the same resultant (moments transferred to a common reference). |
| `cut.moment_about(point)` | Transfer the moment to a different reference point — what `compare_to` uses internally. |

Both validators return `(ok, residual)` so the caller can inspect the
per-step residual when a check fails.

---

## SectionCutSpec

::: STKO_to_python.cuts.specs.SectionCutSpec

---

## SectionCut

::: STKO_to_python.cuts.section_cut.SectionCut

---

## Plane

::: STKO_to_python.cuts.plane.Plane

---

## SectionSweep

::: STKO_to_python.cuts.sweep.SectionSweep

---

## MultiCutResult

::: STKO_to_python.cuts.multi_cut.MultiCutResult

---

## DriftSpec

::: STKO_to_python.cuts.specs.DriftSpec
