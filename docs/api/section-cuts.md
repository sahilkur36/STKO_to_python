# Section cuts

Compute the resultant `(F, M)` force/moment carried across a plane sliced
through the discretized model. Beams contribute through the line-element
internal force at the intersection; shells contribute via a chord
integral of `section.force` along the intersection chord; continuum
solids contribute via a surface integral of the traction `t = σ · n_cut`
over the planar polygon clipped from each crossing element's volume.
All three are composed in one pass and aggregated about a shared
centroid.

The library's section-cut surface grew in three milestones:

- **v1.5.0** — beams only.
- **v1.6.0** — adds the **shell kernel** (`ASDShellQ4`, `ASDShellT3`;
  layered variants transparent) and an optional **convex bounding
  polygon** on the cut plane.
- **v1.7.0** — adds the **solid (continuum) kernel** for `Brick`,
  `BbarBrick`, `SSPbrick`, and `FourNodeTetrahedron`; adds the
  **per-layer breakdown** for layered shells
  (`SectionCut.per_layer_force(k, ds)`).
- **v1.8.0** — adds **higher-order hex support** (`Brick20`, `Brick27`)
  with triquadratic 27-IP stress sampling; adds the **per-fiber
  breakdown** for fibered layers
  (`SectionCut.per_fiber_force(L, F, ds)`).

For engineering walkthroughs:

- [Section cuts through frames with shells](../cookbook/10-section-cut-shells.md)
  — story shears with shells, validators, `bounding_polygon`.
- [Section cuts through brick continua](../cookbook/11-section-cut-solids.md)
  — solid kernel, beam + shell + solid composition, higher-order hex
  notes.
- [Per-layer & per-fiber decomposition of layered-shell cuts](../cookbook/12-section-cut-layered-shells.md)
  — through-thickness slicing of layered shells.

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
| `SectionCut` | `STKO_to_python.cuts.section_cut` | Dataset-bound result with `F`, `M`, `time`, validators, `per_layer_force` / `per_fiber_force` (v1.7+/v1.8+), and a `.plot` namespace. |
| `SectionSweep` | `STKO_to_python.cuts.sweep` | Many planes against one dataset — story-shear-vs-elevation profiles. |
| `MultiCutResult` | `STKO_to_python.cuts.multi_cut` | One spec, many cases (ground-motion ensembles). |
| `DriftSpec` | `STKO_to_python.cuts.specs` | Node-pair drift spec — paired here because it's the second "saved cut definition" type. |
| `LayerInfo` | `STKO_to_python.model.layered_section_reader` | One `LayeredShell` layer — `(material_id, thickness, z_offset)`. Returned by `MPCODataSet.layered_sections`. |

The **dataset entry points** are:

| Call | Returns |
|---|---|
| `ds.section_cut(plane=..., ..., model_stage=...)` | `SectionCut` |
| `ds.section_cut(spec=spec, model_stage=...)` | `SectionCut` |
| `ds.section_cut(..., per_layer=k, model_stage=...)` | `SectionCut` (v1.7+; per-layer view of layered shells) |
| `ds.section_cut(..., per_layer=k, per_fiber=f, model_stage=...)` | `SectionCut` (v1.8+; per-fiber-in-layer view) |
| `ds.section_sweep(planes=[...], ..., model_stage=...)` | `SectionSweep` |
| `ds.layered_sections` | `{section_id: tuple[LayerInfo, ...]}` parsed from `sections.tcl` (v1.7+) |

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

## What ships in v1.7.0

### Solid (continuum) kernel

For each crossing solid element, the kernel:

1. classifies the eight (or four) corner nodes against the cut plane,
2. walks the element's edges and collects plane crossings + on-plane
   vertices,
3. sorts the unique crossings CCW around the plane normal to get the
   planar intersection polygon (3–6 vertices for a hex, 3–4 for a tet),
4. fan-triangulates the polygon and places a 3-point Gauss rule on
   each triangle,
5. inverts the element shape function at each quadrature point to get
   natural coords (Newton for hex trilinear, closed-form 3×3 solve for
   tet linear),
6. samples the `material.stress` Voigt vector (six components per IP)
   via trilinear interpolation between IPs and expands to a symmetric
   3×3 tensor,
7. computes the traction `t = σ · n_cut` with `n_cut` oriented from
   kept → discarded, weights by triangle area × Gauss weight, and
   accumulates `(F, M)` about the polygon centroid.

Supported solid classes (`SOLID_ELEMENT_CLASSES`, v1.7):

- `Brick`, `BbarBrick`, `SSPbrick` — 8-node trilinear hexahedra (8-IP
  2×2×2 Gauss-Legendre).
- `FourNodeTetrahedron` — 4-node linear tetrahedron (1-IP centroid or
  4-IP).

### `SectionCut.compute` composition over three kernels

Beam + shell + solid kernels run in one pass, share a `PolygonClipper`
when a `bounding_polygon` is set, and aggregate `(F, M)` about a
common centroid (mean of beam intersection points, shell chord
midpoints, and solid polygon centroids). New fields:

- `solid_intersections` — `tuple[SolidIntersection, ...]`,
- `per_solid_F` — `dict[int, ndarray]`,
- `per_solid_M_at_centroid` — `dict[int, ndarray]`,
- `contributing_element_ids` — walks beams + shells + solids in id
  order.

The `consistency_check` and `compare_to` validators automatically
extend to the composed cut — same residual semantics.

### Sutherland-Hodgman clipping for solid polygons

When a `bounding_polygon` is set, the solid kernel clips the planar
intersection polygon against the bounding polygon using a
Sutherland-Hodgman polygon-vs-polygon pass — extension of the
Cyrus-Beck shell-chord clip that already ships in v1.6.

### Per-layer breakdown for layered shells

`SectionCut.per_layer_force(layer_idx, dataset)` returns a derivative
cut whose shell contributions come from a single through-thickness
layer:

```text
F_layer^(k) = ∫_chord (σ_layer^(k) · n_cut) · t_k · dl
M_layer^(k) = ∫_chord (σ_layer^(k) · n_cut) · t_k · z_offset_k · dl
```

Sum across layers recovers the standard through-thickness `section.force`
cut. The shortcut `ds.section_cut(..., per_layer=k)` does the
compose-then-slice in one call.

### `MPCODataSet.layered_sections` + `LayerInfo`

Lazy property that locates and parses the `sections.tcl` script beside
the recorder output. Returns
`{section_id: tuple[LayerInfo, ...]}` where each `LayerInfo` carries
`(material_id, thickness, z_offset_from_centroid)`. Empty dict when
no script is found — the per-layer surface then raises a clear error
rather than silently zeroing.

The `LayerInfo.z_offset` is the signed distance from the layer
midplane to the **section midplane**; bottom layer has the
most-negative offset.

---

## What ships in v1.8.0

### Higher-order hex support

`Brick20`, `Brick27`, plus the OpenSees aliases `TwentyNodeBrick` /
`TwentySevenNodeBrick` join `SOLID_ELEMENT_CLASSES`. The geometry phase
uses the first 8 nodes (corners) for the plane-vs-polyhedron polygon
math — sound because the corners define the convex hull. Stress
sampling dispatches on IP count:

- 8-IP 2×2×2 bucket → trilinear weights (same as the 8-node hex case),
- 27-IP 3×3×3 bucket → **triquadratic Lagrange** weights at the
  tensor-product nodes (`±√(3/5)`, 0 per axis).

Curvature induced by midpoint / face / centre nodes is ignored at the
geometry layer — an O(midpoint-deviation²) approximation that's
adequate for any well-conditioned higher-order hex.

### Per-fiber breakdown for fibered layers

`SectionCut.per_fiber_force(layer_idx, fiber_idx, dataset)` returns a
derivative cut from one fiber inside one through-thickness layer.
Required when a layer is itself a `section Fiber` — the MPCO recorder
then writes columns named `<comp>_f<F>_l<L>_ip<K>`. The fiber's
tributary thickness defaults to `t_layer / n_fibers_in_layer`
(uniform distribution); the z-offset is the centroid of that sub-band.

Non-fibered layers raise a clear `ValueError` pointing at
`per_layer_force` instead.

The shortcut `ds.section_cut(..., per_layer=k, per_fiber=f)` does the
compose-then-slice in one call. `per_fiber` without `per_layer` is
rejected — fibers are indexed within a single layer.

### Three column-naming conventions on layered-shell stress

The per-layer / per-fiber readers tolerate three column-naming
conventions for `section.fiber.stress`:

| Pattern | When it appears |
|---|---|
| `sigma<ij>_l<L>_ip<K>` | Most explicit form (per MPCO format docs §17). |
| `sigma<ij>_f<L>_ip<K>` | Alternate when the recorder treats the layer axis as a fiber axis. |
| `UnknownStress(n)_f<L>_ip<K>` | `nDMaterial` fallback. Mapped to `(σ11, σ22, σ12, σ13, σ23)` per the PlateFiber convention. |

The `Test_NLShell` fixture uses the third form because its layered
materials (concrete, smeared rebar) don't register OpenSees response
codes. The reader handles this transparently.

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

---

## LayerInfo

::: STKO_to_python.model.layered_section_reader.LayerInfo
