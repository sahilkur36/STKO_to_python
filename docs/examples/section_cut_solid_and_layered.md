# Section cuts — solids + layered shells

End-to-end demonstration of the v1.7 + v1.8 section-cut surface:
the solid (continuum) kernel, the per-layer breakdown for layered
shells, and the per-fiber breakdown for fibered layers. Composed
(beam + shell + solid) cuts in a single `SectionCut.compute` call;
`SectionCut.per_layer_force` and `per_fiber_force` for through-
thickness slicing.

**Script:** [`examples/section_cut_solid_and_layered_example.py`](https://github.com/nmorabowen/STKO_to_python/blob/main/examples/section_cut_solid_and_layered_example.py)

```bash
python examples/section_cut_solid_and_layered_example.py
```

The script auto-discovers `stko_results_examples/` under the repo root
(with a worktree-aware fallback to the main checkout so it runs from a
developer worktree too), and gracefully exits each fixture's section
when the heavy `.mpco` files are absent locally.

---

## Fixtures

- **`stko_results_examples/Test_NLShell`** — 4-partition layered
  reinforced-concrete wall meshed with `ASDShellT3` below `z=870` and
  `ASDShellQ4` above. Section 16 is a 7-layer `LayeredShell`
  (concrete cover / smeared rebar / concrete core / smeared rebar /
  concrete cover, total thickness 102 mm). 3 model stages of
  pushover analysis, 10 steps per stage. Drives the per-layer +
  per-fiber-error-path sections.
- **`stko_results_examples/solid_partition_example`** — 2-partition
  mesh with `56-Brick` continuum + `64-DispBeamColumn3d` beams.
  Drives the solid kernel + composed cut sections. Gitignored;
  skip-if-absent.

---

## Part 1 — Layered-shell decomposition (v1.7 + v1.8)

### Standard cut (recap of v1.6)

```python
cut = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage="MODEL_STAGE[1]",
)
print(cut)
# SectionCut(stage='MODEL_STAGE[1]', n_steps=10, n_intersections=6, side='positive')
print(cut.F[0])
# [1.443e+01  8.403e-12  3.784e+04]
```

### Parsed LayeredShell table

`MPCODataSet.layered_sections` lazily parses the `sections.tcl`
sidecar:

```python
print(sorted(ds.layered_sections.keys()))
# [15, 16]

# Test_NLShell section 16 — concrete cover + cover rebar + core + ...
for layer in ds.layered_sections[16]:
    print(f"mat={layer.material_id:>2}  t={layer.thickness:>9.4f}  "
          f"z_offset={layer.z_offset:>+9.4f}")
# mat= 3  t= 20.0000  z_offset=-41.0000
# mat= 4  t=  0.2313  z_offset=-30.8843
# mat=11  t=  1.4266  z_offset=-30.0554
# mat= 3  t= 58.6841  z_offset= +0.0000
# mat=11  t=  1.4266  z_offset=+30.0554
# mat= 4  t=  0.2313  z_offset=+30.8843
# mat= 3  t= 20.0000  z_offset=+41.0000
```

### Per-layer breakdown

```python
per_layer = [cut.per_layer_force(k, ds) for k in range(7)]
for k, p in enumerate(per_layer):
    print(f"  layer {k}: F = {p.F[0]}")
# layer 0: F = [2.84e+00  1.56e-12  7.01e+03]
# layer 1: F = [0.        0.        0.       ]
# layer 2: F = [0.        2.91e-13  1.31e+03]
# layer 3: F = [8.75e+00  4.72e-12  2.12e+04]
# layer 4: F = [0.        2.91e-13  1.31e+03]
# layer 5: F = [0.        0.        0.       ]
# layer 6: F = [2.84e+00  1.56e-12  7.01e+03]
```

The concrete core (layer 3) carries ~56% of F_z; cover layers (0 and 6)
~19% each; smeared rebar layers (2 and 4) ~3.5% each. Layers 1 and 5
(cover rebar in the orthogonal direction) carry zero on this cut.

### Sum-of-layers identity

```python
F_sum = sum(p.F for p in per_layer)
np.testing.assert_allclose(F_sum, cut.F, atol=1.0)
# Max abs diff: ~6e-11  (essentially exact)
```

### Inline `per_layer=k` shortcut

```python
top_layer = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage="MODEL_STAGE[1]",
    per_layer=6,         # top layer = concrete cover
)
# Equivalent to cut.per_layer_force(6, ds).
```

### Per-fiber error path on a non-fibered layer

Test_NLShell's layers are each a single `nDMaterial` — no fibers.
Calling `per_fiber_force` surfaces a clear error pointing at
`per_layer_force` instead:

```python
try:
    cut.per_fiber_force(0, 0, ds)
except ValueError as exc:
    print(exc)
# Element 168, layer 0: no fiber-in-layer columns (`_f<F>_l<L>_ip<K>`)
# available. Use compute_shell_cut_per_layer for non-fibered layers.
```

For a section whose layers ARE fibered, the same call returns the
per-fiber cut; summing across fibers within a layer recovers that
layer's `per_layer_force(L)`.

### Through-thickness plot

The script saves a bar chart of per-layer F_x vs layer midplane
z-coordinate:

```python
fig, ax = plt.subplots(figsize=(5, 5))
ax.barh(z_offsets, F_x_per_layer, height=[l.thickness * 0.9 for l in layers])
ax.set_xlabel("F_x carried by layer  (step 0, N)")
ax.set_ylabel("Layer midplane z (mm)")
plt.savefig("examples/_out_layered_shell_per_layer.png", dpi=120)
```

---

## Part 2 — Solid continuum kernel (v1.7) + higher-order hex (v1.8)

> Skipped at runtime when `solid_partition_example` isn't present
> locally — the section below is what the script does when the
> fixture is available.

### Brick-only cut + per-element diagnostics

```python
brick_only = ds.section_cut(
    plane=Plane.horizontal(z=z_mid),
    element_ids=brick_ids,
    model_stage="MODEL_STAGE[1]",
)
# Heaviest contributor by |per_solid_F|:
heaviest = max(
    brick_only.solid_intersections,
    key=lambda ix: float(np.max(np.abs(brick_only.per_solid_F[ix.element_id]))),
)
print(f"element {heaviest.element_id}  polygon_area={heaviest.polygon_area:.3f}")
```

A `SolidIntersection` carries both the planar polygon in global coords
(`polygon_global`, shape `(K, 3)` for a `K`-vertex polygon) and the
same polygon mapped into element natural coords (`polygon_natural`).

### Composed beam + solid cut + `consistency_check`

```python
composed = ds.section_cut(
    plane=Plane.horizontal(z=z_mid),
    element_ids=all_ids,           # bricks + dispBeamCol3d
    model_stage="MODEL_STAGE[1]",
)
print(f"beams  : {len(composed.intersections)}")
print(f"solids : {len(composed.solid_intersections)}")

ok, residual = composed.consistency_check(ds, atol=scale * 1e-3, rtol=1e-6)
# Newton's 3rd law across the three-kernel composition
```

`consistency_check` recomputes the cut with the opposite `side` and
verifies that the two sums to zero (moments transferred to a common
reference). Independent of the analysis type — works for static,
dynamic, DRM, PML, explicit dynamics.

### `SectionSweep` across elevations

```python
zs = np.linspace(z_min, z_max, 6)
sweep = ds.section_sweep(
    planes=[Plane.horizontal(z=float(z)) for z in zs],
    element_ids=all_ids,
    model_stage="MODEL_STAGE[1]",
)
env = sweep.envelope()
ax.plot(env["Fz_peak_abs"], sweep.plane_locators("z"), "o-")
```

### `bounding_polygon` over a half-slab region

```python
right_half = (
    (mid_x, -big, z_mid),
    (   big, -big, z_mid),
    (   big,  big, z_mid),
    (mid_x,  big, z_mid),
)
right = ds.section_cut(
    plane=Plane.horizontal(z=z_mid),
    element_ids=all_ids,
    model_stage="MODEL_STAGE[1]",
    bounding_polygon=right_half,
)
```

For solid intersections the kernel runs a Sutherland-Hodgman clip of
the intersection polygon against the bounding polygon — so the
per-element contribution comes from only the portion of the cut
polygon inside the region of interest.

### Higher-order hex (Brick20 / Brick27)

The same surface accepts higher-order hexes transparently:

```python
from STKO_to_python.cuts.kernels import SOLID_ELEMENT_CLASSES
print(SOLID_ELEMENT_CLASSES)
# ('Brick', 'BbarBrick', 'SSPbrick', 'Brick20', 'Brick27',
#  'TwentyNodeBrick', 'TwentySevenNodeBrick', 'FourNodeTetrahedron')
```

The geometry phase uses the first 8 corner nodes for the
plane-vs-polyhedron polygon math; stress sampling dispatches on the
IP count (8-IP trilinear vs 27-IP triquadratic Lagrange). No user
opt-in needed.

---

For the engineering walkthroughs see the cookbook recipes [Section
cuts through brick continua](../cookbook/11-section-cut-solids.md) and
[Per-layer & per-fiber decomposition of layered-shell
cuts](../cookbook/12-section-cut-layered-shells.md); for the full API
reference see [Section cuts](../api/section-cuts.md).
