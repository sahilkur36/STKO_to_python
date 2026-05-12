# Per-layer & per-fiber decomposition of layered-shell cuts

> Take a `SectionCut` through a layered reinforced-concrete wall and
> peel it apart through-thickness — one layer at a time, then (when
> the layers are themselves fibered) one fiber at a time.

The engineering question is: *the integrated section-cut tells me the
total force a layered shell carries across the cut chord — but how is
that force distributed through the wall thickness?* Layered shells
typically combine concrete cover, rebar layers, and concrete core; the
per-layer view answers *what does layer K alone contribute?* and the
per-fiber view (when applicable) drills one level deeper into the
fiber discretisation of a single layer.

The fixture is `stko_results_examples/Test_NLShell` — a 4-partition
wall with `ASDShellT3` below `z=870` and `ASDShellQ4` above, three
analysis stages. Sections 15 and 16 are 7-layer `LayeredShell`
definitions whose materials alternate concrete cover, rebar, concrete
core, rebar, concrete cover (`section LayeredShell 15/16 7 3 t1 4 t2
11 t3 3 t4 11 t5 4 t6 3 t7`, where mat 3 is the core concrete, 4 the
cover, 11 the rebar smear).

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from STKO_to_python import LayerInfo, MPCODataSet
from STKO_to_python.cuts import Plane, SectionCutSpec, SectionCut

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "Test_NLShell"
STAGE = "MODEL_STAGE[1]"

ds = MPCODataSet(str(DATASET), "Results", verbose=False)
shell_eids = tuple(ds.elements_info["dataframe"]["element_id"].tolist())
```

---

## 1. Where does the layer table come from?

Layered-shell **results** live in the `.mpco` file
(`section.fiber.stress` records the per-layer stress field), but the
layered-shell **geometry** (the per-layer thicknesses and material
ids) lives in the `sections.tcl` script that built the model. The
dataset auto-discovers `sections.tcl` beside the recorder output:

```python
sections = ds.layered_sections    # lazy property, parses sections.tcl on first touch
print(set(sections.keys()))
# {15, 16}

for layer in sections[16]:
    print(f"mat={layer.material_id:>3}  t={layer.thickness:>8.4f}  z={layer.z_offset:>+9.4f}")
# mat=  3  t= 20.0000  z= -40.5000
# mat=  4  t=  0.2313  z= -30.3833
# mat= 11  t=  1.4266  z= -29.5535
# mat=  3  t= 58.6841  z=  -0.5092
# mat= 11  t=  1.4266  z=  28.5351
# mat=  4  t=  0.2313  z=  29.3649
# mat=  3  t= 20.0000  z=  39.4815
```

Each `LayerInfo` carries:

| Field | Type | Meaning |
|---|---|---|
| `material_id` | `int` | OpenSees material tag used in the layer. |
| `thickness` | `float` | Layer thickness (length units of the model). |
| `z_offset` | `float` | Signed distance from the layer midplane to the **section midplane**. Bottom layer has the most-negative offset; top layer the most-positive. |

`z_offset` is what the per-layer kernel uses to compute the
through-thickness moment contribution
(`M_layer = σ_layer · t_layer · z_offset_layer`).

---

## 2. The standard layered-shell cut (re-cap)

The basic `SectionCut` against a layered-shell wall already exists
since v1.6 — the kernel reads the through-thickness-integrated
`section.force` (`Fxx, Fyy, ..., Vyz`) and integrates along the chord.
This is the natural baseline against which the per-layer breakdown is
validated:

```python
cut = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage=STAGE,
)
print(cut)
# SectionCut(stage='MODEL_STAGE[1]', n_steps=10, n_intersections=6, side='positive')
print(f"F[0] = {cut.F[0]}")
```

---

## 3. Per-layer breakdown

`cut.per_layer_force(layer_idx, ds)` returns a *derivative cut* in
which the shell contributions come from a single through-thickness
layer. Beams and solids in the original cut are dropped — per-layer
is a shell-only concept.

```python
per_layer = [cut.per_layer_force(k, ds) for k in range(7)]

# Each is a SectionCut with the same shape — built from one layer's stress
print(per_layer[0].F.shape)        # (n_steps, 3)
print(per_layer[0].M.shape)        # (n_steps, 3)
print(per_layer[0].shell_intersections == cut.shell_intersections)  # True
```

The math: the standard `section.force` 8-vector is the
through-thickness integral of the layer-wise stress::

```text
N_xx = ∫ σ_xx dz  ≈  Σ_k σ_xx^(k) · t_k
M_xx = ∫ σ_xx · z dz  ≈  Σ_k σ_xx^(k) · t_k · z_offset_k
```

Replacing the standard 8-vector with the kth-layer-only contribution
gives `per_layer_force(k)`. The rest of the math (chord integration,
rotation to global) is identical to the standard shell kernel.

### Sum-of-layers identity

A consequence of the math: summing every per-layer cut recovers the
standard shell cut (within numerical tolerance from quadrature
roundoff and the layer-stress recorder precision).

```python
F_sum = sum(p.F for p in per_layer)
np.testing.assert_allclose(F_sum, cut.F, atol=max(1.0, np.max(np.abs(cut.F))) * 0.01)
```

This is the cleanest check that the per-layer reads are consistent
with the through-thickness integration.

### Plot the through-thickness force distribution

The natural visualisation: per-layer F_x at step 0 plotted against
the layer midplane offset. Concrete layers carry the bulk of the
membrane force; rebar layers spike to high stress over thin
thicknesses.

```python
sec = ds.layered_sections[16]
z_offsets = [layer.z_offset for layer in sec]
F_per_layer = [p.F[0, 0] for p in per_layer]

fig, ax = plt.subplots(figsize=(4, 6))
ax.barh(z_offsets, F_per_layer, height=[l.thickness * 0.9 for l in sec])
ax.axvline(0.0, color="k", lw=0.5)
ax.set_xlabel("F_x carried by layer  (step 0)")
ax.set_ylabel("Layer midplane z (mm)")
ax.set_title("Through-thickness force distribution")
plt.tight_layout()
plt.show()
```

---

## 4. The inline `per_layer=k` shortcut

For one-shot per-layer queries the dataset entry point grows a
`per_layer=` kwarg that short-circuits the two-step
`compute → per_layer_force` flow:

```python
top_layer = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage=STAGE,
    per_layer=6,        # top layer = concrete cover
)

# Equivalent to:
full = ds.section_cut(plane=Plane.horizontal(z=2500.0), element_ids=shell_eids, model_stage=STAGE)
also_top = full.per_layer_force(6, ds)
np.testing.assert_allclose(top_layer.F, also_top.F)
```

For batch workflows, build a `SectionCutSpec` once and dispatch the
per-layer call across stages / cases — same pattern as the standard
cut.

---

## 5. Per-fiber view (when layers are themselves fibered)

A layer that's defined as a `section Fiber ...` in OpenSees carries
multiple discrete fibers. The MPCO recorder then writes columns named
`<comp>_f<F>_l<L>_ip<K>` (one fiber `F` inside one layer `L`), and
`cut.per_fiber_force(layer_idx, fiber_idx, ds)` returns a derivative
cut from one fiber alone.

The fiber's tributary thickness defaults to
`t_layer / n_fibers_in_layer` (uniform distribution within the
layer's thickness); the fiber's z-offset is the centroid of that
sub-band relative to the section midplane. Summing all fibers in a
layer recovers that layer's per-layer cut.

```python
# When fibers exist in layer L (NOT the case for Test_NLShell —
# its layers are single nDMaterial each):
per_fiber_0 = cut.per_fiber_force(layer_idx=L, fiber_idx=0, dataset=ds)
n_fibers = number_of_fibers_in_layer_L           # discovered from columns
F_sum_of_fibers = sum(
    cut.per_fiber_force(L, f, ds).F for f in range(n_fibers)
)
F_layer_L = cut.per_layer_force(L, ds).F
np.testing.assert_allclose(F_sum_of_fibers, F_layer_L, atol=1e-3)
```

**Non-fibered layers raise a clear error.** Test_NLShell's layers are
each a single `nDMaterial`, so requesting a per-fiber cut on it
surfaces:

```python
try:
    cut.per_fiber_force(0, 0, ds)
except ValueError as exc:
    print(exc)
# Element ..., layer 0: no fiber-in-layer columns (`_f<F>_l<L>_ip<K>`)
# available. Use compute_shell_cut_per_layer for non-fibered layers.
```

Use `per_layer_force` for non-fibered sections; `per_fiber_force` is
the right tool only when the recorder carries the `_f<F>_l<L>_ip<K>`
column layout.

### Inline form

```python
# per_fiber requires per_layer
cut_f0_l3 = ds.section_cut(
    plane=Plane.horizontal(z=2500.0),
    element_ids=shell_eids,
    model_stage=STAGE,
    per_layer=3,
    per_fiber=0,
)

# per_fiber without per_layer raises:
ds.section_cut(plane=..., element_ids=..., model_stage=..., per_fiber=0)
# ValueError: per_fiber requires per_layer to also be specified ...
```

---

## 6. Three column-naming conventions

The MPCO format isn't perfectly uniform on layered-shell stress
columns. The per-layer reader tries three conventions in priority
order; users normally don't need to think about this, but it's useful
to know if you're inspecting raw columns:

| Pattern | When it appears |
|---|---|
| `sigma<ij>_l<L>_ip<K>` | Most explicit form, per the MPCO format docs §17. |
| `sigma<ij>_f<L>_ip<K>` | Alternate when the recorder treats the layer axis as a fiber axis. |
| `UnknownStress(n)_f<L>_ip<K>` | `nDMaterial` fallback (no registered response codes). Mapped to `(σ11, σ22, σ12, σ13, σ23)` per the PlateFiber convention. |

Test_NLShell uses the third form — its layered concrete / steel
materials don't register OpenSees response codes, so MPCO writes
indexed placeholders. The reader maps them transparently; the user-
facing surface (`per_layer_force`) doesn't change shape.

---

## Variations

- **Per-element layer F**: `per_layer.per_shell_F[eid]` carries the
  per-layer force contribution of one shell — useful for diagnosing
  which element drives a layer's resultant.
- **Multi-stage per-layer sweep**: loop over `ds.model_stages` and
  rebuild the per-layer view per stage. The `LayerInfo` table is
  fixed (it comes from `sections.tcl`), so a cached layer-table dict
  works across stages.
- **Custom fiber distribution**: the v1.8 per-fiber view assumes
  uniform distribution within a layer. For real fiber positions
  (e.g. asymmetric reinforcement), the v1.9 candidate path is to
  read the actual fiber geometry from the OpenSees model and pass it
  to the kernel — see the v1.9 Out-of-scope note in the project
  [`CHANGELOG.md`](https://github.com/nmorabowen/STKO_to_python/blob/main/CHANGELOG.md).
- **Moment about the section midplane**: the per-layer F and M are
  returned about the cut centroid (mean of shell chord midpoints).
  Use `cut.moment_about(point)` to transfer the moment to e.g. the
  base of the wall for direct comparison against support reactions.

For solids and composed (beam + shell + solid) cuts see
[recipe 11](11-section-cut-solids.md); for the full API reference of
the cuts subpackage see [Section cuts API](../api/section-cuts.md).
