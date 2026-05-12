# Render beam elements as 3D extruded solids

> Turn 1D beam elements into 3D extruded section solids using the
> `*BEAM_PROFILE` + `*BEAM_PROFILE_ASSIGNMENT` + `*LOCAL_AXES` data
> STKO writes to the `.cdata` sidecar. One call to
> `ds.plot.beam_solids(...)` does the entire pipeline.

The engineering question is: *I have a frame of `5-ElasticBeam3d`
elements — line elements in the recorder — and I want to see the
actual section shape extruded along each member.* STKO already records
everything we need: a 2D cross-section per profile, an element →
profile assignment, and a local-frame rotation per element. v1.4.0+
parses these into the cdata reader; this recipe wires them through to
a 3D matplotlib plot.

The fixture used here is
`stko_results_examples/elasticFrame/QuadFrame_results` — 75
`5-ElasticBeam3d` elements arranged as a frame, plus 625
`203-ASDShellQ4` shell elements in the cladding. The renderer
**silently filters out** anything that isn't in
`beam_profile_assignments`, so the shells stay out of the way.

```python
from pathlib import Path
import matplotlib.pyplot as plt

from STKO_to_python import MPCODataSet

REPO_ROOT = Path("path/to/STKO_to_python")
DATASET = REPO_ROOT / "stko_results_examples" / "elasticFrame" / "QuadFrame_results"

ds = MPCODataSet(str(DATASET), "results", verbose=False)
```

---

## 1. Render every beam in the model

The default call extrudes every element with a profile assignment:

```python
ax, meta = ds.plot.beam_solids()
print(meta["element_count"])     # 75
print(meta["triangle_count"])    # 900
print(meta["profile_ids"])       # [1]
plt.show()
```

`meta["skipped_elements"]` lists IDs that were filtered but lacked
`*LOCAL_AXES` (rare in practice — STKO writes axes for every line
element) or referenced a profile id that wasn't defined.

The structural-edge overlay (section perimeter at both ends + sweep
longitudinals) is on by default in dark gray. Disable it for a flat
filled look:

```python
ax, _ = ds.plot.beam_solids(edge_color=None, face_color="C2")
```

---

## 2. Narrow the rendered set with selection inputs

Mix and match `element_ids`, `selection_set_id`, and
`selection_set_name`; the union is intersected with the set of
elements that actually have profile assignments. Selection-set names
go through the same case-insensitive resolver the rest of the library
uses, so the lookup matches STKO's `*SELECTION_SET` blocks verbatim.

```python
# Just the columns:
ax, _ = ds.plot.beam_solids(selection_set_name="Columns")

# A small subset by explicit id:
ax, _ = ds.plot.beam_solids(element_ids=[1, 2, 3])
```

If the filter produces no matching beam elements the renderer raises
`ValueError("No beam elements with profile assignments matched the filter.")`
— the same shape as the rest of the plot facade.

---

## 3. Compose with the shell mesh

`ds.plot.beam_solids` accepts a pre-existing 3D `matplotlib` axes via
the `ax` parameter, so the beam solids can sit on top of (or under) a
shell mesh on the same canvas. Both `ds.plot.mesh` and
`ds.plot.deformed_shape` auto-select 3D when the model is
non-planar, so the combined plot looks consistent:

```python
ax, _ = ds.plot.mesh(element_type="ASDShellQ4")
ds.plot.beam_solids(ax=ax, face_color="C3", alpha=0.7)
```

The order matters for matplotlib's depth sort — draw the wireframe
first, then the solids, so the column volumes occlude the shell edges
that pass behind them.

---

## 4. Render the deformed configuration

`ds.plot.beam_solids_deformed(model_stage, step, scale)` is the
deformed twin of `beam_solids`. It fetches `DISPLACEMENT` at the
requested step, shifts every end node by `scale * disp`, and feeds
those into the same extrusion pipeline. `meta` carries `model_stage`,
`step`, and `scale` on top of the standard keys.

```python
ax, meta = ds.plot.beam_solids_deformed(
    model_stage="MODEL_STAGE[2]",
    step=9,
    scale=2000.0,
    face_color="C3",
)
```

Two things to know:

1. **Pick a scale that makes sense for your model.** Elastic frames
   under service-load levels produce displacements that are orders of
   magnitude smaller than member lengths — at true-to-life scale, the
   deformation is invisible. The default in literature is 50-1000×;
   the fixture used here needs ~2000× because the imposed load is
   very small.
2. **The cross-section's local frame is taken from the *undeformed*
   `*LOCAL_AXES` quaternion.** STKO does not record a deformed local
   frame, so we cannot rotate the section with the deformed tangent.
   Translations look correct; large rotations on slender members may
   show a slightly off section orientation. For visualization this is
   acceptable; for downstream geometry export, use the undeformed
   render plus a custom transform.

`scale=0` collapses to the undeformed configuration without fetching
DISPLACEMENT — useful for code paths that toggle deformed/undeformed
behind a flag.

---

## 5. Per-element profile vs. uniform section

STKO supports variable cross-section through
`*BEAM_PROFILE_ASSIGNMENT` entries with weight pairs. **This release
picks the first profile per element**; weighted blending is a future
extension. For elements with one assignment (the common case)
behavior is unchanged.

If you need the lower-level building blocks — say, to export the
triangulated mesh to a different renderer — call into the geometry
kernel directly:

```python
from STKO_to_python.plotting.beam_solid import extrude_beam_geometry

eid = 1
pid, _weight = ds.cdata.beam_profile_assignments[eid][0]
profile = ds.cdata.beam_profiles[pid]
R = ds.cdata.rotation_matrix(eid)
# end-node coords from your favourite source...
vertices, faces = extrude_beam_geometry(
    profile, axis_start=n0, axis_end=n1, R=R,
    section_offset=ds.cdata.section_offsets.get(eid),
)
```

`vertices` is `(2 * n_pts, 3)` global coordinates and `faces` is
`(n_tri, 3)` zero-based indices — drop them into any triangle-mesh
viewer.

---

## Variations

- **Color beams by force** — fetch element results, build a per-element
  scalar, normalize, and pass `face_color` as an array; the
  `Poly3DCollection` accepts a colormap-compatible value array via
  `set_array`.
- **Animate a pushover** — call `beam_solids_deformed` in a loop over
  step indices and use `matplotlib.animation.FuncAnimation` to drive
  the timeline. Hold the axes fixed (don't reset between frames) so
  the camera stays steady.
- **Export to STL** — `vertices` and `faces` from
  `extrude_beam_geometry` are the canonical STL inputs; concatenate
  the per-element batches with a vertex offset and write with
  ``numpy-stl`` or similar.
