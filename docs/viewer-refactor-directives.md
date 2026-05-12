# Viewer Refactor — Directives

**Status:** conceptual model — locked. The implementation plan now lives
under [`docs/viewer/`](viewer/00-roadmap.md):

- [Roadmap](viewer/00-roadmap.md) — phased delivery, semver targets, definition of done per phase.
- [Architecture](viewer/01-architecture.md) — renderer-agnostic layering, `Backend` protocol, three deployment targets.
- [Porting from apeGmsh](viewer/02-porting-from-apegmsh.md) — file-by-file lift plan with adaptation notes.
- [Deployment targets](viewer/03-deployment-targets.md) — local Qt, headless CLI, VNC, Trame web, SSH-cluster guidance.

**Goal:** define the conceptual model for the next-generation STKO_to_python
viewer so an implementation plan can hang off concrete contracts.
**Scope:** post-processing only — we render what's in `.mpco`, never edit the model.

The current `Plot` / `NodalResultsPlotter` / `ElementResultsPlotter` trio
covers the easy 30%: a single x–y line, one mesh, one IP scatter. Everything
beyond that — combining a deformed mesh with a stress contour, drawing a
moment diagram on top of a 3D frame, slicing a solid model, rendering a
fiber section — is currently either impossible or one-off-and-fragile.

This document fixes the *vocabulary* before code is written.

---

## 1. The output zoo

Eight output families need first-class rendering. Each row maps a class
of MPCO output to the renderable it produces.

| Family                       | Examples                                                      | Geometry support             | Native mode  |
|------------------------------|---------------------------------------------------------------|------------------------------|--------------|
| **Mesh wireframe**           | undeformed model, optional part filter                        | 2D / 3D                      | line/edge    |
| **Deformed mesh**            | mesh + DISPLACEMENT × scale at step                            | 2D / 3D                      | line/edge    |
| **Nodal scalar/vector**      | DISPLACEMENT, REACTION, VELOCITY                               | nodes coloured / arrows      | point/vector |
| **Zero-length elements**     | springs, dashpots, contact, hinges (2 coincident nodes)        | symbol on top of mesh        | marker       |
| **Line-element diagrams**    | M, V, N along a beam at one step                               | curve perpendicular to axis  | line         |
| **Gauss-point fields**       | shells / planes / solids — IP positions coloured by σ, ε       | scatter or contour           | point/contour|
| **Surface contour**          | shell membrane/bending fields, properly triangulated           | filled patches per element   | contour      |
| **Volume fields**            | solid stress / strain — slice planes, iso-surfaces             | 3D                           | slice/iso    |
| **Fiber section response**   | force-based beam fibers at a station — σ over the cross-section| 2D plot of (y,z) fibers      | scatter      |
| **Time histories (XY)**      | response vs time for one or many DOFs/elements                 | classic 2D plot              | line         |

The taxonomy matters because each family has a different rendering primitive
(line, point, polygon, slice) and a different *cost model* (cheap edges vs.
expensive triangulation vs. very expensive iso-surfacing). The current
plotter API conflates "the plot type" with "the mpl call" — that's why every
new family today means a new top-level method on `Plot`.

---

## 2. Conceptual model: Scene + Layers

Adopt a **Scene / Layer** decomposition. This is the standard separation
in every visualization toolkit (matplotlib's `Figure`/`Artist`,
ParaView's `View`/`Representation`, pyvista's `Plotter`/`Actor`):

```
   Figure
     └── Scene (one drawable region — 2D or 3D)
           ├── MeshLayer            (model edges, cached)
           ├── DeformedMeshLayer    (mesh + displacement at step)
           ├── ContourLayer         (shell / plane field as filled patches)
           ├── GaussLayer           (IPs scattered or contoured)
           ├── DiagramLayer         (per-beam M/V/N curves)
           ├── ZeroLengthLayer      (springs as symbols)
           ├── NodeLayer            (selected nodes as markers)
           ├── VectorLayer          (nodal vectors as arrows)
           └── ...
```

A **Scene** owns:
- one set of axes (mpl 2D, mpl 3D, or — Phase 2 — a pyvista `Plotter`)
- the model's bounding box + projection plane
- a current step / model_stage
- the ordered list of layers
- the colorbar / legend registry

A **Layer** is a renderable: it knows how to draw itself, how to update
when the step changes, and how to query its data source. Layers are
*backend-agnostic* — they emit primitives (`segments`, `points`,
`polygons`, `arrows`, `contour_field`) that a backend translates to mpl
or pyvista calls.

This split fixes today's pain points:

- **Composition** is the default, not an afterthought. `scene.add(MeshLayer())`
  then `scene.add(ContourLayer(er, "membrane_xx", step=10))` produces a
  single coherent view. No more passing `ax` between two unrelated functions.
- **Animation** is `scene.set_step(k)` — every layer updates its artists
  in place. Works for matplotlib `FuncAnimation` *and* a pyvista interactor.
- **Multiple panels** are multiple Scenes in one Figure. Useful for
  side-by-side undeformed/deformed/contour or for building a 4-up
  pushover/hysteresis/profile/mesh report.

---

## 3. Selection — "plot parts of the model"

Already solved at the data layer (`SelectionSetResolver`). Lift it to the
viewer:

- Every layer accepts a **selection** parameter:
  `selection_set_name=`, `selection_set_id=`, `node_ids=`, `element_ids=`,
  `element_type=` (or any combination, AND-ed).
- A `Part(scene, selection)` convenience wraps a selection so a user can
  do `slab = scene.part(selection_set_name="slab"); slab.contour(...)`.
- A scene can hold many parts. A part is *not* a separate scene — same
  axes, same bounds — it's just a filter shorthand.
- The `mesh` layer drawn under a contour respects the same selection so
  the contour-on-mesh composite always lines up.

This handles the two real workflows:

1. *"Show me only the slab elements with their bending moment."* —
   one part, one mesh layer, one contour layer.
2. *"Show the whole frame in light gray, and overlay the plastic-hinge
   demand on the perimeter columns."* — one full-model mesh layer +
   one contour layer filtered to a perimeter selection set.

---

## 4. Multiple diagrams simultaneously

Two distinct meanings, both supported:

**(a) Same axes, multiple layers.** This is the Scene story above.
Layers stack in z-order; the colorbar/legend registry deconflicts them.
Examples that should "just work":

- mesh + deformed mesh (translucent original under solid deformed)
- mesh + contour + zero-length spring symbols + node labels
- frame mesh + perpendicular moment diagrams along selected beams + axial-force colored line elements

**(b) Multiple panels in one figure.** A `MultiScene` (or just helper
`scenes_grid(n_rows, n_cols)`) builds a Figure with several Scenes that
optionally share bounds / step / camera. Examples:

- 2×1: undeformed | deformed-with-contour
- 2×2: floor plan | elevation X | elevation Y | 3D iso
- 1×N: contour at step 10, 50, 100, 200 (animation strip)

A panel grid is *not* a single scene with subplots — each panel is its
own Scene with its own layer stack. That's the only way colorbars and
animation stay sane.

---

## 5. Layer catalog (initial)

Detailed per-family contracts. Each entry lists data source, rendering
primitive, the canonical method on the existing API that maps to it
(if any), and known unknowns.

### 5.1 `MeshLayer`
- **In:** dataset, optional selection, optional `model_stage` (advisory).
- **Out:** edge segments per element class.
- **Today:** `ds.plot.mesh()` — keep, wrap inside the layer.
- **Caching:** geometry per (selection-hash) so re-rendering at a new
  step doesn't rebuild segments.

### 5.2 `DeformedMeshLayer`
- **In:** as `MeshLayer` + `model_stage`, `step`, `scale`,
  `show_undeformed` toggle.
- **Out:** edge segments using `node_xyz + scale * displacement`.
- **Today:** `ds.plot.deformed_shape()`.
- **Animation:** `set_step(k)` re-evaluates displacement only; segment
  topology is cached.

### 5.3 `ZeroLengthLayer`
- **In:** dataset, selection (defaults to all `zeroLength*` element classes).
- **Out:** marker per element placed at the (coincident) node — symbol
  encodes element subtype: spring (◇), dashpot (▷), contact (■).
  Optional `color_by` couples it to a force/displacement at the IP.
- **Today:** *not supported.* `_edge_topology(2)` produces a degenerate
  segment of zero length — invisible.
- **Open:** how to read element subtype out of `element_type`. We
  already split on `[`; we need a small mapping `element_type -> symbol`
  in `format/`.

### 5.4 `NodeLayer` / `VectorLayer`
- **In:** dataset, selection, optional `NodalResults`, optional
  `component`, `step`, `scale_arrow`.
- **Out:** point or arrow per node. Color by scalar (e.g. |U|) or by
  vector magnitude. Arrows for true vector fields (DISPLACEMENT,
  REACTION). Quiver in 2D, `Line3DCollection` arrows in 3D.
- **Today:** *not supported* as a discrete layer. `xy()` / `plot_TH()`
  cover the time-series side.

### 5.5 `DiagramLayer` (line elements)
- **In:** `ElementResults` for line elements + canonical name +
  `step` + `scale`.
- **Out:** for each element, a polyline drawn perpendicular to the
  element axis whose offset from the axis = `scale * value(s)`. The
  classic structural-engineering moment / shear / axial diagram, *on
  top of the 3D mesh*.
- **Today:** `er.plot.diagram()` exists but is single-element, single
  axes, draws value vs. position rather than perpendicular-on-mesh.
  Keep the old `diagram()` (still useful for one-off inspection),
  add `DiagramLayer` for whole-model overlays.
- **Open:** sign convention per axis (is +Mz drawn on the +y or –y side
  of the local axis?). Default to "draws on the tension side" for
  bending; `flip=True` for the other convention.

### 5.6 `GaussLayer` (IPs scattered)
- **In:** `ElementResults` (gp_dim ≥ 1) + canonical + `step` + projection
  plane.
- **Out:** scatter at physical IP coords coloured by value.
- **Today:** `er.plot.scatter()` — keep, wrap inside the layer.

### 5.7 `ContourLayer` (shells / planes — proper contour)
- **In:** `ElementResults` (gp_dim == 2) + canonical + `step`.
- **Out:** *filled* per-element patches (one constant color per element
  using the element-mean of its IPs) **or** triangulated smooth contour
  (IP values extrapolated to nodes via shape functions, then
  `ax.tricontourf`). Default to per-element flat fill (faithful, fast,
  no smoothing artifact); offer smooth as opt-in.
- **Today:** *not supported.* `er.plot.scatter()` is a stand-in.
- **Open:** node-averaging policy at shared edges (mean? max-abs?
  per-side?). Default to mean for visual continuity, surface a
  `node_averaging="mean"|"max_abs"|"per_element"` knob.

### 5.8 `VolumeLayer` (solids)
- **In:** `ElementResults` (gp_dim == 3) + canonical + `step`,
  + `mode ∈ {"points", "slice", "iso"}`.
- **Out:**
  - `points` — IP scatter in 3D (works in mpl).
  - `slice` — IP values resampled onto a plane (needs a grid; fine in
    mpl with `imshow` after re-gridding, better in pyvista).
  - `iso` — iso-surface (needs marching cubes; **pyvista-only**).
- **Today:** *not supported.*
- **Reality check:** `slice` and `iso` are why we want a backend
  abstraction. Don't try to ship iso-surfacing in matplotlib.

### 5.9 `FiberLayer`
- **In:** `ElementResults` for a force-based beam with a fiber bucket,
  + `element_id` + `station` (line IP index) + `step` + `component`
  (e.g. `"stress_11"`).
- **Out:** scatter of the fibers in the section (y, z) plane coloured
  by value. Optional outline of the section if a section geometry
  description is provided.
- **Today:** *not supported.*
- **Open:** **fiber (y,z) coordinates are not in MPCO by default.**
  We either (a) require the user to supply a `FiberSection` geometry
  object alongside the result, or (b) read fiber positions from the
  STKO `.scd` definition file if the user has it. Land (a) first;
  (b) is a stretch goal.

### 5.10 `XYLayer` (time histories)
- 2D-only Scene type. Already covered by `NodalResultsPlotter.xy` and
  `plot_TH`. The Scene/Layer model still wraps these so we can stack
  multiple results, multi-stage annotations, and engineering recipes
  (pushover curve, hysteresis loop) under the same API.

---

## 6. Backend abstraction

A thin **`Backend` protocol** that layers call. Two implementations:

- `MplBackend` — what we ship. 2D and 3D, slow on big models, no
  interactive rotation in notebooks (3D mpl is bad), but zero new deps.
- `PyVistaBackend` (Phase 2, opt-in via extra `[viewer-3d]`) — fast 3D,
  iso-surfaces, screenshot-quality renders, optional interactive
  windows. Used for `VolumeLayer` slice/iso modes and for any 3D scene
  with > ~5k elements.

Backend protocol surface (sketch):

```python
class Backend(Protocol):
    def make_scene(self, *, is_3d: bool) -> SceneHandle: ...
    def add_segments(self, scene, segs, *, color, linewidth, ...): ...
    def add_points(self, scene, pts, *, color, size, ...): ...
    def add_polygons(self, scene, polys, *, values, cmap, ...): ...
    def add_arrows(self, scene, origins, vectors, *, scale, ...): ...
    def add_slice(self, scene, plane, field): ...   # may raise NotImplementedError
    def add_iso(self, scene, level, field): ...     # may raise NotImplementedError
    def set_bounds(self, scene, bbox): ...
    def show(self, scene): ...
    def save(self, scene, path, *, dpi=...): ...
```

Layers that need a primitive the active backend doesn't provide raise a
clear `BackendCapabilityError("VolumeIso requires the pyvista backend")`.
We don't silently degrade.

**Phase 1 hard constraint:** every existing top-level `ds.plot.*` /
`nr.plot.*` / `er.plot.*` method continues to work and produces the same
output. The Scene/Layer machinery sits *under* these methods; we don't
break the convenience API to ship the new architecture.

---

## 7. Coordinate handling

- 2D vs 3D: extend the current `_decide_3d` to consider both element
  topology (any 8-node brick → 3D) *and* node-z spread.
- Projection plane: explicit `view ∈ {"xy", "xz", "yz", "3d"}` parameter
  on `Scene`. `"xy"` is the natural default for floor plans; `"xz"`
  for elevations. `"3d"` always uses an mpl 3D axes.
- Aspect: `ax.set_aspect("equal")` already applied for 2D; mpl 3D needs
  the post-3.6 `ax.set_aspect("equal")` — guard for older versions.

---

## 8. Style + theming

Extend `PlotSettings` (today: a small Line2D-kwargs struct) to be
**hierarchical**:

```
SceneStyle
  - background, grid, font_size, theme ("light"/"dark")
  - per-layer-class defaults:
      MeshLayer: { color: "lightgray", linewidth: 0.5 }
      ContourLayer: { cmap: "viridis", node_averaging: "mean" }
      DiagramLayer: { tension_color: "red", compression_color: "blue" }
```

A user sets project defaults once on the dataset (`ds.plot_style = ...`),
overrides per-scene with `Scene(style=...)`, and overrides per-layer
with kwargs at `add()` time. Three levels — same precedence ladder as
the existing `_build_line_kwargs` in `NodalResultsPlotter`.

**Colormap policy by quantity type** (a small registry keyed on the
canonical name):
- Sequential (`viridis`, `plasma`) for non-signed scalars: |U|, σ_vm,
  ε_eq.
- Diverging (`RdBu_r`, `coolwarm`) for signed scalars where 0 is
  meaningful: M, σ_11, ε_xx.
- Categorical (`tab10`) for failure-mode / hinge-state plots.

---

## 9. Engineering recipes (free with the layer model)

Once Scene + Layers exist, these are 5–15 lines each — write them as
top-level helpers under `viewer/recipes.py`:

- `plot_pushover(ds, control_node, base_set, ax=...)` — base shear vs
  roof drift.
- `plot_drift_profile(ds, story_z_levels, step=..., ax=...)` — story
  drift up the building.
- `plot_hysteresis(er, element_id, x="rotation", y="moment", ax=...)`
  — moment-rotation loop for a beam.
- `plot_mode_shape(ds, mode_index, scale=...)` — scene with deformed
  layer driven by the eigenvector at that mode.
- `plot_plastic_hinge_map(ds, dcr_provider, step=..., threshold=1.0)`
  — element wireframe coloured by demand/capacity ratio.
- `plot_section_response(er, element_id, station, step)` — fiber scatter
  in the (y, z) cross-section plane.

---

## 10. Performance budget

The current renderers are O(n_elements) per call — fine for the
checked-in `elasticFrame` (~30 elements) and the `QuadFrame_results`
fixture (~600 elements). The targets that matter:

- ≤ 50k elements, 2D mesh: < 0.5 s render (mpl, vectorized
  `LineCollection`).
- ≤ 50k elements, 3D mesh: < 2 s render (mpl 3D), < 0.5 s (pyvista).
- Animation, 200 steps, ≤ 50k elements: ≥ 5 fps interactive (pyvista
  required).
- Cold contour on shells, ≤ 10k elements: < 1 s.

These are real-target numbers, not guesses. Bench them in `bench/`
once `Scene` exists.

**Cache discipline:**
- Mesh segments per `(selection_hash, deformed_or_not)` — invalidate
  on selection change, not on step change.
- IP physical positions per `ElementResults` (already implicit via
  `physical_coords()` — make it cached).
- Triangulation for shell contours — cache per selection.

---

## 11. What we explicitly **don't** ship in v1

To keep the first cut buildable, the following are deferred:

- 3D iso-surfacing (`VolumeLayer mode="iso"`) — pyvista-only, Phase 2.
- True interactive picking (click an element to print its results) —
  wants pyvista hooks; not in the matplotlib leg.
- Time-step animation as a built-in (`scene.animate(...)`) — wire it up
  when both mpl and pyvista paths exist; for now ship `set_step` and
  let users drive `FuncAnimation` themselves.
- Fiber section geometry from `.scd` files — require the user to pass
  a `FiberSection` object until we know what the read path looks like.
- VTK / XDMF export. Not a viewer concern; will live in `io/` if it
  ever lands.
- 3D arrow rendering in mpl. mpl `quiver` in 3D is poor; use
  `Line3DCollection` segments with arrow caps as a workaround.

---

## 12. Open questions for the user

1. **Backend priority.** Is the pyvista path acceptable as a Phase 2
   add-on (requires `pyvista` extra), or should v1 be matplotlib-only?
   Pyvista doubles the dep surface but unlocks real 3D and iso-surfaces.
2. **Fiber section geometry.** Are the STKO `.scd` definitions (which
   carry fiber y/z positions) something we can rely on existing next
   to the `.mpco`, or should we ask the user to construct a small
   `FiberSection` data class manually?
3. **Diagram sign convention.** "Tension side" vs. "+y local" for
   moment diagrams — which default do you use in your reports?
4. **Per-element vs. node-averaged contour default.** Per-element flat
   fill is "honest" (no averaging artefact); node-averaged smooth is
   prettier and matches what most commercial post-processors do.
   Which should be the default?
5. **Scope of "Part"**. Is a per-scene `Part` enough, or do we want
   parts to persist across scenes (e.g. `slab = ds.parts["slab"]` —
   reusable across every figure in a report)?

Answers steer Phase 1 vs. Phase 2 splits in the implementation plan.

---

## Next step

Once these directives are signed off, the implementation plan should:

1. Define the `Backend` protocol and the `MplBackend` (one PR).
2. Land `Scene` + `MeshLayer` + `DeformedMeshLayer` + `NodeLayer`
   under it; keep `ds.plot.mesh()` / `deformed_shape()` working as
   thin wrappers (one PR).
3. Add `ContourLayer` (per-element flat fill first), `GaussLayer`,
   `DiagramLayer`, `ZeroLengthLayer` — one PR each, each with its
   own bench + integration test.
4. Land `VolumeLayer` modes `points` and `slice` against `MplBackend`;
   leave `iso` for the pyvista PR.
5. Add the recipes module on top once layers stabilize.

Each PR follows the existing semver discipline: backwards-compatible
adds are MINOR, anything that touches a public method signature is
MAJOR. The legacy `Plot` / `NodalResultsPlotter` / `ElementResultsPlotter`
methods stay working until at least the next MAJOR release.
