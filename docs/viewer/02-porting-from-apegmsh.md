# Porting from apeGmsh

The apeGmsh `ResultsViewer` is a mature ~20k-line PyVista/Qt desktop
post-processor. We are not vendoring it as a dependency — we're **lifting
selected pieces** into `STKO_to_python.viewer` and adapting them to our
data model.

This document is the file-by-file plan, with adapter notes and an
overlap audit against the work already shipped in v1.8.0.

---

## 1. Scope

**Lifted** = source code copied (or significantly inspired) and adapted
to STKO_to_python's data model and module layout.

**Not lifted** = either reimplemented from scratch (better fit with our
existing code) or skipped entirely (out of scope).

Three tiers, increasing in complexity and dep cost:

- **Tier 1** — pure numpy / math. No Qt, no VTK. Lands in Phase 1.
- **Tier 2** — Scene-builder + diagrams (renderable layers). Needs
  PyVista. Lands in Phase 2–3.
- **Tier 3** — Qt UI. Needs PySide6 + pyvistaqt. Lands in Phase 4.

apeGmsh paths below are relative to its `src/apeGmsh/`. STKO_to_python
target paths are relative to `src/STKO_to_python/viewer/`.

---

## 2. Attribution & license

- apeGmsh is your own codebase, so attribution is informational rather
  than legal. Still, every lifted file should carry a one-line header:

  ```python
  # Adapted from apeGmsh viewers/diagrams/_contour.py (commit <sha>).
  ```

- If apeGmsh ever ships under a different license than STKO_to_python
  (currently MIT), the lift becomes a license question. Today it is not.

---

## 3. Tier 1 — pure math (Phase 1)

Pure numpy. No rendering dependencies. Most valuable for the section-cut
work that's already in `cuts/` and for the eventual contour layers.

| apeGmsh source | Target | Adaptation |
|---|---|---|
| `results/_gauss_extrapolation.py` | `viewer/math/gauss_extrapolation.py` | Drop apeGmsh's `Slab` typing; accept `(values: np.ndarray, gp_natural: np.ndarray, element_node_coords: np.ndarray)` shaped the same way `ElementResults.physical_coords()` already returns. Reuse `format/shape_functions.py` registries — don't carry apeGmsh's parallel implementation. |
| `results/_shape_functions.py` | merge into `format/shape_functions.py` | Audit gaps first: STKO has shape fns for the elements in `cuts/` already. If apeGmsh covers a type we don't, port that entry into the existing registry. |
| `results/_gauss_world_coords.py` | already exists as `format/shape_functions.compute_physical_coords` | **Don't port.** Audit instead — apeGmsh may handle element types we don't. |
| `results/_shell_geometry.py` | `viewer/math/shell_frame.py` | Adapter point: STKO's shells live in `cuts/kernels/shell.py` and `format/shape_functions.py` — keep the local-axes computation backed by a quaternion (good numerics) but expose it as a free function, not a class. |
| `viewers/diagrams/_beam_geometry.py` | `viewer/math/beam_frame.py` | apeGmsh derives beam local frame from endpoint coords + a chord-vector heuristic. STKO's beam frame today is handled per-kernel in `cuts/kernels/beam.py`. Port the apeGmsh version once into `viewer/math/` and let both `cuts/` and viewer code consume it. **Audit `cuts/kernels/beam.py` first** to make sure the conventions match. |
| `viewers/core/results_pick.py` (display-space box-pick only) | `viewer/math/picking.py` | The clever vectorized projection trick: build camera matrix once, batch-multiply all candidate world points to screen space, intersect with the drag rectangle. ~40× faster than per-point `WorldToDisplay`. **Critical for picking thousands of Gauss markers.** Strip out the VTK-specific actor handling — we just need the projection math. |

**Effort:** ~1 week.

**Files NOT to port at Tier 1:**

- `viewers/scene/glyph_points.py` — needs PyVista; defer to Tier 2.
- `results/_slabs.py` — apeGmsh's frozen Slab dataclasses. Our equivalent
  is `NodalResults` / `ElementResults`. Don't duplicate the data model.

---

## 4. Tier 2 — Scene builder + diagrams (Phase 2–3)

| apeGmsh source | Target | Adaptation |
|---|---|---|
| `viewers/scene/fem_scene.py` | `viewer/scene_3d/fem_scene.py` | Heavy adapter. apeGmsh consumes `FEMData` (custom); we consume `MPCODataSet`. Two changes: (1) input adapter — pull node coords and connectivities from `ds.nodes` and `ds.elements`; (2) linearization table — extend apeGmsh's `GMSH_LINEAR` lookup to cover the STKO element-type-tag set (OpenSees class tags, not Gmsh element codes). |
| `viewers/diagrams/_base.py` | `viewer/core/layer.py` | Rename `Diagram` → `Layer` to match STKO/directive vocabulary. Keep the `attach` / `update_to_step` / `detach` lifecycle exactly. The `DiagramSpec` becomes `LayerSpec`. |
| `viewers/diagrams/_director.py` | `viewer/core/director.py` | Slim down — apeGmsh's director has TimeMode = SINGLE/RANGE/ENVELOPE/ANIMATION; Phase 2 ships SINGLE only. Drop the multi-stage "combined" pseudo-stage logic until we know we need it for STKO. |
| `viewers/diagrams/_contour.py` | `viewer/layers/contour.py` | **High-value port.** Five-path dispatch (nodal, GP-cell, GP-cell-averaged, GP-node-extrap, GP-node-discrete). Adapter: feed it from `ElementResults` instead of apeGmsh slabs. |
| `viewers/diagrams/_deformed_shape.py` | `viewer/layers/deformed_mesh.py` | Adapter: pull displacements via `NodalResults`. Keep in-place point mutation (perf contract). |
| `viewers/diagrams/_vector_glyph.py` | `viewer/layers/node.py::VectorLayer` | Direct port. Wrap in our `NodeLayer` / `VectorLayer` split (directive §5.4). |
| `viewers/diagrams/_line_force.py` | `viewer/layers/diagram.py` | Hatched fill perpendicular to beam axis (M, V, N diagrams). Sign convention — see open question #3 in roadmap. |
| `viewers/diagrams/_fiber_section.py` | `viewer/layers/fiber.py` | **Compare with v1.8.0 per-fiber shell views first.** Likely a 60–80% port. The 3D fiber cloud + 2D side-panel architecture is good; the data accessor needs to read STKO's fiber slabs. |
| `viewers/diagrams/_layer_stack.py` | `viewer/layers/layer_stack.py` | **Compare with v1.8.0 per-layer shell views first.** Apeg's aggregation strategies (mid_layer / mean / max_abs) are worth lifting; our v1.8.0 work may already have its own conventions — reconcile before merging. |
| `viewers/diagrams/_gauss_marker.py` | `viewer/layers/gauss.py` | World-coord mapping via shape functions — but we already have `format/shape_functions.compute_physical_coords`; consume that instead of porting apeGmsh's parallel pathway. Sphere markers as real geometry (not billboards) — keep. |
| `viewers/diagrams/_spring_force.py` | `viewer/layers/zerolength.py` (partial) | Spring-only flavor of the directive's `ZeroLengthLayer`. Useful starting point for the broader symbol-based renderer. |
| `viewers/diagrams/_loads.py` | `viewer/layers/loads.py` *(optional)* | Static load glyphs. Lower priority since MPCO doesn't carry the applied loads — would need to read from a separate source. Defer until there's a use case. |
| `viewers/diagrams/_reactions.py` | `viewer/layers/reactions.py` | Reaction arrows + curved-arrow moment glyphs. Pulls from `NodalResults.REACTION` — direct adapter. |
| `viewers/diagrams/_scalar_bar_support.py` | `viewer/core/style.py` (extend) | Merge into `LayerStyle` — every layer with a scalar field has a colormap and clim. |
| `viewers/diagrams/_selectors.py` | `viewer/core/selection.py` | Replace `SlabSelector` with a wrapper around our existing `SelectionSetResolver`. Same grammar (pg / label / selection / ids) maps to ours (selection_set_name / selection_set_id / node_ids / element_ids). |
| `viewers/diagrams/_compositions.py` | skip | Specific to apeGmsh's CAD geometry manager; not relevant to STKO. |
| `viewers/diagrams/_kind_catalog.py` | `viewer/layers/__init__.py` (`LAYER_KINDS` dict) | Simple registry; trivial port. |
| `viewers/diagrams/_registries.py` | `viewer/core/scene.py::Scene.layers` | Absorbed into the `Scene` class — we don't need a separate `DiagramRegistry`. |
| `viewers/core/clipping_controller.py` | `viewer/layers/clipping.py` | **Audit against `cuts/` first.** v1.8.0 has section-cut kernels and plotting helpers; the interactive clipping plane widget is what's missing. Port the VTK plane-widget glue, but the cut math itself stays in `cuts/`. |
| `viewers/core/color_manager.py` | `viewer/core/style.py` (extend) | Merge — `SceneStyle` already has theming. |
| `viewers/core/element_visibility.py` | `viewer/core/layer.py::Layer.set_visible_subset()` | Per-cell vtkGhostType masking. Useful for "hide this element type" controls. |
| `viewers/core/opacity_controller.py` | `viewer/core/style.py` (extend) | Per-actor alpha + depth-peeling enable. |
| `viewers/animation.py` | `viewer/headless/runner.py` | Direct port for Phase 5. MP4 via imageio-ffmpeg, GIF via Pillow. |

**Effort:** ~3 weeks for Phase 2 + Phase 3 combined.

**Files NOT to port at Tier 2:**

- `viewers/scene/brep_scene.py` — CAD geometry; not in STKO scope.
- `viewers/scene/mesh_scene.py` — pre-solve mesh; not in STKO scope (we're
  post-processing only, by directive).
- `viewers/results/live/*` — live recording during analysis. Out of scope.

---

## 5. Tier 3 — Qt UI (Phase 4)

Most expensive tier. Doing this is what "full experience" means.

| apeGmsh source | Target | Adaptation |
|---|---|---|
| `viewers/results_viewer.py` (orchestrator, 1955 lines) | `viewer/qt/main_window.py` (split) | Don't port as one file. The orchestrator does dock layout + Director wiring + signal plumbing — split into `main_window.py`, `app.py`, and controller files. |
| `viewers/ui/_results_window.py` | `viewer/qt/main_window.py` | Dock layout. Direct port; rename to match our module naming. |
| `viewers/ui/_outline_tree.py` | `viewer/qt/widgets/outline_tree.py` | Stage/layer/geometry tree. Drop apeGmsh's geometry section (we don't have pre-solve geometry). |
| `viewers/ui/_plot_pane.py` | `viewer/qt/widgets/viewport.py` | Embeds the `QtInteractor`. Simple. |
| `viewers/ui/_details_panel.py` | `viewer/qt/widgets/details_panel.py` | Right-dock; routes to layer-specific settings tabs. |
| `viewers/ui/_diagram_settings_tab.py` | `viewer/qt/widgets/layer_settings_tab.py` | One tab per active layer. Each layer contributes a `settings_widget()` factory. |
| `viewers/ui/_geometry_settings_panel.py` | `viewer/qt/widgets/scene_settings_panel.py` | Global scene knobs: deformation scale, show-undeformed toggle, background. |
| `viewers/ui/_time_scrubber.py` | `viewer/qt/widgets/time_scrubber.py` | Slider + play/pause + step counter. Drives the Director. |
| `viewers/ui/_time_history.py` | `viewer/qt/widgets/time_history_tab.py` | Matplotlib in a Qt tab. Shows TH for shift-clicked node. Uses our existing `NodalResultsPlotter.xy` machinery. |
| `viewers/ui/_add_diagram_dialog.py` | `viewer/qt/widgets/add_layer_dialog.py` | Multi-select dialog. Layer kind catalog drives the choices. |
| `viewers/ui/_session_panel.py` | `viewer/qt/controllers/session.py` | Save/restore window layout + scene specs to disk. Uses QSettings + our `SceneSpec` round-trip. |
| `viewers/ui/_pick_readout_hud.py` | `viewer/qt/widgets/pick_readout_hud.py` | HUD overlay. Direct port. |
| `viewers/ui/_viewport_hud.py` | `viewer/qt/widgets/viewport_hud.py` | FPS + mode + key hints. Direct port. |
| `viewers/ui/theme.py` | `viewer/qt/theme.py` | Qt palette. Direct port; adjust accent colors to match STKO branding if desired. |
| `viewers/ui/preferences.py` + `preferences_manager.py` | `viewer/qt/preferences.py` | Persisted settings. Drop apeGmsh-specific keys. |
| `viewers/core/navigation.py` | `viewer/qt/controllers/navigation.py` | Camera presets (front, top, iso). Direct port. |
| `viewers/core/pick_engine.py` | `viewer/qt/controllers/picking.py` | Click + box-pick controller; uses `viewer/math/picking.py` underneath. |
| `viewers/core/results_pick_engine.py` | merge into `picking.py` | Combine; the two are very close in apeGmsh. |
| `viewers/overlays/*` | defer | Probe / measure / constraint overlays. Phase 4+ nice-to-have. |

**Effort:** ~4 weeks.

**Files NOT to port at Tier 3:**

- `viewers/ui/_add_diagram_dialog.py`'s geometry-aware logic. STKO has no
  pre-solve geometry.
- `viewers/overlays/constraint_overlay.py` — boundary condition markers.
  MPCO doesn't carry these; would need a separate input path. Defer.

---

## 6. Non-obvious algorithms worth preserving verbatim

These are the "wouldn't reinvent" tricks. Carry them across with care
(unit-test them on day one):

1. **Gauss → nodal extrapolation using linear shape functions.** apeGmsh
   notes that even when the substrate VTK cells are higher-order, using
   *linear* shape functions for the extrapolation is more stable under
   `pinv`. Without this insight you get noisy contours on quadratic
   elements.
2. **In-place actor scalar mutation.** Every layer pre-allocates its
   per-step scalar array at attach time and only `Modified()`-flags it
   on update. Recreating actors per step is what makes naive viewers
   choke on big models.
3. **Display-space box pick.** Vectorize via the camera matrix; never
   loop `WorldToDisplay` per point. 40× speedup at 1M Gauss points.
4. **Shell local-axes quaternion.** Avoids gimbal lock in the
   normal-rotation pipeline when shells are deformed.
5. **Sphere markers, not billboards, for Gauss points.** Avoids
   z-fighting at glancing angles when the user rotates around dense
   GP clouds.
6. **Layer aggregation strategies (mid_layer / mean / max_abs)**
   precomputed at attach as dot-product weights — per-step cost is one
   dot per cell.

---

## 7. Overlap audit with v1.8.0 work

We've already shipped section cuts and per-layer/per-fiber shell views.
Before porting the apeGmsh equivalents, **diff what exists**.

The beam-local-frame row was audited at the start of Phase 1; the
finding is captured below. The other rows are still on the "audit before
porting" list and will be settled when their respective layers land.

| Topic | STKO_to_python today | apeGmsh equivalent | Action |
|---|---|---|---|
| Section cuts (math) | `cuts/kernels/{beam,shell,solid}.py` | none (apeGmsh has only an interactive clipping plane; no resultant integration) | **Keep STKO's** — apeGmsh's clipping is a viz feature, not an integration kernel. |
| Section cuts (interactive plane) | none | `viewers/core/clipping_controller.py` | **Port apeGmsh's** — wrap STKO's `Plane` so the GUI exposes it. |
| Per-layer shell views | shipped in v1.8.0 | `viewers/diagrams/_layer_stack.py` | **Audit + reconcile** before Phase 3. Likely keep STKO's data model + lift apeGmsh's aggregation tactics. |
| Per-fiber views | shipped in v1.8.0 | `viewers/diagrams/_fiber_section.py` | **Audit + reconcile** before Phase 3. Likely keep STKO's data model + lift apeGmsh's 3D-cloud + 2D-side-panel UI pattern. |
| Beam local frame | `model/transforms.py` + `CDataReader.rotation_matrix` (quaternion from STKO `.cdata` `*LOCAL_AXES`) | `viewers/diagrams/_beam_geometry.py` (vecxz Gram-Schmidt from endpoint coords) | **STKO's wins** — the quaternion is the exact frame OpenSees used during analysis; apeGmsh's reconstructs it. Port apeGmsh's `compute_local_axes` to `viewer/math/beam_frame.py` as a **fallback** for datasets without `.cdata` and for unit tests. Fill-axis policy (`COMPONENT_TO_LOCAL_AXIS`, `fill_axis_for`, `resolve_fill_direction`) is rendering policy, not geometry — defer to Phase 3's `DiagramLayer`. |

**Note on `cuts/kernels/beam.py`:** earlier drafts of this table listed
that module as the local-frame counterpart to port from apeGmsh. The
audit found it is unrelated — `cuts/kernels/beam.py` solves
**plane–segment intersection** geometry for section cuts, not local-frame
math. Different problem; nothing to reconcile.

**This audit is a precondition for each porting PR.** It's a few hours
of reading per topic, not a separate phase, but skip it and we end up
with two parallel implementations of the same physics.

---

## 8. Adapter strategy: STKO DataFrames → array-shaped inputs

apeGmsh layers expect `Slab` objects — numpy arrays with a fixed
shape per topology (nodes / element / GP / fiber / layer).
STKO returns pandas DataFrames with MultiIndex columns.

The `MPCODataSourceAdapter` (`viewer/core/datasource.py`) is the *only*
place this conversion happens:

```python
def element_values(self, name, component, step, *, topology="element", ...):
    er = self.dataset.elements.get_element_results(
        results_name=name,
        # ... selection ...
    )
    df = er.df  # MultiIndex
    if topology == "gp":
        return er.gp_values(component, step)         # (sum_gp,) array
    if topology == "fiber":
        return er.fiber_values(component, step)      # (sum_fiber,) array
    # ... etc.
```

**Rules:**

- Adapter methods return **contiguous numpy arrays** in element order
  consistent with the grid built by `viewer/scene_3d/fem_scene.py`.
- The adapter caches array views — every call doesn't trigger a fresh
  DataFrame query. LRU around `(name, component, step, stage,
  selection_hash, topology)`.
- Slab-like dataclasses are an *internal* implementation detail of
  layers if they need them; they are not part of the DataSource API.

This is where STKO's existing query-engine caching pays off — the
adapter is mostly a thin shape-translator over an already-cached DataFrame.

---

## 9. Files we explicitly do **not** port

- `viewers/scene/brep_scene.py`, `mesh_scene.py` — pre-solve geometry.
- `results/live/*` — live recording during analysis.
- `results/transcoders/_recorder.py` — apeGmsh ingests OpenSees `.out`
  files. STKO ingests `.mpco` only.
- `results/readers/_mpco*` — apeGmsh has its own MPCO reader. We have
  ours; we keep ours. Not the other way around.
- `viewers/ui/preferences*` — port the *pattern* but write our own
  config (we're not inheriting apeGmsh's settings keys).
- `architecture/*.md` docs — they're internal to apeGmsh.

---

## 10. Process for each port

1. Open the apeGmsh source file. Read it. Note the SHA you read from.
2. Write a STKO_to_python skeleton in the target path with `pass` bodies
   and the adapted signatures.
3. Land that skeleton with imports + type stubs only. CI green.
4. Port the body, swapping apeGmsh data accesses for `DataSource` calls.
5. Add a unit test against the smallest fixture that exercises the
   layer (`elasticFrame/results/` for 1D, `QuadFrame_results/` for
   shells, `solid_partition_example/` for solids).
6. Add the source-attribution header.
7. PR — small enough to review in one sitting (≤ 600 LOC diff).

---

## 11. Lift estimates by tier

| Tier | Files | apeGmsh LOC (approx) | Target LOC (after adaptation) | Effort |
|---|---|---|---|---|
| Tier 1 — math | 5 | ~600 | ~500 | 1 week |
| Tier 2 — scene + layers | 18 | ~4000 | ~3500 | 3 weeks (Phase 2 + 3) |
| Tier 3 — Qt UI | 16 | ~5000 | ~4000 | 4 weeks |
| **Total** | **~40** | **~9600** | **~8000** | **~8 weeks** |

The total is consistent with the roadmap estimate of ~10–12 weeks to
Phase 4 (the difference is Phase 0 setup, Phase 2 refactor work that
isn't a direct port, and integration testing).
