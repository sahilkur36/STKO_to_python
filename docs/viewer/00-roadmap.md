# Viewer Implementation Roadmap

**Status:** plan of record. Graduates `viewer-refactor-directives.md` from
brainstorm into phased delivery.
**Anchor docs:**
[`viewer-refactor-directives.md`](../viewer-refactor-directives.md) (conceptual
model — Scene/Layer, layer catalog),
[`01-architecture.md`](01-architecture.md) (renderer-agnostic layering, Backend
protocol, three deployment targets),
[`02-porting-from-apegmsh.md`](02-porting-from-apegmsh.md) (concrete file
lift list and adaptation notes),
[`03-deployment-targets.md`](03-deployment-targets.md) (local, headless,
cluster, web).

---

## Vision

Ship a **full desktop post-processing experience** — Qt window with embedded
3D viewport, time scrubber, diagram settings, picking, animation export — on
top of the existing matplotlib 2D library, **without** breaking the lightweight
notebook usage and **without** forcing every user to install Qt/VTK.

The viewer is **opt-in via extras**: `pip install stko_to_python` keeps the
current numpy/pandas/matplotlib footprint. `pip install stko_to_python[viewer]`
adds the Qt/PyVista stack.

The architecture is **renderer-agnostic from day one** so the same
`Scene`/`Layer` graph that drives the Qt window also drives off-screen
animation export on a cluster, and (later) a Trame-based browser viewer for
SSH-only workflows.

---

## Current state (anchor: v1.8.0)

What's already shipped on `main`:

- 2D matplotlib plotting: `Plot`, `NodalResultsPlotter`, `ElementResultsPlotter`.
- `cuts/` section-cut kernels (beam, shell, solid) + their plotting helpers.
- Per-layer / per-fiber shell views (v1.8.0).
- Selection sets, query engines, aggregation engine, multi-case `MPCOResults`.
- No 3D rendering. No animation. No picking. No GUI.

What this plan adds: everything above the "no" line, behind an opt-in extra.

---

## Semver mapping

Adding the viewer subpackage is **purely additive** — no public API of v1.8.0
changes. New code lives under `STKO_to_python.viewer` (new namespace) and is
gated behind extras. So everything below stays on the v1.x track until we
actually break something.

| Phase | Target tag | Why |
|---|---|---|
| Phase 0 — Foundation | `v1.9.0` | New optional extras, new empty subpackage. |
| Phase 1 — Algorithm tier | `v1.10.0` | New pure-numpy modules; matplotlib renderers may opt to use them. |
| Phase 2 — Backend + 2D layers | `v1.11.0` | Scene/Layer machinery under existing `Plot.*` methods. Convenience API unchanged. |
| Phase 3 — PyVista backend + 3D layers | `v1.12.0` | New `[viewer-3d]` extra. Qt not yet required. |
| Phase 4 — Qt desktop | `v1.13.0` | New `[viewer]` extra. Adds CLI entry point `stko-viewer`. |
| Phase 5 — Headless CLI | `v1.14.0` | New `[viewer-headless]` extra; animation/screenshot CLI. |
| Phase 6 — Trame web | `v2.0.0` *(optional)* | Only bumps MAJOR if it forces a refactor of the public viewer API. Otherwise stays MINOR. |

A breaking API change anywhere in this plan triggers `v2.0.0` immediately.
Nothing in the current outline requires that.

---

## Phase 0 — Foundation

**Goal:** stand up the empty viewer subpackage, optional extras, CI gates,
docs skeleton. No rendering yet.

**Deliverables:**

- `src/STKO_to_python/viewer/__init__.py` — empty namespace, `__all__ = []`.
- `src/STKO_to_python/viewer/_version.py` — pinned schema versions for
  saved sessions (forward-compat).
- `pyproject.toml` — new optional extras:
  - `viewer-3d` → `pyvista>=0.43`, `vtk>=9.2`
  - `viewer` → `viewer-3d` + `PySide6>=6.5`, `pyvistaqt>=0.11`, `qtpy`
  - `viewer-headless` → `viewer-3d` + `imageio>=2.30`, `imageio-ffmpeg>=0.4`
  - `viewer-web` (placeholder) → `trame>=3.0`, `trame-vtk`, `trame-vuetify`
- `tests/viewer/` directory with one smoke test (import + `__all__` check).
- `.github/workflows/test.yml` — add a job matrix entry that installs
  `[viewer]` and runs the viewer smoke tests under `xvfb-run` on Linux only.
- `docs/viewer/` — these four planning docs (this PR).

**Definition of done:**

- `pip install -e .[viewer]` succeeds on Linux, macOS, Windows.
- `python -c "import STKO_to_python.viewer"` succeeds without any heavy
  imports happening at module-load time.
- CI green on the new viewer job.

**Estimate:** ~2 days. **Risk:** low.

---

## Phase 1 — Algorithm tier (pure numpy)

**Goal:** lift the math-only pieces from apeGmsh that we'll need later
(Gauss extrapolation, shape functions for missing element types, shell
local-axes quaternion, beam local-frame). Wire what's useful into the
existing `cuts/` and `format/` packages now, ahead of any rendering.

**Deliverables:**

- `src/STKO_to_python/viewer/math/` (new module):
  - `gauss_extrapolation.py` — pinv-based GP → nodal smoothing with
    cross-element averaging. Adapted from apeGmsh's
    `results/_gauss_extrapolation.py`.
  - `beam_frame.py` — element local frame from endpoint coords + chord
    vector heuristic. Adapted from `viewers/diagrams/_beam_geometry.py`.
  - `shell_frame.py` — shell mid-surface local-axes quaternion. Adapted
    from `results/_shell_geometry.py`.
- Possible expansion of `src/STKO_to_python/format/shape_functions.py` to
  cover any element types apeGmsh handles that we don't yet.
- `tests/viewer/math/` — unit tests against analytic cases and against
  the existing `cuts/` kernel outputs (regression).

**Definition of done:**

- 100% unit-test coverage on the three new modules.
- A bench comparing GP → nodal extrapolation cost on the
  `Test_NLShell` fixture (target: < 50 ms for 10k shell elements).
- The existing `cuts/kernels/shell.py` optionally consumes the new
  extrapolation routine (gated by a kwarg, default off) for a follow-up
  PR to wire in. **Not coupled** to this phase landing.

**Estimate:** ~1 week. **Risk:** low (pure math).

---

## Phase 2 — Backend protocol + 2D layers

**Goal:** introduce `Scene` + `Layer` + `Backend` from
[`01-architecture.md`](01-architecture.md). Re-implement the current
matplotlib plotters as layers on top of `MplBackend`. No new user-visible
features — pure refactor under the existing API.

**Deliverables:**

- `src/STKO_to_python/viewer/core/`:
  - `backend.py` — `Backend` protocol.
  - `scene.py` — `Scene`, `MultiScene`.
  - `layer.py` — `Layer` base class.
  - `style.py` — hierarchical `SceneStyle` (extends `PlotSettings`).
  - `datasource.py` — `DataSource` adapter: `MPCODataSet` → layer-ready
    arrays.
- `src/STKO_to_python/viewer/backends/mpl/` — `MplBackend` implementation,
  one file per primitive (`segments.py`, `points.py`, `polygons.py`,
  `arrows.py`).
- `src/STKO_to_python/viewer/layers/`:
  - `mesh.py` — `MeshLayer` (wraps current `plot_mesh`).
  - `deformed_mesh.py` — `DeformedMeshLayer` (wraps current `plot_deformed_shape`).
  - `node.py` — `NodeLayer` + `VectorLayer`.
  - `xy.py` — `XYLayer` (wraps `NodalResultsPlotter.xy`).
- Existing `Plot.*` / `NodalResultsPlotter.*` / `ElementResultsPlotter.*`
  rewired to build a `Scene` under the hood. **No signature changes.**

**Definition of done:**

- Every existing test in `tests/integration/test_plotting*.py` still passes
  byte-identically (visual regression via image diff under tolerance).
- New `tests/viewer/test_scene_layer.py` covers composition cases
  (mesh + deformed; mesh + node; multiple scenes in a figure).
- `docs/cookbook/` — one new recipe demonstrating direct Scene/Layer use
  (no behavior change to the old recipes).

**Estimate:** ~2 weeks. **Risk:** medium (refactor under live API).

---

## Phase 3 — PyVista backend + 3D layers

**Goal:** add the `PyVistaBackend` and the layer types that require 3D
rendering. Off-screen mode only — no Qt yet.

**Deliverables:**

- `src/STKO_to_python/viewer/backends/pyvista/` — `PyVistaBackend`,
  off-screen mode, snapshot/save to PNG, animation as image sequence.
- `src/STKO_to_python/viewer/scene_3d/`:
  - `fem_scene.py` — `MPCODataSet` → `pyvista.UnstructuredGrid` builder.
    Adapted from apeGmsh `viewers/scene/fem_scene.py`.
- `src/STKO_to_python/viewer/layers/`:
  - `contour.py` — `ContourLayer` with five-path dispatch (nodal,
    GP-cell, GP-cell-averaged, GP-node, GP-node-discrete).
  - `gauss.py` — `GaussLayer` (IP markers in 3D).
  - `volume.py` — `VolumeLayer` modes `points` and `slice` (3D) plus `iso`
    (PyVista-only via marching cubes).
  - `diagram.py` — `DiagramLayer` for line-element M/V/N on 3D mesh.
  - `fiber.py` — `FiberLayer` for force-based beam fiber sections.
  - `layer_stack.py` — through-thickness shell view.
  - `zerolength.py` — `ZeroLengthLayer` symbols.
  - `clipping.py` — interactive clipping plane (plugs into existing
    `cuts/` so the viewer's section-cut UI uses the kernels we already have).
- `tests/viewer/test_pyvista_backend.py` — off-screen render regression
  tests against golden PNGs (tolerance-bounded image diff).

**Definition of done:**

- All eight layer types render against the
  `elasticFrame/QuadFrame_results` fixture in off-screen mode.
- `Scene(backend="pyvista", off_screen=True).save("out.png")` works on CI
  with `xvfb-run`.
- Bench: 50k-element 3D mesh under 0.5 s cold render. 200-step animation
  at ≥ 5 fps as image sequence.

**Estimate:** ~3 weeks. **Risk:** medium (lots of layer types). Mitigated
because the algorithms are mostly already done by Phase 1.

---

## Phase 4 — Qt desktop UI

**Goal:** the "full experience." Ship the `stko-viewer` GUI: dock layout,
time scrubber, diagram tree, picking, settings tabs, animation export.

**Deliverables:**

- `src/STKO_to_python/viewer/qt/`:
  - `app.py` — `QApplication` bootstrap, theming.
  - `main_window.py` — `ResultsWindow` (QMainWindow, dock layout).
  - `widgets/`:
    - `outline_tree.py` — left dock: stages, layers, geometry.
    - `viewport.py` — center: `QtInteractor` wrapping the PyVista plotter.
    - `details_panel.py` — right dock: per-layer settings tabs.
    - `time_scrubber.py` — bottom: play/pause/slider/step count.
    - `add_layer_dialog.py` — choose layer type + selection + component.
    - `pick_readout_hud.py` — HUD on picked node/element/GP.
  - `controllers/`:
    - `director.py` — step orchestration; layers register for
      `step_changed` callbacks.
    - `picking.py` — port of apeGmsh's vectorized box-pick math.
    - `session.py` — save/restore window layout + layer specs to disk.
- `pyproject.toml` — add `[project.scripts]` entry: `stko-viewer = STKO_to_python.viewer.qt.cli:main`
- `stko-viewer <file.mpco>` — open the file in the GUI.
- `MPCODataSet.viewer()` — open the current dataset in the GUI (blocks).

**Definition of done:**

- Open any committed fixture in the GUI and use every Phase-3 layer
  interactively.
- Time scrubber drives layers at ≥ 10 fps on QuadFrame.
- Box-pick selects ≥ 1k Gauss markers in < 100 ms.
- Save/restore session produces a byte-stable file across runs.
- Smoke test on CI under `pytest-qt` + `xvfb-run`.

**Estimate:** ~4 weeks. **Risk:** medium-high (Qt UI breadth, lots of
small panels).

---

## Phase 5 — Headless / batch CLI

**Goal:** make the viewer usable on an SSH-only cluster for animation
export, screenshots, and dump-to-image workflows. Same Scene/Layer graph
as the GUI, no Qt loop.

**Deliverables:**

- `stko-viewer animate run.mpco --config view.toml --out anim.mp4`
- `stko-viewer screenshot run.mpco --step 250 --out frame.png`
- `stko-viewer batch run.mpco --config batch.toml --out frames/`
- `view.toml` schema — declarative spec for layers + camera + step range.
  Matches the in-memory `Scene` layout 1:1 so the GUI can save → run on
  cluster → reload result.
- Cluster docs: how to install `[viewer-headless]` without a display, how
  to use Xvfb/EGL/Mesa, recommended SLURM stanza.

**Definition of done:**

- Animation runs on a headless Linux box with no `$DISPLAY` set.
- `stko-viewer screenshot` produces the same PNG (within tolerance) as
  the GUI's "Save snapshot" button does on the same step.

**Estimate:** ~1.5 weeks. **Risk:** low. Most of the work is CLI wiring;
rendering uses Phase 3 off-screen mode.

---

## Phase 6 — Trame web viewer *(deferred / optional)*

**Goal:** browser-based viewer for cluster + SSH workflows.
**Status:** not committed. Land Phase 5 first; revisit only if there's
demonstrated demand.

**Outline:**

- `src/STKO_to_python/viewer/web/` — Trame server.
- Same `Scene`/`Layer` model; the renderer is a `TrameBackend` that uses
  `pyvista.trame` and exposes the scene over WebSocket.
- One-port SSH tunnel: `ssh -L 8080:localhost:8080 cluster` →
  `localhost:8080` in browser.
- Two render modes: server-side rendering (encoded frames over
  WebSocket) and client-side via `vtk.js` (geometry shipped once).

**Estimate:** ~3–4 weeks **if** the layering from Phase 2/3 stays clean.
Up to twice that if layers leaked VTK assumptions.

---

## Open questions still gating decisions

Five questions from `viewer-refactor-directives.md` §12 are still open.
The apeGmsh review settles one of them; the rest still need a call:

| # | Question | Status |
|---|---|---|
| 1 | Backend priority — mpl-only v1 or pyvista as Phase 2? | **Settled** by user: full experience, so pyvista is committed. |
| 2 | Fiber section (y, z) geometry — from `.scd` or user-supplied? | Open. Phase 3 default: user-supplied via a `FiberSection` data object. Read-from-`.scd` is a stretch goal. |
| 3 | Moment-diagram sign convention — "tension side" or "+y_local"? | Open. Default proposed: tension side, with `flip=True` toggle. |
| 4 | Contour default — per-element flat or node-averaged smooth? | Open. Default proposed: per-element flat (faithful, fast), smooth opt-in. |
| 5 | `Part` scope — per-scene or persist across scenes? | Open. Default proposed: per-scene first, promote later if needed. |

Each "default proposed" can be revisited before its phase ships — none
block Phase 0 or Phase 1.

---

## What this plan does **not** commit to

- Replacing or deprecating the existing matplotlib renderers. They stay,
  and they get rewired through `Scene`/`Layer` so the public API is
  unchanged.
- VTK / XDMF export. Mentioned in the directive as "lives in `io/` if it
  ever lands" — still not in scope.
- 3D arrow rendering inside matplotlib. Use PyVista for that.
- Edit-the-model UI. Pure post-processing; viewer never writes back.
- A separate viewer process / server architecture for the Qt app. We
  embed PyVista in Qt; one process.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| PyVista/VTK version drift breaks rendering | Medium | Pin minor versions in extras; CI matrix covers `pyvista` 0.43 + 0.44. |
| Qt + matplotlib interactive coexistence (event loop conflicts) | Medium | Keep Qt scenes and mpl scenes in separate windows; never mix in one figure. |
| Apple Silicon / VTK GL issues | Low | Bench Phase 3 on macOS arm64 before declaring Phase 3 done. |
| Layer model can't accommodate v1.8.0 section cuts cleanly | Medium | Phase 3 `clipping.py` plugs into existing `cuts/` rather than reimplementing; audit overlap during Phase 2. |
| Saved session files break across minor releases | High if unchecked | `_version.py` pins schema; session loader rejects unknown majors and warns on minors. |
| apeGmsh internals we lifted shift under us | Low | We're forking the code, not depending on the package. Track upstream changes manually if they matter. |

---

## Sequencing

```
Phase 0 ── Phase 1 ── Phase 2 ── Phase 3 ── Phase 4 ── Phase 5
                                   │
                                   └── (Phase 6, optional)
```

Phase 1 and Phase 2 can overlap (different files). Phase 3 strictly
depends on Phase 2 (`Scene`/`Layer` must exist). Phase 4 depends on
Phase 3. Phase 5 depends on Phase 3 but **not** on Phase 4.

Total estimate to "full experience" (Phase 4 done): **~10–12 weeks** of
focused work. Headless CLI (Phase 5) adds ~1.5 weeks. Trame (Phase 6)
adds ~3–4 weeks beyond that.
