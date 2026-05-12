# Viewer Architecture

**Anchor:** `viewer-refactor-directives.md` fixes the *vocabulary* (Scene,
Layer, Backend, layer catalog). This doc fixes the *implementation* — the
concrete protocols, the data flow, the module map, and how the same code
runs in three deployment targets.

---

## 1. Renderer-agnostic from day one

Three deployment targets, one Scene/Layer graph, three backends:

| Target | What it is | When | Backend |
|---|---|---|---|
| **Local Qt desktop** | Full GUI, embedded PyVista viewport, dock layout, time scrubber, picking. | Default for the engineer at their workstation. | `PyVistaBackend(qt_widget=...)` |
| **Headless / batch** | CLI that produces PNGs, MP4s, GIFs from a saved scene spec. No window, no Qt loop. | Cluster jobs, CI snapshots, automated reports. | `PyVistaBackend(off_screen=True)` |
| **Trame web** *(deferred)* | Server renders, browser displays. One SSH tunnel, one port. | SSH-only clusters; collaborative review. | `PyVistaBackend(trame=True)` or dedicated `TrameBackend` |
| *(unchanged)* **Matplotlib 2D** | Today's `Plot.*` / `nr.plot.*` / `er.plot.*`. | Notebooks, lightweight 2D plots, reports. | `MplBackend` |

The contract: **a `Scene` doesn't know its backend**. A layer doesn't know
its backend. They emit primitives (`segments`, `points`, `polygons`,
`arrows`, `field`); the backend translates them. Same scene, four render
targets.

This is **not** what apeGmsh does — apeGmsh is PyVista-only by design.
Keeping the matplotlib leg is a STKO_to_python improvement: it lets the
existing 2D notebook workflow stay light while the new 3D pipeline goes
through a heavier optional dep stack.

---

## 2. Layered design

Five layers, top to bottom. Data flows down on initialization, up on render.

```
   +----------------------------------+
   |  Layer 5: User-facing API        |   MPCODataSet.viewer(),
   |                                  |   ds.scene(), Scene(), MultiScene(),
   |                                  |   stko-viewer CLI
   +----------------------------------+
                   |
                   v
   +----------------------------------+
   |  Layer 4: Scene + Layers         |   Scene, MeshLayer, ContourLayer,
   |  (backend-neutral)               |   DeformedMeshLayer, GaussLayer,
   |                                  |   DiagramLayer, FiberLayer, ...
   +----------------------------------+
                   |
                   v
   +----------------------------------+
   |  Layer 3: DataSource adapter     |   MPCODataSourceAdapter:
   |                                  |   MPCODataSet -> Slabs/grids/arrays
   +----------------------------------+
                   |
                   v
   +----------------------------------+
   |  Layer 2: Math + algorithms      |   gauss_extrapolation, shape_fns,
   |  (pure numpy)                    |   beam_frame, shell_frame, picking math
   +----------------------------------+
                   |
                   v
   +----------------------------------+
   |  Layer 1: Backends               |   MplBackend, PyVistaBackend,
   |                                  |   (later) TrameBackend
   +----------------------------------+
```

Each upper layer depends only on its immediate neighbor below. No layer
reaches across.

---

## 3. The `Backend` protocol

Surface every layer in the catalog can express, written so the matplotlib
backend can either render or `raise BackendCapabilityError`:

```python
class Backend(Protocol):
    name: str  # "mpl" | "pyvista" | "trame"
    is_3d_capable: bool
    is_interactive: bool

    # Scene lifecycle
    def make_scene(self, *, is_3d: bool, view: str = "auto",
                   off_screen: bool = False) -> SceneHandle: ...
    def set_bounds(self, scene: SceneHandle, bbox: BBox) -> None: ...
    def set_camera(self, scene: SceneHandle, cam: CameraSpec) -> None: ...
    def set_style(self, scene: SceneHandle, style: SceneStyle) -> None: ...

    # Primitives — every layer is built from these
    def add_segments(self, scene, segs, *, color, width, alpha, label) -> ActorRef: ...
    def add_points(self, scene, pts, *, color, size, scalars=None, cmap=None) -> ActorRef: ...
    def add_polygons(self, scene, polys, *, values=None, cmap=None,
                     edge_color=None) -> ActorRef: ...
    def add_arrows(self, scene, origins, vectors, *, scale, color, cmap) -> ActorRef: ...
    def add_glyphs(self, scene, pts, glyph_kind, *, color, size, scale_by=None) -> ActorRef: ...
    def add_field(self, scene, grid, *, scalars, cmap, clim, mode) -> ActorRef:
        # mode: "points" | "cells" | "smooth"
        ...

    # 3D-only — may raise BackendCapabilityError on MplBackend
    def add_clipped(self, scene, actor: ActorRef, plane: Plane) -> None: ...
    def add_slice(self, scene, grid, plane: Plane, field: str) -> ActorRef: ...
    def add_iso(self, scene, grid, field: str, level: float) -> ActorRef: ...

    # In-place updates (for animation; no actor recreation)
    def update_scalars(self, actor: ActorRef, scalars: np.ndarray) -> None: ...
    def update_points(self, actor: ActorRef, pts: np.ndarray) -> None: ...
    def set_visible(self, actor: ActorRef, visible: bool) -> None: ...
    def remove(self, scene: SceneHandle, actor: ActorRef) -> None: ...

    # Output
    def show(self, scene: SceneHandle) -> None: ...                 # interactive
    def save(self, scene: SceneHandle, path: Path, *, dpi=300) -> None: ...
    def snapshot(self, scene: SceneHandle) -> np.ndarray: ...        # H×W×3 uint8
```

**Hard rules:**

- `update_*` methods must mutate in place and re-render in O(modified
  data), not O(scene). This is the apeGmsh perf contract from
  `_contour.py:32-41` — porting it is non-negotiable for animation
  performance.
- Layers never import from a backend module. They get a `Backend`
  instance through the scene.
- A backend that can't do something raises `BackendCapabilityError`
  with a precise message ("VolumeLayer mode='iso' requires the pyvista
  backend"). No silent fallback.

---

## 4. `Scene` and `Layer`

```python
class Scene:
    backend: Backend
    style: SceneStyle
    layers: list[Layer]
    current_step: int | None
    current_stage: str | None

    def add(self, layer: Layer) -> Layer: ...
    def remove(self, layer: Layer) -> None: ...
    def set_step(self, step: int) -> None:
        """Fires layer.update_to_step on every layer in order."""
    def show(self) -> None: ...
    def save(self, path: Path) -> None: ...
    def to_spec(self) -> SceneSpec: ...
    @classmethod
    def from_spec(cls, spec: SceneSpec, ds: MPCODataSet) -> "Scene": ...


class Layer(ABC):
    name: str
    visible: bool
    z_order: int
    selection: SelectionSpec
    style: LayerStyle

    @abstractmethod
    def attach(self, scene: Scene, source: DataSource) -> None:
        """Build VTK/mpl actors. Called once."""

    @abstractmethod
    def update_to_step(self, step: int) -> None:
        """In-place mutate scalars/points for animation. No recreation."""

    @abstractmethod
    def detach(self) -> None: ...

    def to_spec(self) -> LayerSpec: ...
```

**Key contracts (lifted from apeGmsh, adapted):**

1. `attach` is the only place a layer may allocate VTK actors or mpl
   artists. After attach, the actor set is frozen.
2. `update_to_step` reads from the `DataSource`, scatters into pre-allocated
   scalar arrays, and calls `backend.update_scalars` (or
   `update_points` for deformed shape). It does **not** call
   `add_*`.
3. `detach` removes actors and releases references.
4. `to_spec` / `from_spec` make every layer round-trippable through a
   TOML/JSON config so the GUI can save → run on cluster.

---

## 5. The `DataSource` adapter

The chokepoint between STKO's DataFrame world and the apeGmsh-style
Slab-and-array world that the layers expect.

```python
class DataSource(Protocol):
    @property
    def dataset(self) -> MPCODataSet: ...

    # Geometry
    def grid(self, *, selection: SelectionSpec | None = None) -> Grid:
        """UnstructuredGrid-shaped: nodes, connectivities by type, ids."""

    def node_coords(self, ids: np.ndarray | None = None) -> np.ndarray: ...
    def element_centroids(self, ids: np.ndarray | None = None) -> np.ndarray: ...
    def model_bbox(self) -> BBox: ...

    # Result fields — per step, per topology
    def nodal_values(self, name: str, component: str | int,
                     step: int, stage: str | None = None,
                     selection: SelectionSpec | None = None
                     ) -> np.ndarray: ...

    def element_values(self, name: str, component: str,
                       step: int, stage: str | None = None,
                       selection: SelectionSpec | None = None,
                       topology: str = "element"  # "element" | "gp" | "fiber" | "layer"
                       ) -> np.ndarray: ...

    # Time
    def n_steps(self, stage: str | None = None) -> int: ...
    def time(self, stage: str | None = None) -> np.ndarray: ...

    # Selection resolution
    def resolve_selection(self, spec: SelectionSpec) -> ResolvedSelection: ...
```

The default implementation `MPCODataSourceAdapter` wraps an existing
`MPCODataSet` and delegates to:

- `ds.nodes` for node geometry,
- `ds.elements` for connectivity and element results,
- `ds.cuts` for clipping primitives (reuses Phase v1.8.0 work),
- `ds.selection_set_resolver` for named groups.

This is the **one** place STKO's pandas DataFrames are converted to
contiguous numpy arrays shaped how the layers want. Doing it here means
layers stay backend-agnostic *and* data-source-agnostic — a future
non-MPCO source (e.g., direct OpenSeesPy stream) just implements
`DataSource`.

---

## 6. The Director (step orchestration)

```python
class Director:
    scene: Scene
    n_steps: int
    current_step: int

    def play(self, *, fps: int = 30, step_stride: int = 1) -> None: ...
    def pause(self) -> None: ...
    def set_step(self, step: int) -> None: ...
    def export_animation(self, path: Path, *, fps: int = 30,
                         step_stride: int = 1, format: str = "mp4") -> None: ...

    # Signals (Qt-style; matplotlib backend uses callbacks)
    on_step_changed: Signal[int]
    on_layer_added: Signal[Layer]
    on_layer_removed: Signal[Layer]
```

The Director is what the time scrubber drives in Qt and what the headless
CLI iterates programmatically. **Same class either way.**

The apeGmsh Director ships with TimeMode = SINGLE | RANGE | ENVELOPE |
ANIMATION (only SINGLE in their Phase 0). Our Phase 4 ships SINGLE only;
RANGE/ENVELOPE are deferred.

---

## 7. How each deployment target wires up

### 7.1 Local Qt desktop

```
QApplication
  └── ResultsWindow (QMainWindow)
        ├── OutlineTree (left dock)
        ├── ViewportWidget (center)   ── pyvistaqt.QtInteractor
        │     │                            │
        │     │                            └── PyVistaBackend(qt_widget=interactor)
        │     │
        │     └── Scene  ── Director ── Layer x N
        ├── DetailsPanel (right dock)
        ├── TimeScrubber (bottom dock)
        └── PickReadoutHud (overlay)
```

Single process. One scene per main window. Picking and time scrubber
talk to the Director. Director calls `scene.set_step`, which fans out to
each layer's `update_to_step`, which mutates VTK arrays in place. Qt
event loop is the master clock.

### 7.2 Headless / batch

```
stko-viewer animate run.mpco --config view.toml --out anim.mp4
        │
        └── load view.toml -> SceneSpec
            scene = Scene.from_spec(spec, ds)
            scene.backend = PyVistaBackend(off_screen=True)
            director = Director(scene)
            director.export_animation(out, fps=..., step_stride=...)
```

No Qt. No window. Backend renders to an off-screen framebuffer (Xvfb on
Linux, EGL where available, native off-screen on macOS/Windows).
`export_animation` walks steps, snapshots each, hands the frame stream
to `imageio-ffmpeg`.

### 7.3 Trame web *(Phase 6)*

```
trame_app
  └── VtkRemoteView    ── pyvista.trame.PyVistaRemoteView
        │                       │
        │                       └── PyVistaBackend(plotter=...)
        │
        └── Scene ── Director ── Layer x N
```

Trame's reactive state binds UI controls (sliders, layer toggles) to
the Director and to layer properties. Browser opens
`http://cluster:8080`; server stream-encodes frames.

The key payoff of the layering: **only the `Backend` instantiation
changes** between targets. Scene/Layer/Director/DataSource are
byte-identical.

---

## 8. Proposed module map

```
src/STKO_to_python/viewer/
├── __init__.py              # exports Scene, Layer, etc. — lazy
├── _version.py              # schema versions for SceneSpec
├── api.py                   # high-level entry: MPCODataSet.viewer()
│
├── core/
│   ├── backend.py           # Backend protocol, BackendCapabilityError
│   ├── scene.py             # Scene, MultiScene
│   ├── layer.py             # Layer base class
│   ├── style.py             # SceneStyle, LayerStyle (extends PlotSettings)
│   ├── datasource.py        # DataSource protocol + MPCODataSourceAdapter
│   ├── director.py          # Director + signals
│   ├── selection.py         # SelectionSpec dataclass; merges with SelectionSetResolver
│   ├── specs.py             # SceneSpec, LayerSpec — TOML/JSON round-trip
│   └── errors.py            # BackendCapabilityError, LayerAttachError
│
├── math/                    # Phase 1 (pure numpy)
│   ├── gauss_extrapolation.py
│   ├── beam_frame.py
│   ├── shell_frame.py
│   └── picking.py           # box-pick projection math
│
├── backends/
│   ├── mpl/
│   │   ├── __init__.py
│   │   ├── backend.py       # MplBackend
│   │   ├── segments.py
│   │   ├── points.py
│   │   ├── polygons.py
│   │   └── arrows.py
│   ├── pyvista/
│   │   ├── __init__.py
│   │   ├── backend.py       # PyVistaBackend
│   │   ├── primitives.py
│   │   └── offscreen.py
│   └── trame/               # Phase 6
│       └── ...
│
├── scene_3d/
│   └── fem_scene.py         # MPCODataSet -> pyvista.UnstructuredGrid
│
├── layers/
│   ├── mesh.py              # MeshLayer
│   ├── deformed_mesh.py     # DeformedMeshLayer
│   ├── node.py              # NodeLayer + VectorLayer
│   ├── contour.py           # ContourLayer (5-path dispatch)
│   ├── gauss.py             # GaussLayer
│   ├── diagram.py           # DiagramLayer (line elements)
│   ├── fiber.py             # FiberLayer
│   ├── layer_stack.py       # LayerStackLayer (shell through-thickness)
│   ├── volume.py            # VolumeLayer (points/slice/iso)
│   ├── zerolength.py        # ZeroLengthLayer
│   ├── clipping.py          # interactive clipping; plugs into cuts/
│   └── xy.py                # XYLayer (2D time history)
│
├── recipes/                 # high-level helpers (Phase 4+)
│   ├── pushover.py
│   ├── drift_profile.py
│   ├── mode_shape.py
│   └── section_response.py
│
├── qt/                      # Phase 4
│   ├── app.py
│   ├── main_window.py
│   ├── cli.py               # stko-viewer entry point
│   ├── widgets/
│   │   ├── outline_tree.py
│   │   ├── viewport.py
│   │   ├── details_panel.py
│   │   ├── time_scrubber.py
│   │   ├── add_layer_dialog.py
│   │   └── pick_readout_hud.py
│   ├── controllers/
│   │   ├── picking.py
│   │   └── session.py
│   └── theme.py
│
├── headless/                # Phase 5
│   ├── cli.py               # stko-viewer animate/screenshot/batch
│   └── runner.py
│
└── web/                     # Phase 6 (deferred)
    └── ...
```

---

## 9. What this architecture buys

1. **One scene graph runs in four targets.** Mpl notebook, off-screen
   batch, Qt desktop, web (eventually). Same layer code, different
   backend instance.
2. **Existing API stays.** `ds.plot.*` rewires onto Scene/MplBackend.
   Cookbook recipes don't break.
3. **Section cuts already in v1.8.0 plug in cleanly.** `ClippingLayer`
   wraps `cuts.Plane` + `cuts.SectionCut`; no duplication.
4. **Animation export and the desktop time scrubber share one Director.**
   Saving a scene from the GUI and running it on a cluster produces the
   same frames.
5. **The matplotlib leg stays usable in notebooks.** Users who don't
   want Qt/VTK never install it.

---

## 10. Non-goals

- A custom rendering engine. We use VTK via PyVista for all 3D.
- A web framework other than Trame. (Trame is Kitware's own — same
  shop as VTK; lowest impedance match.)
- Edit-the-model UI. Post-processing only.
- Cross-process viewer (server + remote client over IPC). We have
  in-process Qt and we have HTTP/Trame; we don't need a third pattern.
- Replacing the matplotlib renderers. They stay.

---

## 11. Coupling boundaries (don't cross these)

| From → To | Allowed? |
|---|---|
| `layers/*` → `backends/*` | **No** (use `Scene.backend` indirection) |
| `layers/*` → `qt/*` | **No** |
| `qt/*` → `layers/*` | **Yes** (UI configures layers) |
| `qt/*` → `backends/*` | **Yes**, but only `pyvista` (Qt embeds PyVista) |
| `core/*` → `backends/*` | **No** |
| `headless/*` → `qt/*` | **No** (headless must run without Qt installed) |
| `web/*` → `qt/*` | **No** |
| `cuts/*` → `viewer/*` | **No** (viewer wraps cuts, not the other way) |

`pyproject.toml` enforces some of these via optional extras (no Qt
import survives `pip install stko_to_python[viewer-headless]` without
the user also installing `[viewer]`). The rest are enforced by review.
