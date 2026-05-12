# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/) as
spelled out in [`CLAUDE.md`](CLAUDE.md#versioning-policy):

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features.
- **PATCH** (`v1.x.Y`) — bug fixes, docs, internal refactors with no API change.

## [Unreleased]

## [1.8.0] — 2026-05-12

Extends v1.7 with **higher-order solid section cuts** (20- and 27-node
hexahedra) and the **per-fiber breakdown for fibered layers** in
layered shells. Both are additive — the v1.7 surface (beam + shell +
solid composition, per-layer shell view) is unchanged; v1.8 adds new
element classes the solid kernel accepts and a new
`SectionCut.per_fiber_force` accessor.

### Added

- **Higher-order solid section cuts** — `Brick20`, `Brick27`, and the
  OpenSees aliases `TwentyNodeBrick` / `TwentySevenNodeBrick` join the
  solid kernel's registry. The geometry phase uses the first 8 nodes
  (corners) for the plane-vs-polyhedron polygon math — sound because
  the corners define the element's convex hull and the cut polygon
  math is on convex polyhedra. Stress sampling dispatches on IP count:
  8-pt 2×2×2 stays trilinear; 27-pt 3×3×3 uses **triquadratic
  Lagrange** interpolation over the tensor-product grid. Curvature
  induced by midpoint / face / centre nodes is ignored at the geometry
  layer — adequate for any well-conditioned higher-order hex.
- **27-IP triquadratic sampler** — `_brick_27ip_weights(ξ, η, ζ)`
  returns the 27 tensor-product Lagrange weights at the 3×3×3
  Gauss-Legendre IPs (1-D nodes at `±√(3/5)` and 0). Partition of
  unity, unit response at each IP, and exact reproduction of any
  separable triquadratic polynomial are pinned by tests.
- **Per-fiber breakdown for fibered layers** —
  `SectionCut.per_fiber_force(layer_idx, fiber_idx, dataset)` returns
  a derivative cut from one fiber inside one through-thickness layer
  of the layered shells in the cut. Required when a layer is itself a
  Fiber section; the MPCO recorder then writes columns named
  `<comp>_f<F>_l<L>_ip<K>`. The fiber's tributary thickness defaults
  to `t_layer / n_fibers_in_layer` (uniform distribution within the
  layer); the fiber's z-offset is the centroid of that sub-band.
  Summing all fibers in a layer recovers the per-layer cut for that
  layer.
- **`ds.section_cut(..., per_layer=k, per_fiber=f)`** — both shortcuts
  resolved together: with only `per_layer` the dataset returns the
  per-layer view; with both, it returns the per-fiber-in-layer view.
  `per_fiber` without `per_layer` raises — fibers are indexed within a
  single layer.

### Added (internal)

- `STKO_to_python.cuts.kernels.solid._brick_27ip_1d_lagrange` and
  `_brick_27ip_weights` — pure-math triquadratic Lagrange basis at the
  3-pt Gauss-Legendre nodes and the resulting 27-IP tensor-product
  weights. Both are independently tested.
- `_CORNER_NODES_PER_CLASS` — fixed mapping from element class to the
  number of corner nodes the geometry phase consumes (8 for any hex
  variant; 4 for the tet). Higher-order classes carry richer
  connectivity but the geometry sub-frame is always 8 corners.
- `STKO_to_python.cuts.kernels.shell._fiber_in_layer_column_candidates`,
  `_read_fiber_in_layer_stress_array`, and
  `_discover_fiber_count_in_layer` — read `<comp>_f<F>_l<L>_ip<K>`
  columns under both naming conventions (`sigma11_f<F>_l<L>_ip<K>` and
  the `nDMaterial`-fallback `UnknownStress(n)_f<F>_l<L>_ip<K>`). The
  fiber count is discovered from the available columns rather than
  hard-coded — the same kernel works for sections with arbitrary
  fiber counts per layer.
- `STKO_to_python.cuts.kernels.shell.compute_shell_cut_per_fiber` —
  parallel to `compute_shell_cut_per_layer`. Uses the uniform fiber
  distribution to compute the fiber's tributary `LayerInfo` (subset
  of the layer's thickness centred at the fiber's z-band centroid)
  and delegates to `_shell_cut_per_element_for_layer`.

### Test coverage

- `tests/unit/cuts/test_solid_kernel.py` gains 8 new tests covering
  the higher-order machinery: registry contents (Brick20 / Brick27 /
  OpenSees aliases), node-count mapping (20 vs 27 connectivity, 8
  corners for the geometry phase), 1-D Lagrange basis (unit response,
  partition of unity), 27-IP weights (partition of unity at origin,
  unit response at each of the 27 IPs, exact reproduction of a
  separable triquadratic), dispatch into the 27-IP sampler from
  `_sample_solid_stress`, and a corner-only plane-vs-polyhedron test
  on a synthetic 20-node hex.
- `tests/unit/cuts/test_per_layer_shell.py` gains 8 new tests covering
  the fiber-in-layer reader (`sigma11_f<F>_l<L>_ip<K>` and
  `UnknownStress(n)_f<F>_l<L>_ip<K>` variants, missing-layer fallback
  to zero, fiber count discovery, the `per_fiber` inline-form
  argument contract, and the "non-fibered layer + `per_fiber_force`"
  error path).

### Out of scope (v1.9 candidates)

- Curved-edge geometry on higher-order hexes — `Brick20` / `Brick27`
  midpoints define an actual curved face on the element interior,
  which the current corner-only polygon math ignores. For
  well-conditioned models this is an O(midpoint-deviation²) error
  that doesn't affect resultants meaningfully; for badly-shaped
  higher-order elements it can.
- Explicit fiber positions for fibered layers — the v1.8 kernel
  assumes uniform distribution within a layer. Real fiber positions
  may be specified in the OpenSees model and would arrive via the
  `.cdata` or `sections.tcl`.
- 10-node tetrahedra (`TenNodeTetrahedron` and friends), wedge, and
  pyramid solids — not in the static catalog.

## [1.7.0] — 2026-05-12

Adds the **solid (continuum) section-cut kernel** and the **per-layer
breakdown for layered shells**. Both are additive — no breaking changes
to the public surface. Section cuts now compose contributions from
beams, shells, and solids in a single `SectionCut.compute` call; the
per-layer view exposes the through-thickness slice of a layered-shell
cut.

### Added

- **Solid section-cut kernel** — `compute_solid_cut` integrates
  `material.stress` (six Voigt components per IP — `σ11, σ22, σ33,
  σ12, σ23, σ13` in global frame) over the planar polygon clipped from
  each crossing continuum element. Plane-vs-polyhedron geometry walks
  the element's edges (12 for an 8-node hex, 6 for a 4-node tet),
  collects crossings + on-plane vertices, dedupes, and sorts CCW around
  the plane normal. The polygon is fan-triangulated; each triangle
  gets a 3-point Gauss rule. Trilinear inversion at each quadrature
  point pulls (ξ, η, ζ) for stress sampling; the traction
  `t = σ · n_cut` integrates to the resultant force in global frame.
  Supports `Brick`, `BbarBrick`, `SSPbrick` (all 8-node hexes) and
  `FourNodeTetrahedron`.
- **Side-aware shared-face resolution for solids** — when a cut plane
  lands on the shared face between two adjacent solids, the same
  side-aware filter used in the shell kernel skips elements whose
  interior is entirely on the kept side; avoids double-counts at mesh
  boundaries.
- **`SectionCut.compute` composition over three kernels** — beam +
  shell + solid in one pass, sharing a `PolygonClipper` so the plane
  basis isn't recomputed. New `solid_intersections`, `per_solid_F`,
  `per_solid_M_at_centroid` fields on `SectionCut`;
  `contributing_element_ids` walks all three kernels.
- **Per-layer breakdown for layered shells** —
  `SectionCut.per_layer_force(layer_idx, dataset)` returns a
  derivative cut from only one through-thickness layer of the
  contributing layered shells. The math replaces the through-thickness
  integrated `section.force` with the per-layer contribution
  `(σ_11^(k) · t_k, σ_22^(k) · t_k, σ_12^(k) · t_k, σ_11^(k) · t_k ·
  z_k, ...)` and reuses the same chord integration + rotation. Summing
  every layer's contribution recovers the standard cut to numerical
  tolerance (verified on Test_NLShell, 7-layer wall sections 15/16).
  `dataset.section_cut(..., per_layer=k)` short-circuits to the
  per-layer view.
- **`MPCODataSet.layered_sections`** — lazy property that locates and
  parses `sections.tcl` beside the recorder output. Returns
  `{section_id: tuple[LayerInfo, ...]}` where each `LayerInfo` carries
  `material_id`, `thickness`, and `z_offset` (signed distance from the
  section midplane). Empty dict when no script is found — the
  per-layer surface raises a clear error in that case rather than
  silently zeroing.
- **`STKO_to_python.model.layered_section_reader`** — new module with
  `LayerInfo` dataclass, `parse_sections_tcl(path)` parser, and
  `find_sections_tcl(directory)` locator. Handles Tcl line
  continuations, multiple section blocks, comments, and the
  per-section sanity check that the body has 2·n tokens for n declared
  layers.

### Added (internal)

- `STKO_to_python.cuts.kernels.solid` — `SOLID_ELEMENT_CLASSES`
  registry, `SolidIntersection` record carrying both the planar
  polygon in global coords and its natural-coord image,
  `find_solid_intersections`, `compute_solid_cut`, `SolidCutResult`.
  Helpers cover plane-vs-polyhedron polygon math, hex trilinear /
  tet linear inverse maps, Voigt-to-tensor stress expansion, and a
  Sutherland-Hodgman polygon-vs-polygon clip used when a
  `bounding_polygon` is set.
- `STKO_to_python.cuts.kernels.shell.compute_shell_cut_per_layer` —
  parallel to `compute_shell_cut`. The standard `section.force`
  8-vector is built from a single layer's stress and geometric weights
  rather than read from the recorder. Recognises three column-naming
  conventions for `section.fiber.stress`: explicit
  `sigma11_l<L>_ip<K>`, explicit `sigma11_f<L>_ip<K>`, and the
  `nDMaterial`-fallback `UnknownStress(n)_f<L>_ip<K>` mapped to
  `(σ11, σ22, σ12, σ13, σ23)` per the PlateFiber convention.

### Test coverage

- `tests/unit/cuts/test_solid_kernel.py` — 33 tests split into
  pure-math and real-fixture groups. Pure-math (23 tests) covers
  the trilinear IP weights (partition of unity, unit response at each
  IP, linear-field reproduction), the Voigt-to-tensor mapping
  (symmetry, traction against unit normal), the brick / tet inverse
  shape-function maps, and the plane-vs-polyhedron polygon math
  (horizontal cut → square, diagonal cut → hexagon, face-aligned cut
  → 4 on-plane vertices, plane missing the polyhedron → None, tet
  cut → triangle). Real-fixture tests (10 tests, gated on
  `solid_partition_example`) verify the geometry phase, Newton's-3rd-
  law consistency, side-flip equivalence, the mixed beam + solid
  composition through `SectionCut.compute`, and that `per_solid_F`
  actually pulls real stress data.
- `tests/unit/cuts/test_per_layer_shell.py` — 21 tests covering
  the layered Voigt stress reader (single layer, multi-layer
  disambiguation, multi-IP), the `_sample_layer_stress` IP-weight
  dispatch on Q4/T3, and the `sections.tcl` parser (parses
  Test_NLShell's two `LayeredShell` blocks, layer thicknesses,
  symmetric z_offsets, missing-file error, truncated-body error,
  comment skipping). Real-fixture tests against `Test_NLShell`
  validate the dataset's `layered_sections` accessor, the per-layer
  cut's shape and contents, the **sum-of-layers = full-cut**
  identity (Σ_k F^(k) ≈ cut.F to ~1% of the cut magnitude across the
  step axis), error paths for out-of-range layer indices and
  shell-less cuts, and the `dataset.section_cut(..., per_layer=k)`
  inline form.

### Out of scope (v1.8 candidates)

- Higher-order solids (Brick20, Brick27 connectivity beyond 8-node
  hex). The 27-IP catalog entry already exists for 2×2×2 sampling on
  a Brick8; full Brick27 connectivity is v1.8.
- Per-fiber stress on layered shells (one level deeper than per-
  layer). The MPCO format already supports it (the column-name parser
  handles `_f<F>_l<L>_ip<K>` suffixes); the surface to expose it is
  v1.8.
- Non-convex intersection polygons. The current solid kernel assumes
  convexity, which all standard FE element shapes satisfy.
- Wedge / pyramid solids — not in the static catalog. Treat as
  follow-up.

## [1.6.0] — 2026-05-12

Extends the v1.5.0 section-cut subpackage with the **shell kernel**
and an optional **bounding polygon** on the cut plane. Section cuts
through models that mix beams and shells now produce a single
combined resultant; cuts through analyses where the recorded
selection sets don't pre-filter to a structural sub-region can be
narrowed via a convex polygon on the cut plane.

### Added

- **Shell section-cut kernel** — `compute_shell_cut` integrates
  `section.force` (8 components per IP — `Fxx, Fyy, Fxy, Mxx, Myy,
  Mxy, Vxz, Vyz`) along the chord at which the cut plane crosses
  each shell midsurface. 2-point Gauss-Legendre line quadrature
  along the chord, bilinear sampling between IPs for `ASDShellQ4`,
  linear sampling for `ASDShellT3`. Rotates element-local traction
  to global via `cdata.rotation_matrix(eid)`. Layered-shell
  variants are transparent (`section.force` is already through-
  thickness-integrated regardless of layer count).
- **Shared-edge resolution** — when a cut plane lands exactly on
  the shared edge between two adjacent shells, `find_shell_intersections`
  applies a side-aware geometric filter: only the shell whose interior
  lies on the **discarded** side contributes. Avoids the double-count
  that the naive geometry pipeline produced on cuts at mesh-row
  elevations (verified on Test_NLShell's z=870 T3/Q4 interface).
- **Bounding polygon on the cut plane** — `SectionCutSpec` gains an
  optional `bounding_polygon=` field, a convex polygon on the cut
  plane restricting the cut to elements whose intersection falls
  inside it. Validation at construction rejects off-plane vertices,
  degenerate polygons, fewer than three vertices, and non-convex
  shapes. Threaded through `MPCODataSet.section_cut(...,
  bounding_polygon=...)`.
- **`SectionCut.compute` composition** — runs the beam and shell
  kernels in one pass, shares a `PolygonClipper` so the plane basis
  isn't recomputed, and aggregates `(F, M)` about a mixed centroid
  (mean of beam intersection points and shell chord midpoints). New
  `shell_intersections`, `per_shell_F`, `per_shell_M_at_midpoint`
  fields on the result; `contributing_element_ids` walks both
  kernels.

### Added (internal)

- `STKO_to_python.cuts.geometry` — plane-basis projection,
  shoelace-signed-area, inward-edge-normal computation, Cyrus-Beck
  segment clipping, point-in-polygon test, and `PolygonClipper` (a
  pre-built struct of plane basis + 2-D polygon for reuse across
  the kernels). All testable without an `.mpco` fixture.
- `STKO_to_python.cuts.kernels.shell` — `SHELL_ELEMENT_CLASSES`
  registry, `ShellIntersection` record, `find_shell_intersections`,
  `compute_shell_cut`, `ShellCutResult`. Internal helpers cover
  bilinear quad-IP / linear tri-IP sampling, inverse shape-function
  maps for ASDShellQ4 (Newton on a planar 2-D embedding) and
  ASDShellT3, midsurface-normal computation, and cut-normal
  orientation that resolves both normal cuts and edge-coincident
  cuts.

### Test coverage

- `tests/unit/cuts/test_geometry.py` — 30 tests for plane-basis
  orthonormality, batch projection, signed-area and convexity
  predicates, inward-edge-normal orientation (CCW + CW), point-in-
  polygon with edge / vertex / exterior cases, and Cyrus-Beck
  clipping covering segment-through-polygon, fully-inside, fully-
  outside, starting-inside, parallel-to-edge, and grazing-edge
  geometries. End-to-end `PolygonClipper` smoke on horizontal and
  oblique planes.
- `tests/unit/cuts/test_shell_kernel.py` — 29 tests split into
  pure-math (bilinear/linear sampling weight identities, inverse
  shape-function helpers, `_sample_shell_section_force` recovers
  IP values at the IP coords) and real-fixture
  (`Test_NLShell`) groups. Real-fixture coverage exercises the
  registry, the geometry phase, end-to-end consistency (`Newton's
  3rd law` residual = 0 at z=2500 across all three model stages),
  side-flip equivalence, and the shared-edge side-aware filter at
  z=870 (positive matches just-below T3, negative matches just-
  above Q4, no double-count).
- `tests/unit/cuts/test_bounding_polygon.py` — 10 integration
  tests: half-plate beam cut returns half the gravity force
  (5000 → 2500 against the `elasticFrame_mesh_displacementBased`
  fixture), shell chord clipping at the polygon boundary on
  `Test_NLShell`, empty-polygon edge cases, the `ds.section_cut(...,
  bounding_polygon=...)` inline form, and the rejection rule that
  bans mixing `spec=` with inline kwargs.
- `tests/unit/cuts/test_specs.py` gains 11 tests covering the
  `bounding_polygon` validation contract (≥3 vertices, on-plane
  within tol, non-degenerate area, convex), `ndarray` coercion,
  CW polygon acceptance, hash / equality semantics, and pickle
  round-trip.

### Documentation

- New cookbook recipe
  [10 — Section cuts through frames with shells](docs/cookbook/10-section-cut-shells.md)
  walks through the v1.6.0 surface end-to-end: single-elevation cut,
  `consistency_check` + `compare_to` validators, a `SectionSweep` for
  story-shear-vs-elevation profiles, a `bounding_polygon` clip
  restricting the cut to the left half of the wall, spec-driven
  reuse + pickle, and the shared-edge behavior with a note on
  consistency at load-discontinuity elevations.
- New API reference page
  [Section cuts API](docs/api/section-cuts.md) covers the public
  surface (`Plane`, `SectionCutSpec`, `SectionCut`, `SectionSweep`,
  `MultiCutResult`, `DriftSpec`), the dataset entry points, the
  v1.6.0 additions (shell kernel, shared-edge resolution, bounding
  polygon), and pulls full mkdocstrings autodoc per class. Wired
  into the nav under "API reference → Section cuts".
- `docs/MPCODataSet.md` gains a "Resource management" section showing
  the `with MPCODataSet(...) as ds:` pattern, when it's worth reaching
  for, and how `ds.clear_result_caches()` fits as a finer-grained
  alternative. The context-manager feature already existed (since the
  partition-pool work in v1.4.x) but wasn't surfaced anywhere users
  would notice it.
- `examples/usage_tour.py` gains a new section 15 demonstrating the
  context-manager form end-to-end with cache-clearing observable
  through `_nodal_query_engine.cached_result_count` before and after
  `__exit__`. Runs against the elasticFrame fixture; no behavior
  change.
- `__enter__` / `__exit__` docstrings on `MPCODataSet` updated to
  describe what they do today (close pool + drop engine caches)
  instead of the historical "Phase 0 stub" placeholder. Class-level
  context-manager-support paragraph rewritten accordingly.

### Added (internal — bench)

- `bench/test_construction_bench.py` — three new pytest-benchmark
  cases isolating `MPCODataSet.__init__` from the first fetch:
  single-partition construction, multi-partition construction (via
  the QuadFrame fixture), and first `selection_set` access (which
  triggers the lazy .cdata parse). Fills the spec §6 Phase 5 gap
  where dataset construction was previously bundled into
  `test_bench_fetch_cold`.
- `bench/README.md` documents the v1.5.0 baseline numbers and a
  table mapping each bench to the spec §6 target it sits under. The
  per-partition incremental cost (~3 ms) is now a recorded data
  point so future refactors that change construction can be
  evaluated against a known reference.

### Out of scope (v2.0 candidates)

- Solid (continuum) section-cut kernel — surface integration over
  the polygon clipped from each crossing element's volume.
- Non-convex `bounding_polygon` — would need general clipping
  (Sutherland-Hodgman) instead of the convex-only Cyrus-Beck.
- Bounding half-spaces (combine multiple planes) as an alternative
  to a polygon.
- Per-layer / per-fiber breakdown of shell cuts via
  `material.fiber.stress`.

---

## [1.5.0] — 2026-05-12

Builds 3D beam visualization on top of the v1.3 / v1.4 cdata-sidecar
work: `ds.plot.beam_solids` and `ds.plot.beam_solids_deformed` extrude
beam elements as section solids, driven by the parsed
`*BEAM_PROFILE`, `*BEAM_PROFILE_ASSIGNMENT`, `*LOCAL_AXES`, and
`*SECTION_OFFSET` blocks. Cdata parsing also becomes lazy on the
dataset, so workflows that don't touch selection sets or beam viz
skip the parse entirely.

### Added

- **`ds.plot.beam_solids(...)`** — new method on the dataset plot
  facade that renders beam elements as 3D extruded section solids,
  using the cdata sidecar's `*BEAM_PROFILE`,
  `*BEAM_PROFILE_ASSIGNMENT`, `*LOCAL_AXES`, and `*SECTION_OFFSET`
  blocks. Builds one triangle batch per beam, accumulates a single
  `Poly3DCollection` for the fill, and overlays the section
  perimeters + sweep longitudinals as a `Line3DCollection` (interior
  triangulation edges intentionally suppressed). Accepts the same
  `element_ids` / `selection_set_id` / `selection_set_name` filter as
  the rest of the plot facade and silently filters out elements that
  aren't beams. Auto-sets the 3D box aspect from the data ranges so
  beams don't render squashed by matplotlib's default unit cube.
  Returns `(ax, meta)` with `element_count`, `triangle_count`,
  `skipped_elements`, and `profile_ids`
  ([#70](https://github.com/nmorabowen/STKO_to_python/pull/70)).
- **`ds.plot.beam_solids_deformed(model_stage, step, scale=1.0, ...)`** —
  deformed twin of `ds.plot.beam_solids`. Fetches `DISPLACEMENT` at the
  requested step, shifts every end node by `scale * disp`, and feeds
  the displaced coordinates through the same extrusion pipeline.
  `scale=0` collapses to the undeformed configuration without
  fetching displacements. `meta` carries `model_stage`, `step`, and
  `scale` on top of the standard keys. The cross-section's local
  frame is taken from the undeformed `*LOCAL_AXES` quaternion — STKO
  does not record a deformed local frame, so translations are exact
  but large rotations on slender members may show a slightly off
  section orientation (documented in the cookbook recipe)
  ([#71](https://github.com/nmorabowen/STKO_to_python/pull/71)).
- New cookbook recipe
  [09 — Render beams as 3D extruded solids](docs/cookbook/09-render-beam-solids.md)
  covers the default render, selection-set filtering, composition
  with the shell mesh, the low-level geometry-kernel escape hatch,
  and the deformed-configuration render with scale-selection
  guidance
  ([#70](https://github.com/nmorabowen/STKO_to_python/pull/70),
  [#71](https://github.com/nmorabowen/STKO_to_python/pull/71)).

### Added (internal)

- `STKO_to_python.plotting.beam_solid.extrude_beam_geometry(...)` —
  pure-numpy helper that sweeps a `BeamProfile` between two beam
  endpoints (with optional `*SECTION_OFFSET`) and returns
  `(vertices, faces)` as a 3D triangle mesh. Foundation for the
  `ds.plot.beam_solids` rendering wrapper; matplotlib-free so it can
  be unit-tested without a display backend
  ([#69](https://github.com/nmorabowen/STKO_to_python/pull/69)).

### Changed (internal)

- `MPCODataSet.selection_set` and `MPCODataSet._selection_resolver` are
  now lazy `@cached_property` attributes. Dataset construction no
  longer parses the `.cdata` selection-set blocks; the parse triggers
  on first access. Workflows that only fetch element results without
  passing `selection_set_*` arguments now skip the cdata parse
  entirely
  ([#68](https://github.com/nmorabowen/STKO_to_python/pull/68)).
- `NodalResultsQueryEngine` and `ElementResultsQueryEngine` no longer
  take `resolver=` in their constructor; they read
  `dataset._selection_resolver` at query time. Internal refactor —
  the public managers (`Nodes.get_nodal_results`,
  `Elements.get_element_results`) behave identically
  ([#68](https://github.com/nmorabowen/STKO_to_python/pull/68)).

### Test coverage

- Two regression guards under
  `tests/unit/selection/test_dataset_resolver_integration.py` pin the
  laziness contract: `selection_set` and `_selection_resolver` absent
  from `ds.__dict__` immediately after construction, and repeat
  access returns the same cached object
  ([#68](https://github.com/nmorabowen/STKO_to_python/pull/68)).
- `tests/unit/test_beam_extrude.py` (13 tests) covers the
  geometry-helper contract: vertex placement under identity / rotated
  frames, section-offset translation, cap winding, side-surface
  triangulation from the sweep loop, and degenerate profiles.
  Includes a real-fixture smoke check on the `elasticFrame/results`
  beam profile
  ([#69](https://github.com/nmorabowen/STKO_to_python/pull/69)).
- `tests/integration/test_beam_solids.py` (12 tests) exercises the
  `ds.plot.beam_solids` and `ds.plot.beam_solids_deformed` renderers
  on single-partition (`elasticFrame/results`, 3 beams) and
  multi-class (`elasticFrame/QuadFrame_results`, 75 beams + 625
  shells) fixtures. Pins the `(ax, meta)` contract, filter behavior,
  the `edge_color=None` switch, user-supplied 3D axes composition,
  the deformed `scale=0` short-circuit, and a monkeypatched
  displacement test that proves the scaled displacement flows into
  the rendered vertex positions
  ([#70](https://github.com/nmorabowen/STKO_to_python/pull/70),
  [#71](https://github.com/nmorabowen/STKO_to_python/pull/71)).

---

## [1.4.0] — 2026-05-12

Turns the v1.3.0 `*LOCAL_AXES` parsing into a usable rotation:
public `quaternion_to_rotation_matrix` helper and reader-side
conveniences. Round-trips ``localForce`` against the
OpenSees-emitted ``force`` (global) recorder to machine precision
on the bundled fixture.

### Added

- **`STKO_to_python.quaternion_to_rotation_matrix(q)`** — pure-numpy
  helper accepting a single quaternion (shape `(4,)`) or a batch
  (`(N, 4)`) in `(qw, qx, qy, qz)` order and returning the
  corresponding `(3, 3)` / `(N, 3, 3)` rotation matrix. Normalizes
  the input internally so STKO's 6-digit quaternions still produce
  an exactly orthogonal matrix. Convention: `v_global = R @ v_local`
  ([#65](https://github.com/nmorabowen/STKO_to_python/pull/65)).
- **`CDataReader.rotation_matrix(element_id)`** — convenience
  wrapping the `local_axes` lookup + `quaternion_to_rotation_matrix`
  for a single element ([#65](https://github.com/nmorabowen/STKO_to_python/pull/65)).
- **`CDataReader.rotation_matrices(element_ids=None)`** — batched
  version returning `(ids: (N,), R: (N, 3, 3))` aligned row-for-row.
  Powers vectorized rotation of `(n_steps, n_elements, 3)` arrays
  via `np.einsum`
  ([#65](https://github.com/nmorabowen/STKO_to_python/pull/65)).
- New module `STKO_to_python.model.transforms` housing the
  rotation utilities ([#65](https://github.com/nmorabowen/STKO_to_python/pull/65)).
- Cookbook recipe [08 — Rotate beam-local forces and moments to the
  global frame](docs/cookbook/08-rotate-beam-forces-to-global.md).
  Demonstrates the round-trip end-to-end against the
  `elasticFrame_mesh_results` fixture and includes a vectorized
  batch-rotation pattern.

### Fixed

- Two broken cross-references in `docs/selector_and_mask_pipeline.md`
  (`../api/element-results.md` should have been `api/element-results.md`
  after the file moved to the docs root in [#55](https://github.com/nmorabowen/STKO_to_python/pull/55)).
  `mkdocs build --strict` was aborting; it now exits clean
  ([#66](https://github.com/nmorabowen/STKO_to_python/pull/66)).

### Test coverage

- 10 new unit tests for `quaternion_to_rotation_matrix` (identity,
  180° rotations, batched/single equivalence, orthogonality,
  off-unit-input normalization, conjugate inverse, error paths).
- 7 new unit tests for the reader convenience methods plus a real
  fixture round-trip that rotates `localForce` and verifies it
  matches `force` to machine precision on both a column (non-trivial
  rotation) and a beam (identity rotation).
- Total unit suite: **630 passed**, 1 pre-existing skip.

---

## [1.3.0] — 2026-05-11

Adds full parsing of the `.cdata` sidecar file (every section, not just
`*SELECTION_SET`) and turns the new metadata into a user-facing
"select by STKO geometry/property name" workflow on `ElementSelector`.
Verified end-to-end against the bundled 95k-line example.

### Added

- **`CDataReader` now parses every `.cdata` section** in one pass per
  partition. New `@cached_property` accessors on the reader (and
  reachable via `dataset.cdata.X`):
  - `local_axes` — `{elem_id: ndarray([qw, qx, qy, qz])}` per-element
    rotation quaternion ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `section_offsets` — `{elem_id: ndarray([yOff, zOff])}` in
    element-local coords ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `element_info` — `{elem_id: ElementInfo}` parent geometry,
    sub-geometry type, and physical/element property names
    ([#58](https://github.com/nmorabowen/STKO_to_python/pull/58))
  - `beam_profiles` — `{profile_id: BeamProfile}` 2D cross-section
    geometry (points, triangulation, edge outline, sweep indices)
    ([#59](https://github.com/nmorabowen/STKO_to_python/pull/59))
  - `beam_profile_assignments` — `{elem_id: [(profile_id, weight), ...]}`
    element-to-profile mapping ([#59](https://github.com/nmorabowen/STKO_to_python/pull/59))
- **`ElementInfo`** and **`BeamProfile`** frozen dataclasses,
  re-exported at the top level (`STKO_to_python.ElementInfo`,
  `STKO_to_python.BeamProfile`).
- **Four new `ElementSelector` anchor primitives** resolving against
  `.cdata` `*ELEMENT_INFO` ([#60](https://github.com/nmorabowen/STKO_to_python/pull/60)):
  - `.of_geometry(name)` — STKO parent geometry name
  - `.of_physical_property(name)` — material/section property name
  - `.of_element_property(name)` — element class property name
  - `.of_sub_geom_type(t)` — `"Edge"` / `"Face"` / `"Solid"`
- **`STKO_to_python.model.cdata_format.CDataFormatPolicy`** — pure-
  functional policy class mirroring `MpcoFormatPolicy`. Owns the six
  section marker tokens plus `known_markers()`, `is_section_marker()`,
  `is_any_marker()` ([#61](https://github.com/nmorabowen/STKO_to_python/pull/61)).
- Cookbook recipe
  [`07-select-by-geometry-and-property.md`](docs/cookbook/07-select-by-geometry-and-property.md) —
  end-to-end walk-through of the new selector anchors.

### Changed

- **Selection-set id-list parsing is now wrap-width agnostic.** The
  previous parser assumed exactly 10 ids per line via `(n + 9) // 10`;
  the new `_consume_ids` helper scans forward consuming integers until
  the expected count is reached, so any wrap width parses correctly
  ([#61](https://github.com/nmorabowen/STKO_to_python/pull/61)).
- **`.cdata` parse failures fail fast.** Previously a malformed file
  silently returned `[]`, producing a partial `dataset.selection_set`
  that broke downstream queries far from the cause. Now the parser
  logs the offending file via `logger.exception` and re-raises, so
  dataset construction fails loudly at the source
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).

### Fixed

- `.cdata` files are now opened with `encoding="utf-8", errors="replace"` —
  fixes a `UnicodeDecodeError` on Windows when files contain non-ASCII
  bytes (`cp1252` was the previous default)
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Latent `UnboundLocalError` when a selection set had `NNODES=0` and
  `NELEMENTS>0` (the element parsing branch referenced an unbound
  `nodes_end_line`). Not triggered by current STKO output but real
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- `print_selection_set_names` now routes through the module logger
  (`logger.info`) instead of `print`, matching every sibling reader
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Dropped a wasteful `np.array(lines, dtype=str)` wrapper in the
  selection-set parser — it allocated a fixed-width `U`-array sized
  to the longest line for no speedup ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).
- Removed a stale docstring on `_extract_selection_set_ids`
  referencing a `fileName` argument that the method does not take
  ([#57](https://github.com/nmorabowen/STKO_to_python/pull/57)).

### Test coverage

- 30 new unit tests for the `.cdata` parser (selection-set, every new
  section, the width-agnostic id consumer, format policy, error paths).
- 13 new unit tests for the `ElementSelector` anchors backed by
  `*ELEMENT_INFO`.
- Real-file smoke checks against the bundled `QuadFrame_results` and
  the 95k-line `examples/New/results_nodes.mpco.cdata` sidecar.
- Total unit suite: **613 passed** (was 566 before the stack), 1
  pre-existing skip.

---

## [1.2.0] — 2026-05-09

### Added

- **`ElementSelector`** — lazy, chainable, composable element-id
  queries with spatial primitives (`within_box`, `within_distance`,
  `nearest_to`, `on_plane`, `near_line`, `centroid_in`, `where`)
  and Boolean composition (`&` / `|` / `~`).
- **`NodeSelector` + `NodeResultMask`** — node-side equivalents,
  with the same anchor / filter-op / boolean-algebra design.
- **`ResultMask`** — per-element boolean mask built from value
  conditions over a time window, applied via `er[mask]`.
- Top-level [selector + mask pipeline guide](docs/selector_and_mask_pipeline.md)
  promoted to a primary doc.
- Cookbook recipes 05 (element pipeline) and 06 (node pipeline).
- Test CI workflow, benchmark CI workflow.

### Changed

- Group B file renames: managers are now on canonical paths
  (`nodes.node_manager`, `elements.element_manager`, etc.).
  Legacy paths emit `DeprecationWarning` via PEP 562 `__getattr__`
  shims and continue to import.
- `Gauss` / `shape` modules relocated from `utilities/` to `format/`.
- `__slots__` applied to `NodalResults` and `_ResultView` for memory
  footprint (Phase 4a).

---

## [1.1.0] — 2026-04 / 2026-05

### Added

- Layered shells support (MPCO recorder + `ElementResults`).
- Per-element fixtures and demo notebooks under `examples/`.
- API stubs filled in across the doc tree.
- Spatial-query polish for `ElementResults`.

### Changed

- Examples nav and index restructuring; element-results API and
  navigation overhauled.

See the merge log for full detail:
`git log --merges v1.0.0..v1.1.0`.

---

## [1.0.0] — initial public release

Initial release. `MPCODataSet` opens MPCO HDF5 recorder output;
`NodalResults` / `ElementResults` expose result DataFrames;
`Aggregator` provides cross-element/cross-step engineering reductions
(interstory drift, envelopes, residuals, orbits); `NodalResultsPlotter`
produces publication figures. Multi-partition MP output is read
transparently.

See `git log --merges v1.0.0` for the full pre-1.0 history.

[Unreleased]: https://github.com/nmorabowen/STKO_to_python/compare/v1.8.0...HEAD
[1.8.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/nmorabowen/STKO_to_python/releases/tag/v1.0.0
