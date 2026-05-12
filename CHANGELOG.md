# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/) as
spelled out in [`CLAUDE.md`](CLAUDE.md#versioning-policy):

- **MAJOR** (`vX.0.0`) — breaking changes to the public API.
- **MINOR** (`v1.X.0`) — new backward-compatible features.
- **PATCH** (`v1.x.Y`) — bug fixes, docs, internal refactors with no API change.

## [Unreleased]

_Nothing pending._

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

[Unreleased]: https://github.com/nmorabowen/STKO_to_python/compare/v1.4.0...HEAD
[1.4.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/nmorabowen/STKO_to_python/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/nmorabowen/STKO_to_python/releases/tag/v1.0.0
